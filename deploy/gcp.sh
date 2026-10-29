#!/usr/bin/env bash
# The distributed setup on Google Cloud: MongoDB in a container on a VM, and the workers, the
# loader, the monitor and the report as Cloud Run jobs that reach it over the VPC. See
# "Distributed workers" in readme.md.
#
#   deploy/gcp.sh up                  create everything and push the worker image
#   deploy/gcp.sh image               push the worker image again, after changing the code
#   deploy/gcp.sh load ARGS...        load a DAG, with the arguments of the load command
#   deploy/gcp.sh run TASKS THREADS   run TASKS x THREADS workers, with the monitor alongside
#   deploy/gcp.sh report              print the report on the last run
#   deploy/gcp.sh down                delete everything again
set -euo pipefail

PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
REGION=${REGION:-europe-west1}
ZONE=${ZONE:-$REGION-b}
IMAGE=$REGION-docker.pkg.dev/$PROJECT/dag-experiments/worker
VM=dag-mongo
IN_REGION=(--project "$PROJECT" --region "$REGION")
TAG_FILE=${TMPDIR:-/tmp}/dag-experiments-image

up() {
    gcloud artifacts repositories create dag-experiments --project "$PROJECT" --location "$REGION" \
        --repository-format docker
    gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
    _push

    # MongoDB runs from the same image as on a development system, on Container-Optimized OS.
    # MongoDB takes up to 80% of its open-files limit in connections, so the container gets a
    # high one. In case `down` is forgotten, the VM deletes itself after 6 hours.
    gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" \
        --machine-type c3-standard-8 --image-family cos-stable --image-project cos-cloud \
        --boot-disk-size 50GB --boot-disk-type pd-balanced \
        --max-run-duration 6h --instance-termination-action DELETE \
        --metadata startup-script='docker run --detach --name mongo --restart always --publish 27017:27017 --ulimit nofile=100000:100000 mongo:8.2'

    # The jobs join the default network's subnet through Direct VPC egress, and reach the VM on
    # its internal address, which the default-allow-internal firewall rule lets through.
    local address
    address=$(gcloud compute instances describe "$VM" --project "$PROJECT" --zone "$ZONE" \
        --format 'value(networkInterfaces[0].networkIP)')
    local job=("${IN_REGION[@]}" --image "$(cat "$TAG_FILE")" --network default --subnet default
        --vpc-egress private-ranges-only --set-env-vars "MONGO_URL=mongodb://$address:27017"
        --max-retries 0 --task-timeout 3h)
    gcloud run jobs create dag-workers "${job[@]}" --cpu 1 --memory 1Gi
    # Loading the 1.2M-node graph, and reading it back for the report, hold it in memory a few times.
    gcloud run jobs create dag-tool "${job[@]}" --cpu 2 --memory 8Gi
}

image() {
    _push
    gcloud run jobs update dag-workers "${IN_REGION[@]}" --image "$(cat "$TAG_FILE")"
    gcloud run jobs update dag-tool "${IN_REGION[@]}" --image "$(cat "$TAG_FILE")"
}

load() {
    _tool "load,$(IFS=,; echo "$*")"
}

run() {
    local tasks=$1 threads=$2
    gcloud run jobs execute dag-tool "${IN_REGION[@]}" --async --args monitor
    gcloud run jobs execute dag-workers "${IN_REGION[@]}" --wait --tasks "$tasks" --args "work,--threads,$threads"
}

report() {
    _tool report
}

down() {
    # Each step goes on when the one before fails, so `down` also cleans up after a broken `up`.
    gcloud run jobs delete dag-workers "${IN_REGION[@]}" --quiet || true
    gcloud run jobs delete dag-tool "${IN_REGION[@]}" --quiet || true
    gcloud compute instances delete "$VM" --project "$PROJECT" --zone "$ZONE" --quiet || true
    gcloud artifacts repositories delete dag-experiments --project "$PROJECT" --location "$REGION" --quiet || true
}

# Jobs pin the image they were given, so every push gets its own tag.
_push() {
    local tag
    tag=$IMAGE:$(date +%Y%m%d-%H%M%S)
    docker build --tag "$tag" .
    docker push "$tag"
    echo "$tag" > "$TAG_FILE"
}

# Run the tool job with the given comma-separated arguments and print what it printed.
_tool() {
    local execution output=""
    execution=$(gcloud run jobs execute dag-tool "${IN_REGION[@]}" --wait --args "$1" --format 'value(metadata.name)')
    # The output reaches Cloud Logging a little after the execution ends.
    for _ in $(seq 30); do
        output=$(gcloud logging read --project "$PROJECT" --order asc --format 'value(textPayload)' \
            "resource.type=cloud_run_job AND labels.\"run.googleapis.com/execution_name\"=$execution")
        [[ -n $output ]] && break
        sleep 5
    done
    echo "$output"
}

"$@"
