FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
WORKDIR /app
# Dependencies first, so changing the code doesn't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
ENV PATH=/app/.venv/bin:$PATH
# Print each line as it happens, so logs arrive in order.
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "dag_experiments.distributed"]
