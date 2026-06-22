FROM python:3.11-slim

# pin uv to the same version used locally so builds are reproducible
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

WORKDIR /app

# copy dependency files first so Docker can cache this layer
# the install step only reruns when pyproject.toml or uv.lock actually change
COPY pyproject.toml uv.lock ./

# install exact locked versions, skip dev dependencies in production
RUN uv sync --frozen --no-dev

# copy the rest of the application code
COPY . .

RUN mkdir -p store/chroma

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
