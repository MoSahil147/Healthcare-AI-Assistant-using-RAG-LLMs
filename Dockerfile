# recipe to bake a cake
# 1, runs first build image

# 1, start with Python, pulling a light weight Python 
FROM python:3.11-slim

# 2. pin uv to the same version used locally so builds are reproducible
# uv package manegr 
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

# /app directory of the container
WORKDIR /app

# 3. copy dependency files first so Docker can cache this layer
# the install step only reruns when pyproject.toml or uv.lock actually change
COPY pyproject.toml uv.lock ./

# 4. install exact locked versions, skip dev dependencies in production
RUN uv sync --frozen --no-dev

# 5. copy the rest of the application code
COPY . .

# 6. create the chroma folder, so it exitst before the server starts
RUN mkdir -p store/chroma

# telling docker that this container uses port 8000
EXPOSE 8000

# 7 start the server
# --host 0.0.0.0  = accept connections from outside the container
# --port 8000     = run on port 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
