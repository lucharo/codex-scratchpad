FROM python:3.11-slim

ENV UV_LINK_MODE=copy
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update \ 
    && apt-get install -y --no-install-recommends curl \ 
    && rm -rf /var/lib/apt/lists/* \ 
    && curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

CMD ["uv", "run", "python", "-m", "src.bot"]
