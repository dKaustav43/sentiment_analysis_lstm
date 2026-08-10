FROM python:3.14-trixie

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv sync --locked

COPY . . 

EXPOSE 8000 
# port for fastapi

CMD ["sleep", "infinity"]