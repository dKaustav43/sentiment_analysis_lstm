FROM python:3.14-trixie

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv sync --locked

COPY . . 

EXPOSE 8000 
# port for fastapi

CMD ["uv", "run", "uvicorn", "sentiment_analysis.inference_api.py", "--host", "0.0.0.0", "--port", "8000"]
