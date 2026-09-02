FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the application dependencies.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

EXPOSE 8000

CMD ["uvicorn","rag_chatbot.api:app","--host", "0.0.0.0","--port", "8000"]