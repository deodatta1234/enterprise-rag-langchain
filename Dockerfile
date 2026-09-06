FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Cache dependencies and the model independently of application source changes.
COPY pyproject.toml ./
RUN python -c "import subprocess, sys, tomllib; dependencies = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; subprocess.check_call([sys.executable, '-m', 'pip', 'install', *dependencies])"

ARG RERANKER_MODEL_ID=cross-encoder/ms-marco-MiniLM-L6-v2
ENV RAG_RERANKER_MODEL=/opt/models/reranker

# Download at build time and export weights, tokenizer, and configuration together.
# A temporary download cache avoids storing a second copy in the image.
RUN python -c "import os, tempfile; from sentence_transformers import CrossEncoder; cache = tempfile.TemporaryDirectory(); model = CrossEncoder(os.environ['RERANKER_MODEL_ID'], device='cpu', max_length=512, cache_folder=cache.name); model.save_pretrained(os.environ['RAG_RERANKER_MODEL']); cache.cleanup()"

# Runtime model loading must use the bundled files, with no Hub requests.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
RUN python -c "import os; from sentence_transformers import CrossEncoder; model = CrossEncoder(os.environ['RAG_RERANKER_MODEL'], device='cpu', local_files_only=True, max_length=512); scores = model.predict([('How long is probation?', 'The probation period is 90 days.')], show_progress_bar=False); assert len(scores) == 1"

COPY src ./src
RUN pip install --no-deps .

EXPOSE 8000

CMD ["uvicorn","rag_chatbot.api:app","--host", "0.0.0.0","--port", "8000"]
