# Runs PartCheck anywhere Docker runs (and as a free Hugging Face "Docker" Space, which has more memory than Streamlit Community Cloud).
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PARTCHECK_CACHE=/tmp/partcheck HF_HOME=/tmp/hf
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=user . .
USER user
EXPOSE 7860
# XSRF/CORS are switched off ONLY because Hugging Face shows the app inside an iframe, where uploads are otherwise refused (HTTP 403).
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true --server.enableXsrfProtection=false --server.enableCORS=false"]
