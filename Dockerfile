FROM python:3.11-slim

WORKDIR /app

# Install deps first — separate layer so it's cached on rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download weights at build time so runtime starts instantly
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download( \
    repo_id='taanmaay/GPT-2-124M-weights', \
    filename='gpt_pytorch_weights.pth', \
    local_dir='/app/weights' \
)"

# Copy application code
COPY app.py blocks.py model.py sampling.py config.py ./
COPY .streamlit/ .streamlit/

EXPOSE 7860

CMD ["streamlit", "run", "app.py"]
