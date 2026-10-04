FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CODEFORMER_ROOT=/opt/codeformer \
    REALESRGAN_WEIGHTS=/app/models/RealESRGAN_x4plus.pth \
    REALESRGAN_TILE=256

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user \
    && mkdir -p /app/models /opt/codeformer \
    && chown -R user:user /app /opt/codeformer

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && python -m pip install -r requirements.txt

COPY --chown=user:user app.py .

ADD --chown=user:user https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth /app/models/RealESRGAN_x4plus.pth

USER user

RUN git clone --depth 1 --branch v0.1.0 https://github.com/sczhou/CodeFormer.git /opt/codeformer

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
