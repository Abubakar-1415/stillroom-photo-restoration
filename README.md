---
title: Stillroom Photo Restoration
emoji: ✨
colorFrom: gray
colorTo: yellow
sdk: docker
---

# Stillroom — AI Photo Restoration

A server-backed photo restoration studio. Upload a JPEG, PNG, or WEBP image,
run Real-ESRGAN upscaling and CodeFormer face restoration, and download the
result as a PNG.

The hosted Space runs the FastAPI application on CPU. Uploaded images are sent
to the Hugging Face-hosted app for processing and are not kept as a permanent
gallery; do not upload sensitive photos. CodeFormer can reconstruct plausible
facial detail, but it cannot know the exact details lost to blur and may alter
the person's appearance. Its default fidelity weight is 0.1 for stronger
restoration. Real-ESRGAN provides a 4× upscale.

## Open the public demo

Open <https://huggingface.co/spaces/Abusiddiq/sidd>. The first request may take
longer while the free CPU Space starts and loads its models.

## Run the FastAPI version locally

Python 3.11 is recommended. Install a CPU build of PyTorch and the Python
dependencies:

```powershell
cd "C:\path\to\photo enhancer"
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Open <http://127.0.0.1:8001/studio>. The FastAPI app uses local
Real-ESRGAN and CodeFormer checkpoints; set `REALESRGAN_WEIGHTS` and
`CODEFORMER_ROOT` if they are not at the default paths.

## Public Space model file

`Dockerfile` builds the FastAPI Space with the Real-ESRGAN upscaler and
CodeFormer face-restoration project. The browser-only `index.html` and ONNX
model are retained for local static-demo reference; the live Space uses the
server-backed FastAPI interface.

## API

- `GET /` returns the health summary for API clients and the studio page for
  browsers.
- `GET /studio` serves the photo-restoration interface.
- `GET /health` reports whether the upscaling model has loaded.
- `POST /enhance` accepts a multipart image file with optional `strength` and
  `fidelity` values from `0` to `1`; it returns the restored image as PNG.

## Model and image notes

The app processes uploaded images in memory and temporary files needed by
CodeFormer; it does not maintain a permanent image gallery. Because the public
demo sends images to its hosting server for inference, avoid uploading
sensitive photos.

Real-ESRGAN, CodeFormer, and their checkpoints are third-party projects and
assets. Review their upstream licenses and terms before redistribution or
commercial use.
