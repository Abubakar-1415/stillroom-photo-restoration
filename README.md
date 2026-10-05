---
title: Stillroom Photo Restoration
emoji: ✨
colorFrom: gray
colorTo: yellow
sdk: static
---

# Stillroom — AI Photo Upscaling

A browser-based photo upscaling demo. Upload a JPEG, PNG, or WEBP image, run
Real-ESRGAN 4× upscaling, and download the result as a PNG.

The public Space runs inference in your browser. Uploaded images remain on
your device. Real-ESRGAN enlarges and sharpens images, but does not reliably
remove severe motion blur or recover the exact original details. The separate
FastAPI app can run CodeFormer face reconstruction locally. CodeFormer may
invent facial features that were not captured in the original.

The browser app also links to the official CodeFormer demo as a separate,
optional service for face reconstruction. Users must upload images directly
there; this app does not transmit them. CodeFormer may invent facial details,
and external service privacy and model terms apply.

## Open the public demo

Open <https://huggingface.co/spaces/Abusiddiq/sidd>. The first AI upscale
downloads the approximately 64 MB ONNX model into the browser cache; WebGPU is
used when supported, otherwise the app falls back to WebAssembly.

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

The Hugging Face account currently has no free CPU Space quota, so CodeFormer
cannot run as a hosted server on this account. `index.html` is the live static
browser demo. The included `Dockerfile` is for deployments with an available
server runtime; it is not used by the public static Space.

## API

- `GET /` returns the health summary for API clients and the studio page for
  browsers.
- `GET /studio` serves the photo-restoration interface.
- `GET /health` reports whether the upscaling model has loaded.
- `POST /enhance` accepts a multipart image file with optional `strength` and
  `fidelity` values from `0` to `1`; it returns the restored image as PNG.

## Model and image notes

The public static demo processes uploads in the browser and does not send them
to a server. The optional local FastAPI app processes images locally, including
temporary files needed by CodeFormer.

Real-ESRGAN, CodeFormer, and their checkpoints are third-party projects and
assets. Review their upstream licenses and terms before redistribution or
commercial use.
