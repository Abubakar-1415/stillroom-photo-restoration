---
title: Stillroom Photo Restoration
emoji: ✨
colorFrom: gray
colorTo: yellow
sdk: static
---

# Stillroom — AI Photo Restoration

A browser-based photo restoration studio. Upload a JPEG, PNG, or WEBP image,
preview a Real-ESRGAN 4× upscale, and download the result as a PNG.

The public Hugging Face Space is static, so inference runs in each visitor's
browser through ONNX Runtime Web and WebGPU when available. Uploaded
photographs are not sent to an inference server. The recommended starting
settings are **AI detail blend** at 100% and **Source preservation** at 45%.
Higher source preservation tightly limits AI color changes and can make the
result look almost unchanged. At 0% AI detail blend, inference is skipped and
the app returns only a conventional 4× interpolation of the original.
Real-ESRGAN estimates plausible detail; it cannot recover the exact original
from information lost to severe blur or motion blur.

## Open the public demo

Open <https://huggingface.co/spaces/Abusiddiq/sidd>. The first AI restoration
downloads the approximately 64 MB ONNX model into the browser cache; WebGPU is
used when the browser supports it, otherwise the app falls back to WebAssembly.
For reliable performance, use an up-to-date desktop browser and photos up to
1.5 megapixels.

## Run the FastAPI version locally

Python 3.11 is recommended. Install a CPU build of PyTorch and the Python
dependencies:

```powershell
cd "C:\path\to\photo enhancer"
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Open <http://127.0.0.1:8001/studio>. The FastAPI version uses local
Real-ESRGAN and CodeFormer checkpoints; set `REALESRGAN_WEIGHTS` and
`CODEFORMER_ROOT` if they are not at the default paths.

## Public Space model file

`index.html` is the static browser demo. `RealESRGAN_x4plus.onnx` is generated
from the original Real-ESRGAN checkpoint for browser inference. The ONNX model
is uploaded to the Hugging Face Space separately; model binaries, local
environment files, and original checkpoints are excluded from the GitHub
repository.

## API

- `GET /` returns the health summary for API clients and the studio page for
  browsers.
- `GET /studio` serves the photo-restoration interface.
- `GET /health` reports whether the upscaling model has loaded.
- `POST /enhance` accepts a multipart image file with optional `strength` and
  `fidelity` values from `0` to `1`; it returns the restored image as PNG.

## Model and image notes

The app processes uploaded images in memory and temporary files needed by
CodeFormer. It does not maintain an image gallery or save uploads as a
permanent collection. Public demo links can be used by anyone who can access
them, so avoid uploading sensitive photos.

Real-ESRGAN, CodeFormer, and their checkpoints are third-party projects and
assets. Review their upstream licenses and terms before redistribution or
commercial use.
