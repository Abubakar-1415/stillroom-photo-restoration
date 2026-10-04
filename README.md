---
title: Stillroom Photo Restoration
emoji: ✨
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
---

# Stillroom — AI Photo Restoration

A small, local-first photo restoration studio. Upload a JPEG, PNG, or WEBP
image, preview the restored result, and download a high-resolution PNG.

The enhancement pipeline combines Real-ESRGAN upscaling with optional,
conservative CodeFormer face restoration. AI may estimate fine detail that
cannot be recovered from a blurry original. Set **AI detail blend** to **0%**
to skip generative restoration and return a conventional 4× upscale of the
original.

## Run locally on Windows

Python 3.11 is recommended. Install a CPU build of PyTorch, then the app
dependencies:

```powershell
cd "C:\path\to\photo enhancer"
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the Real-ESRGAN `RealESRGAN_x4plus.pth` checkpoint in `models`, or set
`REALESRGAN_WEIGHTS` to its path. Make the CodeFormer source checkout available
in `CodeFormer-src`, or set `CODEFORMER_ROOT` to its path. The CodeFormer
checkpoint is downloaded by CodeFormer on first use.

Start the app:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Open <http://127.0.0.1:8001/studio>. If port 8001 is occupied, choose another
port and use it in the URL.

## Deploy on Hugging Face Spaces

1. Create a **public Docker Space** on Hugging Face.
2. Upload or push this project's tracked files to the Space repository.
3. Wait for the Docker build and model initialization to finish, then share the
   Space URL ending in `.hf.space/studio`.

The Docker image downloads the Real-ESRGAN checkpoint during its build. The
CodeFormer source is cloned from its upstream `v0.1.0` release. CodeFormer
downloads its own face-restoration checkpoints when face restoration is first
used. Model checkpoints and local environment files are excluded from this
project's Git history. Free CPU hosting is suitable for a demo, but image
restoration may be slow; the first face-restoration request also needs to
download its model checkpoints.

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
