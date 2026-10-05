---
title: Stillroom Photo Restoration
emoji: ✨
colorFrom: gray
colorTo: yellow
sdk: static
---

# Stillroom — AI Photo Restoration

The local FastAPI studio offers a staged workflow: optional noise reduction,
Restormer motion deblurring, optional CodeFormer generative face reconstruction,
then Real-ESRGAN 4× upscaling. Review the side-by-side result before downloading.
Face reconstruction is opt-in because it can invent facial details.

Restormer is trained on benchmark motion blur and may perform poorly on real
phone blur; it cannot reliably recover severe blur. Generative face restoration
can produce plausible details, not evidence of the exact original appearance.
The public Hugging Face Space remains a browser-only Real-ESRGAN upscaler and
does not run this server pipeline.

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

Open <http://127.0.0.1:8001/studio>. The FastAPI app uses local Real-ESRGAN
and CodeFormer checkpoints. Set `REALESRGAN_WEIGHTS`, `CODEFORMER_ROOT`, and
`RESTORMER_ROOT` if they are not at their defaults. On the first deblur
request it downloads the public
Restormer motion-deblurring checkpoint (about 105 MB) from Hugging Face into
the local Hub cache. A Restormer source checkout is also required:

```powershell
git clone --depth 1 https://github.com/swz30/Restormer.git .\Restormer
```

### Optional OpenAI image editing

The local studio also offers an optional OpenAI image-editing engine. It is
disabled unless the backend has an API key. Set the key only in the backend's
environment (never in `index.html`, browser JavaScript, or a public repository)
and start the server in that same PowerShell session:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_IMAGE_MODEL = "gpt-image-2.5-sunburst"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

The default model can be changed with `OPENAI_IMAGE_MODEL` if your API account
has access to a different supported image-edit model. In the studio, select
**OpenAI cloud image edit** and check the explicit consent box before
processing. The image is sent to OpenAI only after that confirmation; API
requests may incur charges. OpenAI's API data policy describes possible
retention for abuse monitoring, so do not upload sensitive photos. Although
the prompt asks the model to preserve the source faithfully, generative edits
can change faces or invent details and cannot prove what the camera captured.
Review the result before downloading. The normal local restoration option
continues to use this app's server and does not call OpenAI.

## Public Space model file

The Hugging Face account currently has no free CPU Space quota. The live
`index.html` Space therefore remains the browser-only upscaler; the Dockerfile
is for a server host with sufficient compute. Server requests upload the photo
to that host for processing.

## API

- `GET /` returns the health summary for API clients and the studio page for
  browsers.
- `GET /studio` serves the photo-restoration interface.
- `GET /health` reports whether the upscaling model has loaded.
- `POST /enhance` accepts an image and optional `strength`, `fidelity`,
  `denoise_strength`, `deblur`, and `face_recovery` settings; it returns the
  processed image as PNG. `face_recovery` defaults to false. Set `engine=openai`
  and `openai_consent=true` to use the optional cloud-edit route; it requires
  `OPENAI_API_KEY` on the backend.

## Model and image notes

The public static demo processes uploads in the browser and does not send them
to a server. The FastAPI app sends uploads to its own server process and uses
temporary files for optional CodeFormer processing. Restormer and CodeFormer
are separate optional stages; neither can guarantee exact recovery of details
lost from a single blurry frame.

Real-ESRGAN, CodeFormer, and their checkpoints are third-party projects and
assets. Review their upstream licenses and terms before redistribution or
commercial use.
