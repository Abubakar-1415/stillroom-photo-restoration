import io
import importlib.util
import os
import tempfile
import threading
import asyncio
import base64
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
import shutil
import subprocess
import sys

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from huggingface_hub import hf_hub_download
import requests
from safetensors.torch import load_file



# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_REALESRGAN_WEIGHTS = (
    BASE_DIR / "models" / "RealESRGAN_x4plus.pth"
)
LOCAL_REALESRGAN_WEIGHTS = Path(
    r"C:\Users\siddi\Downloads\weights\RealESRGAN_x4plus.pth"
)
if (
    not DEFAULT_REALESRGAN_WEIGHTS.exists()
    and LOCAL_REALESRGAN_WEIGHTS.exists()
):
    DEFAULT_REALESRGAN_WEIGHTS = LOCAL_REALESRGAN_WEIGHTS

DEFAULT_CODEFORMER_ROOT = BASE_DIR / "CodeFormer-src"
LOCAL_CODEFORMER_ROOT = BASE_DIR.parent / "CodeFormer-src"
if (
    not (DEFAULT_CODEFORMER_ROOT / "inference_codeformer.py").exists()
    and (LOCAL_CODEFORMER_ROOT / "inference_codeformer.py").exists()
):
    DEFAULT_CODEFORMER_ROOT = LOCAL_CODEFORMER_ROOT

DEFAULT_RESTORMER_ROOT = BASE_DIR / "Restormer"
RESTORMER_ROOT = Path(
    os.environ.get(
        "RESTORMER_ROOT",
        str(DEFAULT_RESTORMER_ROOT),
    )
)
REALESRGAN_WEIGHTS = Path(
    os.environ.get(
        "REALESRGAN_WEIGHTS",
        str(DEFAULT_REALESRGAN_WEIGHTS),
    )
)
CODEFORMER_ROOT = Path(
    os.environ.get(
        "CODEFORMER_ROOT",
        str(DEFAULT_CODEFORMER_ROOT),
    )
)
REALESRGAN_TILE = int(os.environ.get("REALESRGAN_TILE", "256"))
RESTORMER_REPO_ID = "mlx-community/Restormer-motion-deblurring-fp32"
RESTORMER_TILE_SIZE = 192
RESTORMER_TILE_OVERLAP = 32
OPENAI_IMAGE_MODEL = os.environ.get(
    "OPENAI_IMAGE_MODEL",
    "gpt-image-2.5-sunburst",
)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Image Enhancer API",
    description="Staged noise reduction, deblurring, face recovery, and upscaling API",
    version="1.0.0",
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("AI IMAGE ENHANCER BACKEND")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Real-ESRGAN weights: {REALESRGAN_WEIGHTS}")
print(f"CodeFormer root: {CODEFORMER_ROOT}")
print(f"Restormer root: {RESTORMER_ROOT}")
print("=" * 60)


# ============================================================
# MODEL
# ============================================================

upsampler = None
restormer = None
restormer_lock = threading.Lock()


def initialize_models():
    global upsampler

    print("\nLoading Real-ESRGAN...")

    if not REALESRGAN_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Real-ESRGAN weights not found:\n"
            f"{REALESRGAN_WEIGHTS}"
        )

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=str(REALESRGAN_WEIGHTS),
        model=model,
        tile=REALESRGAN_TILE,
        tile_pad=10,
        pre_pad=0,
        half=(DEVICE == "cuda"),
    )

    print("Real-ESRGAN loaded successfully.")


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    initialize_models()


# ============================================================
# BLUR DETECTION
# ============================================================

def calculate_blur_score(image: np.ndarray) -> float:
    """
    Higher value = sharper image.
    Lower value = blurrier image.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    return float(score)


# ============================================================
# CODEFORMER
# ============================================================

def run_codeformer(input_path: str, output_path: str, fidelity: float):
    output_dir = Path(output_path).parent / "codeformer_output"

    command = [
        sys.executable,
        str(CODEFORMER_ROOT / "inference_codeformer.py"),
        "--input_path", input_path,
        "--output_path", str(output_dir),
        "--fidelity_weight", str(fidelity),
        "--upscale", "1",
    ]

    subprocess.run(command, cwd=CODEFORMER_ROOT, check=True)

    generated = (
        output_dir
        / "final_results"
        / f"{Path(input_path).stem}.png"
    )

    if not generated.is_file():
        raise FileNotFoundError(
            f"CodeFormer output was not created: {generated}"
        )

    shutil.copyfile(generated, output_path)


def load_restormer():
    global restormer

    if restormer is not None:
        return restormer

    with restormer_lock:
        if restormer is not None:
            return restormer

        architecture_path = (
            RESTORMER_ROOT
            / "basicsr"
            / "models"
            / "archs"
            / "restormer_arch.py"
        )
        if not architecture_path.is_file():
            raise FileNotFoundError(
                "Restormer source was not found. Set RESTORMER_ROOT "
                f"to a Restormer checkout; expected {architecture_path}."
            )

        spec = importlib.util.spec_from_file_location(
            "_stillroom_restorer_arch",
            architecture_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Could not load Restormer architecture from {architecture_path}."
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Restormer(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            dual_pixel_task=False,
        )

        checkpoint_path = hf_hub_download(
            repo_id=RESTORMER_REPO_ID,
            filename="model.safetensors",
        )
        state_dict = load_file(checkpoint_path)
        pytorch_state = {
            name: (
                value.permute(0, 3, 1, 2).contiguous()
                if value.ndim == 4
                else value
            )
            for name, value in state_dict.items()
        }
        model.load_state_dict(pytorch_state, strict=True)
        model.to(DEVICE)
        model.eval()
        restormer = model

    return restormer


def run_restormer(image: np.ndarray) -> np.ndarray:
    model = load_restormer()
    height, width = image.shape[:2]
    output = np.empty_like(image)
    tile_size = RESTORMER_TILE_SIZE
    overlap = RESTORMER_TILE_OVERLAP

    for core_top in range(0, height, tile_size):
        for core_left in range(0, width, tile_size):
            core_bottom = min(core_top + tile_size, height)
            core_right = min(core_left + tile_size, width)
            tile_top = max(0, core_top - overlap)
            tile_left = max(0, core_left - overlap)
            tile_bottom = min(height, core_bottom + overlap)
            tile_right = min(width, core_right + overlap)

            tile_rgb = cv2.cvtColor(
                image[tile_top:tile_bottom, tile_left:tile_right],
                cv2.COLOR_BGR2RGB,
            )
            tensor = (
                torch.from_numpy(tile_rgb.transpose(2, 0, 1).copy())
                .float()
                .div_(255.0)
                .unsqueeze(0)
                .to(DEVICE)
            )
            pad_height = (-tensor.shape[-2]) % 8
            pad_width = (-tensor.shape[-1]) % 8
            if pad_height or pad_width:
                pad_mode = (
                    "reflect"
                    if tensor.shape[-2] > pad_height
                    and tensor.shape[-1] > pad_width
                    else "replicate"
                )
                tensor = torch.nn.functional.pad(
                    tensor,
                    (0, pad_width, 0, pad_height),
                    mode=pad_mode,
                )

            with torch.inference_mode():
                restored = model(tensor)

            restored = (
                restored[..., :tile_bottom - tile_top, :tile_right - tile_left]
                .clamp(0, 1)
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            restored_bgr = cv2.cvtColor(
                np.round(restored * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )

            output[core_top:core_bottom, core_left:core_right] = restored_bgr[
                core_top - tile_top:core_bottom - tile_top,
                core_left - tile_left:core_right - tile_left,
            ]

    return output


def process_image(
    image_bytes: bytes,
    strength: float,
    fidelity: float,
    denoise_strength: int,
    deblur: bool,
    face_recovery: bool,
) -> tuple[bytes, float, float, str, bool, int, str]:
    """Denoise, deblur, optionally reconstruct faces, then upscale."""

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Could not decode the uploaded image."
        )

    # --------------------------------------------------------
    # Blur analysis
    # --------------------------------------------------------

    blur_score = calculate_blur_score(image)

    applied_strength = strength
    processed = image.copy()
    stages = []

    if denoise_strength > 0:
        processed = cv2.fastNlMeansDenoisingColored(
            processed,
            None,
            h=denoise_strength,
            hColor=denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        stages.append("Noise reduction")

    if deblur:
        processed = run_restormer(processed)
        stages.append("Motion deblurring (Restormer)")

    face_restoration = "not_requested"
    if face_recovery:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "deblurred.png"
            restored_path = Path(temp_dir) / "face-reconstructed.png"
            if not cv2.imwrite(str(input_path), processed):
                raise RuntimeError(
                    "Could not prepare the image for CodeFormer face recovery."
                )

            run_codeformer(
                input_path=str(input_path),
                output_path=str(restored_path),
                fidelity=fidelity,
            )
            restored = cv2.imread(str(restored_path))
            if restored is None:
                raise RuntimeError(
                    "CodeFormer did not produce a readable face-recovery image."
                )
            if restored.shape != processed.shape:
                restored = cv2.resize(
                    restored,
                    (processed.shape[1], processed.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            processed = restored
            face_restoration = "success"
            stages.append("Generative face reconstruction (CodeFormer)")

    try:
        ai_output, _ = upsampler.enhance(
            processed,
            outscale=4,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Real-ESRGAN upscaling failed: {exc}"
        ) from exc
    stages.append("Real-ESRGAN 4x upscaling")

    original_upscaled = cv2.resize(
        image,
        (ai_output.shape[1], ai_output.shape[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )
    output = cv2.addWeighted(
        ai_output,
        applied_strength,
        original_upscaled,
        1.0 - applied_strength,
        0.0,
    )

    # --------------------------------------------------------
    # Encode final image
    # --------------------------------------------------------

    success, encoded = cv2.imencode(
        ".png",
        output,
    )

    if not success:
        raise RuntimeError(
            "Could not encode final image."
        )

    return (
        encoded.tobytes(),
        applied_strength,
        blur_score,
        face_restoration,
        deblur,
        denoise_strength,
        " > ".join(stages),
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root(request: Request):
    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(STUDIO_HTML)

    return {
        "status": "online",
        "service": "AI Image Enhancer API",
        "device": DEVICE,
    }


@app.get("/studio", response_class=HTMLResponse)
async def studio():
    return HTMLResponse(STUDIO_HTML)


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model_loaded": upsampler is not None,
        "deblur_model_loaded": restormer is not None,
        "device": DEVICE,
    }


@app.get("/capabilities")
async def capabilities():
    return {
        "openai_image_edit_available": bool(os.environ.get("OPENAI_API_KEY")),
        "openai_image_model": OPENAI_IMAGE_MODEL,
    }


class OpenAIConfigurationError(RuntimeError):
    pass


def edit_image_with_openai(
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> bytes:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIConfigurationError(
            "OpenAI image editing is not configured. Set OPENAI_API_KEY "
            "on the backend and restart it."
        )

    try:
        response = requests.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "model": OPENAI_IMAGE_MODEL,
                "prompt": (
                    "Restore this exact photograph as faithfully as possible. "
                    "Reduce motion blur and improve natural clarity while "
                    "preserving the same people, identity, age, facial "
                    "geometry, expression, pose, clothing, objects, "
                    "background, lighting, and framing. Do not beautify, "
                    "redesign, replace, or add objects or facial features. "
                    "Do not invent fine details that are not supported by "
                    "the input; when information is missing, keep the result "
                    "conservative and consistent with the source. Keep a "
                    "natural photographic look."
                ),
                "quality": "high",
                "output_format": "png",
                "n": "1",
            },
            files={
                "image": (
                    filename,
                    io.BytesIO(image_bytes),
                    content_type,
                )
            },
            timeout=(15, 300),
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        status_code = (
            exc.response.status_code
            if exc.response is not None
            else "network"
        )
        raise RuntimeError(
            f"OpenAI image edit request failed ({status_code})."
        ) from exc
    except ValueError as exc:
        raise RuntimeError("OpenAI returned an invalid response.") from exc

    data = payload.get("data")
    if not data or not data[0].get("b64_json"):
        raise RuntimeError("OpenAI returned no edited image.")

    try:
        result = base64.b64decode(data[0]["b64_json"], validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("OpenAI returned invalid image data.") from exc

    if not result:
        raise RuntimeError("OpenAI returned an empty image.")
    return result


# ============================================================
# ENHANCE ENDPOINT
# ============================================================

@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    fidelity: float = Form(
        0.1,
        ge=0.0,
        le=1.0,
    ),
    strength: float = Form(
        1.0,
        ge=0.0,
        le=1.0,
    ),
    denoise_strength: int = Form(
        2,
        ge=0,
        le=10,
    ),
    deblur: bool = Form(True),
    face_recovery: bool = Form(False),
    engine: str = Form("local"),
    openai_consent: bool = Form(False),
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image type. "
                "Use JPG, PNG or WEBP."
            ),
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        image_bytes = await file.read()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded file: {exc}",
        )

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Choose a file under 20 MB.",
        )

    if engine not in {"local", "openai"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported restoration engine.",
        )

    if engine == "openai":
        if not openai_consent:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Confirm that you want to send this image to OpenAI "
                    "before using cloud restoration."
                ),
            )
        try:
            result_bytes = await asyncio.to_thread(
                edit_image_with_openai,
                image_bytes,
                file.filename or "photo",
                file.content_type,
            )
        except OpenAIConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            print(f"OpenAI image editing failed: {type(exc).__name__}")
            raise HTTPException(
                status_code=502,
                detail="OpenAI image editing failed. Please try again.",
            ) from exc

        headers = {
            "X-Restoration-Engine": "openai",
            "X-Processing-Stages": (
                f"OpenAI image edit ({OPENAI_IMAGE_MODEL})"
            ),
        }
        return StreamingResponse(
            io.BytesIO(result_bytes),
            media_type="image/png",
            headers=headers,
        )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    try:

        (
            result_bytes,
            applied_strength,
            blur_score,
            restoration_status,
            deblur_used,
            applied_denoise,
            processing_stages,
        ) = process_image(
            image_bytes=image_bytes,
            strength=strength,
            fidelity=fidelity,
            denoise_strength=denoise_strength,
            deblur=deblur,
            face_recovery=face_recovery,
        )

    except Exception as exc:

        print(
            f"Image processing error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    headers = {
        "X-Face-Strength": f"{applied_strength:.2f}",
        "X-Blur-Score": f"{blur_score:.2f}",
        "X-Face-Restoration": restoration_status,
        "X-Deblur-Used": str(deblur_used).lower(),
        "X-Denoise-Strength": str(applied_denoise),
        "X-Processing-Stages": processing_stages,
    }

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/png",
        headers=headers,
    )


STUDIO_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#101419">
  <title>Stillroom — AI Photo Restoration</title>
  <style>
    :root {
      color-scheme: dark;
      --canvas: #101419;
      --surface: #171d23;
      --surface-raised: #1c242b;
      --line: rgba(226, 232, 236, .10);
      --muted: #929da5;
      --text: #f1f0e9;
      --accent: #c9aa75;
      --accent-light: #e4c998;
      --success: #91b49c;
      --danger: #e49b91;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      min-width: 320px;
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(ellipse at 50% -18%, rgba(109, 128, 137, .18), transparent 45%),
        var(--canvas);
    }
    button, input { font: inherit; }
    .shell { width: min(1140px, calc(100% - 48px)); margin: 0 auto; }
    .topbar {
      height: 76px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-mark {
      width: 34px; height: 34px; display: grid; place-items: center;
      border: 1px solid rgba(201, 170, 117, .55); border-radius: 11px;
      color: var(--accent-light); font-size: 17px;
    }
    .brand-name { font-size: 14px; font-weight: 650; letter-spacing: .13em; }
    .brand-sub { margin-top: 3px; color: var(--muted); font-size: 10px; letter-spacing: .12em; text-transform: uppercase; }
    .local-badge {
      display: inline-flex; align-items: center; gap: 8px; padding: 8px 11px;
      border: 1px solid var(--line); border-radius: 99px; color: #c5cdc9;
      font-size: 11px; letter-spacing: .04em;
    }
    .local-badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--success); box-shadow: 0 0 12px rgba(145, 180, 156, .55); }
    main { padding: 56px 0 70px; }
    .intro { max-width: 690px; margin: 0 auto 34px; text-align: center; }
    .eyebrow { color: var(--accent-light); font-size: 10px; font-weight: 650; letter-spacing: .22em; text-transform: uppercase; }
    h1 { margin: 14px 0 12px; font-family: Georgia, "Times New Roman", serif; font-size: clamp(38px, 6vw, 60px); font-weight: 400; letter-spacing: -.045em; line-height: 1.06; }
    h1 span { color: var(--accent-light); font-style: italic; }
    .intro p { margin: 0; color: #a7afb3; font-size: 14px; line-height: 1.75; }
    .workspace {
      display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 18px;
      align-items: start;
    }
    .panel { overflow: hidden; border: 1px solid var(--line); border-radius: 16px; background: rgba(23, 29, 35, .88); }
    .panel-heading { min-height: 55px; padding: 0 18px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--line); }
    .panel-title { font-size: 12px; font-weight: 620; letter-spacing: .035em; }
    .panel-note { color: var(--muted); font-size: 11px; }
    .image-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line); }
    .image-card { min-width: 0; background: #14191e; }
    .image-label { height: 43px; padding: 0 14px; display: flex; align-items: center; justify-content: space-between; color: #c6ccca; font-size: 11px; }
    .image-label span:last-child { color: var(--muted); font-size: 10px; }
    .image-stage { min-height: 284px; padding: 12px; display: grid; place-items: center; position: relative; }
    .image-stage img { display: block; width: 100%; max-height: 420px; object-fit: contain; border-radius: 8px; }
    .empty-state { width: 100%; min-height: 258px; padding: 24px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; color: var(--muted); }
    .empty-icon { width: 46px; height: 46px; margin-bottom: 14px; display: grid; place-items: center; border: 1px solid var(--line); border-radius: 14px; color: var(--accent-light); font-size: 19px; }
    .empty-title { color: #d5d8d5; font-size: 12px; }
    .empty-copy { max-width: 190px; margin-top: 7px; font-size: 11px; line-height: 1.6; }
    .upload-zone {
      margin: 14px; padding: 15px 17px; display: flex; align-items: center; gap: 13px;
      border: 1px dashed rgba(201, 170, 117, .38); border-radius: 11px;
      background: rgba(201, 170, 117, .035); cursor: pointer; transition: border-color .18s, background .18s;
    }
    .upload-zone:hover, .upload-zone.is-over { border-color: var(--accent-light); background: rgba(201, 170, 117, .09); }
    .upload-icon { width: 34px; height: 34px; flex: 0 0 auto; display: grid; place-items: center; border-radius: 9px; background: rgba(201, 170, 117, .11); color: var(--accent-light); }
    .upload-copy { min-width: 0; flex: 1; }
    .upload-copy strong { display: block; color: #e4e4df; font-size: 11px; font-weight: 580; }
    .upload-copy span { display: block; margin-top: 4px; color: var(--muted); font-size: 10px; }
    .browse { color: var(--accent-light); font-size: 10px; }
    .settings { padding: 17px; }
    .setting + .setting { margin-top: 23px; }
    .setting-top { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .setting-label { color: #e0e1dc; font-size: 11px; font-weight: 560; }
    .setting-value { color: var(--accent-light); font-variant-numeric: tabular-nums; font-size: 11px; }
    .setting-hint { margin-top: 6px; color: var(--muted); font-size: 10px; line-height: 1.55; }
    input[type="range"] { width: 100%; height: 3px; margin: 15px 0 3px; display: block; accent-color: var(--accent); cursor: pointer; }
    .scale-labels { display: flex; justify-content: space-between; color: #78828a; font-size: 9px; }
    .divider { height: 1px; margin: 20px 0; background: var(--line); }
    .enhance-button {
      width: 100%; min-height: 44px; display: flex; align-items: center; justify-content: center; gap: 9px;
      border: 0; border-radius: 9px; background: var(--accent); color: #181813;
      font-size: 11px; font-weight: 680; letter-spacing: .035em; cursor: pointer;
      transition: background .18s, transform .18s, opacity .18s;
    }
    .enhance-button:hover:not(:disabled) { background: var(--accent-light); transform: translateY(-1px); }
    .enhance-button:disabled { opacity: .42; cursor: not-allowed; }
    .button-spinner { width: 13px; height: 13px; border: 2px solid rgba(24, 24, 19, .28); border-top-color: #181813; border-radius: 50%; animation: spin .7s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .status { min-height: 34px; padding-top: 12px; color: var(--muted); font-size: 10px; line-height: 1.5; }
    .status.error { color: var(--danger); }
    .status.success { color: var(--success); }
    .privacy { display: flex; align-items: flex-start; gap: 8px; color: #87918f; font-size: 10px; line-height: 1.55; }
    .privacy span:first-child { color: var(--success); }
    .result-bar { min-height: 58px; padding: 10px 15px; display: flex; align-items: center; justify-content: space-between; gap: 12px; border-top: 1px solid var(--line); }
    .result-info { color: var(--muted); font-size: 10px; line-height: 1.6; }
    .download {
      padding: 9px 12px; border: 1px solid rgba(201, 170, 117, .44); border-radius: 8px;
      color: var(--accent-light); text-decoration: none; font-size: 10px; white-space: nowrap;
    }
    .download:hover { background: rgba(201, 170, 117, .08); }
    .footer { margin-top: 23px; display: flex; justify-content: center; gap: 9px; color: #747e84; font-size: 10px; }
    .footer span + span::before { content: "·"; margin-right: 9px; color: #586168; }
    [hidden] { display: none !important; }
    @media (max-width: 760px) {
      .shell { width: min(100% - 30px, 560px); }
      main { padding-top: 42px; }
      .workspace { grid-template-columns: 1fr; }
      .image-stage { min-height: 220px; }
      .empty-state { min-height: 195px; }
      .settings { padding: 17px; }
    }
    @media (max-width: 480px) {
      .shell { width: calc(100% - 24px); }
      .topbar { height: 66px; }
      .local-badge { padding: 7px 9px; font-size: 9px; }
      .intro { margin-bottom: 25px; }
      .intro p { font-size: 12px; }
      .image-grid { grid-template-columns: 1fr; }
      .image-stage { min-height: 190px; }
      .empty-state { min-height: 160px; }
      .footer { flex-wrap: wrap; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">✳</div>
        <div>
          <div class="brand-name">STILLROOM</div>
          <div class="brand-sub">Photo restoration studio</div>
        </div>
      </div>
      <div class="local-badge">Server-powered AI restoration</div>
    </header>

    <main>
      <section class="intro">
        <div class="eyebrow">A second life for your photographs</div>
        <h1>Bring the details <span>back.</span></h1>
        <p>Reduce noise and motion blur, then upscale. Optional generative face recovery can create plausible details, but those details may differ from the original.</p>
      </section>

      <section class="workspace" aria-label="Photo restoration workspace">
        <div class="panel">
          <div class="panel-heading">
            <div class="panel-title">Image workspace</div>
            <div class="panel-note">Original &nbsp; / &nbsp; Restored</div>
          </div>
          <div class="image-grid">
            <div class="image-card">
              <div class="image-label"><span>Original</span><span id="original-size">Waiting for image</span></div>
              <div class="image-stage" id="original-stage">
                <div class="empty-state">
                  <div class="empty-icon" aria-hidden="true">◇</div>
                  <div class="empty-title">Your photograph begins here</div>
                  <div class="empty-copy">Choose an image below to see it in the workspace.</div>
                </div>
                <img id="original-image" alt="Original uploaded photograph" hidden>
              </div>
            </div>
            <div class="image-card">
              <div class="image-label"><span>Restored</span><span id="result-size">Your finished image</span></div>
              <div class="image-stage">
                <div class="empty-state" id="result-empty">
                  <div class="empty-icon" aria-hidden="true">✧</div>
                  <div class="empty-title">Clarity, carefully restored</div>
                  <div class="empty-copy">Your enhanced image will appear here, ready to download.</div>
                </div>
                <img id="result-image" alt="AI-restored photograph" hidden>
              </div>
            </div>
          </div>
          <label class="upload-zone" id="drop-zone" for="file-input">
            <span class="upload-icon" aria-hidden="true">↑</span>
            <span class="upload-copy">
              <strong id="upload-name">Choose an image or drop it here</strong>
              <span>JPG, PNG or WEBP · sent to the server for processing</span>
            </span>
            <span class="browse">Browse files</span>
          </label>
          <input id="file-input" type="file" accept="image/jpeg,image/png,image/webp" hidden>
          <div class="result-bar">
            <div class="result-info" id="result-info">Your original stays untouched. The finished image is exported as a high-resolution PNG.</div>
            <a class="download" id="download-link" href="#" download="stillroom-restored.png" hidden>↓ &nbsp; Download PNG</a>
          </div>
        </div>

        <aside class="panel">
          <div class="panel-heading">
            <div class="panel-title">Restoration settings</div>
            <div class="panel-note">Fine tune</div>
          </div>
          <div class="settings">
            <div class="setting">
              <label class="setting-label" for="engine">Restoration engine</label>
              <select id="engine" class="engine-select" aria-describedby="engine-note">
                <option value="local">Local restoration pipeline</option>
                <option value="openai" id="openai-option" disabled>OpenAI cloud image edit</option>
              </select>
              <div class="setting-hint" id="engine-note">Local images stay on this server. OpenAI cloud editing requires a configured API key.</div>
            </div>
            <div class="setting" id="openai-consent-setting" hidden>
              <label class="setting-label"><input id="openai-consent" type="checkbox"> I agree to send this photo to OpenAI for AI editing.</label>
              <div class="setting-hint">The image will leave this app’s server. OpenAI API image data may be retained for up to 30 days for abuse monitoring. API usage may incur charges. The generated face may differ from the real person; review it before saving.</div>
            </div>
            <div id="local-settings">
            <div class="setting">
              <div class="setting-top">
                <label class="setting-label" for="strength">AI detail blend</label>
                <output class="setting-value" id="strength-value" for="strength">100%</output>
              </div>
              <div class="setting-hint">Blend the complete AI pipeline with the original. Higher values apply more denoising, deblurring, and upscale detail.</div>
              <input id="strength" type="range" min="0" max="100" value="100">
              <div class="scale-labels"><span>Original-only upscale</span><span>More AI detail</span></div>
            </div>
            <div class="setting">
              <div class="setting-top">
                <label class="setting-label" for="denoise">Noise reduction</label>
                <output class="setting-value" id="denoise-value" for="denoise">2</output>
              </div>
              <div class="setting-hint">Light denoising before deblurring. Increase only if the photo has visible grain.</div>
              <input id="denoise" type="range" min="0" max="10" value="2">
              <div class="scale-labels"><span>Off</span><span>Stronger</span></div>
            </div>
            <div class="setting">
              <label class="setting-label"><input id="deblur" type="checkbox" checked> Motion deblur (Restormer)</label>
              <div class="setting-hint">Trained for benchmark motion blur. Results vary on real phone photos, and severe blur may remain.</div>
            </div>
            <div class="setting">
              <label class="setting-label"><input id="face-recovery" type="checkbox"> Generative face recovery (CodeFormer)</label>
              <div class="setting-hint">Optional reconstruction for faces. It can invent facial details; review the preview carefully before downloading.</div>
            </div>
            <div class="setting">
              <div class="setting-top">
                <label class="setting-label" for="fidelity">Source preservation</label>
                <output class="setting-value" id="fidelity-value" for="fidelity">10%</output>
              </div>
              <div class="setting-hint">Used only when generative face recovery is enabled. Higher values favor identity preservation over visual quality.</div>
              <input id="fidelity" type="range" min="0" max="100" value="10">
              <div class="scale-labels"><span>Creative</span><span>Faithful</span></div>
            </div>
            </div>
            <div class="divider"></div>
            <button class="enhance-button" id="enhance-button" type="button" disabled>
              <span aria-hidden="true">✧</span><span id="button-label">Select a photo to begin</span>
            </button>
            <div class="status" id="status" role="status" aria-live="polite">Choose a photo to prepare your restoration.</div>
            <div class="divider"></div>
            <div class="privacy"><span aria-hidden="true">!</span><span id="privacy-copy">Your image is sent to this app's server for local AI processing. It is not kept as a permanent upload. Generative face recovery may change facial details; avoid sensitive photos.</span></div>
          </div>
        </aside>
      </section>

      <div class="footer">
        <span>Real-ESRGAN upscaling</span>
        <span>Optional generative face recovery</span>
        <span>High-resolution PNG export</span>
      </div>
    </main>
  </div>

  <script>
    const fileInput = document.getElementById("file-input");
    const dropZone = document.getElementById("drop-zone");
    const originalImage = document.getElementById("original-image");
    const resultImage = document.getElementById("result-image");
    const enhanceButton = document.getElementById("enhance-button");
    const status = document.getElementById("status");
    const downloadLink = document.getElementById("download-link");
    const faceRecoveryInput = document.getElementById("face-recovery");
    const fidelityInput = document.getElementById("fidelity");
    const engineInput = document.getElementById("engine");
    const openAIConsentInput = document.getElementById("openai-consent");
    let selectedFile = null;
    let originalUrl = null;
    let resultUrl = null;

    function updateEngineControls() {
      const useOpenAI = engineInput.value === "openai";
      document.getElementById("local-settings").hidden = useOpenAI;
      document.getElementById("openai-consent-setting").hidden = !useOpenAI;
      document.getElementById("engine-note").textContent = useOpenAI
        ? "Cloud edit uses OpenAI’s image model. Your image leaves this server only after you confirm below."
        : "Local images stay on this server. Noise reduction, motion deblurring, optional face recovery, then upscaling.";
      document.getElementById("privacy-copy").textContent = useOpenAI
        ? "After you consent, this app sends your image to OpenAI for editing. API usage may incur charges, and image data may be retained for abuse monitoring. Avoid sensitive photos; generated details may differ from the original."
        : "Your image is sent to this app's server for local AI processing. It is not kept as a permanent upload. Generative face recovery may change facial details; avoid sensitive photos.";
      if (selectedFile && (!useOpenAI || openAIConsentInput.checked)) {
        enhanceButton.disabled = false;
      } else if (useOpenAI) {
        enhanceButton.disabled = true;
      }
    }

    fetch("/capabilities")
      .then(response => {
        if (!response.ok) throw new Error(`Could not load app capabilities (${response.status}).`);
        return response.json();
      })
      .then(data => {
        document.getElementById("openai-option").disabled = !data.openai_image_edit_available;
        if (!data.openai_image_edit_available) {
          document.getElementById("engine-note").textContent =
            "OpenAI cloud editing is unavailable until OPENAI_API_KEY is configured on the backend.";
        }
      })
      .catch(error => {
        console.error(error);
        document.getElementById("engine-note").textContent =
          "Could not check OpenAI availability. Local restoration is still available.";
      });

    engineInput.addEventListener("change", updateEngineControls);
    openAIConsentInput.addEventListener("change", updateEngineControls);

    function setStatus(message, kind = "") {
      status.textContent = message;
      status.className = `status ${kind}`.trim();
    }

    function imageDimensions(image, target) {
      image.addEventListener("load", () => {
        target.textContent = `${image.naturalWidth.toLocaleString()} × ${image.naturalHeight.toLocaleString()} px`;
      }, { once: true });
    }

    function chooseFile(file) {
      if (!file) return;
      if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
        setStatus("Please choose a JPG, PNG or WEBP image.", "error");
        return;
      }
      selectedFile = file;
      if (originalUrl) URL.revokeObjectURL(originalUrl);
      originalUrl = URL.createObjectURL(file);
      originalImage.src = originalUrl;
      originalImage.hidden = false;
      document.querySelector("#original-stage .empty-state").hidden = true;
      imageDimensions(originalImage, document.getElementById("original-size"));
      document.getElementById("upload-name").textContent = file.name;
      document.getElementById("result-empty").hidden = false;
      resultImage.hidden = true;
      document.getElementById("result-size").textContent = "Your finished image";
      document.getElementById("result-info").textContent = "Your original stays untouched. The finished image is exported as a high-resolution PNG.";
      downloadLink.hidden = true;
      if (resultUrl) {
        URL.revokeObjectURL(resultUrl);
        resultUrl = null;
      }
      enhanceButton.disabled = engineInput.value === "openai" && !openAIConsentInput.checked;
      document.getElementById("button-label").textContent = "Restore my photograph";
      setStatus("Image ready. Processing sends it to this server for the selected restoration stages.");
    }

    fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
    faceRecoveryInput.addEventListener("change", () => {
      fidelityInput.disabled = !faceRecoveryInput.checked;
    });
    fidelityInput.disabled = !faceRecoveryInput.checked;
    for (const eventName of ["dragenter", "dragover"]) {
      dropZone.addEventListener(eventName, event => {
        event.preventDefault();
        dropZone.classList.add("is-over");
      });
    }
    for (const eventName of ["dragleave", "drop"]) {
      dropZone.addEventListener(eventName, event => {
        event.preventDefault();
        dropZone.classList.remove("is-over");
      });
    }
    dropZone.addEventListener("drop", event => chooseFile(event.dataTransfer.files[0]));

    for (const [inputId, outputId, suffix] of [
      ["strength", "strength-value", "%"],
      ["fidelity", "fidelity-value", "%"],
      ["denoise", "denoise-value", ""]
    ]) {
      const input = document.getElementById(inputId);
      input.addEventListener("input", () => {
        document.getElementById(outputId).value = `${input.value}${suffix}`;
        document.getElementById(outputId).textContent = `${input.value}${suffix}`;
      });
    }

    enhanceButton.addEventListener("click", async () => {
      if (!selectedFile) return;
      const data = new FormData();
      data.append("file", selectedFile);
      data.append("strength", (Number(document.getElementById("strength").value) / 100).toString());
      data.append("fidelity", (Number(document.getElementById("fidelity").value) / 100).toString());
      data.append("denoise_strength", document.getElementById("denoise").value);
      data.append("deblur", document.getElementById("deblur").checked.toString());
      data.append("face_recovery", document.getElementById("face-recovery").checked.toString());
      data.append("engine", engineInput.value);
      data.append("openai_consent", openAIConsentInput.checked.toString());
      enhanceButton.disabled = true;
      document.getElementById("button-label").textContent = "Restoring image…";
      enhanceButton.querySelector("span").className = "button-spinner";
      setStatus(engineInput.value === "openai"
        ? "Sending your photo to OpenAI for generative editing…"
        : "Reducing noise, deblurring, and upscaling. The first deblur may take longer while the model downloads.");
      try {
        const response = await fetch("/enhance", { method: "POST", body: data });
        if (!response.ok) {
          let message = `Restoration failed (${response.status}).`;
          try {
            const detail = await response.json();
            if (detail.detail) message = detail.detail;
          } catch (_) {
            // Keep the HTTP status message when the server does not return JSON.
          }
          throw new Error(message);
        }
        const blob = await response.blob();
        if (resultUrl) URL.revokeObjectURL(resultUrl);
        resultUrl = URL.createObjectURL(blob);
        resultImage.src = resultUrl;
        resultImage.hidden = false;
        document.getElementById("result-empty").hidden = true;
        imageDimensions(resultImage, document.getElementById("result-size"));
        const stem = selectedFile.name.replace(/\.[^.]+$/, "") || "photo";
        downloadLink.href = resultUrl;
        downloadLink.download = `${stem}-restored.png`;
        downloadLink.hidden = false;
        const blur = response.headers.get("X-Blur-Score");
        const restoration = response.headers.get("X-Face-Restoration");
        const stages = response.headers.get("X-Processing-Stages");
        const notes = stages ? [stages] : [];
        if (restoration === "success") notes.push("Review reconstructed facial details carefully.");
        if (blur) notes.push(`Input sharpness score ${Number(blur).toFixed(1)}`);
        document.getElementById("result-info").textContent = notes.join(" · ") || "Your restored image is ready to download.";
        setStatus(engineInput.value === "openai"
          ? "OpenAI edit complete. Review the generated image before downloading."
          : "Restoration complete. Your high-resolution image is ready.", "success");
      } catch (error) {
        setStatus(error.message || "Could not restore this image. Please try again.", "error");
      } finally {
        enhanceButton.disabled = !selectedFile
          || (engineInput.value === "openai" && !openAIConsentInput.checked);
        document.getElementById("button-label").textContent = "Restore my photograph";
        enhanceButton.querySelector("span").className = "";
      }
    });
  </script>
</body>
</html>"""


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )