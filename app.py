import io
import os
import tempfile
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


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Image Enhancer API",
    description="Real-ESRGAN + CodeFormer image enhancement API",
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
print("=" * 60)


# ============================================================
# MODEL
# ============================================================

upsampler = None


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


def calculate_safe_strength(
    requested_strength: float,
    blur_score: float,
) -> float:
    """
    Prevent aggressive face restoration on extremely
    blurry images.
    """

    if blur_score < 20:
        return min(requested_strength, 0.25)

    if blur_score < 50:
        return min(requested_strength, 0.40)

    if blur_score < 100:
        return min(requested_strength, 0.60)

    return requested_strength


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


def process_image(
    image_bytes: bytes,
    strength: float,
    fidelity: float,
) -> tuple[bytes, float, float, str]:
    """Upscale an image and apply conservative face restoration."""

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

    applied_strength = calculate_safe_strength(
        strength,
        blur_score,
    )

    # --------------------------------------------------------
    # Real-ESRGAN
    # --------------------------------------------------------

    print(
        f"Processing image | "
        f"Blur={blur_score:.2f} | "
        f"Strength={applied_strength:.2f}"
    )

    try:

        ai_output, _ = upsampler.enhance(
            image,
            outscale=4,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Real-ESRGAN processing failed: {exc}"
        ) from exc

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
    # Temporary files for CodeFormer
    # --------------------------------------------------------

    face_restoration = "skipped"

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        input_path = temp_dir / "upscaled.png"
        restored_path = temp_dir / "restored.png"

        success = cv2.imwrite(
            str(input_path),
            output,
        )

        if not success:
            raise RuntimeError(
                "Could not save the upscaled image."
            )

        # ----------------------------------------------------
        # CodeFormer
        # ----------------------------------------------------

        if applied_strength > 0:

            try:

                run_codeformer(
                    input_path=str(input_path),
                    output_path=str(restored_path),
                    fidelity=fidelity,
                )

                if restored_path.exists():

                    restored = cv2.imread(
                        str(restored_path)
                    )

                    if restored is not None:
                        if restored.shape != output.shape:
                            restored = cv2.resize(
                                restored,
                                (output.shape[1], output.shape[0]),
                                interpolation=cv2.INTER_LANCZOS4,
                            )
                        output = cv2.addWeighted(
                            restored,
                            applied_strength,
                            output,
                            1.0 - applied_strength,
                            0.0,
                        )
                        face_restoration = "success"

                    else:
                        face_restoration = "failed"

                else:
                    face_restoration = "failed"

            except Exception as exc:

                print(
                    f"CodeFormer warning: {exc}"
                )

                face_restoration = "failed"

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
        "device": DEVICE,
    }


# ============================================================
# ENHANCE ENDPOINT
# ============================================================

@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    fidelity: float = Form(
        0.85,
        ge=0.0,
        le=1.0,
    ),
    strength: float = Form(
        0.6,
        ge=0.0,
        le=1.0,
    ),
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

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    try:

        (
            result_bytes,
            applied_strength,
            blur_score,
            restoration_status,
        ) = process_image(
            image_bytes=image_bytes,
            strength=strength,
            fidelity=fidelity,
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
      <div class="local-badge">AI restoration studio</div>
    </header>

    <main>
      <section class="intro">
        <div class="eyebrow">A second life for your photographs</div>
        <h1>Bring the details <span>back.</span></h1>
        <p>Restore softness and upscale with a careful AI finish. Balanced defaults favor your original image; AI may estimate fine detail that was missing from a blurry photo.</p>
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
              <span>JPG, PNG or WEBP</span>
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
              <div class="setting-top">
                <label class="setting-label" for="strength">AI detail blend</label>
                <output class="setting-value" id="strength-value" for="strength">25%</output>
              </div>
              <div class="setting-hint">At 0%, only the original is conventionally upscaled. Increase gradually to blend in AI-estimated detail.</div>
              <input id="strength" type="range" min="0" max="100" value="25">
              <div class="scale-labels"><span>Original-only upscale</span><span>More AI detail</span></div>
            </div>
            <div class="setting">
              <div class="setting-top">
                <label class="setting-label" for="fidelity">Original fidelity</label>
                <output class="setting-value" id="fidelity-value" for="fidelity">98%</output>
              </div>
              <div class="setting-hint">Higher fidelity asks face restoration to stay closer to the original features.</div>
              <input id="fidelity" type="range" min="0" max="100" value="98">
              <div class="scale-labels"><span>Creative</span><span>Faithful</span></div>
            </div>
            <div class="divider"></div>
            <button class="enhance-button" id="enhance-button" type="button" disabled>
              <span aria-hidden="true">✧</span><span id="button-label">Select a photo to begin</span>
            </button>
            <div class="status" id="status" role="status" aria-live="polite">Choose a photo to prepare your restoration.</div>
            <div class="divider"></div>
            <div class="privacy"><span aria-hidden="true">✓</span><span>Your image is processed by this app and is not saved as a permanent upload.</span></div>
          </div>
        </aside>
      </section>

      <div class="footer">
        <span>Real-ESRGAN upscaling</span>
        <span>Conservative face restoration</span>
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
    let selectedFile = null;
    let originalUrl = null;
    let resultUrl = null;

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
      enhanceButton.disabled = false;
      document.getElementById("button-label").textContent = "Restore my photograph";
      setStatus("Image ready. Adjust the finish or start with the balanced defaults.");
    }

    fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
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

    for (const [inputId, outputId] of [["strength", "strength-value"], ["fidelity", "fidelity-value"]]) {
      const input = document.getElementById(inputId);
      input.addEventListener("input", () => {
        document.getElementById(outputId).value = `${input.value}%`;
        document.getElementById(outputId).textContent = `${input.value}%`;
      });
    }

    enhanceButton.addEventListener("click", async () => {
      if (!selectedFile) return;
      const data = new FormData();
      data.append("file", selectedFile);
      data.append("strength", (Number(document.getElementById("strength").value) / 100).toString());
      data.append("fidelity", (Number(document.getElementById("fidelity").value) / 100).toString());
      enhanceButton.disabled = true;
      document.getElementById("button-label").textContent = "Restoring image…";
      enhanceButton.querySelector("span").className = "button-spinner";
      setStatus("Upscaling and refining details. This may take a little while on the first image.");
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
        const notes = [];
        if (blur) notes.push(`Input detail score ${Number(blur).toFixed(1)}`);
        if (restoration === "success") notes.push("face detail refined");
        if (restoration === "skipped") notes.push("upscaled, face refinement not needed");
        if (restoration === "failed") notes.push("upscaled; face refinement unavailable");
        document.getElementById("result-info").textContent = notes.join(" · ") || "Your restored image is ready to download.";
        setStatus("Restoration complete. Your high-resolution image is ready.", "success");
      } catch (error) {
        setStatus(error.message || "Could not restore this image. Please try again.", "error");
      } finally {
        enhanceButton.disabled = !selectedFile;
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