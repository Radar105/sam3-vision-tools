#!/usr/bin/env python3
"""
SAM3 Vision Tools - Web UI
Text-prompted segmentation with configurable host for network access.

Usage:
    python web_ui.py                          # localhost:8080
    python web_ui.py --host 0.0.0.0           # all interfaces
    python web_ui.py --host 0.0.0.0 --port 9000

Environment:
    SAM3_HOST=0.0.0.0    # Override default host
    SAM3_PORT=8080       # Override default port
    CAMERA_DEVICE=/dev/video0  # Camera device
"""

import os
import sys
import io
import base64
import argparse
import subprocess
import logging
import time
from pathlib import Path
from typing import Optional
from collections import deque
from datetime import datetime

# Log buffer for web UI
LOG_BUFFER = deque(maxlen=500)
LOG_BUFFER_LOCK = None

class WebUILogHandler(logging.Handler):
    """Custom log handler that captures logs for web UI."""
    def emit(self, record):
        try:
            msg = self.format(record)
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            LOG_BUFFER.append({
                'time': timestamp,
                'level': record.levelname,
                'message': msg
            })
        except Exception:
            pass

# Setup logging to capture everything
def setup_logging():
    # Root logger - capture all
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Web UI handler
    web_handler = WebUILogHandler()
    web_handler.setLevel(logging.DEBUG)
    web_handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
    root.addHandler(web_handler)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    root.addHandler(console)

setup_logging()
logger = logging.getLogger('sam3_web')

# Add SAM3 to path - check multiple locations
SAM3_PATH = os.environ.get('SAM3_PATH', None)
if SAM3_PATH:
    sys.path.insert(0, SAM3_PATH)
else:
    # Try common locations
    for path in [
        Path.home() / 'aurora_vision' / 'sam3',  # Jetson
        Path(__file__).parent / 'sam3',           # Local
        Path.home() / 'sam3',                     # Home
    ]:
        if path.exists():
            sys.path.insert(0, str(path))
            break

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from PIL import Image
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Config
DEFAULT_HOST = os.environ.get('SAM3_HOST', '127.0.0.1')
DEFAULT_PORT = int(os.environ.get('SAM3_PORT', '8080'))
CAMERA_DEVICE = os.environ.get('CAMERA_DEVICE', '/dev/video0')
OUTPUT_DIR = Path('/tmp/sam3_web')
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="SAM3 Vision Tools", version="1.0.0")

# CORS for network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model (lazy loaded)
sam3_model = None


class SAM3Wrapper:
    """Simple SAM3 wrapper for web UI."""
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.processor = None
        self.log = logging.getLogger('SAM3')

    def load_model(self):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.log.info(f"Initializing SAM3 on device: {self.device}")
        self.log.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.log.info("Loading SAM3 model architecture...")
        t0 = time.time()
        self.model = build_sam3_image_model()
        self.log.info(f"Model architecture loaded in {time.time()-t0:.2f}s")

        if self.device == 'cuda':
            self.log.info("Moving model to CUDA...")
            t0 = time.time()
            self.model = self.model.cuda()
            self.log.info(f"Model moved to GPU in {time.time()-t0:.2f}s")

        self.log.info("Creating SAM3 processor...")
        self.processor = Sam3Processor(self.model, device=self.device)
        self.log.info("SAM3 ready for inference")

    def segment(self, image: Image.Image, prompt: str) -> dict:
        if self.model is None:
            self.load_model()

        self.log.info(f"=== Starting segmentation ===")
        self.log.info(f"Prompt: '{prompt}'")
        self.log.info(f"Image size: {image.size[0]}x{image.size[1]}")

        # Set image
        self.log.info("Encoding image through backbone...")
        t0 = time.time()
        state = self.processor.set_image(image)
        self.log.info(f"Image encoding: {time.time()-t0:.2f}s")

        # Run text prompt
        self.log.info(f"Running text-prompted segmentation...")
        t0 = time.time()
        output = self.processor.set_text_prompt(prompt=prompt, state=state)
        self.log.info(f"Segmentation: {time.time()-t0:.2f}s")

        # Log results
        n_masks = len(output['masks'])
        self.log.info(f"Found {n_masks} mask(s)")
        if n_masks > 0:
            scores = output['scores']
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            for i, s in enumerate(scores):
                self.log.info(f"  Mask {i+1}: confidence {s*100:.1f}%")

        self.log.info(f"=== Segmentation complete ===")

        return {
            'masks': output['masks'],
            'boxes': output['boxes'],
            'scores': output['scores'],
            'count': n_masks
        }


def get_model():
    """Lazy load SAM3 model."""
    global sam3_model
    if sam3_model is None:
        sam3_model = SAM3Wrapper()
    return sam3_model


def image_to_base64(img: Image.Image, format='JPEG') -> str:
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()


def create_visualization(image: Image.Image, result: dict, prompt: str, viz_mode: str = 'both') -> str:
    """Create visualization and return as base64.

    viz_mode: 'mask', 'box', or 'both'
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(result['masks']), 1)))

    # Draw masks if mode includes masks
    if viz_mode in ('mask', 'both'):
        for i, mask in enumerate(result['masks']):
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            # Handle different mask shapes
            while mask_np.ndim > 2:
                mask_np = mask_np.squeeze(0)

            color = colors[i % len(colors)]
            color_with_alpha = np.array([*color[:3], 0.5])
            h, w = mask_np.shape
            mask_image = (mask_np > 0.5).reshape(h, w, 1) * color_with_alpha.reshape(1, 1, -1)
            ax.imshow(mask_image)

    # Draw boxes if mode includes boxes
    if viz_mode in ('box', 'both'):
        for i, box in enumerate(result['boxes']):
            box_np = box.cpu().numpy() if isinstance(box, torch.Tensor) else box
            x0, y0, x1, y1 = box_np
            rect = plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                 edgecolor=colors[i % len(colors)],
                                 facecolor='none', lw=3)
            ax.add_patch(rect)

    ax.set_title(f"Prompt: '{prompt}' | Found: {len(result['masks'])} objects", fontsize=14)
    ax.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='JPEG', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    return HTML_TEMPLATE


@app.get("/api/status")
async def status():
    """Check system status."""
    cuda_available = torch.cuda.is_available()
    camera_available = os.path.exists(CAMERA_DEVICE)
    model_loaded = sam3_model is not None and sam3_model.model is not None

    return {
        "cuda": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "camera": camera_available,
        "camera_device": CAMERA_DEVICE,
        "model_loaded": model_loaded
    }


@app.get("/api/logs")
async def get_logs(since: int = 0, clear: bool = False):
    """Get log entries. Use since=index to get only new logs."""
    if clear:
        LOG_BUFFER.clear()
        return {"logs": [], "total": 0}

    logs = list(LOG_BUFFER)
    if since > 0 and since < len(logs):
        logs = logs[since:]

    return {
        "logs": logs,
        "total": len(LOG_BUFFER)
    }


@app.post("/api/logs/clear")
async def clear_logs():
    """Clear all logs."""
    LOG_BUFFER.clear()
    logger.info("Logs cleared")
    return {"success": True}


@app.post("/api/segment")
async def segment(
    image: UploadFile = File(None),
    image_path: str = Form(None),
    prompt: str = Form(...),
    use_camera: bool = Form(False),
    viz_mode: str = Form('box')
):
    """
    Segment image with text prompt.

    Accepts:
    - Uploaded file (image)
    - File path on server (image_path)
    - Camera capture (use_camera=true)
    - viz_mode: 'mask', 'box', or 'both'
    """
    try:
        # Get image from one of the sources
        if use_camera:
            img = capture_from_camera()
            if img is None:
                raise HTTPException(400, "Camera capture failed")
        elif image is not None:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents)).convert('RGB')
        elif image_path:
            if not os.path.exists(image_path):
                raise HTTPException(400, f"File not found: {image_path}")
            img = Image.open(image_path).convert('RGB')
        else:
            raise HTTPException(400, "No image source provided")

        # Run segmentation
        model = get_model()
        result = model.segment(img, prompt)

        # Create visualization
        viz_base64 = create_visualization(img, result, prompt, viz_mode)

        # Extract scores
        scores = result['scores']
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy().tolist()

        return {
            "success": True,
            "prompt": prompt,
            "count": len(result['masks']),
            "scores": scores,
            "image_size": list(img.size),
            "visualization": f"data:image/jpeg;base64,{viz_base64}"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/camera/capture")
async def camera_capture():
    """Capture image from OBSBOT camera."""
    img = capture_from_camera()
    if img is None:
        raise HTTPException(500, "Camera capture failed")

    return {
        "success": True,
        "image": f"data:image/jpeg;base64,{image_to_base64(img)}",
        "size": list(img.size)
    }


@app.post("/api/camera/control")
async def camera_control(
    pan: int = Form(None),
    tilt: int = Form(None),
    zoom: int = Form(None)
):
    """Control OBSBOT PTZ camera."""
    try:
        cmds = []
        if pan is not None:
            cmds.append(f"pan_absolute={pan}")
        if tilt is not None:
            cmds.append(f"tilt_absolute={tilt}")
        if zoom is not None:
            cmds.append(f"zoom_absolute={zoom}")

        if cmds:
            cmd = f"v4l2-ctl -d {CAMERA_DEVICE} --set-ctrl={','.join(cmds)}"
            subprocess.run(cmd, shell=True, check=True)

        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def capture_from_camera() -> Optional[Image.Image]:
    """Capture single frame from camera."""
    try:
        import cv2
        cap = cv2.VideoCapture(CAMERA_DEVICE)
        if not cap.isOpened():
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    except:
        # Fallback to ffmpeg
        try:
            output = OUTPUT_DIR / 'capture.jpg'
            cmd = f"ffmpeg -y -f v4l2 -i {CAMERA_DEVICE} -frames:v 1 {output}"
            subprocess.run(cmd, shell=True, check=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if output.exists():
                return Image.open(output).convert('RGB')
        except:
            pass
    return None


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAM3 Vision Tools</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #00d4ff;
        }
        .status-bar {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .status-item {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
        }
        .status-ok { background: #00c853; color: #000; }
        .status-warn { background: #ff9100; color: #000; }
        .status-err { background: #ff1744; color: #fff; }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 900px) {
            .main-grid { grid-template-columns: 1fr; }
        }

        .panel {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
        }
        .panel h2 {
            margin-bottom: 15px;
            color: #00d4ff;
            font-size: 18px;
        }

        .drop-zone {
            border: 2px dashed #444;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 15px;
        }
        .drop-zone:hover, .drop-zone.drag-over {
            border-color: #00d4ff;
            background: rgba(0, 212, 255, 0.1);
        }
        .drop-zone input { display: none; }

        .input-group {
            margin-bottom: 15px;
        }
        .input-group label {
            display: block;
            margin-bottom: 5px;
            color: #aaa;
            font-size: 14px;
        }
        .input-group input, .input-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #333;
            border-radius: 6px;
            background: #0f0f23;
            color: #fff;
            font-size: 16px;
        }
        .input-group input:focus {
            outline: none;
            border-color: #00d4ff;
        }

        .btn-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            flex: 1;
            min-width: 120px;
        }
        .btn-primary {
            background: #00d4ff;
            color: #000;
        }
        .btn-primary:hover { background: #00a8cc; }
        .btn-secondary {
            background: #333;
            color: #fff;
        }
        .btn-secondary:hover { background: #444; }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .result-image {
            width: 100%;
            border-radius: 8px;
            margin-top: 15px;
        }
        .result-info {
            margin-top: 15px;
            padding: 15px;
            background: #0f0f23;
            border-radius: 8px;
        }
        .result-info p {
            margin: 5px 0;
        }

        .camera-controls {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        .camera-controls .btn {
            padding: 15px;
            font-size: 20px;
        }

        .preview-img {
            max-width: 100%;
            max-height: 200px;
            border-radius: 8px;
            margin-top: 10px;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading.show { display: block; }
        .spinner {
            border: 3px solid #333;
            border-top: 3px solid #00d4ff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .config-bar {
            background: #0f0f23;
            padding: 10px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        .config-bar label { color: #aaa; }
        .config-bar input {
            padding: 8px;
            border: 1px solid #333;
            border-radius: 4px;
            background: #1a1a2e;
            color: #fff;
            width: 200px;
        }

        .log-panel {
            margin-top: 20px;
        }
        .log-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .log-header h2 { margin: 0; }
        .log-container {
            background: #0a0a15;
            border: 1px solid #333;
            border-radius: 8px;
            height: 300px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 12px;
            padding: 10px;
        }
        .log-entry {
            padding: 2px 0;
            border-bottom: 1px solid #1a1a2e;
        }
        .log-time {
            color: #666;
            margin-right: 8px;
        }
        .log-level {
            padding: 1px 6px;
            border-radius: 3px;
            margin-right: 8px;
            font-size: 10px;
            font-weight: bold;
        }
        .log-level.DEBUG { background: #333; color: #aaa; }
        .log-level.INFO { background: #1565c0; color: #fff; }
        .log-level.WARNING { background: #ff9100; color: #000; }
        .log-level.ERROR { background: #ff1744; color: #fff; }
        .log-message { color: #ccc; }
        .log-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .log-controls label {
            display: flex;
            align-items: center;
            gap: 5px;
            color: #aaa;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SAM3 Vision Tools</h1>

        <div class="config-bar">
            <label>API URL:</label>
            <input type="text" id="apiUrl" value="" placeholder="http://localhost:8080">
            <button class="btn btn-secondary" onclick="checkStatus()">Check Status</button>
        </div>

        <div class="status-bar" id="statusBar">
            <span class="status-item" id="cudaStatus">CUDA: checking...</span>
            <span class="status-item" id="cameraStatus">Camera: checking...</span>
            <span class="status-item" id="modelStatus">Model: not loaded</span>
        </div>

        <div class="main-grid">
            <div class="panel">
                <h2>Input</h2>

                <div class="drop-zone" id="dropZone">
                    <p>Drop image here or click to browse</p>
                    <input type="file" id="fileInput" accept="image/*">
                    <img class="preview-img" id="previewImg" style="display:none;">
                </div>

                <div class="input-group">
                    <label>Or enter image path:</label>
                    <input type="text" id="imagePath" placeholder="/path/to/image.jpg">
                </div>

                <div class="input-group">
                    <label>Text Prompt:</label>
                    <input type="text" id="prompt" placeholder="laptop, person wearing red, coffee mug..." value="object">
                </div>

                <div class="input-group">
                    <label>Visualization:</label>
                    <select id="vizMode">
                        <option value="box" selected>Bounding Box</option>
                        <option value="mask">Mask Only</option>
                        <option value="both">Both</option>
                    </select>
                </div>

                <div class="btn-row">
                    <button class="btn btn-primary" id="segmentBtn" onclick="runSegmentation()">
                        Segment
                    </button>
                    <button class="btn btn-secondary" onclick="captureFromCamera()">
                        📷 Camera
                    </button>
                </div>

                <div class="camera-controls" id="cameraControls" style="display:none;">
                    <button class="btn btn-secondary" onclick="moveCamera(0, 20000)">↑</button>
                    <button class="btn btn-secondary" onclick="moveCamera(0, 0, 1)">+</button>
                    <button class="btn btn-secondary" onclick="resetCamera()">⟲</button>
                    <button class="btn btn-secondary" onclick="moveCamera(-20000, 0)">←</button>
                    <button class="btn btn-secondary" onclick="moveCamera(0, -20000)">↓</button>
                    <button class="btn btn-secondary" onclick="moveCamera(20000, 0)">→</button>
                    <button class="btn btn-secondary" onclick="moveCamera(0, 0, -1)">−</button>
                </div>
            </div>

            <div class="panel">
                <h2>Result</h2>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing with SAM3...</p>
                </div>

                <img class="result-image" id="resultImg" style="display:none;">

                <div class="result-info" id="resultInfo" style="display:none;">
                    <p><strong>Prompt:</strong> <span id="resultPrompt"></span></p>
                    <p><strong>Objects found:</strong> <span id="resultCount"></span></p>
                    <p><strong>Confidence:</strong> <span id="resultScores"></span></p>
                </div>
            </div>
        </div>

        <!-- Log Panel -->
        <div class="panel log-panel">
            <div class="log-header">
                <h2>Model Logs</h2>
                <div class="log-controls">
                    <button class="btn btn-secondary" style="padding: 6px 12px; font-size: 12px;" onclick="fetchLogs()">Refresh</button>
                    <button class="btn btn-secondary" style="padding: 6px 12px; font-size: 12px;" onclick="clearLogs()">Clear</button>
                    <label>
                        <input type="checkbox" id="autoRefresh"> Auto
                    </label>
                </div>
            </div>
            <div class="log-container" id="logContainer">
                <div class="log-entry">
                    <span class="log-time">--:--:--.---</span>
                    <span class="log-level INFO">INFO</span>
                    <span class="log-message">Waiting for logs...</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentImage = null;
        let cameraZoom = 0;

        // Set default API URL
        document.getElementById('apiUrl').value = window.location.origin;

        function getApiUrl() {
            return document.getElementById('apiUrl').value || window.location.origin;
        }

        // Check status on load
        window.onload = checkStatus;

        async function checkStatus() {
            try {
                const resp = await fetch(getApiUrl() + '/api/status');
                const data = await resp.json();

                document.getElementById('cudaStatus').className =
                    'status-item ' + (data.cuda ? 'status-ok' : 'status-warn');
                document.getElementById('cudaStatus').textContent =
                    'CUDA: ' + (data.cuda ? data.cuda_device : 'CPU only');

                document.getElementById('cameraStatus').className =
                    'status-item ' + (data.camera ? 'status-ok' : 'status-err');
                document.getElementById('cameraStatus').textContent =
                    'Camera: ' + (data.camera ? 'Ready' : 'Not found');

                if (data.camera) {
                    document.getElementById('cameraControls').style.display = 'grid';
                }

                document.getElementById('modelStatus').className =
                    'status-item ' + (data.model_loaded ? 'status-ok' : 'status-warn');
                document.getElementById('modelStatus').textContent =
                    'Model: ' + (data.model_loaded ? 'Loaded' : 'Will load on first use');

            } catch (e) {
                console.error('Status check failed:', e);
            }
        }

        // Drag and drop
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const previewImg = document.getElementById('previewImg');

        dropZone.onclick = () => fileInput.click();

        dropZone.ondragover = (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        };

        dropZone.ondragleave = () => {
            dropZone.classList.remove('drag-over');
        };

        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        };

        fileInput.onchange = () => {
            if (fileInput.files.length) {
                handleFile(fileInput.files[0]);
            }
        };

        function handleFile(file) {
            currentImage = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                previewImg.style.display = 'block';
            };
            reader.readAsDataURL(file);
            document.getElementById('imagePath').value = '';
        }

        async function runSegmentation() {
            const prompt = document.getElementById('prompt').value;
            const imagePath = document.getElementById('imagePath').value;

            if (!prompt) {
                alert('Please enter a text prompt');
                return;
            }

            if (!currentImage && !imagePath) {
                alert('Please select an image or enter a path');
                return;
            }

            document.getElementById('loading').classList.add('show');
            document.getElementById('resultImg').style.display = 'none';
            document.getElementById('resultInfo').style.display = 'none';
            document.getElementById('segmentBtn').disabled = true;

            try {
                const formData = new FormData();
                formData.append('prompt', prompt);
                formData.append('viz_mode', document.getElementById('vizMode').value);

                if (currentImage) {
                    formData.append('image', currentImage);
                } else {
                    formData.append('image_path', imagePath);
                }

                const resp = await fetch(getApiUrl() + '/api/segment', {
                    method: 'POST',
                    body: formData
                });

                const data = await resp.json();

                if (data.success) {
                    document.getElementById('resultImg').src = data.visualization;
                    document.getElementById('resultImg').style.display = 'block';
                    document.getElementById('resultPrompt').textContent = data.prompt;
                    document.getElementById('resultCount').textContent = data.count;
                    document.getElementById('resultScores').textContent =
                        data.scores.map(s => (s * 100).toFixed(1) + '%').join(', ');
                    document.getElementById('resultInfo').style.display = 'block';

                    // Update model status
                    document.getElementById('modelStatus').className = 'status-item status-ok';
                    document.getElementById('modelStatus').textContent = 'Model: Loaded';
                } else {
                    alert('Error: ' + data.error);
                }

            } catch (e) {
                alert('Request failed: ' + e.message);
            } finally {
                document.getElementById('loading').classList.remove('show');
                document.getElementById('segmentBtn').disabled = false;
            }
        }

        async function captureFromCamera() {
            try {
                const resp = await fetch(getApiUrl() + '/api/camera/capture', {
                    method: 'POST'
                });
                const data = await resp.json();

                if (data.success) {
                    previewImg.src = data.image;
                    previewImg.style.display = 'block';

                    // Convert base64 to blob for upload
                    const base64 = data.image.split(',')[1];
                    const binary = atob(base64);
                    const array = new Uint8Array(binary.length);
                    for (let i = 0; i < binary.length; i++) {
                        array[i] = binary.charCodeAt(i);
                    }
                    currentImage = new Blob([array], {type: 'image/jpeg'});
                    document.getElementById('imagePath').value = '';
                } else {
                    alert('Camera capture failed');
                }
            } catch (e) {
                alert('Camera error: ' + e.message);
            }
        }

        async function moveCamera(pan, tilt, zoomDelta) {
            const formData = new FormData();
            if (pan) formData.append('pan', pan);
            if (tilt) formData.append('tilt', tilt);
            if (zoomDelta) {
                cameraZoom = Math.max(0, Math.min(12, cameraZoom + zoomDelta));
                formData.append('zoom', cameraZoom);
            }

            await fetch(getApiUrl() + '/api/camera/control', {
                method: 'POST',
                body: formData
            });
        }

        async function resetCamera() {
            cameraZoom = 0;
            const formData = new FormData();
            formData.append('pan', 0);
            formData.append('tilt', 0);
            formData.append('zoom', 0);
            await fetch(getApiUrl() + '/api/camera/control', {
                method: 'POST',
                body: formData
            });
        }

        // Enter key triggers segmentation
        document.getElementById('prompt').onkeypress = (e) => {
            if (e.key === 'Enter') runSegmentation();
        };

        // === Log handling ===
        let logIndex = 0;
        let logInterval = null;
        const MAX_LOG_ENTRIES = 100;

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function fetchLogs() {
            try {
                const resp = await fetch(getApiUrl() + '/api/logs?since=' + logIndex);
                const data = await resp.json();

                if (data.logs && data.logs.length > 0) {
                    const container = document.getElementById('logContainer');

                    // Build all entries at once
                    const fragment = document.createDocumentFragment();
                    data.logs.forEach(log => {
                        const entry = document.createElement('div');
                        entry.className = 'log-entry';
                        entry.innerHTML = `<span class="log-time">${log.time}</span>` +
                            `<span class="log-level ${log.level}">${log.level}</span>` +
                            `<span class="log-message">${escapeHtml(log.message)}</span>`;
                        fragment.appendChild(entry);
                    });

                    // Clear placeholder if first load
                    if (container.innerHTML.includes('Waiting for logs')) {
                        container.innerHTML = '';
                    }

                    container.appendChild(fragment);
                    logIndex = data.total;

                    // Trim old entries
                    while (container.children.length > MAX_LOG_ENTRIES) {
                        container.removeChild(container.firstChild);
                    }

                    // Auto-scroll to bottom
                    container.scrollTop = container.scrollHeight;
                }
            } catch (e) {
                console.error('Failed to fetch logs:', e);
            }
        }

        async function clearLogs() {
            try {
                await fetch(getApiUrl() + '/api/logs/clear', { method: 'POST' });
                document.getElementById('logContainer').innerHTML =
                    '<div class="log-entry"><span class="log-time">--:--:--.---</span>' +
                    '<span class="log-level INFO">INFO</span>' +
                    '<span class="log-message">Logs cleared</span></div>';
                logIndex = 0;
            } catch (e) {
                console.error('Failed to clear logs:', e);
            }
        }

        function startLogPolling() {
            if (logInterval) clearInterval(logInterval);
            logInterval = setInterval(() => {
                if (document.getElementById('autoRefresh').checked) {
                    fetchLogs();
                }
            }, 3000);
        }

        // Start polling but auto-refresh is OFF by default
        startLogPolling();
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description='SAM3 Vision Tools Web UI')
    parser.add_argument('--host', default=DEFAULT_HOST,
                        help=f'Host to bind (default: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'Port to bind (default: {DEFAULT_PORT})')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload for development')
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"SAM3 Vision Tools - Web UI")
    print(f"{'='*50}")
    print(f"Local:    http://127.0.0.1:{args.port}")
    if args.host == '0.0.0.0':
        print(f"Network:  http://<your-ip>:{args.port}")
    print(f"{'='*50}\n")

    uvicorn.run(
        "web_ui:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == '__main__':
    main()
