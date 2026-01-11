#!/usr/bin/env python3
"""
SAM3 Live Camera Interface
Real-time text-prompted segmentation with PTZ camera control.

Controls:
    WASD    - Pan/Tilt camera
    Z/X     - Zoom in/out
    R       - Reset to center
    C       - Capture and enter prompt mode
    Q       - Quit

In prompt mode, type what you want to find (e.g., "laptop", "person").

Requirements:
    - SAM3 installed
    - v4l2-utils for camera control
    - ffmpeg for capture

Environment:
    CAMERA_DEVICE: Video device (default: /dev/video0)
    SAM3_OUTPUT: Output directory (default: /tmp/sam3_output)

@license MIT
"""

import os
import sys
import gc
import time
import subprocess
import threading
import select
import termios
import tty

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configuration from environment
CAMERA_DEVICE = os.environ.get('CAMERA_DEVICE', '/dev/video0')
OUTPUT_DIR = os.environ.get('SAM3_OUTPUT', '/tmp/sam3_output')
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, 'sam3_live_output.jpg')
CAPTURE_FILENAME = os.path.join(OUTPUT_DIR, 'sam3_capture.jpg')

# PTZ Limits (OBSBOT Tiny SE defaults, adjust for your camera)
PAN_MIN, PAN_MAX = -468000, 468000
TILT_MIN, TILT_MAX = -324000, 324000
ZOOM_MIN, ZOOM_MAX = 0, 12
PAN_STEP = 20000
TILT_STEP = 20000


def run_cmd(cmd):
    """Execute shell command silently."""
    try:
        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pass


class CameraController:
    """PTZ camera controller via v4l2-ctl."""

    def __init__(self, device=CAMERA_DEVICE):
        self.device = device
        self.pan = 0
        self.tilt = 0
        self.zoom = 0
        self.sync_state()

    def sync_state(self):
        """Read current position from hardware."""
        try:
            out = subprocess.check_output(
                f"v4l2-ctl -d {self.device} --get-ctrl=pan_absolute,tilt_absolute,zoom_absolute",
                shell=True, text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                if "pan_absolute" in line:
                    self.pan = int(line.split(":")[1].strip())
                if "tilt_absolute" in line:
                    self.tilt = int(line.split(":")[1].strip())
                if "zoom_absolute" in line:
                    self.zoom = int(line.split(":")[1].strip())
        except Exception:
            pass

    def move(self, pan_delta=0, tilt_delta=0):
        """Move relative to current position."""
        self.pan = max(PAN_MIN, min(PAN_MAX, self.pan + pan_delta))
        self.tilt = max(TILT_MIN, min(TILT_MAX, self.tilt + tilt_delta))
        run_cmd(f"v4l2-ctl -d {self.device} --set-ctrl=pan_absolute={self.pan},tilt_absolute={self.tilt}")

    def set_zoom(self, zoom):
        """Set absolute zoom level."""
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
        run_cmd(f"v4l2-ctl -d {self.device} --set-ctrl=zoom_absolute={self.zoom}")

    def zoom_step(self, direction):
        """Zoom in or out by one step."""
        if direction == 'in':
            self.set_zoom(self.zoom + 1)
        elif direction == 'out':
            self.set_zoom(self.zoom - 1)

    def reset(self):
        """Return to center position."""
        self.pan = 0
        self.tilt = 0
        self.zoom = 0
        run_cmd(f"v4l2-ctl -d {self.device} --set-ctrl=pan_absolute=0,tilt_absolute=0,zoom_absolute=0")


def capture_frame(filename):
    """Capture frame from camera using ffmpeg."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cmd = f"ffmpeg -f video4linux2 -video_size 1920x1080 -i {CAMERA_DEVICE} -vframes 1 -q:v 2 -y {filename}"
    run_cmd(cmd)
    return os.path.exists(filename)


def save_result(image, masks, prompt, filename):
    """Save visualization with masks overlaid."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    if masks is not None and len(masks) > 0:
        for mask in masks:
            m = mask.cpu().numpy() > 0
            if m.ndim == 3:
                m = m[0]
            color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)
            h, w = m.shape
            mask_image = m.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

    plt.title(f"Prompt: {prompt}")
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


class SAM3Controller:
    """SAM3 model controller with lazy loading."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.loading = True
        self.inference_state = None
        self.current_image = None

    def load_model(self):
        """Load SAM3 model in background."""
        print("\n[System] Loading SAM3 Model... (may take 30+ seconds)")
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"

            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            model = build_sam3_image_model()
            if torch.cuda.is_available():
                model = model.cuda()

            self.processor = Sam3Processor(model)
            self.model = model
            self.loading = False
            print("\n[System] SAM3 Ready!")

        except Exception as e:
            print(f"\n[Error] Failed to load SAM3: {e}")

    def process_image(self, image_path):
        """Load and process image for inference."""
        if self.loading:
            print("[Wait] Model still loading...")
            return False

        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.current_image = Image.open(image_path).convert("RGB")

            with torch.amp.autocast('cuda'):
                self.inference_state = self.processor.set_image(self.current_image)
            return True

        except Exception as e:
            print(f"[Error] Image processing failed: {e}")
            return False

    def run_prompt(self, text_prompt):
        """Run text-prompted segmentation."""
        if not self.inference_state:
            print("[Error] No image loaded")
            return 0

        try:
            with torch.amp.autocast('cuda'):
                output = self.processor.set_text_prompt(
                    state=self.inference_state,
                    prompt=text_prompt
                )
            masks = output["masks"]
            save_result(self.current_image, masks, text_prompt, OUTPUT_FILENAME)
            return len(masks)

        except Exception as e:
            print(f"[Error] Inference failed: {e}")
            return 0


def get_key():
    """Non-blocking key read."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    # Initialize
    camera = CameraController()
    controller = SAM3Controller()

    # Load model in background
    loader_thread = threading.Thread(target=controller.load_model)
    loader_thread.start()

    print("=" * 50)
    print("SAM3 LIVE INTERFACE")
    print("=" * 50)
    print("Controls:")
    print("  W/A/S/D : Pan/Tilt Camera")
    print("  Z/X     : Zoom In/Out")
    print("  R       : Reset Position")
    print("  C       : Capture & Analyze")
    print("  Q       : Quit")
    print("-" * 50)

    mode = "CAMERA"

    try:
        while True:
            if mode == "CAMERA":
                key = get_key()
                if key:
                    k = key.lower()
                    if k == 'w':
                        camera.move(tilt_delta=TILT_STEP)
                    elif k == 's':
                        camera.move(tilt_delta=-TILT_STEP)
                    elif k == 'a':
                        camera.move(pan_delta=-PAN_STEP)
                    elif k == 'd':
                        camera.move(pan_delta=PAN_STEP)
                    elif k == 'z':
                        camera.zoom_step('in')
                    elif k == 'x':
                        camera.zoom_step('out')
                    elif k == 'r':
                        camera.reset()
                        print("[Camera] Reset")
                    elif k == 'q':
                        break
                    elif k == 'c':
                        print("\n[Capture] Taking photo...")
                        if capture_frame(CAPTURE_FILENAME):
                            if controller.loading:
                                print("[Wait] Model loading, please wait...")
                                loader_thread.join()

                            if controller.process_image(CAPTURE_FILENAME):
                                mode = "PROMPT"
                                print("\n--- PROMPT MODE ---")
                                print("Type what to find (e.g., 'laptop')")
                                print("Type 'exit' to return to camera")
                        else:
                            print("[Error] Capture failed")

            elif mode == "PROMPT":
                prompt = input("SAM3 > ").strip()
                if prompt.lower() in ['exit', 'quit', 'back']:
                    mode = "CAMERA"
                    print("\n--- CAMERA MODE ---")
                    continue

                if not prompt:
                    continue

                print(f"Searching for '{prompt}'...")
                count = controller.run_prompt(prompt)
                print(f"Found {count} matches. Saved to {OUTPUT_FILENAME}")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nExiting...")


if __name__ == "__main__":
    main()
