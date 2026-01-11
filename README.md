# SAM3 Vision Tools

Text-prompted segmentation and PTZ camera control using Meta's SAM3 (Segment Anything 3).

SAM3 features a 270K concept vocabulary - describe what you want to find in natural language.

## Features

- **Text-Prompted Segmentation**: Find objects by description
- **Live Camera Integration**: Real-time PTZ camera control
- **Edge Optimized**: Memory-efficient for Jetson and embedded devices
- **Zero-Shot Detection**: No training required

## Requirements

### Hardware
- NVIDIA GPU (CUDA) or CPU
- Tested on: Jetson Orin Nano, RTX 3090/4090
- PTZ Camera (optional, for live mode)

### Software
```bash
# Core dependencies
pip install torch torchvision pillow matplotlib numpy

# SAM3 (install from Meta's repo)
git clone https://github.com/meta/sam3
cd sam3 && pip install -e .

# For camera control
sudo apt install v4l-utils ffmpeg
```

## Usage

### Web UI (Recommended)

Launch the web interface for easy testing:

```bash
# Install dependencies
pip install -r requirements.txt

# Run on localhost
python web_ui.py

# Run on all interfaces (for network access)
python web_ui.py --host 0.0.0.0

# Custom port
python web_ui.py --host 0.0.0.0 --port 9000
```

Open `http://localhost:8080` in your browser.

**Features:**
- Drag & drop image upload
- File path input for server-side images
- OBSBOT camera capture and PTZ control
- Configurable API URL for Tailscale/network access
- Real-time segmentation visualization

**Environment Variables:**
```bash
export SAM3_HOST=0.0.0.0      # Bind to all interfaces
export SAM3_PORT=8080          # Custom port
export CAMERA_DEVICE=/dev/video0  # Camera device
```

### Static Image Inference
```bash
python sam3/sam3_inference.py --image photo.jpg --prompt "laptop"
python sam3/sam3_inference.py -i scene.jpg -p "person wearing blue" -o result.jpg
```

### Live Camera Mode
```bash
# Set camera device (optional)
export CAMERA_DEVICE=/dev/video0

# Run interactive session
python sam3/sam3_live.py
```

**Controls:**
- `WASD` - Pan/Tilt camera
- `Z/X` - Zoom in/out
- `R` - Reset to center
- `C` - Capture frame and enter prompt mode
- `Q` - Quit

**Prompt Mode:**
Type any object description:
- "laptop"
- "red coffee mug"
- "person wearing glasses"
- "open book on table"

### Python API
```python
from sam3.sam3_inference import SAM3Inference
from PIL import Image

# Initialize
sam3 = SAM3Inference(device='cuda')

# Load image
image = Image.open('photo.jpg').convert('RGB')

# Segment by text prompt
result = sam3.segment(image, "laptop computer")
print(f"Found {result['count']} matches")

# Visualize
sam3.visualize(image, result, "laptop", output_path="result.jpg")
```

## SAM3 vs SAM1/SAM2

| Feature | SAM1 | SAM2 | SAM3 |
|---------|------|------|------|
| Text Prompts | No | No | Yes (270K concepts) |
| Point Prompts | Yes | Yes | Yes |
| Box Prompts | Yes | Yes | Yes |
| Video Support | No | Yes | Yes |
| Zero-Shot | Limited | Limited | Full vocabulary |

## Memory Optimization

For edge devices (Jetson, etc.):
```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"
```

The live interface uses automatic mixed precision (AMP) for reduced memory.

## Camera Compatibility

Works with UVC PTZ cameras supporting v4l2 controls:
- OBSBOT Tiny SE
- OBSBOT Meet
- Logitech PTZ Pro 2
- Any camera with `pan_absolute`, `tilt_absolute`, `zoom_absolute`

Check your camera:
```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-ctrls
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_DEVICE` | `/dev/video0` | Camera device path |
| `SAM3_OUTPUT` | `/tmp/sam3_output` | Output directory |

## Performance

**Jetson Orin Nano (8GB):**
- Model load: ~30 seconds
- Inference: ~2-3 seconds per image
- Memory: ~4GB VRAM

**RTX 4090:**
- Model load: ~10 seconds
- Inference: <1 second per image

## License

MIT License

## Related Projects

- [sam-angle-detection](https://github.com/Radar105/sam-angle-detection) - SAM-based angle measurement
- [obsbot-camera-mcp](https://github.com/Radar105/obsbot-camera-mcp) - MCP camera control server
