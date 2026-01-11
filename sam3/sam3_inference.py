#!/usr/bin/env python3
"""
SAM3 Text-Prompted Segmentation
Uses Meta's SAM3 with text prompts for zero-shot object detection.

SAM3 has a 270K concept vocabulary - just describe what you want to find.

Usage:
    python sam3_inference.py --image photo.jpg --prompt "laptop"
    python sam3_inference.py --image photo.jpg --prompt "person wearing red"

Requirements:
    pip install torch torchvision pillow matplotlib
    # SAM3 from: https://github.com/meta/sam3

@license MIT
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Memory optimization for edge devices
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def show_mask(mask, ax, random_color=False):
    """Overlay segmentation mask on plot."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    """Draw bounding box on plot."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                                facecolor=(0, 0, 0, 0), lw=2))


class SAM3Inference:
    """SAM3 text-prompted segmentation wrapper."""

    def __init__(self, device: str = None):
        """
        Initialize SAM3 model.

        Args:
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None

    def load_model(self):
        """Load SAM3 model (lazy loading for memory efficiency)."""
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            print(f"Loading SAM3 on {self.device}...")
            self.model = build_sam3_image_model()

            if self.device == 'cuda':
                self.model = self.model.cuda()

            self.processor = Sam3Processor(self.model)
            print("SAM3 loaded successfully")

        except ImportError:
            print("Error: SAM3 not installed")
            print("Clone from: https://github.com/meta/sam3")
            sys.exit(1)

    def segment(self, image: Image.Image, prompt: str) -> dict:
        """
        Segment image using text prompt.

        Args:
            image: PIL Image (RGB)
            prompt: Text description of object to find

        Returns:
            dict with 'masks', 'boxes', 'scores'
        """
        if self.model is None:
            self.load_model()

        # Set image
        inference_state = self.processor.set_image(image)

        # Run inference with text prompt
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=prompt
        )

        return {
            'masks': output['masks'],
            'boxes': output['boxes'],
            'scores': output['scores'],
            'count': len(output['masks'])
        }

    def visualize(self, image: Image.Image, result: dict,
                  prompt: str, output_path: str = None):
        """
        Visualize segmentation results.

        Args:
            image: Original PIL image
            result: Output from segment()
            prompt: Text prompt used
            output_path: Save path (optional)
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

        for mask in result['masks']:
            show_mask(mask.cpu().numpy() > 0, ax)

        for box in result['boxes']:
            show_box(box.cpu().numpy(), ax)

        plt.title(f"Prompt: '{prompt}' | Found: {result['count']}")
        plt.axis('off')

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            print(f"Saved: {output_path}")
        else:
            plt.show()

        plt.close()


def main():
    parser = argparse.ArgumentParser(description='SAM3 Text-Prompted Segmentation')
    parser.add_argument('--image', '-i', required=True, help='Input image path')
    parser.add_argument('--prompt', '-p', required=True, help='Text prompt')
    parser.add_argument('--output', '-o', default='sam3_result.jpg',
                        help='Output image path')
    parser.add_argument('--device', '-d', default=None,
                        help='Device: cuda or cpu')
    args = parser.parse_args()

    # Load image
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    image = Image.open(args.image).convert('RGB')
    print(f"Image: {args.image} ({image.size[0]}x{image.size[1]})")

    # Run inference
    sam3 = SAM3Inference(device=args.device)
    result = sam3.segment(image, args.prompt)

    print(f"Found {result['count']} matches for '{args.prompt}'")

    # Visualize
    sam3.visualize(image, result, args.prompt, args.output)


if __name__ == '__main__':
    main()
