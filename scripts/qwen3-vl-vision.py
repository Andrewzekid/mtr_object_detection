#!/usr/bin/env python3
"""
Qwen3-VL Exit Sign detection + bbox visualization.

Sends an image to Qwen3-VL via Ollama API, parses the JSON bbox predictions,
draws them on the image, and saves the annotated output.

USAGE:
    set -a && . ./.env && set +a && \
    python scripts/qwen3-vl-vision.py scripts/test9.jpg

    # Custom output path
    python scripts/qwen3-vl-vision.py scripts/test9.jpg --output output/vis.jpg
"""
import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

API_URL = "https://ollamaapi.ianlo.site/api/chat"
MODEL = "qwen3-vl:235b-a22b-instruct"
API_KEY = os.getenv("IW_OLLAMA_API_KEY", "")

PROMPT = """Analyze this image and detect all objects.
For each object, provide the class name and bounding box coordinates in [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) the bottom-right corner, normalized to the 0-1000 range relative to image width/height. Return the result as a JSON array like: [{"label": "object_name", "bbox_2d": [x1, y1, x2, y2]}]. Only report objects you are confident are actually present in the image; do not invent objects, and do not output duplicate or heavily overlapping boxes for the same object.
User request: Detect all instances of "Exit Sign" in the main image.
a hanging monitor/display showing the lime-green character 出 and text 'EXIT'. It is a hanging LCD screen, not a wall poster.
Return ONLY bounding boxes for this class, as a JSON list of objects each with a "bbox_2d" field in [x1, y1, x2, y2] format (top-left then bottom-right, normalized 0-1000).
Additional guidance:
Detect any hanging overhead exit signage containing the standard Exit icon: a bright lime green square background displaying the white Chinese character '出' stacked above the white English word 'EXIT'. Do not classify normal hanging monitors or tvs without the lime '出' and EXIT text as exit signs. Exit signs ARE NOT posters or tvs or advertisement boards. Exit signs MUST CONTAIN 'EXIT' text and the '出' character and be an OVERHEAD HANGING DISPLAY. Do not detect only the lime square, detect the ENTIRE OVERHEAD MONITOR containing it."""


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def parse_predictions(content: str):
    """Extract JSON bbox list from Qwen response."""
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    content = re.sub(r'```(?:json)?\s*', '', content)
    content = content.strip()
    
    # Find JSON array
    json_match = re.search(r'\[.*\]', content, re.DOTALL)
    if not json_match:
        return []
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return []


def draw_bboxes(image_path: Path, data: list, output_path: Path):
    """Draw bounding boxes on image and save."""
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()

    for item in data:
        bbox = item.get("bbox_2d", item.get("bbox", []))
        label = item.get("label", "unknown")
        if len(bbox) != 4:
            continue

        xmin, ymin, xmax, ymax = bbox

        abs_xmin = int((xmin / 1000.0) * width)
        abs_ymin = int((ymin / 1000.0) * height)
        abs_xmax = int((xmax / 1000.0) * width)
        abs_ymax = int((ymax / 1000.0) * height)

        draw.rectangle(
            [abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline="red", width=3
        )

        text_bbox = draw.textbbox((abs_xmin, abs_ymin), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle(
            [
                abs_xmin,
                abs_ymin - text_height - 6,
                abs_xmin + text_width + 6,
                abs_ymin,
            ],
            fill="red",
        )

        draw.text(
            (abs_xmin + 3, abs_ymin - text_height - 4),
            label,
            fill="white",
            font=font,
        )

    image.save(str(output_path))
    print(f"Annotated image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output annotated image path (default: <stem>_output.jpg)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path.resolve()}")

    if not API_KEY:
        sys.exit("Missing IW_OLLAMA_API_KEY in environment. Source .env first.")

    output_path = Path(args.output) if args.output else image_path.parent / f"{image_path.stem}_output.jpg"

    headers = {
        "IW-Ollama-API-Key": API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [encode_image(image_path)],
            }
        ],
        "stream": False,
    }

    print("Connecting to campus computing node platform...")
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        sys.exit(f"Execution failed: {response.status_code}")

    content = response.json().get("message", {}).get("content", "")
    print(f"\nConnection successful!")
    print(content)

    # Parse predictions
    data = parse_predictions(content)
    print(f"\nDetected {len(data)} object(s).")

    # Draw and save
    if data:
        draw_bboxes(image_path, data, output_path)
    else:
        print("No detections to draw.")


if __name__ == "__main__":
    main()