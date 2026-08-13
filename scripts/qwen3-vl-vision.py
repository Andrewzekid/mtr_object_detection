import base64
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

URL = "https://ollamaapi.ianlo.site/api/chat"
IMAGE_PATH = Path("test8.jpg")

api_key = os.getenv("IW_OLLAMA_API_KEY")
if not api_key:
    raise ValueError("Missing IW_OLLAMA_API_KEY in environment.")

headers = {
    "IW-Ollama-API-Key": api_key,
    "Content-Type": "application/json",
}


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH.resolve()}")

payload = {
    "model": "qwen3-vl:235b-a22b-instruct",
    "messages": [
        {
            "role": "user",
            "content": """Analyze this image and detect all objects. 
            For each object, provide the class name and bounding box coordinates in [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) the bottom-right corner, normalized to the 0-1000 range relative to image width/height. Return the result as a JSON array like: [{"label": "object_name", "bbox_2d": [x1, y1, x2, y2]}]. Only report objects you are confident are actually present in the image; do not invent objects, and do not output duplicate or heavily overlapping boxes for the same object.
            User request: Detect all instances of "Exit Sign" in the main image.
            a hanging monitor/display showing the lime-green character 出 and text 'EXIT'. It is a hanging LCD screen, not a wall poster.
            Return ONLY bounding boxes for this class, as a JSON list of objects each with a "bbox_2d" field in [x1, y1, x2, y2] format (top-left then bottom-right, normalized 0-1000).
            Additional guidance:
            Detect any hanging overhead exit signage containing the standard Exit icon: a bright lime green square background displaying the white Chinese character '出' stacked above the white English word 'EXIT'. Do not classify normal hanging monitors or tvs without the lime '出' and EXIT text as exit signs. Exit signs ARE NOT posters or tvs or advertisement boards. Exit signs MUST CONTAIN 'EXIT' text and the '出' character and be an OVERHEAD HANGING DISPLAY. Do not detect only the lime square, detect the ENTIRE OVERHEAD MONITOR containing it.""",
            "images": [encode_image(IMAGE_PATH)],
        }
    ],
    "stream": False,
}

print("Connecting to campus computing node platform...")
response = requests.post(URL, headers=headers, json=payload)

if response.status_code == 200:
    print("\nConnection successful!")
    print(response.json().get("message", {}).get("content"))
else:
    print(f"Execution failed: {response.status_code}")
    # print(response.text)