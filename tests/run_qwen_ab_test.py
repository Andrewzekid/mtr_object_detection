#!/usr/bin/env python3
"""
Run an A/B comparison of Qwen3.8 inference settings via llama.cpp.

The script starts a temporary llama-server for each variant, runs inference on a
sample of images, writes a JSON report, and draws side-by-side comparison images.

Example:
    python scripts/run_qwen_ab_test.py \
        --variant-a "1024-think" --image-min-tokens-a 1024 \
        --variant-b "2048-no-think" --image-min-tokens-b 2048 --no-think-b \
        --images-dir output/2026-08-20_22-06-52_pipeline/keyframes/left \
        --n-images 10 \
        --report output/my_ab_test.json \
        --vis-dir output/my_ab_test_vis
"""

import argparse
import base64
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

DEFAULT_PROMPT = """Sofa: Detect any upholstered multi-person seating furniture located in lounge or waiting areas. They feature dark teal or dark green fabric, blocky rectangular structures, padded seat cushions, and low-profile backrests. Do not classify individual office chairs or bare wooden benches as sofas.

Wooden Door: Detect dark brown or reddish-brown wood-grain entrance doors set flush or recessed into interior walls. Key features include metallic horizontal lever handles and rectangular room designation or notice plaques mounted near eye level. Do not classify open glass doors or metal security gates as wooden doors.

Overhead Signage: Detect large rectangular directional or informational boards suspended high above corridors and walkways. They feature dark brown or black panels with light-colored text, arrows, or floor level indicators (e.g., "LG"). Do not classify wall-mounted posters, digital displays, or normal TV monitors as overhead signage.

Sprinkler (on the ceiling): Detect small, localized fire safety sprinkler heads protruding from or recessed into flat ceiling surfaces. They feature small circular metallic deflector plates or white ceiling escutcheon rings, often arranged in a spaced grid alongside ceiling lights or smoke detectors. Do not classify standard recessed spotlight fixtures, cameras, or ceiling air vents as sprinklers.

Analyze this image and detect all objects. For each object, provide the class name and bounding box coordinates in [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) the bottom-right corner, normalized to the 0-1000 range relative to image width/height. Return the result as a JSON array like: [{"label": "object_name", "bbox_2d": [x1, y1, x2, y2]}]. Only report objects you are confident are actually present in the image; do not invent objects, and do not output duplicate or heavily overlapping boxes for the same object.

Please format your response as valid JSON."""

NO_THINK_PREFIX = "Do not think or explain. Output only the final JSON.\n\n"


def parse_args():
    parser = argparse.ArgumentParser(description="A/B test Qwen3.8 vision settings")
    parser.add_argument("--server-bin", default="/home/wangyiming/code/llama/llama.cpp/build/bin/llama-server")
    parser.add_argument("--model", default="/home/wangyiming/code/llama/llama.cpp/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--mmproj", default="/home/wangyiming/code/llama/llama.cpp/Qwen3.8-mmproj-F16.gguf")
    parser.add_argument("--port", default="8089")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--n-gl", type=int, default=999)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=1024)
    parser.add_argument("--n-predict", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=16)

    parser.add_argument("--images-dir", required=True, help="Directory of images to sample from")
    parser.add_argument("--n-images", type=int, default=10, help="Number of images to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="output/qwen_ab_test_report.json")
    parser.add_argument("--vis-dir", default="output/qwen_ab_test_visualizations")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)

    parser.add_argument("--variant-a", default="A", help="Label for variant A")
    parser.add_argument("--image-min-tokens-a", type=int, default=1024)
    parser.add_argument("--no-think-a", action="store_true", help="Disable thinking for variant A")

    parser.add_argument("--variant-b", default="B", help="Label for variant B")
    parser.add_argument("--image-min-tokens-b", type=int, default=2048)
    parser.add_argument("--no-think-b", action="store_true", help="Disable thinking for variant B")

    return parser.parse_args()


def make_server_args(args, image_min_tokens):
    return [
        args.server_bin,
        "-m", args.model,
        "--mmproj", args.mmproj,
        "-ngl", str(args.n_gl),
        "--flash-attn", "on",
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-n", str(args.n_predict),
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "-t", str(args.threads),
        "--port", args.port,
        "--host", args.host,
        "--image-min-tokens", str(image_min_tokens),
    ]


def start_server(args, image_min_tokens):
    cmd = make_server_args(args, image_min_tokens)
    print(f"Starting llama-server with --image-min-tokens {image_min_tokens} ...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = f"http://{args.host}:{args.port}/health"
    for _ in range(120):
        if proc.poll() is not None:
            out, _ = proc.communicate()
            print("Server exited early:", out[-2000:])
            raise RuntimeError("llama-server failed to start")
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print("Server ready")
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("llama-server did not become ready in time")


def stop_server(proc):
    print("Stopping server ...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    time.sleep(2)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_dets(raw):
    s = raw
    if "```json" in s:
        s = s.split("```json")[1].split("```")[0]
    elif "```" in s:
        s = s.split("```")[1].split("```")[0]
    s = s.strip()
    try:
        parsed = json.loads(s)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def infer_one(args, image_path, prompt):
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    b64 = encode_image(image_path)
    payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(url, json=payload, headers={"Authorization": "Bearer local"}, timeout=300)
    dt = time.time() - t0
    data = r.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    rc = msg.get("reasoning_content", "")
    tim = data.get("timings", {})
    return {
        "wall_time_s": dt,
        "predicted_n": tim.get("predicted_n"),
        "predicted_per_second": tim.get("predicted_per_second"),
        "prompt_n": tim.get("prompt_n"),
        "reasoning_chars": len(rc),
        "content_len": len(content),
        "detections": len(parse_dets(content)),
        "raw_content": content,
    }


def run_variant(args, images, label, image_min_tokens, no_think):
    proc = start_server(args, image_min_tokens)
    try:
        prefix = NO_THINK_PREFIX if no_think else ""
        prompt = prefix + args.prompt
        results = []
        for img in images:
            print(f"[{label}] {img.name} ...")
            results.append({"image": img.name, **infer_one(args, img, prompt)})
            time.sleep(0.5)
        return results
    finally:
        stop_server(proc)


def draw_boxes(image_path, dets, color, title):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    out = img.copy()
    for d in dets:
        bbox = d.get("bbox_2d") or d.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1 = int(bbox[0] / 1000.0 * w)
        y1 = int(bbox[1] / 1000.0 * h)
        x2 = int(bbox[2] / 1000.0 * w)
        y2 = int(bbox[3] / 1000.0 * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, d.get("label", ""), (x1, max(y1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.rectangle(out, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(out, title, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    return out


def main():
    args = parse_args()
    img_dir = Path(args.images_dir)
    if not img_dir.is_dir():
        print(f"Images directory not found: {img_dir}", file=sys.stderr)
        sys.exit(1)

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    random.seed(args.seed)
    if len(images) < args.n_images:
        print(f"Only {len(images)} images available; using all of them.")
        selected = images
    else:
        selected = random.sample(images, args.n_images)

    print(f"Selected {len(selected)} images for A/B test\n")

    results_a = run_variant(args, selected, args.variant_a, args.image_min_tokens_a, args.no_think_a)
    results_b = run_variant(args, selected, args.variant_b, args.image_min_tokens_b, args.no_think_b)

    combined = []
    for ra, rb in zip(results_a, results_b):
        combined.append({
            "image": ra["image"],
            args.variant_a: ra,
            args.variant_b: rb,
        })

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    for item in combined:
        img_path = img_dir / item["image"]
        dets_a = parse_dets(item[args.variant_a]["raw_content"])
        dets_b = parse_dets(item[args.variant_b]["raw_content"])
        ta = item[args.variant_a]["wall_time_s"]
        tb = item[args.variant_b]["wall_time_s"]
        na = item[args.variant_a]["detections"]
        nb = item[args.variant_b]["detections"]
        left = draw_boxes(img_path, dets_a, (0, 255, 0), f"{args.variant_a} ({ta:.1f}s, {na} dets)")
        right = draw_boxes(img_path, dets_b, (0, 0, 255), f"{args.variant_b} ({tb:.1f}s, {nb} dets)")
        if left is None or right is None:
            continue
        hr = min(left.shape[0], right.shape[0])
        left_r = cv2.resize(left, (int(left.shape[1] * hr / left.shape[0]), hr))
        right_r = cv2.resize(right, (int(right.shape[1] * hr / right.shape[0]), hr))
        comp = np.hstack([left_r, right_r])
        out_path = vis_dir / f"{Path(item['image']).stem}_compare.jpg"
        cv2.imwrite(str(out_path), comp)

    print(f"Visualizations saved to: {vis_dir}\n")

    print("=== Summary ===")
    cols = ["image", f"{args.variant_a}_t", f"{args.variant_b}_t",
            f"{args.variant_a}_dets", f"{args.variant_b}_dets"]
    print(f"{cols[0]:<40} {cols[1]:>12} {cols[2]:>12} {cols[3]:>14} {cols[4]:>14}")
    for item in combined:
        print(f"{item['image']:<40} "
              f"{item[args.variant_a]['wall_time_s']:>12.2f} "
              f"{item[args.variant_b]['wall_time_s']:>12.2f} "
              f"{item[args.variant_a]['detections']:>14} "
              f"{item[args.variant_b]['detections']:>14}")
    avg_a = sum(item[args.variant_a]["wall_time_s"] for item in combined) / len(combined)
    avg_b = sum(item[args.variant_b]["wall_time_s"] for item in combined) / len(combined)
    print(f"\nAverages: {args.variant_a} {avg_a:.2f}s, {args.variant_b} {avg_b:.2f}s")


if __name__ == "__main__":
    main()
