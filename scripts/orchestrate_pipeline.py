#!/usr/bin/env python3
"""
End-to-end rosbag annotation pipeline orchestrator.

Automates the full workflow from a Metacam rosbag to a YOLO-format detection
dataset:

    undistort -> time-based split -> keyframes -> Qwen -> human review
    -> propagate -> YOLO export

Example:
    python scripts/orchestrate_pipeline.py \
        --rosbag 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52 \
        --camera left \
        --qwen-model Qwen3.8-27B-Q4_K_M.gguf \
        --qwen-mmproj Qwen3.8-mmproj-F16.gguf \
        --llamacpp-url http://127.0.0.1:8089 \
        --sam3-model core/sam3/models/sam3-model/sam3.pt

The default propagation method is ``interpolation+sam3``: reviewed keyframe
boxes are first interpolated to every train frame with optical flow, then SAM3
refines each interpolated box into a segmentation mask using the box as an
exemplar and the class name as a text concept.
"""

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

DEFAULT_PROMPT = """Sofa: Detect any upholstered multi-person seating furniture located in lounge or waiting areas. They feature dark teal or dark green fabric, blocky rectangular structures, padded seat cushions, and low-profile backrests. Do not classify individual office chairs or bare wooden benches as sofas.

Wooden Door: Detect dark brown or reddish-brown wood-grain entrance doors set flush or recessed into interior walls. Key features include metallic horizontal lever handles and rectangular room designation or notice plaques mounted near eye level. Do not classify open glass doors or metal security gates as wooden doors.

Overhead Signage: Detect large rectangular directional or informational boards suspended high above corridors and walkways. They feature dark brown or black panels with light-colored text, arrows, or floor level indicators (e.g., "LG"). Do not classify wall-mounted posters, digital displays, or normal TV monitors as overhead signage.

Sprinkler (on the ceiling): Detect small, localized fire safety sprinkler heads protruding from or recessed into flat ceiling surfaces. They feature small circular metallic deflector plates or white ceiling escutcheon rings, often arranged in a spaced grid alongside ceiling lights or smoke detectors. Sprinklers can be very small in the image; examine the ceiling carefully, especially near lights, smoke detectors, and ceiling corners, and report each individual sprinkler head even if it is only a few pixels wide. Do not classify standard recessed spotlight fixtures, cameras, or ceiling air vents as sprinklers."""

NO_THINK_PREFIX = "Do not think or explain. Output only the final JSON.\n\n"


def get_project_root() -> Path:
    """Return the repository root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="End-to-end rosbag annotation pipeline orchestrator."
    )
    parser.add_argument(
        "--rosbag",
        required=True,
        help="Path to rosbag root with camera/left, camera/right, info/calibration.json",
    )
    parser.add_argument(
        "--camera",
        default="left",
        choices=["left", "right", "both"],
        help="Camera(s) to process (default: left)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output directory (default: <rosbag>_pipeline)",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen3.8-27B-Q4_K_M.gguf",
        help="Path to Qwen GGUF weights",
    )
    parser.add_argument(
        "--qwen-mmproj",
        default="Qwen3.8-mmproj-F16.gguf",
        help="Path to Qwen mmproj file (must be loaded by the llama.cpp server)",
    )
    parser.add_argument(
        "--llamacpp-url",
        default="http://127.0.0.1:8089",
        help="llama.cpp server URL",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Detection prompt for Qwen",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class names (default: parsed from --prompt)",
    )
    parser.add_argument(
        "--keyframe-stride",
        type=int,
        default=10,
        help="Extract 1 keyframe every N frames (default: 10)",
    )
    parser.add_argument(
        "--splits",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test time ratios (default: 0.8 0.1 0.1)",
    )
    parser.add_argument(
        "--sam3-model",
        default="core/sam3/models/sam3-model/sam3.pt",
        help="Path to SAM3 weights",
    )
    parser.add_argument(
        "--propagation-method",
        default="interpolation+sam3",
        choices=["interpolation", "sam3", "interpolation+sam3"],
        help="Propagation method (default: interpolation+sam3)",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        help="Run only this stage, e.g. undistort, split, keyframes, qwen, qwen_coco, review, propagate, export",
    )
    parser.add_argument(
        "--skip-stage",
        action="append",
        default=None,
        help="Skip this stage (can be repeated)",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=None,
        help="Resume Qwen batch from this 1-indexed image number (passed to 07_run_qwen.py)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun stages even if marker exists",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for SAM3 (cuda or cpu, default: cuda)",
    )
    return parser.parse_args(argv)


def parse_classes_from_prompt(prompt: str) -> list:
    """Extract class names from a prompt formatted as 'Class Name: description...'."""
    classes = []
    for line in prompt.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            name = line.split(":", 1)[0].strip()
            # Remove parenthetical qualifiers like "(on the ceiling)"
            name = name.split("(")[0].strip()
            if name:
                classes.append(name)
    return classes


def time_based_split(src_dir: Path, out_dir: Path, ratios: list) -> dict:
    """Symlink images from src_dir into out_dir/images/{train,val,test} by time order."""
    if abs(sum(ratios) - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
    image_files = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_files:
        raise RuntimeError(f"No images found in {src_dir}")
    total = len(image_files)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    partitions = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:],
    }
    counts = {}
    for split, files in partitions.items():
        split_img_dir = out_dir / "images" / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = split_img_dir / src.name
            if not dst.exists():
                dst.symlink_to(src.resolve())
        counts[split] = len(files)
    return counts


def read_marker(marker_path: Path) -> dict | None:
    if not marker_path.exists():
        return None
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_marker(marker_path: Path, data: dict):
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["completed_at"] = datetime.now(timezone.utc).isoformat()
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_stage(stage_name: str, cmd: list, cwd: Path, marker_path: Path, force: bool = False):
    """Run a stage command unless its marker exists."""
    if not force and read_marker(marker_path):
        print(f"[skip] {stage_name}: marker exists at {marker_path}")
        return None
    print(f"[run] {stage_name}: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Stage {stage_name} failed with exit code {result.returncode}")
    write_marker(marker_path, {"stage": stage_name})
    return result


def check_llamacpp_server(url: str, timeout: float = 2.0) -> bool:
    """Return True if the llama.cpp server /health endpoint responds."""
    health_url = url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def should_run(stage: str, args) -> bool:
    if args.skip_stage and stage in args.skip_stage:
        return False
    if args.stage:
        return stage in args.stage
    return True


def export_yolo_from_coco(
    coco_path: Path,
    img_dir: Path,
    labels_dir: Path,
    class_names: list,
):
    """Convert a COCO detection file to YOLO txt labels in labels_dir."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cat_id_to_yolo_id = {
        cat["id"]: class_names.index(cat["name"])
        for cat in coco.get("categories", [])
        if cat["name"] in class_names
    }
    img_id_to_info = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, img_info in img_id_to_info.items():
        img_w = img_info.get("width")
        img_h = img_info.get("height")
        if not img_w or not img_h:
            try:
                with Image.open(img_dir / img_info["file_name"]) as im:
                    img_w, img_h = im.size
            except Exception:
                continue
        lines = []
        for ann in anns_by_image.get(img_id, []):
            yolo_id = cat_id_to_yolo_id.get(ann["category_id"])
            if yolo_id is None:
                continue
            x, y, w, h = ann["bbox"]
            xc = (x + w / 2.0) / img_w
            yc = (y + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{yolo_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        stem = Path(img_info["file_name"]).stem
        label_path = labels_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")


def write_data_yaml(split_dir: Path, class_names: list):
    yaml_path = split_dir / "data.yaml"
    config = {
        "path": str(split_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def coco_to_per_image_annotations(
    coco_path: Path,
    out_annotations: Path,
    image_dir: Path,
):
    """Convert a COCO file into the per-image per-class layout expected by 06_run_sam3.py."""
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    img_id_to_name = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    anns_by_image = {}
    for ann in coco.get("annotations", []):
        img_name = img_id_to_name.get(ann["image_id"])
        if not img_name:
            continue
        cat_name = cat_id_to_name.get(ann["category_id"], "object")
        x, y, w, h = ann["bbox"]
        bbox = [x, y, x + w, y + h]
        anns_by_image.setdefault(img_name, {}).setdefault(cat_name, []).append(bbox)

    out_annotations.mkdir(parents=True, exist_ok=True)
    for img_name, classes in anns_by_image.items():
        img_stem = Path(img_name).stem
        img_folder = out_annotations / img_stem
        img_folder.mkdir(parents=True, exist_ok=True)
        for class_name, bboxes in classes.items():
            safe_name = class_name.replace(" ", "_")
            json_path = img_folder / f"{safe_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "class_name": class_name,
                        "image": str(image_dir / img_name),
                        "bboxes": bboxes,
                    },
                    f,
                    indent=2,
                )


def _nearest_keyframe_for_frames(
    keyframe_names: set,
    all_train_names: list,
) -> dict:
    """Map every train frame filename to the nearest preceding keyframe filename."""
    keyframes_sorted = sorted(keyframe_names)
    mapping = {}
    for name in all_train_names:
        # Find the last keyframe whose timestamp <= this frame's timestamp.
        nearest = None
        for kf in keyframes_sorted:
            if kf <= name:
                nearest = kf
            else:
                break
        mapping[name] = nearest
    return mapping


def propagate_interpolation(
    train_img_dir: Path,
    keyframe_dir: Path,
    reviewed_coco: Path,
    propagated_coco: Path,
    project_root: Path,
):
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "13_interpolate_tracks.py"),
        "--keyframes-coco",
        str(reviewed_coco),
        "--manifest",
        str(keyframe_dir / "keyframe_manifest.json"),
        "--image-folder",
        str(train_img_dir),
        "--output-coco",
        str(propagated_coco),
    ]
    result = subprocess.run(cmd, cwd=project_root, check=False)
    if result.returncode != 0:
        raise RuntimeError("Interpolation stage failed")


def propagate_sam3(
    train_img_dir: Path,
    keyframe_dir: Path,
    reviewed_coco: Path,
    propagated_dir: Path,
    sam3_model: str,
    device: str,
    project_root: Path,
):
    """Propagate keyframe boxes to all train frames using nearest-keyframe bboxes + SAM3."""
    per_image_dir = propagated_dir / "per_image_annotations"
    # Convert reviewed keyframe COCO into per-image per-class format.
    coco_to_per_image_annotations(reviewed_coco, per_image_dir, train_img_dir)

    # For every non-keyframe train image, create an annotation folder using the
    # nearest preceding keyframe's boxes as bbox exemplars.
    keyframe_names = {p.name for p in keyframe_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    train_files = sorted(
        p for p in train_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    train_names = [p.name for p in train_files]
    nearest_map = _nearest_keyframe_for_frames(keyframe_names, train_names)

    keyframe_annotation_dirs = {p.name: p for p in per_image_dir.iterdir() if p.is_dir()}
    for train_name in train_names:
        if train_name in keyframe_names:
            continue
        nearest_keyframe = nearest_map.get(train_name)
        if not nearest_keyframe:
            continue
        src_dir = keyframe_annotation_dirs.get(Path(nearest_keyframe).stem)
        if not src_dir:
            continue
        dst_dir = per_image_dir / Path(train_name).stem
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        # Update image path in copied JSONs to point at the current train frame.
        for json_file in dst_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["image"] = str(train_img_dir / train_name)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    sam3_out = propagated_dir / "sam3_masks"
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "06_run_sam3.py"),
        "--annotations-folder",
        str(per_image_dir),
        "--image-folder",
        str(train_img_dir),
        "--segmented-output",
        str(sam3_out),
        "--model",
        sam3_model,
        "--device",
        device,
    ]
    result = subprocess.run(cmd, cwd=project_root, check=False)
    if result.returncode != 0:
        raise RuntimeError("SAM3 propagation stage failed")


def propagate_interpolation_plus_sam3(
    train_img_dir: Path,
    keyframe_dir: Path,
    reviewed_coco: Path,
    propagated_dir: Path,
    propagated_coco: Path,
    sam3_model: str,
    device: str,
    project_root: Path,
):
    """Interpolate boxes to all train frames, then run SAM3 on every frame."""
    propagate_interpolation(
        train_img_dir, keyframe_dir, reviewed_coco, propagated_coco, project_root
    )
    per_image_dir = propagated_dir / "per_image_annotations_interpolated"
    coco_to_per_image_annotations(propagated_coco, per_image_dir, train_img_dir)

    sam3_out = propagated_dir / "sam3_masks"
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "06_run_sam3.py"),
        "--annotations-folder",
        str(per_image_dir),
        "--image-folder",
        str(train_img_dir),
        "--segmented-output",
        str(sam3_out),
        "--model",
        sam3_model,
        "--device",
        device,
    ]
    result = subprocess.run(cmd, cwd=project_root, check=False)
    if result.returncode != 0:
        raise RuntimeError("SAM3 refinement stage failed")


def propagate(
    args,
    camera: str,
    train_img_dir: Path,
    keyframe_dir: Path,
    reviewed_coco: Path,
    propagated_dir: Path,
    propagated_coco: Path,
    project_root: Path,
):
    propagated_dir.mkdir(parents=True, exist_ok=True)
    if args.propagation_method == "interpolation":
        propagate_interpolation(
            train_img_dir, keyframe_dir, reviewed_coco, propagated_coco, project_root
        )
    elif args.propagation_method == "sam3":
        propagate_sam3(
            train_img_dir,
            keyframe_dir,
            reviewed_coco,
            propagated_dir,
            args.sam3_model,
            args.device,
            project_root,
        )
    elif args.propagation_method == "interpolation+sam3":
        propagate_interpolation_plus_sam3(
            train_img_dir,
            keyframe_dir,
            reviewed_coco,
            propagated_dir,
            propagated_coco,
            args.sam3_model,
            args.device,
            project_root,
        )
    else:
        raise ValueError(f"Unknown propagation method: {args.propagation_method}")


def run_camera_pipeline(args, camera: str):
    rosbag = Path(args.rosbag).resolve()
    out_root = Path(args.output_root or f"{rosbag}_pipeline").resolve()
    project_root = get_project_root()

    raw_cam_dir = rosbag / "camera" / camera
    calibration = rosbag / "info" / "calibration.json"
    if not raw_cam_dir.exists():
        raise FileNotFoundError(f"Camera folder not found: {raw_cam_dir}")
    if not calibration.exists():
        raise FileNotFoundError(f"Calibration not found: {calibration}")

    # Stage 1: Undistort
    undistorted_dir = out_root / f"{camera}_undistorted"
    marker = undistorted_dir / "stage_completed.json"
    if should_run("undistort", args):
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "undistort_rosbag.py"),
            "--images-root",
            str(raw_cam_dir),
            "--output-root",
            str(undistorted_dir),
            "--calibration",
            str(calibration),
            "--camera-name",
            camera,
        ]
        run_stage("undistort", cmd, project_root, marker, args.force)

    # Stage 2: Time-based split
    split_dir = out_root / "splits" / camera
    marker = split_dir / "stage_completed.json"
    if should_run("split", args):
        counts = time_based_split(undistorted_dir, split_dir, args.splits)
        print(f"Split counts: {counts}")
        write_marker(marker, {"stage": "split", "ratios": args.splits, "counts": counts})

    train_img_dir = split_dir / "images" / "train"

    # Stage 3: Extract keyframes
    keyframe_dir = out_root / "keyframes" / camera
    marker = keyframe_dir / "stage_completed.json"
    if should_run("keyframes", args):
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "12_extract_keyframes.py"),
            "--image-folder",
            str(train_img_dir),
            "--output-dir",
            str(keyframe_dir),
            "--every",
            str(args.keyframe_stride),
            "--mode",
            "symlink",
        ]
        run_stage("keyframes", cmd, project_root, marker, args.force)

    # Stage 4: Qwen annotation
    qwen_dir = out_root / "qwen" / camera
    marker = qwen_dir / "stage_completed.json"
    if should_run("qwen", args):
        if not check_llamacpp_server(args.llamacpp_url):
            raise RuntimeError(
                f"llama.cpp server not reachable at {args.llamacpp_url}. "
                "Start it with: llama-server -m <model> --mmproj <mmproj> --image-min-tokens 2048 --port 8089"
            )
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "07_run_qwen.py"),
            "--backend",
            "llamacpp",
            "--llamacpp-url",
            args.llamacpp_url,
            "--llamacpp-model",
            args.qwen_model,
            "--prompt",
            NO_THINK_PREFIX + args.prompt,
            "--template",
            "object_detection",
            "--format",
            "json",
            "--image-folder",
            str(keyframe_dir),
            "--annotations-output",
            str(qwen_dir),
        ]
        if args.resume_from is not None:
            cmd.extend(["--resume-from", str(args.resume_from)])
        run_stage("qwen", cmd, project_root, marker, args.force)

    # Stage 4b: Combine per-image Qwen results into a single COCO file
    # that gui/label_review can open directly.
    qwen_coco = qwen_dir / "labels_coco.json"
    marker = qwen_dir / "qwen_coco_completed.json"
    if should_run("qwen_coco", args):
        qwen_results_dir = qwen_dir
        if not any(qwen_dir.glob("*_result.json")):
            # Older runs kept per-image results next to the keyframes.
            alt = keyframe_dir / "qwen_results"
            if any(alt.glob("*_result.json")):
                qwen_results_dir = alt
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "08c_qwen_results_to_coco.py"),
            "--qwen-results-dir",
            str(qwen_results_dir),
            "--output",
            str(qwen_coco),
            "--side",
            camera,
        ]
        run_stage("qwen_coco", cmd, project_root, marker, args.force)

    # Stage 5: Human review
    reviewed_dir = out_root / "reviewed" / camera
    reviewed_coco = reviewed_dir / "coco_reviewed.json"
    reviewed_yolo = reviewed_dir / "yolo_reviewed"
    marker = reviewed_dir / "stage_completed.json"
    if should_run("review", args):
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "08_click_review_coco.py"),
            "--qwen-annotations-dir",
            str(qwen_dir),
            "--img_dir",
            str(keyframe_dir),
            "--output_json",
            str(reviewed_coco),
            "--output-yolo-dir",
            str(reviewed_yolo),
        ]
        run_stage("review", cmd, project_root, marker, args.force)

    # Stage 6: Propagate
    propagated_dir = out_root / "propagated" / camera
    propagated_coco = propagated_dir / "coco_full.json"
    marker = propagated_dir / "stage_completed.json"
    if should_run("propagate", args):
        propagate(
            args=args,
            camera=camera,
            train_img_dir=train_img_dir,
            keyframe_dir=keyframe_dir,
            reviewed_coco=reviewed_coco,
            propagated_dir=propagated_dir,
            propagated_coco=propagated_coco,
            project_root=project_root,
        )
        write_marker(marker, {"stage": "propagate", "method": args.propagation_method})

    # Stage 7: Export YOLO labels into split layout
    marker = split_dir / "labels_exported.json"
    if should_run("export", args):
        labels_dir = split_dir / "labels" / "train"
        labels_dir.mkdir(parents=True, exist_ok=True)
        if args.propagation_method in ("interpolation", "interpolation+sam3"):
            export_yolo_from_coco(
                coco_path=propagated_coco,
                img_dir=train_img_dir,
                labels_dir=labels_dir,
                class_names=args.classes,
            )
        else:
            # For pure SAM3 mode, export from the SAM3 results JSON files.
            sam3_results_dir = propagated_dir / "sam3_masks"
            export_yolo_from_sam3_results(
                sam3_results_dir=sam3_results_dir,
                labels_dir=labels_dir,
                class_names=args.classes,
            )
        write_data_yaml(split_dir, args.classes)
        write_marker(marker, {"stage": "export"})


def export_yolo_from_sam3_results(
    sam3_results_dir: Path,
    labels_dir: Path,
    class_names: list,
):
    """Convert SAM3 per-image result JSON files to YOLO txt labels."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    for json_file in sorted(sam3_results_dir.glob("*_results.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        image_path = Path(data.get("image", ""))
        if not image_path.name or not image_path.exists():
            continue
        try:
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception:
            continue
        detections = data.get("detections", [])
        lines = []
        for det in detections:
            label = det.get("label")
            if label not in class_names:
                continue
            yolo_id = class_names.index(label)
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            xc = (x1 + w / 2.0) / img_w
            yc = (y1 + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h
            for v in (xc, yc, nw, nh):
                v = min(max(v, 0.0), 1.0)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{yolo_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        label_path = labels_dir / f"{json_file.stem.replace('_results', '')}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")


def main():
    args = parse_args()
    if args.classes is None:
        args.classes = parse_classes_from_prompt(args.prompt)
    print(f"Classes: {args.classes}")
    cameras = ["left", "right"] if args.camera == "both" else [args.camera]
    for camera in cameras:
        print(f"\n{'=' * 60}\nProcessing camera: {camera}\n{'=' * 60}")
        run_camera_pipeline(args, camera)


if __name__ == "__main__":
    main()
