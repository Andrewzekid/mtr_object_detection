#!/usr/bin/env python3
"""
End-to-end keyframe annotation pipeline orchestrator.

Main pipeline (see README "Keyframe pipeline"):

    data folder (or rosbag) -> undistort -> sample -> keyframes
    -> dataset stats -> Qwen seed labels -> combine to COCO
    -> GUI review (human fixes boxes, segments with SAM3, uses the
       autolabel/propagation/interpolation assistants, discards bad frames)
    -> COCO -> flat YOLO-seg dataset (01b) -> train/test/val split (03)
    -> augment TRAIN split only (02, mask-aware) -> assemble final dataset
       + dataset.yaml + dataset statistics CSV (01a)
    -> train (04) -> evaluate per-class metrics (05) -> tracking (11)

Every stage is skip-able/re-runnable: `--stage NAME` runs only that stage,
`--skip-stage NAME` omits it, markers (`stage_completed.json`) make re-runs
incremental, `--force` ignores markers.

Stage names: undistort, sample, keyframes, stats, qwen, qwen_coco, gui,
yolo, split, augment, assemble, train, evaluate, tracking.

Output layout under `<input>_pipeline/`:

    undistorted/<cam>/     full-res working frames (rosbag input only)
    sampled/<cam>/         random sample (--sample-size)
    keyframes/<cam>/       every Nth frame (--keyframe-stride)
    qwen/<cam>/            per-image Qwen results + labels_coco.json
    reviewed/labels_coco.json   human-reviewed COCO (GUI save target)
    dataset/yolo_flat|split|train_augmented|final (+dataset.yaml,
        dataset_statistics.csv)
    training/yolo_training/weights/best.pt
    evaluation/metrics.csv
    tracking/<cam>/

Examples:

    # From a Metacam rosbag, stereo:
    python scripts/orchestrate_pipeline.py \
        --rosbag 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52 \
        --camera both --sample-size 1000 --keyframe-stride 10 \
        --llamacpp-url http://127.0.0.1:8089 \
        --epochs 100 --batch-size 16 --model-type yolo26n --imgsz 768 \
        --tracker deepocsort

    # From an existing image folder (mono), no sampling, label everything:
    python scripts/orchestrate_pipeline.py \
        --images Datasets/HKU_GH/HKU_GH_left --gui-on all

    # Re-run only the post-GUI stages after a review session:
    python scripts/orchestrate_pipeline.py --images /data/left \
        --output-root /data/left_pipeline \
        --stage yolo --stage split --stage augment --stage assemble \
        --stage train --stage evaluate
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

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Format validators shared with run_seg_dataset_pipeline.py — every dataset
# stage is validated so a format mismatch fails at the step that caused it.
from run_seg_dataset_pipeline import (  # noqa: E402
    validate_flat_dataset,
    validate_split_dataset,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

DEFAULT_PROMPT = """Sofa: Detect any upholstered multi-person seating furniture located in lounge or waiting areas. They feature dark teal or dark green fabric, blocky rectangular structures, padded seat cushions, and low-profile backrests. Do not classify individual office chairs or bare wooden benches as sofas.

Wooden Door: Detect dark brown or reddish-brown wood-grain entrance doors set flush or recessed into interior walls. Key features include metallic horizontal lever handles and rectangular room designation or notice plaques mounted near eye level. Do not classify open glass doors or metal security gates as wooden doors.

Overhead Signage: Detect large rectangular directional or informational boards suspended high above corridors and walkways. They feature dark brown or black panels with light-colored text, arrows, or floor level indicators (e.g., "LG"). Do not classify wall-mounted posters, digital displays, or normal TV monitors as overhead signage.

Sprinkler (on the ceiling): Detect small, localized fire safety sprinkler heads protruding from or recessed into flat ceiling surfaces. They feature small circular metallic deflector plates or white ceiling escutcheon rings, often arranged in a spaced grid alongside ceiling lights or smoke detectors. Sprinklers can be very small in the image; examine the ceiling carefully, especially near lights, smoke detectors, and ceiling corners, and report each individual sprinkler head even if it is only a few pixels wide. Do not classify standard recessed spotlight fixtures, cameras, or ceiling air vents as sprinklers."""

NO_THINK_PREFIX = "Do not think or explain. Output only the final JSON.\n\n"

STAGES = [
    "undistort", "sample", "keyframes", "stats", "qwen", "qwen_coco",
    "gui", "yolo", "split", "augment", "assemble", "train", "evaluate",
    "tracking",
]


def get_project_root() -> Path:
    """Return the repository root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="End-to-end keyframe annotation pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages: " + ", ".join(STAGES),
    )

    # ---- input ------------------------------------------------------------
    g = parser.add_argument_group("input")
    g.add_argument("--rosbag", default=None,
                   help="Rosbag root with camera/<cam> + info/calibration.json "
                        "(enables the undistort stage)")
    g.add_argument("--images", default=None,
                   help="Existing image folder (mono), or parent of left/ + "
                        "right/ when --camera both. Skips undistort.")
    g.add_argument("--camera", default="left", choices=["left", "right", "both"],
                   help="Camera(s) to process (default: left)")
    g.add_argument("--output-root", default=None,
                   help="Root output directory (default: <input>_pipeline)")

    # ---- run control ------------------------------------------------------
    g = parser.add_argument_group("run control")
    g.add_argument("--stage", action="append", default=None,
                   help="Run only this stage (repeatable); see epilog")
    g.add_argument("--skip-stage", action="append", default=None,
                   help="Skip this stage (repeatable)")
    g.add_argument("--force", action="store_true",
                   help="Rerun stages even if their marker exists")
    g.add_argument("--copy", action="store_true",
                   help="Copy instead of symlink at the sample/keyframe/"
                        "assemble stages (default: symlink)")

    # ---- sample -----------------------------------------------------------
    g = parser.add_argument_group("sample (00)")
    g.add_argument("--sample-size", type=int, default=None,
                   help="Randomly keep N frames (default: keep all — stage "
                        "skipped). In stereo, right is synced to the sampled "
                        "left timestamps.")
    g.add_argument("--sample-seed", type=int, default=42,
                   help="Sampling random seed (default: 42)")

    # ---- keyframes ----------------------------------------------------------
    g = parser.add_argument_group("keyframes (12)")
    g.add_argument("--keyframe-stride", type=int, default=10,
                   help="Extract 1 keyframe every N frames (default: 10)")

    # ---- qwen seed labels ---------------------------------------------------
    g = parser.add_argument_group("qwen (07 + 08c)")
    g.add_argument("--qwen-model", default="Qwen3.8-27B-Q4_K_M.gguf",
                   help="Qwen GGUF weights name/path served by llama.cpp")
    g.add_argument("--qwen-mmproj", default="Qwen3.8-mmproj-F16.gguf",
                   help="Qwen mmproj file (must be loaded by the server)")
    g.add_argument("--llamacpp-url", default="http://127.0.0.1:8089",
                   help="llama.cpp server URL")
    g.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                   help="Detection prompt for Qwen")
    g.add_argument("--classes", nargs="+", default=None,
                   help="Class names (default: parsed from --prompt)")
    g.add_argument("--resume-from", type=int, default=None,
                   help="Resume Qwen batch from this 1-indexed image number")

    # ---- gui review ---------------------------------------------------------
    g = parser.add_argument_group("gui review")
    g.add_argument("--gui-on", default="keyframes",
                   choices=["keyframes", "all"],
                   help="Frames loaded into the review GUI: keyframes only "
                        "(sparse labeling, default) or every sampled frame "
                        "(use with the GUI's Interpolate/Propagate tools)")

    # ---- coco -> yolo (01b) -------------------------------------------------
    g = parser.add_argument_group("coco->yolo (01b)")
    g.add_argument("--bbox-as-rect", action="store_true",
                   help="Emit mask-less COCO boxes as rectangle polygons")

    # ---- split (03) ---------------------------------------------------------
    g = parser.add_argument_group("split (03)")
    g.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                   metavar=("TRAIN", "TEST", "VAL"),
                   help="Split ratios, must sum to 1.0 (default: 0.7 0.15 0.15)")
    g.add_argument("--split-seed", type=int, default=None,
                   help="Random seed for the split (default: random)")

    # ---- augment (02, train split only) -------------------------------------
    g = parser.add_argument_group("augment (02, train split only)")
    g.add_argument("--skip-augment", action="store_true",
                   help="Skip augmentation; final train split = raw split")
    g.add_argument("--augmentations", "-a", type=str, nargs="+",
                   default=["flip_horizontal", "rotate", "brightness"],
                   choices=["flip_horizontal", "flip_vertical", "rotate",
                            "brightness", "contrast", "hue", "blur", "resize",
                            "mosaic"],
                   help="Augmentation types (default: flip_horizontal rotate "
                        "brightness). Masks/polygons are transformed with "
                        "the image.")
    g.add_argument("--multiplier", "-m", type=int, default=2,
                   help="Augmented copies per train image (1-10, default: 2)")
    g.add_argument("--rotation-range", type=float, nargs=2, default=[-15, 15],
                   metavar=("MIN", "MAX"), help="degrees (default: -15 15)")
    g.add_argument("--brightness-range", type=float, nargs=2,
                   default=[0.8, 1.2], metavar=("MIN", "MAX"),
                   help="factor range (default: 0.8 1.2)")
    g.add_argument("--hue-range", type=float, nargs=2, default=[-15, 15],
                   metavar=("MIN", "MAX"), help="degrees (default: -15 15)")
    g.add_argument("--blur-range", type=int, nargs=2, default=[3, 9],
                   metavar=("MIN", "MAX"), help="kernel range (default: 3 9)")
    g.add_argument("--resize", type=int, nargs=2, default=None,
                   metavar=("WIDTH", "HEIGHT"),
                   help="Target size when 'resize' is among --augmentations")

    # ---- train (04) ---------------------------------------------------------
    g = parser.add_argument_group("train (04)")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--model-type", default="yolo26n",
                   help="Ultralytics model type (default: yolo26n)")
    g.add_argument("--task", default="segment", choices=["detect", "segment"],
                   help="Training task (default: segment — the GUI pipeline "
                        "produces segmentation masks)")
    g.add_argument("--imgsz", type=int, default=640)
    g.add_argument("--pretrained", default=None,
                   help="Pretrained weights to fine-tune from")
    g.add_argument("--lr0", type=float, default=0.01,
                   help="Initial learning rate (default: 0.01)")
    g.add_argument("--device", default="0",
                   help="Training/tracking device: '0', 'cpu', ... (default: 0)")

    # ---- evaluate (05) --------------------------------------------------------
    g = parser.add_argument_group("evaluate (05)")
    g.add_argument("--eval-splits", nargs="+", default=["test", "train"],
                   help="Splits to evaluate (default: test train)")
    g.add_argument("--eval-conf", type=float, default=0.5,
                   help="Confidence threshold (default: 0.5)")
    g.add_argument("--eval-iou", type=float, default=0.5,
                   help="IoU threshold for matching (default: 0.5)")

    # ---- tracking (11) --------------------------------------------------------
    g = parser.add_argument_group("tracking (11)")
    g.add_argument("--tracker", default="deepocsort",
                   choices=["bytetrack", "botsort", "ocsort", "deepocsort"],
                   help="Tracker (default: deepocsort)")
    g.add_argument("--track-conf", type=float, default=0.5,
                   help="Detection confidence threshold (default: 0.5)")
    g.add_argument("--track-iou", type=float, default=0.45,
                   help="NMS IoU threshold (default: 0.45)")
    g.add_argument("--track-imgsz", type=int, default=768,
                   help="Inference image size (default: 768)")
    g.add_argument("--track-fps", type=int, default=10,
                   help="Output video FPS (default: 10)")
    g.add_argument("--track-max-frames", type=int, default=None,
                   help="Track only the first N frames (default: all)")
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


def run_stage(stage_name: str, cmd: list, cwd: Path, marker_path: Path,
              force: bool = False):
    """Run a stage command unless its marker exists."""
    if not force and read_marker(marker_path):
        print(f"[skip] {stage_name}: marker exists at {marker_path}")
        return None
    print(f"[run] {stage_name}: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd], cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Stage {stage_name} failed with exit code {result.returncode}")
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


def iter_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def link_or_copy(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


# ---------------------------------------------------------------------------
# Stereo COCO merge (qwen_coco stage)
# ---------------------------------------------------------------------------

def merge_coco_sides(side_cocos: dict, out_path: Path):
    """Merge per-camera COCO files (each image carries a "side" field) into
    one file the GUI can open for a stereo session. Categories are merged by
    name; image/annotation ids are renumbered."""
    merged = {"images": [], "annotations": [], "categories": []}
    cat_ids: dict = {}  # name -> new id
    next_img = 1
    next_ann = 1
    for side in sorted(side_cocos):
        with open(side_cocos[side], "r", encoding="utf-8") as f:
            coco = json.load(f)
        cat_remap = {}
        for cat in coco.get("categories", []):
            name = cat["name"]
            if name not in cat_ids:
                cat_ids[name] = len(cat_ids)
                merged["categories"].append(
                    {"id": cat_ids[name], "name": name})
            cat_remap[cat["id"]] = cat_ids[name]
        img_remap = {}
        for img in coco.get("images", []):
            img = dict(img)
            img.setdefault("side", side)
            img_remap[img["id"]] = next_img
            img["id"] = next_img
            next_img += 1
            merged["images"].append(img)
        for ann in coco.get("annotations", []):
            ann = dict(ann)
            ann["id"] = next_ann
            next_ann += 1
            ann["image_id"] = img_remap[ann["image_id"]]
            ann["category_id"] = cat_remap[ann["category_id"]]
            merged["annotations"].append(ann)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f)


# ---------------------------------------------------------------------------
# Per-camera working directories
# ---------------------------------------------------------------------------

class CamPaths:
    """Resolve the per-camera folders for one pipeline run."""

    def __init__(self, args, out_root: Path):
        self.out_root = out_root
        self.stereo = args.camera == "both"
        self.cameras = ["left", "right"] if self.stereo else [args.camera]
        # Raw input folders per camera.
        self.raw: dict = {}
        if args.rosbag:
            rosbag = Path(args.rosbag).resolve()
            for cam in self.cameras:
                d = rosbag / "camera" / cam
                if not d.exists():
                    raise FileNotFoundError(f"Camera folder not found: {d}")
                self.raw[cam] = d
            self.calibration = rosbag / "info" / "calibration.json"
            if not self.calibration.exists():
                raise FileNotFoundError(
                    f"Calibration not found: {self.calibration}")
        else:
            images = Path(args.images).resolve()
            if self.stereo:
                for cam in self.cameras:
                    d = images / cam
                    if not d.exists():
                        raise FileNotFoundError(
                            f"--camera both: expected {d} (parent of left/ "
                            f"+ right/)")
                    self.raw[cam] = d
            else:
                if not images.exists():
                    raise FileNotFoundError(
                        f"Images folder not found: {images}")
                self.raw[self.cameras[0]] = images
            self.calibration = None

    def undistorted(self, cam: str) -> Path:
        """Full-resolution working frames (undistorted when applicable)."""
        return (self.out_root / "undistorted" / cam) if self.calibration \
            else self.raw[cam]

    def sampled(self, cam: str) -> Path:
        return self.out_root / "sampled" / cam

    def keyframes(self, cam: str) -> Path:
        return self.out_root / "keyframes" / cam

    def qwen(self, cam: str) -> Path:
        return self.out_root / "qwen" / cam

    def working(self, cam: str, sample: bool) -> Path:
        """The frame pool everything downstream draws from."""
        return self.sampled(cam) if sample else self.undistorted(cam)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_undistort(args, paths: CamPaths):
    if not paths.calibration:
        print("[skip] undistort: no --rosbag given (input used as-is)")
        return
    for cam in paths.cameras:
        marker = paths.undistorted(cam) / "stage_completed.json"
        cmd = [sys.executable, SCRIPTS_DIR / "undistort_rosbag.py",
               "--images-root", paths.raw[cam],
               "--output-root", paths.undistorted(cam),
               "--calibration", paths.calibration,
               "--camera-name", cam]
        run_stage(f"undistort[{cam}]", cmd, get_project_root(), marker,
                  args.force)


def stage_sample(args, paths: CamPaths):
    if args.sample_size is None:
        print("[skip] sample: --sample-size not given (keeping all frames)")
        return
    first = paths.cameras[0]
    marker = paths.sampled(first) / "stage_completed.json"
    cmd = [sys.executable, SCRIPTS_DIR / "00_sample_from_dataset.py",
           "--source-dir", paths.undistorted(first),
           "--out-dir", paths.sampled(first),
           "-n", str(args.sample_size),
           "--seed", str(args.sample_seed)]
    if not args.copy:
        cmd.append("--symlink")
    run_stage(f"sample[{first}]", cmd, get_project_root(), marker, args.force)

    if paths.stereo:
        # Sync the right side to the sampled left timestamps (the GUI pairs
        # frames by timestamp filename, so both sides must carry the same
        # names).
        cam = paths.cameras[1]
        marker = paths.sampled(cam) / "stage_completed.json"
        if not args.force and read_marker(marker):
            print(f"[skip] sample[{cam}]: marker exists")
            return
        names = {p.name for p in iter_images(paths.sampled(first))}
        out_dir = paths.sampled(cam)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for img in iter_images(paths.undistorted(cam)):
            if img.name in names:
                link_or_copy(img, out_dir / img.name, args.copy)
                n += 1
        print(f"[sample:{cam}] synced {n}/{len(names)} frames to the "
              f"{first} timestamps")
        write_marker(marker, {"stage": f"sample[{cam}]", "synced": n})


def stage_keyframes(args, paths: CamPaths):
    for cam in paths.cameras:
        marker = paths.keyframes(cam) / "stage_completed.json"
        cmd = [sys.executable, SCRIPTS_DIR / "12_extract_keyframes.py",
               "--image-folder", paths.working(cam, args.sample_size),
               "--output-dir", paths.keyframes(cam),
               "--every", str(args.keyframe_stride),
               "--mode", "copy" if args.copy else "symlink"]
        run_stage(f"keyframes[{cam}]", cmd, get_project_root(), marker,
                  args.force)


def stage_stats(args, paths: CamPaths):
    """Pre-label statistics: frame counts at each reduction level."""
    marker = paths.out_root / "stats" / "stage_completed.json"
    if not args.force and read_marker(marker):
        print(f"[skip] stats: marker exists at {marker}")
        return
    stats = {}
    for cam in paths.cameras:
        entry = {"raw": len(iter_images(paths.undistorted(cam)))}
        if args.sample_size is not None:
            entry["sampled"] = len(iter_images(paths.sampled(cam)))
        src = paths.working(cam, args.sample_size)
        entry["keyframes"] = len(iter_images(paths.keyframes(cam)))
        entry["gui_frames"] = len(iter_images(
            paths.keyframes(cam) if args.gui_on == "keyframes" else src))
        stats[cam] = entry
    print(json.dumps(stats, indent=2))
    write_marker(marker, {"stage": "stats", "counts": stats})


def stage_qwen(args, paths: CamPaths):
    for cam in paths.cameras:
        marker = paths.qwen(cam) / "stage_completed.json"
        if not (args.force or read_marker(marker)) \
                and not check_llamacpp_server(args.llamacpp_url):
            raise RuntimeError(
                f"llama.cpp server not reachable at {args.llamacpp_url}. "
                "Start it with: llama-server -m <model> --mmproj <mmproj> "
                "--image-min-tokens 2048 --port 8089")
        cmd = [sys.executable, SCRIPTS_DIR / "07_run_qwen.py",
               "--backend", "llamacpp",
               "--llamacpp-url", args.llamacpp_url,
               "--llamacpp-model", args.qwen_model,
               "--prompt", NO_THINK_PREFIX + args.prompt,
               "--template", "object_detection",
               "--format", "json",
               "--image-folder", paths.keyframes(cam),
               "--annotations-output", paths.qwen(cam)]
        if args.resume_from is not None:
            cmd.extend(["--resume-from", str(args.resume_from)])
        run_stage(f"qwen[{cam}]", cmd, get_project_root(), marker, args.force)


def stage_qwen_coco(args, paths: CamPaths, reviewed_coco: Path):
    """Per-camera qwen results -> per-camera COCO -> one (stereo-merged)
    COCO the GUI opens for review."""
    marker = paths.out_root / "reviewed" / "qwen_coco_completed.json"
    if not args.force and read_marker(marker):
        print(f"[skip] qwen_coco: marker exists at {marker}")
        return
    side_cocos = {}
    for cam in paths.cameras:
        coco_path = paths.qwen(cam) / "labels_coco.json"
        cmd = [sys.executable, SCRIPTS_DIR / "08c_qwen_results_to_coco.py",
               "--qwen-results-dir", paths.qwen(cam),
               "--output", coco_path,
               "--side", cam]
        run_stage(f"qwen_coco[{cam}]", cmd, get_project_root(),
                  paths.qwen(cam) / "qwen_coco_completed.json", args.force)
        side_cocos[cam] = coco_path
    if paths.stereo:
        merge_coco_sides(side_cocos, reviewed_coco)
    else:
        reviewed_coco.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(side_cocos[paths.cameras[0]], reviewed_coco)
    print(f"[qwen_coco] review COCO: {reviewed_coco}")
    write_marker(marker, {"stage": "qwen_coco"})


def stage_gui(args, paths: CamPaths, reviewed_coco: Path):
    """Launch the label-review GUI and wait for the human session to end.

    The GUI opens ``reviewed_coco`` (seeded by the qwen_coco stage), the
    human reviews/corrects boxes, segments with SAM3/autolabel assistants,
    and saves on exit. The COCO must exist afterwards for the run to count
    as complete.
    """
    marker = paths.out_root / "reviewed" / "stage_completed.json"
    if not args.force and read_marker(marker):
        print(f"[skip] gui: marker exists at {marker}")
        return
    first = paths.cameras[0]

    def gui_dir(cam):
        return paths.keyframes(cam) if args.gui_on == "keyframes" \
            else paths.working(cam, args.sample_size)

    cmd = [sys.executable, "-m", "gui.label_review.main",
           "--images", gui_dir(first),
           "--output_json", reviewed_coco]
    if paths.stereo:
        cmd += ["--images-right", gui_dir(paths.cameras[1])]
    print(f"[run] gui: {' '.join(str(c) for c in cmd)}")
    print("      (the pipeline resumes when you close the GUI — save first!)")
    result = subprocess.run([str(c) for c in cmd], cwd=get_project_root(),
                            check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"GUI exited with code {result.returncode}; fix the issue and "
            f"re-run with --stage gui (or --skip-stage gui to use the "
            f"existing {reviewed_coco})")
    if not reviewed_coco.exists():
        raise RuntimeError(
            f"GUI closed without saving {reviewed_coco} — re-run with "
            f"--stage gui and save before closing")
    write_marker(marker, {"stage": "gui", "coco": str(reviewed_coco)})


def stage_yolo(args, paths: CamPaths, reviewed_coco: Path):
    """Reviewed COCO -> flat YOLO-seg dataset (01b)."""
    out_dir = paths.out_root / "dataset" / "yolo_flat"
    marker = out_dir / "stage_completed.json"
    first = paths.cameras[0]

    def gui_dir(cam):
        return paths.keyframes(cam) if args.gui_on == "keyframes" \
            else paths.working(cam, args.sample_size)

    # 01b resolves images as <images-dir>/<side>/<file_name> in stereo.
    images_dir = gui_dir(first).parent if paths.stereo else gui_dir(first)
    cmd = [sys.executable, SCRIPTS_DIR / "01b_coco_to_yolo_seg.py",
           "--coco-json", reviewed_coco,
           "--images-dir", images_dir,
           "--output-dir", out_dir]
    if args.bbox_as_rect:
        cmd.append("--bbox-as-rect")
    run_stage("yolo", cmd, get_project_root(), marker, args.force)
    stats = validate_flat_dataset(out_dir)
    print(f"  [validated] {stats['images']} images, "
          f"{stats['label_lines']} polygons")


def stage_split(args, paths: CamPaths):
    out_dir = paths.out_root / "dataset" / "split"
    marker = out_dir / "stage_completed.json"
    cmd = [sys.executable, SCRIPTS_DIR / "03_split_dataset.py",
           "--input-dir", paths.out_root / "dataset" / "yolo_flat",
           "--output-dir", out_dir,
           "--ratios", *(str(v) for v in args.ratios)]
    if args.split_seed is not None:
        cmd += ["--seed", str(args.split_seed)]
    run_stage("split", cmd, get_project_root(), marker, args.force)
    for name, s in validate_split_dataset(out_dir).items():
        print(f"  [validated] {name}: {s['images']} images, "
              f"{s['label_lines']} polygons")


def stage_augment(args, paths: CamPaths):
    """Augment the TRAIN split only (val/test stay clean for honest eval)."""
    if args.skip_augment:
        print("[skip] augment: --skip-augment")
        return
    split_dir = paths.out_root / "dataset" / "split"
    out_dir = paths.out_root / "dataset" / "train_augmented"
    marker = out_dir / "stage_completed.json"
    cmd = [sys.executable, SCRIPTS_DIR / "02_augment_data.py",
           "--input-dir", split_dir,
           "--images-subdir", "train/images",
           "--labels-subdir", "train/labels",
           "--output-dir", out_dir,
           "--augmentations", *args.augmentations,
           "--multiplier", str(args.multiplier),
           "--rotation-range", *(str(v) for v in args.rotation_range),
           "--brightness-range", *(str(v) for v in args.brightness_range),
           "--hue-range", *(str(v) for v in args.hue_range),
           "--blur-range", *(str(v) for v in args.blur_range)]
    if args.resize:
        cmd += ["--resize", *(str(v) for v in args.resize)]
    run_stage("augment", cmd, get_project_root(), marker, args.force)
    stats = validate_flat_dataset(out_dir / "train", len(args.classes))
    print(f"  [validated] augmented train: {stats['images']} images, "
          f"{stats['label_lines']} polygons")


def stage_assemble(args, paths: CamPaths):
    """Build the final dataset tree + dataset.yaml + statistics CSV.

    Layout: final/{train,val,test}/{images,labels}; train points at the
    augmented train split when augmentation ran, val/test at the raw split.
    """
    dataset_dir = paths.out_root / "dataset"
    final_dir = dataset_dir / "final"
    marker = final_dir / "stage_completed.json"
    if not args.force and read_marker(marker):
        print(f"[skip] assemble: marker exists at {marker}")
        return
    split_dir = dataset_dir / "split"
    train_src = split_dir / "train" if args.skip_augment \
        else dataset_dir / "train_augmented" / "train"
    if not train_src.is_dir():
        raise RuntimeError(f"assemble: train split not found at {train_src}")
    for split, src in [("train", train_src),
                       ("val", split_dir / "val"),
                       ("test", split_dir / "test")]:
        for kind in ("images", "labels"):
            src_dir = src / kind
            if not src_dir.is_dir():
                continue
            for f in sorted(src_dir.iterdir()):
                if f.is_file():
                    link_or_copy(f, final_dir / split / kind / f.name,
                                 args.copy)
    yaml_path = final_dir / "dataset.yaml"
    config = {
        "path": str(final_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(args.classes),
        "names": list(args.classes),
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    for name, s in validate_split_dataset(final_dir).items():
        print(f"  [validated] final/{name}: {s['images']} images, "
              f"{s['label_lines']} polygons")
    # Full per-class dataset statistics CSV (01a).
    csv_path = dataset_dir / "dataset_statistics.csv"
    cmd = [sys.executable, SCRIPTS_DIR / "01a_dataset_statistics.py",
           "--input-dir", final_dir, "--csv", csv_path]
    result = subprocess.run([str(c) for c in cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError("assemble: 01a_dataset_statistics failed")
    print(f"  dataset statistics: {csv_path}")
    write_marker(marker, {"stage": "assemble"})


def stage_train(args, paths: CamPaths):
    out_dir = paths.out_root / "training"
    marker = out_dir / "stage_completed.json"
    cmd = [sys.executable, SCRIPTS_DIR / "04_train_model.py",
           "--config", paths.out_root / "dataset" / "final" / "dataset.yaml",
           "--output-dir", out_dir,
           "--epochs", str(args.epochs),
           "--batch-size", str(args.batch_size),
           "--model-type", args.model_type,
           "--task", args.task,
           "--imgsz", str(args.imgsz),
           "--device", args.device,
           "--lr0", str(args.lr0)]
    if args.pretrained:
        cmd += ["--pretrained", args.pretrained]
    run_stage("train", cmd, get_project_root(), marker, args.force)
    if not best_weights(paths).exists():
        raise RuntimeError(
            f"train: best weights not found at {best_weights(paths)}")


def best_weights(paths: CamPaths) -> Path:
    return paths.out_root / "training" / "yolo_training" / "weights" / "best.pt"


def stage_evaluate(args, paths: CamPaths):
    out_dir = paths.out_root / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "stage_completed.json"
    cmd = [sys.executable, SCRIPTS_DIR / "05_evaluate_model.py",
           "--model", best_weights(paths),
           "--data", paths.out_root / "dataset" / "final" / "dataset.yaml",
           "--split", *args.eval_splits,
           "--csv", out_dir / "metrics.csv",
           "--conf", str(args.eval_conf),
           "--iou", str(args.eval_iou)]
    run_stage("evaluate", cmd, get_project_root(), marker, args.force)


def stage_tracking(args, paths: CamPaths):
    """Track on the full (pre-keyframe) frame sequence, per camera."""
    for cam in paths.cameras:
        out_dir = paths.out_root / "tracking" / cam
        marker = out_dir / "stage_completed.json"
        cmd = [sys.executable, SCRIPTS_DIR / "11_run_tracking.py",
               "--tracker", args.tracker,
               "--model", best_weights(paths),
               "--data", paths.working(cam, args.sample_size),
               "--output", out_dir,
               "--conf", str(args.track_conf),
               "--iou", str(args.track_iou),
               "--imgsz", str(args.track_imgsz),
               "--fps", str(args.track_fps),
               "--device", args.device]
        if args.track_max_frames is not None:
            cmd += ["--max-frames", str(args.track_max_frames)]
        run_stage(f"tracking[{cam}]", cmd, get_project_root(), marker,
                  args.force)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_pipeline(args):
    if not args.rosbag and not args.images:
        raise SystemExit("Error: one of --rosbag or --images is required")
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"Error: --ratios must sum to 1.0, got {args.ratios}")
    if not args.skip_augment \
            and "resize" in args.augmentations and not args.resize:
        raise SystemExit(
            "Error: 'resize' augmentation requires --resize WIDTH HEIGHT")

    src = Path(args.rosbag or args.images).resolve()
    out_root = Path(
        args.output_root or f"{src}_pipeline").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    paths = CamPaths(args, out_root)
    reviewed_coco = out_root / "reviewed" / "labels_coco.json"

    print(f"Classes: {args.classes}")
    print(f"Cameras: {', '.join(paths.cameras)}")
    print(f"Output root: {out_root}")

    stages = [
        ("undistort", lambda: stage_undistort(args, paths)),
        ("sample", lambda: stage_sample(args, paths)),
        ("keyframes", lambda: stage_keyframes(args, paths)),
        ("stats", lambda: stage_stats(args, paths)),
        ("qwen", lambda: stage_qwen(args, paths)),
        ("qwen_coco", lambda: stage_qwen_coco(args, paths, reviewed_coco)),
        ("gui", lambda: stage_gui(args, paths, reviewed_coco)),
        ("yolo", lambda: stage_yolo(args, paths, reviewed_coco)),
        ("split", lambda: stage_split(args, paths)),
        ("augment", lambda: stage_augment(args, paths)),
        ("assemble", lambda: stage_assemble(args, paths)),
        ("train", lambda: stage_train(args, paths)),
        ("evaluate", lambda: stage_evaluate(args, paths)),
        ("tracking", lambda: stage_tracking(args, paths)),
    ]
    for name, fn in stages:
        if should_run(name, args):
            fn()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Reviewed COCO:   {reviewed_coco}")
    print(f"  Final dataset:   {out_root / 'dataset' / 'final'}")
    print(f"  Dataset stats:   {out_root / 'dataset' / 'dataset_statistics.csv'}")
    print(f"  Best weights:    {best_weights(paths)}")
    print(f"  Eval metrics:    {out_root / 'evaluation' / 'metrics.csv'}")
    print(f"  Tracking:        {out_root / 'tracking'}")


def main():
    args = parse_args()
    if args.classes is None:
        args.classes = parse_classes_from_prompt(args.prompt)
    run_pipeline(args)


if __name__ == "__main__":
    main()
