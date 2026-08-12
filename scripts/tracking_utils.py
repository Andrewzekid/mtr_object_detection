#!/usr/bin/env python3
"""
Shared helpers for tracking scripts.

Utilities included:
- Ultralytics tracker YAML generation (BoT-SORT).
- COCO-style polygon/area helpers.
- IoU, mask-to-polygon conversions.
- Summary video creation.

These functions are intentionally stateless so they can be reused by multiple
script entry points without creating a tight coupling to any one pipeline.
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def find_ultralytics_trackers_dir() -> Path:
    """Locate ultralytics/cfg/trackers inside the active Python env."""
    import ultralytics

    return Path(ultralytics.__file__).parent / "cfg" / "trackers"


# ---------------------------------------------------------------------------
# DINO ReID encoder
# ---------------------------------------------------------------------------

class DinoReIDEncoder:
    """ReID appearance encoder backed by a DINOv2/DINOv3 ViT.

    Ultralytics' built-in ``ReID`` class only handles YOLO ``.pt`` checkpoints
    or ONNX-style backends, so DINO ViTs need a small adapter that implements the
    same ``(img, dets) -> list[np.ndarray]`` callable interface used by
    BoT-SORT's ``build_encoder``. Each detection crop is resized to the ViT's
    input size, normalized, and passed through the backbone; the pooled token
    embedding (CLS or patch average) is returned per detection.

    Two loading paths are supported:

    * **torch.hub** — for public DINOv2 models (e.g. ``dinov2_vits14``)
      downloaded from the ``facebookresearch/dinov2`` hub repo.
    * **HuggingFace transformers** — for DINOv3 models stored as
      ``config.json`` + ``model.safetensors`` in a local directory (e.g. a
      mirror of a gated repo). Detected when ``model_name`` is a path to a
      directory containing ``config.json``.

    Args:
        model_name: either a torch.hub entry point (e.g. ``"dinov2_vits14"``)
            or a path to a local HuggingFace checkpoint directory.
        device: torch device string (e.g. ``"cuda"`` or ``"cuda:0"``). Defaults
            to CUDA if available.
        imgsz: square input size for the ViT. Defaults to 224 (DINO's native
            pretraining resolution for the small variants).
    """

    def __init__(self, model_name: str = "dinov2_vits14", device: str | None = None,
                 imgsz: int = 224):
        import torch

        self.model_name = model_name
        self.imgsz = imgsz
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model_path = Path(str(model_name))
        if model_path.is_dir() and (model_path / "config.json").exists():
            # HuggingFace transformers loading path (DINOv3 local checkpoint).
            self._load_hf_transformers(model_path)
        elif model_name.startswith("dinov3"):
            repo = "facebookresearch/dinov3"
            self._load_torch_hub(model_name, repo)
        elif model_name.startswith("dinov2"):
            repo = "facebookresearch/dinov2"
            self._load_torch_hub(model_name, repo)
        else:
            raise ValueError(
                f"Unknown DINO model '{model_name}'. Expected a name starting with "
                "'dinov2', 'dinov3', or a path to a HuggingFace checkpoint directory."
            )

        # Standard ImageNet normalization (DINO uses the same).
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        print(f"[DinoReID] {model_name} ready on {self.device}, embed_dim={self.embed_dim}")

    def _load_torch_hub(self, model_name: str, repo: str):
        """Load a DINOv2/DINOv3 model via torch.hub."""
        import torch

        print(f"[DinoReID] Loading {model_name} from {repo} ...")
        self.model = torch.hub.load(repo, model_name, trust_repo=True)
        self.model.eval().to(self.device)
        self.embed_dim = getattr(self.model, "embed_dim", None)
        self._backend = "torch_hub"

    def _load_hf_transformers(self, model_path: Path):
        """Load a DINOv3 model from a local HuggingFace checkpoint directory."""
        import torch
        from transformers import AutoModel, AutoImageProcessor

        print(f"[DinoReID] Loading DINOv3 from HuggingFace checkpoint: {model_path}")
        self.model = AutoModel.from_pretrained(
            str(model_path), trust_remote_code=True
        ).eval().to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(str(model_path))
        self.embed_dim = self.model.config.hidden_size
        self._backend = "hf_transformers"

    @staticmethod
    def _crop_detections(img, dets):
        """Crop detection regions from a BGR image. ``dets`` is xywh (N,>=4)."""
        import torch
        from ultralytics.utils.ops import xywh2xyxy
        from ultralytics.utils.plotting import save_one_box

        return [save_one_box(det, img, save=False)
                for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]

    def __call__(self, img, dets):
        """Extract L2-normalized embeddings for each detection crop.

        Args:
            img: BGR image (H, W, 3) as a numpy array.
            dets: (N, >=4) array of detections in xywh format (only first 4
                  columns are used).

        Returns:
            list of (embed_dim,) numpy arrays, one per detection.
        """
        import torch

        if len(dets) == 0:
            return []

        crops = self._crop_detections(img, dets)

        if self._backend == "hf_transformers":
            return self._encode_hf(crops)

        # torch.hub backend
        batch = torch.empty(len(crops), 3, self.imgsz, self.imgsz,
                            dtype=torch.float32, device=self.device)
        for i, c in enumerate(crops):
            # c is BGR; convert to RGB, to CHW float, normalize, resize.
            t = torch.from_numpy(np.ascontiguousarray(c[..., ::-1])).permute(2, 0, 1).float() / 255.0
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(self.imgsz, self.imgsz),
                mode="bilinear", align_corners=False
            )[0]
            batch[i] = t

        # Normalize per-channel.
        batch = (batch - self.mean) / self.std

        with torch.no_grad():
            out = self.model(batch)
            if isinstance(out, (tuple, list)):
                feats = out[0]
            else:
                feats = out
            if feats.ndim == 3:
                feats = feats[:, 1:].mean(dim=1) if feats.shape[1] > 1 else feats[:, 0]
            feats = feats.cpu().numpy()

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        feats = feats / norms
        return [f for f in feats]

    def _encode_hf(self, crops):
        """Encode crops using the HuggingFace transformers backend."""
        import torch
        from PIL import Image

        pil_crops = [Image.fromarray(c[..., ::-1]) for c in crops]  # BGR -> RGB -> PIL
        inputs = self.processor(images=pil_crops, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            out = self.model(pixel_values=pixel_values)
            # last_hidden_state: (B, N_tokens, D); CLS token at index 0.
            feats = out.last_hidden_state[:, 0]
            feats = feats.cpu().numpy()

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        feats = feats / norms
        return [f for f in feats]


def build_dino_reid_encoder(model_name: str, device: str | None = None, imgsz: int = 224):
    """Convenience factory: return a DinoReIDEncoder or None on failure."""
    try:
        return DinoReIDEncoder(model_name=model_name, device=device, imgsz=imgsz)
    except Exception as e:
        print(f"[DinoReID] Failed to load {model_name}: {e}")
        return None


def build_runtime_tracker_yaml(
    base_yaml: Path,
    tracker_type: str,
    with_cmc: bool,
    cmc_method: str,
    track_buffer: int,
    track_high_thresh: float,
    with_reid: bool,
    output_dir: Path,
    reid_model: str | None = None,
) -> Path:
    """Merge CLI overrides into a copy of a tracker YAML for this run.

    Ultralytics resolves the ``tracker`` argument by name inside its trackers
    dir, so we write a single-file config (with the same fields) next to the
    output and pass its path to ``model.track(...)``.
    """
    import yaml

    runtime_dir = output_dir / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / f"{tracker_type}_runtime.yaml"

    with open(base_yaml, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg["tracker_type"] = tracker_type
    cfg["track_high_thresh"] = float(track_high_thresh)
    cfg["track_buffer"] = int(track_buffer)
    cfg["with_reid"] = bool(with_reid)

    if with_cmc:
        cfg["gmc_method"] = cmc_method
    else:
        cfg["gmc_method"] = "none"

    # Set the ReID model field. "auto" means use the detector's backbone
    # features; any other value is a model name/path that build_encoder
    # resolves. For DINO models (dinov2*/dinov3*) the value is passed through
    # to our patched build_encoder which instantiates DinoReIDEncoder.
    if reid_model is not None:
        cfg["model"] = reid_model

    with open(runtime_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return runtime_path


def bbox_iou(a, b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def mask_to_polygon(mask: np.ndarray) -> list:
    """Convert a binary mask to a flattened COCO-style polygon [x1,y1,x2,y2,...].

    Returns the largest external contour. Empty mask -> empty list.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    eps = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    poly = approx.reshape(-1, 2).astype(float)
    return [float(coord) for pt in poly for coord in pt]


def polygon_area(poly: list) -> float:
    """Shoelace area for a flat polygon list."""
    n = len(poly) // 2
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        x_i = poly[2 * i]
        y_i = poly[2 * i + 1]
        x_j = poly[2 * j]
        y_j = poly[2 * j + 1]
        area += x_i * y_j - x_j * y_i
    return abs(area) / 2.0


def find_image_files(data_path: Path, extensions=None):
    """Return sorted list of image files in a directory."""
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([f for f in data_path.iterdir() if f.suffix.lower() in extensions])


def create_tracking_video(output_dir: Path, image_files, fps: int = 10):
    """Create a video from a list of image paths.

    image_files may be Path objects to either the original images or already
    annotated images. The caller decides which set to pass.
    """
    if not image_files:
        return

    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        return

    height, width = first_img.shape[:2]
    video_path = output_dir / "tracking_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for image_path in image_files:
        frame = cv2.imread(str(image_path))
        if frame is not None:
            out.write(frame)

    out.release()
    print(f"Tracking video saved to: {video_path}")


def write_coco_results(
    output_dir: Path,
    images: list,
    annotations: list,
    categories: list,
    description: str = "Tracking results",
):
    """Write a COCO-style JSON file to output_dir/results.json."""
    coco_output = {
        "info": {
            "description": description,
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        import json

        json.dump(coco_output, f, indent=2)
    print(f"Tracking JSON saved to: {json_path}")
