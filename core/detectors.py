"""Unified open-set detectors for GUI auto-labelling.

Consolidates the three zero-shot / open-vocabulary detector backends that
previously lived in separate scripts:

* **OWLv2**        - ``owlv2_detect`` (text-prompted) and
  ``owlv2_detect_exemplar`` (1-shot image-guided).
* **Grounding DINO** - ``grounding_dino_detect``.
* **Falcon Perception** - ``falcon_detect`` (grounding + instance masks).

All detect functions share one contract. Inputs are an image (path or RGB
numpy array), a list of text queries, optional model override, device, a
confidence/box threshold where applicable, and an optional ``_state`` dict
that lets callers (the batch worker) reuse the loaded model across frames.

Detections come back in the same dict shape the SAM3 autolabel path
produces::

    {label, bbox_xyxy, mask: None-or-HxW-bool, confidence, cat_id (added
    by caller)}

Models load lazily on first use, are cached per (model_id, device), and a
CUDA OOM falls back to CPU once with the fallback remembered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

DEFAULT_OWLV2_MODEL = "google/owlv2-large-patch14-ensemble"
DEFAULT_GDINO_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_FALCON_MODEL = "tiiuae/Falcon-Perception"

_model_cache: Dict[tuple, Any] = {}


def _to_pil(image) -> "PILImage.Image":
    """Accept a path or RGB(HWC uint8) array; return an RGB PIL image."""
    from PIL import Image as PILImage

    if isinstance(image, (str, Path)):
        return PILImage.open(image).convert("RGB")
    return PILImage.fromarray(np.asarray(image)[..., :3])


def _load_cached(key: tuple, loader: Callable[[], Any]):
    """Module-level cache shared by all backends."""
    if key not in _model_cache:
        _model_cache[key] = loader()
    return _model_cache[key]


# --------------------------------------------------------------------------- #
# OWLv2
# --------------------------------------------------------------------------- #

def load_owlv2(model_id: Optional[str] = None, device: str = "cuda"):
    """Load (and cache) the OWLv2 processor + model for one device."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model_id = model_id or DEFAULT_OWLV2_MODEL

    def _load():
        proc = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)
        return proc, model.to(device).eval()

    return _load_cached((model_id, device), _load)


def _owlv2_prepare(state: Optional[dict], model_id: str):
    """Return the OWLv2 (proc, model) pair for this call, handling OOM."""
    state = state if state is not None else {}
    device = state.get("device", "cuda")
    key = (DEFAULT_OWLV2_MODEL if model_id is None else model_id, device)
    if state.get("model_key") != key:
        try:
            state["proc"], state["model"] = load_owlv2(key[0], device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["proc"], state["model"] = load_owlv2(key[0], device)
                state["model_key"] = (key[0], device)
            else:
                raise
    return state["proc"], state["model"], device


def _owlv2_postprocess(results, queries: List[str],
                       label_override: Optional[str] = None,
                       img_size: Optional[tuple] = None
                       ) -> List[Dict[str, Any]]:
    """OWLv2 result dict -> standard detection dicts.

    OWLv2's post-processing rescales boxes to the target size but never
    clips them, so predictions regularly stick out past the image edges;
    when ``img_size`` (w, h) is given, boxes are clamped to the image and
    degenerate ones are dropped."""
    dets: List[Dict[str, Any]] = []
    boxes = results.get("boxes")
    if boxes is None or len(boxes) == 0:
        return dets
    scores = results.get("scores")
    labels = results.get("labels")
    for i, box in enumerate(boxes.tolist()):
        x1, y1, x2, y2 = (float(v) for v in box)
        if img_size is not None:
            w_img, h_img = img_size
            x1, x2 = max(0.0, min(x1, w_img)), max(0.0, min(x2, w_img))
            y1, y2 = max(0.0, min(y1, h_img)), max(0.0, min(y2, h_img))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue  # fully (or almost) outside the image
        dets.append({
            "label": label_override if label_override is not None else (
                queries[int(labels[i])] if labels is not None and
                int(labels[i]) < len(queries) else "object"),
            "bbox_xyxy": [x1, y1, x2, y2],
            "mask": None,
            "confidence": float(scores[i]) if scores is not None else 1.0,
        })
    return dets


def owlv2_detect(image, queries: List[str],
                 model_id: Optional[str] = None, device: str = "cuda",
                 conf: float = 0.3,
                 _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Run text-prompted OWLv2 detection on one image."""
    import torch

    proc, model, device = _owlv2_prepare(_state, model_id)
    pil = _to_pil(image)

    try:
        inputs = proc(text=[list(queries)], images=pil, return_tensors="pt")
    except TypeError:  # older signature without nested query lists
        inputs = proc(text=list(queries), images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([pil.size[::-1]], device=device)
    results = proc.post_process_grounded_object_detection(
        outputs, threshold=conf, target_sizes=target_sizes)[0]
    return _owlv2_postprocess(results, queries, img_size=pil.size)


def owlv2_detect_exemplar(image, exemplar, label: str,
                          model_id: Optional[str] = None,
                          device: str = "cuda", conf: float = 0.3,
                          _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """1-shot image-guided OWLv2 detection against an exemplar crop.

    Image-guided scores run much hotter than text-prompt scores (random
    patches routinely score 0.3+), so ``conf`` here should be ~0.6 (HF's
    image-guided recipe), not the lower text-query threshold. Boxes are
    clamped to the image bounds."""
    import torch

    proc, model, device = _owlv2_prepare(_state, model_id)
    pil, pil_q = _to_pil(image), _to_pil(exemplar)

    inputs = proc(images=pil, query_images=pil_q, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.image_guided_detection(**inputs)
    target_sizes = torch.tensor([pil.size[::-1]], device=device)
    results = proc.post_process_image_guided_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=conf)[0]
    return _owlv2_postprocess(results, [label], label_override=label,
                              img_size=pil.size)


# --------------------------------------------------------------------------- #
# Grounding DINO
# --------------------------------------------------------------------------- #

def load_grounding_dino(model_id: Optional[str] = None, device: str = "cuda"):
    """Load (and cache) the Grounding DINO processor + model for one device."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = model_id or DEFAULT_GDINO_MODEL

    def _load():
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        return proc, model.to(device).eval()

    return _load_cached((model_id, device), _load)


def build_grounding_dino_prompt(queries: List[str]) -> str:
    """Grounding DINO text prompt: labels joined by " . ", trailing period."""
    return " ".join(f"{q.strip().rstrip('.').lower()} ." for q in queries)


def grounding_dino_detect(image, queries: List[str],
                          model_id: Optional[str] = None,
                          device: str = "cuda",
                          box_threshold: float = 0.35,
                          text_threshold: float = 0.25,
                          _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Run text-prompted Grounding DINO detection on one image.

    Returned labels are Grounding DINO's matched text spans; they are
    mapped back to the closest query (case-insensitive substring match),
    so the caller's query->cat_id mapping still applies.
    """
    import torch

    model_id = model_id or DEFAULT_GDINO_MODEL
    state = _state if _state is not None else {}
    device = state.get("device", device)
    key = (model_id, device)
    if state.get("model_key") != key:
        try:
            state["proc"], state["model"] = load_grounding_dino(
                key[0], device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["proc"], state["model"] = load_grounding_dino(
                    key[0], device)
                state["model_key"] = (key[0], device)
            else:
                raise
    proc, model = state["proc"], state["model"]

    pil = _to_pil(image)

    text = build_grounding_dino_prompt(queries)
    inputs = proc(images=pil, text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)

    try:  # transformers 5.x: threshold=, optional input_ids
        results = proc.post_process_grounded_object_detection(
            outputs, threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[pil.size[::-1]])[0]
    except TypeError:  # older: box_threshold=, input_ids positional
        results = proc.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil.size[::-1]])[0]

    dets: List[Dict[str, Any]] = []
    boxes = results.get("boxes")
    if boxes is None or len(boxes) == 0:
        return dets
    scores = results.get("scores")
    # transformers >= 4.51 warns that `labels` will become integer ids —
    # prefer the string `text_labels` when present.
    labels = results.get("text_labels") or results.get("labels")
    lowered = [q.lower() for q in queries]
    for i, box in enumerate(boxes.tolist()):
        raw = str(labels[i]).strip() if labels is not None else ""
        # Map the matched text span back to the session's query: exact
        # (case-insensitive) first, then substring containment either way.
        label = next((q for q in queries if q.lower() == raw.lower()), None)
        if label is None:
            label = next((q for q, ql in zip(queries, lowered)
                          if ql in raw.lower() or raw.lower() in ql), None)
        if label is None:
            label = raw or "object"
        dets.append({
            "label": label,
            "bbox_xyxy": [float(v) for v in box],
            "mask": None,
            "confidence": float(scores[i]) if scores is not None else 1.0,
        })
    return dets


# --------------------------------------------------------------------------- #
# Falcon Perception
# --------------------------------------------------------------------------- #

def _patch_flex_attention_for_smem(model) -> bool:
    """Work around Falcon's attention on consumer GPUs.

    The model's layers always call ``torch.compile``'d flex_attention whose
    default triton config needs ~146KB of shared memory per block — more
    than consumer GPUs have (RTX 4090: ~99KB), so kernel compilation fails
    with "No valid triton configs". When the GPU's limit is below that,
    recompile flex_attention with smaller block sizes that fit. Returns True
    when the patch was applied.

    The default Triton flex-attention kernels use BLOCK_M=BLOCK_N=128 with
    3 pipeline stages (~146KB smem). On the RTX 4090 (~99KB opt-in smem),
    BLOCK=128 + 3 stages does not fit; BLOCK=64 + 1 stage fits comfortably
    but under-utilizes the SM. BLOCK=128 + 1 stage is the sweet spot: the
    larger tile keeps the matmul efficient while the single stage cuts the
    smem footprint enough to compile, giving meaningfully better
    throughput than 64x64 on Ada-class GPUs. Override by setting
    ``FALCON_FLEX_BLOCK_M`` / ``FALCON_FLEX_BLOCK_N`` / ``FALCON_FLEX_STAGES``.
    """
    import os
    import sys
    import torch

    if not torch.cuda.is_available():
        return False
    smem = torch.cuda.get_device_properties(
        torch.cuda.current_device()).shared_memory_per_block_optin
    if smem >= 150_000:
        return False  # datacenter GPU — the default kernels fit

    block_m = int(os.environ.get("FALCON_FLEX_BLOCK_M", "128"))
    block_n = int(os.environ.get("FALCON_FLEX_BLOCK_N", "128"))
    num_stages = int(os.environ.get("FALCON_FLEX_STAGES", "1"))

    # Eager flex_attention is NOT an option (it materializes the full
    # scores matrix — the AnyUp upsampler's cross-attention alone would
    # need ~64GB), so recompile with smaller blocks that fit the smem limit.
    from torch.nn.attention.flex_attention import flex_attention

    def _flex_small_blocks(*args, **kwargs):
        ko = dict(kwargs.get("kernel_options") or {})
        ko.setdefault("BLOCK_M", block_m)
        ko.setdefault("BLOCK_N", block_n)
        ko.setdefault("num_stages", num_stages)
        kwargs["kernel_options"] = ko
        return flex_attention(*args, **kwargs)

    pkg_root = type(model).__module__.split(".modeling_")[0]
    patched = False
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(pkg_root):
            continue
        g = getattr(mod, "__dict__", {})
        for attr, kw in (("compiled_flex_attn_decode",
                          {"fullgraph": True}),
                         ("compiled_flex_attn_prefill",
                          {"dynamic": True})):
            if attr in g:
                g[attr] = torch.compile(_flex_small_blocks, **kw)
                patched = True
    return patched


def _patch_falcon_mask_device(model) -> None:
    """Force the model's device into the remote code's block-mask creation.

    Falcon's remote attention code calls ``create_block_mask`` WITHOUT a
    device, so torch picks the current accelerator (cuda) for the mask
    index tensors even when the model sits on CPU after an OOM fallback —
    the mask_mod closure then mixes CPU tensors with CUDA indices inside
    the compiled trace and dies with "Tensor on device cpu is not on the
    expected device cuda:0!". Wrap ``create_attention_mask`` in both remote
    modules (the definition site in attention.py and the reference imported
    into modeling_falcon_perception.py) so the model's own device is always
    passed.
    """
    import functools
    import sys

    model_device = next(model.parameters()).device
    pkg_root = type(model).__module__.split(".modeling_")[0]
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(pkg_root):
            continue
        g = getattr(mod, "__dict__", {})
        orig = g.get("create_attention_mask")
        if orig is None or getattr(orig, "_falcon_device_patched", False):
            continue

        @functools.wraps(orig)
        def create_attention_mask(*args, _orig=orig, **kwargs):
            kwargs.setdefault("device", model_device)
            return _orig(*args, **kwargs)

        create_attention_mask._falcon_device_patched = True
        g["create_attention_mask"] = create_attention_mask


def load_falcon(model_id: Optional[str] = None, device: str = "cuda"):
    """Load (and cache) the Falcon Perception model for one device."""

    def _load():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map={"": device})
        model.eval()
        _patch_falcon_mask_device(model)
        if device != "cpu" and _patch_flex_attention_for_smem(model):
            print("⚠️ Falcon: GPU shared-memory limit too small for the "
                  "default flex-attention kernels — recompiled with smaller "
                  "blocks (BLOCK=128, 1 stage; set FALCON_FLEX_BLOCK_M/N / "
                  "FALCON_FLEX_STAGES to tune).")
        return model

    return _load_cached((model_id, device), _load)


def _decode_mask(mask_rle: Dict[str, Any]) -> Optional[np.ndarray]:
    """Decode Falcon's COCO RLE dict into an HxW bool array (None on failure)."""
    counts = mask_rle.get("counts")
    size = mask_rle.get("size")
    if not counts or not size:
        return None
    try:
        from pycocotools import mask as mask_utils
        rle = {"size": list(size),
               "counts": counts.encode("utf-8") if isinstance(counts, str)
               else counts}
        return mask_utils.decode(rle).astype(bool)
    except Exception:
        return None


def falcon_detect(image, queries: List[str],
                  model_id: Optional[str] = None, device: str = "cuda",
                  _state: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Run Falcon Perception on one image, one query per category.

    Every detection returned for a query gets that query as its label, so
    the caller's query->cat_id mapping applies directly. Unlike the other
    backends, Falcon also produces real instance masks (RLE-decoded).

    All queries are sent to ``model.generate`` in ONE batched call (one
    prefill + one decode loop), instead of one call per category. The
    model's ``generate`` accepts a list of images and a list of queries
    with one query per image, so we replicate the same image across the
    batch. This cuts per-category Python/compile overhead and keeps the
    GPU busy for the whole multi-category run instead of idling between
    sequential ``generate`` calls.
    """
    model_id = model_id or DEFAULT_FALCON_MODEL
    state = _state if _state is not None else {}
    device = state.get("device", device)
    key = (model_id, device)
    if state.get("model_key") != key:
        try:
            state["model"] = load_falcon(key[0], device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["model"] = load_falcon(key[0], device)
                state["model_key"] = (key[0], device)
            else:
                raise
    model = state["model"]

    pil = _to_pil(image)
    width, height = pil.size

    # Batch all queries in one generate() call: replicate the image so
    # each (image, query) pair becomes one row of the batch. Returns a
    # list-of-lists indexed [image_idx][det_idx]; queries with no
    # detections come back as an empty list. With compile=False we skip
    # Falcon's own torch.compile pass (our small-block patch already
    # covers the flex-attention kernels, and the rest of the graph adds
    # overhead on first-call latency without helping the per-frame path).
    batch_images = [pil] * len(queries)
    results_per_query = model.generate(batch_images, queries, compile=False)

    dets: List[Dict[str, Any]] = []
    for query, preds in zip(queries, results_per_query):
        for pred in preds or []:
            xy, hw = pred.get("xy") or {}, pred.get("hw") or {}
            cx, cy = float(xy.get("x", 0.0)), float(xy.get("y", 0.0))
            bh, bw = float(hw.get("h", 0.0)), float(hw.get("w", 0.0))
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            dets.append({
                "label": query,
                "bbox_xyxy": [x1, y1, x2, y2],
                "mask": _decode_mask(pred.get("mask_rle") or {}),
                "confidence": 1.0,
            })
    return dets
