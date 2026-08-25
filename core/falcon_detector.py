"""Falcon Perception open-set detector for GUI auto-labelling.

Wraps HuggingFace ``AutoModelForCausalLM`` with the repo's remote code
(default checkpoint: ``tiiuae/Falcon-Perception`` — 0.6B early-fusion
vision-language model for open-vocabulary grounding + instance segmentation).
The model's own ``generate(image, query)`` API runs one natural-language
query per call and returns detections as::

    {"xy": {"x": float, "y": float},      # normalized box centre
     "hw": {"h": float, "w": float},      # normalized box size
     "mask_rle": {"counts": str, "size": [H, W]}}

The RLE mask is decoded with pycocotools into a full-resolution HxW bool
array — so unlike OWLv2 / Grounding DINO / Florence-2, this backend produces
real instance masks.

Detections come back in the same dict shape the SAM3/OWLv2 autolabel paths
produce:

    {label, bbox_xyxy, mask, confidence, cat_id (added by caller)}

Falcon emits no confidence scores, so every detection reports
``confidence: 1.0``. The model is loaded lazily on first use and cached per
(model_id, device); a CUDA OOM falls back to CPU once and the fallback is
remembered. ``torch.compile`` is disabled — the first-call kernel build is
slow and pays off only for long batch runs, not interactive GUI use.

Requires: torch>=2.5 (FlexAttention), einops, pycocotools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_FALCON_MODEL = "tiiuae/Falcon-Perception"

_model_cache: Dict[tuple, Any] = {}


def _patch_flex_attention_for_smem(model) -> bool:
    """Work around Falcon's attention on consumer GPUs.

    The model's layers always call ``torch.compile``'d flex_attention whose
    default triton config needs ~146KB of shared memory per block — more
    than consumer GPUs have (RTX 4090: ~99KB), so kernel compilation fails
    with "No valid triton configs". When the GPU's limit is below that,
    swap the module-level compiled variants for eager flex_attention
    (slower, but runs). Returns True when the patch was applied.
    """
    import sys
    import torch

    if not torch.cuda.is_available():
        return False
    smem = torch.cuda.get_device_properties(
        torch.cuda.current_device()).shared_memory_per_block_optin
    if smem >= 150_000:
        return False  # datacenter GPU — the default kernels fit
    # The layer forwards resolve the compiled variants from their own
    # module globals; the remote code spreads them across more than one
    # module (attention.py, modeling_falcon_perception.py, anyup.py), so
    # patch every loaded module of the Falcon package that defines them.
    # Eager flex_attention is NOT an option (it materializes the full
    # scores matrix — the AnyUp upsampler's cross-attention alone would
    # need ~64GB), so recompile with small blocks that fit the smem limit.
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    def _flex_small_blocks(*args, **kwargs):
        ko = dict(kwargs.get("kernel_options") or {})
        ko.setdefault("BLOCK_M", 64)
        ko.setdefault("BLOCK_N", 64)
        ko.setdefault("num_stages", 1)
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


def load_falcon(model_id: Optional[str] = None, device: str = "cuda"):
    """Load (and cache) the Falcon Perception model for one device."""
    model_id = model_id or DEFAULT_FALCON_MODEL
    key = (model_id, device)
    if key in _model_cache:
        return _model_cache[key]
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, device_map={"": device})
    model.eval()
    if device != "cpu" and _patch_flex_attention_for_smem(model):
        print("⚠️ Falcon: GPU shared-memory limit too small for the "
              "default flex-attention kernels — recompiled with smaller "
              "blocks (somewhat slower).")
    _model_cache[key] = model
    return model


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

    ``image`` is a path or an RGB numpy array. ``_state`` lets callers
    (the batch worker) reuse the loaded model across frames and mirrors
    the device fallback; pass a fresh ``{}`` to opt in.

    Every detection returned for a query gets that query as its label, so
    the caller's query->cat_id mapping applies directly.
    """
    from PIL import Image as PILImage

    state = _state if _state is not None else {}
    device = state.get("device", device)
    model_id = model_id or DEFAULT_FALCON_MODEL
    key = (model_id, device)
    if state.get("model_key") != key:
        try:
            state["model"] = load_falcon(model_id, device)
            state["model_key"] = key
        except Exception as e:
            if device != "cpu" and "out of memory" in str(e).lower():
                device = "cpu"
                state["device"] = device
                state["model"] = load_falcon(model_id, device)
                state["model_key"] = (model_id, device)
            else:
                raise
    model = state["model"]

    if isinstance(image, (str, Path)):
        pil = PILImage.open(image).convert("RGB")
    else:
        pil = PILImage.fromarray(np.asarray(image)[..., :3])
    width, height = pil.size

    dets: List[Dict[str, Any]] = []
    for query in queries:
        preds = model.generate(pil, query, compile=False)[0]
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
