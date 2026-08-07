"""Zero-shot segmentation via diffusion-feature correspondence (DIFT).

Implements the core of "Emergent Correspondence from Image Diffusion" (Tang et al.,
NeurIPS 2023): a Stable Diffusion UNet is run one forward pass on an (image-encoded,
noised) latent at a fixed timestep, and the self-attention outputs across the
UNet's transformer blocks are concatenated into a dense per-pixel feature map.
Two pixels with similar DIFT features correspond to the same semantic part, so a
mask labeled on one image (a *seed*) can be transferred to another image by
nearest-neighbour matching in feature space — no training, no text prompt, no
per-image box.

This module is headless and lazy-loads the SD model on first use, mirroring
``run_sam3`` in :mod:`core.models_inference`. It only loads the UNet + VAE +
scheduler (the text encoder is replaced by a zero embedding), so the first run
downloads ~3 GB of SD1.5 weights.

Public surface:
    - ``DIFTModel`` — load-once extractor. ``extract_features(image)`` returns an
      ``(feat_res, feat_res, C)`` L2-normalised float32 array at the 64x64 latent
      resolution (NOT the image size — see ``_DIFT_FEAT_RES``).
    - ``propagate_instance`` — transfer one seed mask (at feat_res) to a target
      feature map; returns a feat_res mask.
    - ``run_dift`` — convenience wrapper that accepts an image-res seed mask,
      resamples internally, and returns an image-res target mask in a
      ``{success, ...}`` dict like ``run_sam3``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Square resolution the image is resized to for the UNet pass. SD1.5 is trained
# at 512; the latent is 64x64 and every attention spatial map is a perfect
# square (64/32/16/8), which lets us infer Hn = int(sqrt(N)) when reshaping.
_DIFT_SIZE = 512
# Spatial resolution of the returned feature map. All self-attention maps are
# resized to this square before channel concatenation. Keeping it at the latent
# resolution (64) — NOT the image size — is what keeps the feature volume
# affordable: ~10k channels x 64x64 x fp32 ~= 170 MB, vs ~100 GB if upsampled
# to a 1500x1700 image. Correspondence is done at this resolution and only the
# final mask is upsampled to the target image size by the caller.
_DIFT_FEAT_RES = 64
# Diffusion timestep at which features are extracted. DIFT uses a single mid-range
# t; 200 works well and keeps the image structure largely intact.
_DIFT_T = 200
# SD1.5 UNet cross-attention dim (used for the zero text embedding).
_SD15_CROSS_DIM = 768
_SD15_TEXT_LEN = 77
_DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


class DIFTModel:
    """Lazy, load-once Stable Diffusion feature extractor.

    Self-attention outputs of every ``attn1`` block (down/mid/up) are captured
    with forward hooks, reshaped to spatial maps, upsampled to ``_DIFT_SIZE``,
    and concatenated into one feature volume. The volume is L2-normalised per
    pixel so dot products are cosine similarities.
    """

    _singleton: Optional["DIFTModel"] = None

    def __init__(self, model_id: str = _DEFAULT_MODEL, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

        self.device = device
        self.dtype = dtype
        self.vae_dtype = torch.float32  # VAE encode in fp32 to avoid NaNs

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae",
                                                 torch_dtype=self.vae_dtype).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet",
                                                        torch_dtype=dtype).to(device)
        self.unet.requires_grad_(False)
        self.unet.eval()

        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(1)

        # Hook every self-attention (attn1) block. Output shape is (B, N, C) where
        # N = Hn*Wn at that resolution; we infer Hn = int(sqrt(N)) when reshaping.
        # We identify self-attention by the ``attn1`` name suffix rather than by
        # ``cross_attention_dim``: in recent diffusers that attribute falls back to
        # ``inner_dim`` when None, so it is no longer a reliable self-attn signal.
        self._captures: List[torch.Tensor] = []
        self._hooks = []
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == "Attention" and _is_self_attn(name, module):
                self._hooks.append(module.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        # `output` is (B, N, C). Some diffusers versions return a tuple.
        if isinstance(output, tuple):
            output = output[0]
        self._captures.append(output.detach())

    # -- preprocessing ---------------------------------------------------

    def _to_tensor(self, image_rgb: np.ndarray) -> torch.Tensor:
        """RGB uint8 (H,W,3) -> (1,3,_DIFT_SIZE,_DIFT_SIZE) in [-1,1], fp32."""
        img = cv2.resize(image_rgb, (_DIFT_SIZE, _DIFT_SIZE), interpolation=cv2.INTER_AREA)
        arr = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        return arr.unsqueeze(0).to(self.device, self.vae_dtype)

    def _empty_text(self) -> torch.Tensor:
        return torch.zeros(1, _SD15_TEXT_LEN, _SD15_CROSS_DIM, device=self.device,
                           dtype=self.dtype)

    # -- feature extraction ----------------------------------------------

    @torch.no_grad()
    def extract_features(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return ``(feat_res, feat_res, C)`` L2-normalised float32 features.

        ``image_rgb`` is the *original* (unresized) RGB uint8 array. The returned
        feature map is at the fixed ``_DIFT_FEAT_RES`` (64x64) latent resolution,
        NOT the image size — this keeps the ~10k-channel volume affordable (see
        ``_DIFT_FEAT_RES``). Masks passed to / returned from
        :func:`propagate_instance` are at this same resolution; the caller
        resamples the final mask to the target image size.
        """
        self._captures.clear()

        x = self._to_tensor(image_rgb)

        # VAE encode -> latent (1,4,64,64)
        latent = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        latent = latent.to(self.dtype)

        # Add noise to timestep _DIFT_T (fixed generator for reproducibility).
        noise = torch.randn(latent.shape, device=self.device, dtype=self.dtype,
                            generator=torch.Generator(self.device).manual_seed(0))
        t = torch.tensor([_DIFT_T], device=self.device, dtype=torch.long)
        latent_t = self.scheduler.add_noise(latent, noise, t)

        _ = self.unet(latent_t, t, encoder_hidden_states=self._empty_text())

        # Concatenate all captured self-attention maps at _DIFT_FEAT_RES.
        feats = []
        for cap in self._captures:
            b, n, c = cap.shape
            hn = int(round(math.sqrt(n)))
            if hn * hn != n:
                continue  # skip non-square (shouldn't happen for SD1.5 attn)
            fmap = cap.reshape(b, hn, hn, c).permute(0, 3, 1, 2).float()
            fmap = torch.nn.functional.interpolate(
                fmap, size=(_DIFT_FEAT_RES, _DIFT_FEAT_RES), mode="bilinear",
                align_corners=False)
            feats.append(fmap)
        feat = torch.cat(feats, dim=1)[0]                  # (C, Fr, Fr)
        feat = feat.permute(1, 2, 0).cpu().numpy()         # (Fr, Fr, C)
        feat = feat.astype(np.float32)

        # L2-normalise per pixel for cosine similarity.
        norm = np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-6
        feat = feat / norm
        return feat

    @classmethod
    def get(cls, model_id: str = _DEFAULT_MODEL, device: str = "cuda",
            dtype: Optional[torch.dtype] = None) -> "DIFTModel":
        # CPU can't run the SD1.5 UNet efficiently in fp16 (and some ops warn);
        # default to fp32 there. On CUDA keep the fp16 default for speed.
        if dtype is None:
            dtype = torch.float32 if device == "cpu" else torch.float16
        if cls._singleton is None:
            cls._singleton = cls(model_id=model_id, device=device, dtype=dtype)
        return cls._singleton


def _is_self_attn(name: str, module) -> bool:
    """An Attention block is self-attention (attn1) if its name ends with
    ``attn1``. Name-based is more robust than ``cross_attention_dim``: in recent
    diffusers that attribute falls back to ``inner_dim`` when None, so it no
    longer distinguishes attn1 from attn2. Falls back to the dim heuristic for
    unnamed/edge cases."""
    if name.endswith("attn1") or ".attn1." in name:
        return True
    cross = getattr(module, "cross_attention_dim", None)
    inner = getattr(module, "inner_dim", None)
    if cross is None:
        return True
    return inner is not None and cross == inner


# ---------------------------------------------------------------------------
# Correspondence: transfer a seed mask to a target feature map
# ---------------------------------------------------------------------------

def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the connected component containing the mass centroid of ``mask``."""
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    # Pick the largest non-background component.
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    return labels == biggest


def propagate_instance(
    seed_feat: np.ndarray,
    seed_mask: np.ndarray,
    target_feat: np.ndarray,
    *,
    sim_floor: float = 0.0,
    rel_thresh: float = 0.75,
    min_score: float = 0.30,
    search_center: Optional[Tuple[float, float]] = None,
    search_radius_frac: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Transfer one seed instance mask to a target image via DIFT correspondence.

    ``seed_feat`` / ``target_feat`` are the ``extract_features`` outputs
    (``feat_res x feat_res x C``); ``seed_mask`` must be at the same
    ``feat_res`` resolution. The returned ``target_mask`` is also at
    ``feat_res`` — resample it to the target image size before drawing.

    Args:
        seed_feat: (Fr, Fr, C) features of the seed (from extract_features).
        seed_mask: (Fr, Fr) bool mask of the single instance on the seed.
        target_feat: (Fr, Fr, C) features of the target.
        sim_floor, rel_thresh: threshold = max(sim_floor, rel_thresh * max_sim).
        min_score: if the best cosine similarity is below this, return empty mask
            (the object is considered absent / not confidently matched).
        search_center, search_radius_frac: optional (y, x) center in target-frame
            fraction coords + radius as a fraction of the smaller target dim, to
            restrict the match to a temporal window (improves instance stability).

    Returns:
        (target_mask (Ht, Wt) bool, best_score float)
    """
    if seed_mask.sum() == 0:
        return np.zeros(target_feat.shape[:2], dtype=bool), 0.0

    # Exemplar feature = mean of seed-mask pixels.
    seed_vec = seed_feat[seed_mask].mean(axis=0)            # (C,)
    seed_vec = seed_vec / (np.linalg.norm(seed_vec) + 1e-6)

    # Cosine similarity of every target pixel to the exemplar.
    sim = target_feat @ seed_vec                            # (Ht, Wt)

    Ht, Wt = sim.shape
    if search_center is not None and search_radius_frac is not None:
        cy, cx = search_center[0] * Ht, search_center[1] * Wt
        r = search_radius_frac * min(Ht, Wt)
        yy, xx = np.mgrid[0:Ht, 0:Wt]
        window = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
        masked_sim = np.where(window, sim, -1.0)
    else:
        masked_sim = sim

    best = float(masked_sim.max())
    if best < min_score:
        return np.zeros((Ht, Wt), dtype=bool), best

    thresh = max(sim_floor, rel_thresh * best)
    bin_mask = masked_sim >= thresh

    # Keep the connected component containing the argmax (the actual match).
    if bin_mask.sum() > 0:
        ys, xs = np.where(masked_sim == best)
        yk, xk = int(ys[0]), int(xs[0])
        n, labels = cv2.connectedComponents(bin_mask.astype(np.uint8), connectivity=8)[:2]
        if n > 1:
            bin_mask = labels == labels[yk, xk]

    # Tidy: open then close.
    k = max(3, int(0.01 * min(Ht, Wt)) | 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bin_mask = cv2.morphologyEx(bin_mask.astype(np.uint8), cv2.MORPH_OPEN, kern)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kern)
    return bin_mask.astype(bool), best


# ---------------------------------------------------------------------------
# run_sam3-style convenience wrapper
# ---------------------------------------------------------------------------

def run_dift(
    seed_image_path: str,
    seed_mask: np.ndarray,
    target_image_path: str,
    model_id: str = _DEFAULT_MODEL,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    **propagate_kwargs,
) -> Dict:
    """Transfer one seed mask to one target image. ``run_sam3``-style return dict.

    Loads the seed and target images (BGR on disk -> RGB), extracts DIFT features
    for both, and returns the propagated target mask plus diagnostics. The model
    is loaded once and cached across calls via ``DIFTModel.get``.
    """
    try:
        seed_bgr = cv2.imread(str(seed_image_path))
        target_bgr = cv2.imread(str(target_image_path))
        if seed_bgr is None or target_bgr is None:
            return {"success": False, "error": "Could not read seed/target image"}
        seed_rgb = cv2.cvtColor(seed_bgr, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

        model = DIFTModel.get(model_id=model_id, device=device, dtype=dtype)
        seed_feat = model.extract_features(seed_rgb)
        target_feat = model.extract_features(target_rgb)

        # seed_mask arrives at image resolution; downsample to feat_res for
        # matching, then upsample the result back to the target image size.
        fr = seed_feat.shape[0]
        seed_mask_fr = cv2.resize(seed_mask.astype(np.uint8), (fr, fr),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_fr, score = propagate_instance(seed_feat, seed_mask_fr, target_feat,
                                            **propagate_kwargs)
        Ht, Wt = target_rgb.shape[:2]
        mask = cv2.resize(mask_fr.astype(np.uint8), (Wt, Ht),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        return {
            "success": True,
            "mask": mask,
            "score": score,
            "seed_features": seed_feat,
            "target_features": target_feat,
        }
    except Exception as e:
        return {"success": False, "error": f"DIFT failed: {e}"}