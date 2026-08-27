#!/usr/bin/env bash
#
# install.sh — one-shot installer for the Object Detection Application.
#
# Sets up a Python environment, installs all pip dependencies, downloads
# the required AI model weights (SAM3 + SAM3.1), and (optionally) builds
# llama.cpp to serve the Qwen VLM used by the seed-labelling pipeline.
#
# Usage:
#   ./install.sh                       # auto: venv, detect GPU, fetch SAM3 weights
#   ./install.sh --gpu                 # force CUDA torch
#   ./install.sh --cpu                 # force CPU torch
#   ./install.sh --conda NAME          # use/create a conda env instead of venv
#   ./install.sh --llamacpp            # also build llama.cpp for the Qwen server
#   ./install.sh --skip-models         # skip SAM3 weight download
#   ./install.sh --skip-apt            # skip apt-get (assume system libs present)
#   ./install.sh --hf-token TOKEN      # HuggingFace token for the gated SAM3 repo
#   ./install.sh -h, --help
#
# See README → Installation for the matching manual steps and the Docker option.
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
GPU="auto"            # auto | yes | no
VENV_DIR=".venv"
CONDA_ENV=""
BUILD_LLAMACPP=0
SKIP_MODELS=0
SKIP_APT=0
HF_TOKEN="${HF_TOKEN:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '\033[1;34m▶\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '3,/^$/p' "$0" | sed 's/^# \?//'
  exit 0
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)        GPU="yes" ;;
    --cpu)        GPU="no" ;;
    --conda)      CONDA_ENV="${2:?--conda needs a name}"; shift ;;
    --llamacpp)   BUILD_LLAMACPP=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --skip-apt)   SKIP_APT=1 ;;
    --hf-token)   HF_TOKEN="${2:?--hf-token needs a value}"; shift ;;
    -h|--help)    usage ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Python interpreter
# ---------------------------------------------------------------------------
if [[ -n "$CONDA_ENV" ]]; then
  command -v conda >/dev/null || die "conda not found on PATH"
  log "Using conda env '$CONDA_ENV'"
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda create -y -n "$CONDA_ENV" python=3.12
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  log "Creating Python venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi
PY="$(command -v python3)"
ok "Python: $PY ($(python3 -V)"

# ---------------------------------------------------------------------------
# 2. System libraries (PyQt6 GL/X + OpenCV + ffmpeg for video)
# ---------------------------------------------------------------------------
if [[ "$SKIP_APT" -eq 0 ]] && command -v apt-get >/dev/null; then
  log "Installing system APT libraries (PyQt6 / OpenCV / ffmpeg)"
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    libgl1 libegl1 libxkbcommon0 libdbus-1-3 \
    libglib2.0-0 libfontconfig1 libxcb-cursor0 \
    ffmpeg wget ca-certificates
  ok "System libraries installed"
elif [[ "$SKIP_APT" -eq 0 ]]; then
  warn "apt-get not found — skipping system libs. On a fresh Linux box install:
    libgl1 libegl1 libxkbcommon0 libdbus-1-3 libglib2.0-0 libfontconfig1 libxcb-cursor0 ffmpeg"
fi

# ---------------------------------------------------------------------------
# 3. Detect GPU + choose torch index
# ---------------------------------------------------------------------------
if [[ "$GPU" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    GPU="yes"; else GPU="no"; fi
fi

TORCH_INDEX_ARGS=()
if [[ "$GPU" == "yes" ]]; then
  log "NVIDIA GPU detected — using CUDA torch (index https://download.pytorch.org/whl/cu121)"
  TORCH_INDEX_ARGS=(--extra-index-url https://download.pytorch.org/whl/cu121)
else
  warn "No NVIDIA GPU — installing CPU torch. GPU stages (SAM3/train/track) will be slow."
fi

# ---------------------------------------------------------------------------
# 4. pip install (torch first so the CUDA build wins, then the rest)
# ---------------------------------------------------------------------------
log "Installing torch / torchvision"
python3 -m pip install --upgrade pip wheel
python3 -m pip install --upgrade torch torchvision "${TORCH_INDEX_ARGS[@]}"

log "Installing requirements.txt"
python3 -m pip install -r requirements.txt

# ultralytics SAM3 text encoder needs the ultralytics CLIP fork; it's listed
# in requirements.txt but some environments choke on the git+ URL — make sure.
python3 -c "import clip" 2>/dev/null || \
  python3 -m pip install "git+https://github.com/ultralytics/CLIP.git"

ok "Python dependencies installed"

# ---------------------------------------------------------------------------
# 5. Verify the deep-learning stack imports
# ---------------------------------------------------------------------------
log "Verifying imports"
python3 - <<'PY'
import importlib, sys
mods = ["torch", "ultralytics", "transformers", "cv2", "PyQt6",
        "rerun_sdk", "numpy", "yaml", "openai"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}: {e}")
if missing:
    print("\033[1;33m! import warnings:\033[0m")
    for x in missing: print(f"  - {x}")
    sys.exit(0)
print("  all core imports OK")
# The SAM3 predictor only exists in ultralytics >= 8.4.83
try:
    from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: F401
    print("  ultralytics SAM3 predictor available")
except Exception as e:
    print(f"\033[1;31m✗\033[0m ultralytics.models.sam.SAM3* missing: {e}")
    print("    -> pip install -U 'ultralytics>=8.4.83'")
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
# 6. Download SAM3 + SAM3.1 weights (gated HF repo — needs HF_TOKEN)
# ---------------------------------------------------------------------------
if [[ "$SKIP_MODELS" -eq 0 ]]; then
  SAM3_DIR="core/sam3/models/sam3-model"
  SAM31_DIR="core/sam3/models/sam3.1-model"
  mkdir -p "$SAM3_DIR" "$SAM31_DIR"

  if [[ -z "$HF_TOKEN" ]] && [[ ! -f "$SAM3_DIR/sam3.pt" ]]; then
    cat >&2 <<'MSG'
!
! SAM3 weights live in the gated HuggingFace repo `facebook/sam3`.
! You must:
!   1. Visit https://huggingface.co/facebook/sam3 and accept the license.
!   2. Create an access token at https://huggingface.co/settings/tokens
!   3. Re-run with:   ./install.sh --hf-token hf_xxx
!      or export:      export HF_TOKEN=hf_xxx && ./install.sh
!
! Skipping model download for now (the GUI will still start but SAM3
! segmentation / autolabel / propagate will not work until you place
! $SAM3_DIR/sam3.pt).
MSG
  fi

  if [[ -n "$HF_TOKEN" ]] || [[ -f "$SAM3_DIR/sam3.pt" ]]; then
    # `huggingface_hub` ships with this script after the pip step above.
    HF_HUB="$(command -v huggingface-cli || true)"
    [[ -n "$HF_HUB" ]] || die "huggingface-cli not found — install huggingface_hub"

    if [[ ! -f "$SAM3_DIR/sam3.pt" ]]; then
      log "Downloading SAM3 weights (~3.4 GB) into $SAM3_DIR"
      "$HF_HUB" download facebook/sam3 sam3.pt \
        --local-dir "$SAM3_DIR" \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
      ok "SAM3 weights: $SAM3_DIR/sam3.pt"
    else
      ok "SAM3 weights already present: $SAM3_DIR/sam3.pt"
    fi

    if [[ ! -f "$SAM31_DIR/sam3.1_multiplex.pt" ]]; then
      log "Downloading SAM3.1 multiplex weights (~3.4 GB) into $SAM31_DIR"
      "$HF_HUB" download facebook/sam3.1 sam3.1_multiplex.pt \
        --local-dir "$SAM31_DIR" \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
      ok "SAM3.1 weights: $SAM31_DIR/sam3.1_multiplex.pt"
    else
      ok "SAM3.1 weights already present: $SAM31_DIR/sam3.1_multiplex.pt"
    fi

    # The HF download also drops a tokenizer/config bundle; the repo expects
    # the full folder layout (config.json, tokenizer.json, ...).  If the
    # gated repo only ships the .pt, copy the bundled config from the repo's
    # checked-in templates (they are present in core/sam3/models/).
  fi
else
  warn "--skip-models: leaving SAM3 weight download to the user."
fi

# ---------------------------------------------------------------------------
# 7. Optional: build llama.cpp for the Qwen VLM server
# ---------------------------------------------------------------------------
if [[ "$BUILD_LLAMACPP" -eq 1 ]]; then
  LLAMA_DIR="${LLAMA_DIR:-$HOME/code/llama/llama.cpp}"
  log "Building llama.cpp at $LLAMA_DIR (for the Qwen seed-label server)"
  if [[ ! -d "$LLAMA_DIR" ]]; then
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
  fi
  cd "$LLAMA_DIR"
  mkdir -p build && cd build
  cmake .. -DLLAMA_CUDA=on -DLLAMA_BUILD_SERVER=on
  cmake --build . --config Release -j"$(nproc)"
  ok "llama-server built at $LLAMA_DIR/build/bin/llama-server"
  cd "$REPO_ROOT"
  cat <<EOF

  Qwen VLM weights are NOT bundled. Download a Qwen3.8 vision GGUF + its
  mmproj, e.g. into $LLAMA_DIR/:
    Qwen3.8-27B-Q4_K_M.gguf      (~17 GB)
    Qwen3.8-mmproj-F16.gguf      (~0.9 GB)
  Then start the server used by the pipeline:
    $LLAMA_DIR/build/bin/llama-server \\
      -m ./Qwen3.8-27B-Q4_K_M.gguf --mmproj ./Qwen3.8-mmproj-F16.gguf \\
      --port 8089
EOF
fi

# ---------------------------------------------------------------------------
# 8. Final summary
# ---------------------------------------------------------------------------
cat <<EOF

────────────────────────────────────────────────────────────
  Installation complete.
────────────────────────────────────────────────────────────

  Python env:    ${CONDA_ENV:-$REPO_ROOT/$VENV_DIR}
  GPU torch:     $([[ "$GPU" == "yes" ]] && echo "yes (CUDA)" || echo "no (CPU)")
  SAM3 weights:  $([[ -f core/sam3/models/sam3-model/sam3.pt ]] && echo "present" || echo "MISSING — re-run with --hf-token")

  Launch the GUI (label review app):
    python -m gui.label_review.main --images /path/to/frames

  Or run the full orchestrated pipeline (needs the Qwen server running
  on http://127.0.0.1:8089 — see the llama.cpp step above):
    python scripts/orchestrate_pipeline.py --images Datasets/YourData --camera both

  The HF autolabel backends (owlv2 / grounding-dino / falcon)
  download automatically on first use in the GUI's Settings → Autolabel.

  See README → Installation for the Docker option and manual steps.
EOF