#!/usr/bin/env bash
# Convenience wrapper around scripts/orchestrate_pipeline.py.
#
# Runs the end-to-end rosbag annotation pipeline:
#   undistort -> split -> keyframes -> qwen -> qwen_coco -> review
#   -> propagate -> export
#
# Usage (from the repo root):
#   ./scripts/run_pipeline.sh <rosbag_path> [extra orchestrator args...]
#
# Examples:
#   ./scripts/run_pipeline.sh 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52
#   ./scripts/run_pipeline.sh <rosbag> --camera both
#   ./scripts/run_pipeline.sh <rosbag> --stage qwen_coco        # one stage only
#   ./scripts/run_pipeline.sh <rosbag> --stage qwen --resume-from 500
#
# Environment overrides:
#   LLAMACPP_URL   llama.cpp server URL  (default http://127.0.0.1:8089)
#   QWEN_MODEL     Qwen GGUF weights     (default Qwen3.8-27B-Q4_K_M.gguf)
#   QWEN_MMPROJ    Qwen mmproj file      (default Qwen3.8-mmproj-F16.gguf)
#   SAM3_MODEL     SAM3 weights path     (default core/sam3/models/sam3-model/sam3.pt)
#
# The qwen stage needs the llama.cpp server running first:
#   llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
#       --mmproj Qwen3.8-mmproj-F16.gguf --image-min-tokens 2048 --port 8089
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <rosbag_path> [extra orchestrator args...]" >&2
    echo "e.g.: $0 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52 --camera left" >&2
    exit 1
fi

ROSBAG="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMACPP_URL="${LLAMACPP_URL:-http://127.0.0.1:8089}"
QWEN_MODEL="${QWEN_MODEL:-Qwen3.8-27B-Q4_K_M.gguf}"
QWEN_MMPROJ="${QWEN_MMPROJ:-Qwen3.8-mmproj-F16.gguf}"
SAM3_MODEL="${SAM3_MODEL:-core/sam3/models/sam3-model/sam3.pt}"

exec python3 "${SCRIPT_DIR}/orchestrate_pipeline.py" \
    --rosbag "${ROSBAG}" \
    --llamacpp-url "${LLAMACPP_URL}" \
    --qwen-model "${QWEN_MODEL}" \
    --qwen-mmproj "${QWEN_MMPROJ}" \
    --sam3-model "${SAM3_MODEL}" \
    "$@"
