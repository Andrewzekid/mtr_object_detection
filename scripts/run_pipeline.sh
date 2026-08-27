#!/usr/bin/env bash
# Convenience wrapper around scripts/orchestrate_pipeline.py.
#
# Runs the end-to-end keyframe annotation pipeline:
#   undistort -> sample -> keyframes -> stats -> qwen -> qwen_coco -> gui
#   -> yolo -> split -> augment -> assemble -> train -> evaluate -> tracking
#
# Usage (from the repo root):
#   ./scripts/run_pipeline.sh <rosbag_path> [extra orchestrator args...]
#   ./scripts/run_pipeline.sh --coco-json <coco> --images-dir <dir> ...   # dataset-only mode
#
# Examples:
#   ./scripts/run_pipeline.sh 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52
#   ./scripts/run_pipeline.sh <rosbag> --camera both --sample-size 1000
#   ./scripts/run_pipeline.sh <rosbag> --stage qwen_coco        # one stage only
#   ./scripts/run_pipeline.sh <rosbag> --stage qwen --resume-from 500
#   ./scripts/run_pipeline.sh --coco-json /data/run/labels_coco.json \
#       --images-dir /data/run/camera --output-root output/my_dataset --skip-augment
#
# Environment overrides:
#   LLAMACPP_URL   llama.cpp server URL  (default http://127.0.0.1:8089)
#   QWEN_MODEL     Qwen GGUF weights     (default Qwen3.8-27B-Q4_K_M.gguf)
#   QWEN_MMPROJ    Qwen mmproj file      (default Qwen3.8-mmproj-F16.gguf)
#
# The qwen stage needs the llama.cpp server running first:
#   llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
#       --mmproj Qwen3.8-mmproj-F16.gguf --image-min-tokens 2048 --port 8089
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMACPP_URL="${LLAMACPP_URL:-http://127.0.0.1:8089}"
QWEN_MODEL="${QWEN_MODEL:-Qwen3.8-27B-Q4_K_M.gguf}"
QWEN_MMPROJ="${QWEN_MMPROJ:-Qwen3.8-mmproj-F16.gguf}"

# Dataset-only mode (--coco-json ...): no rosbag, pass everything straight
# through to the orchestrator.
if [[ "${1:-}" == "--coco-json" ]]; then
    exec python3 "${SCRIPT_DIR}/orchestrate_pipeline.py" "$@"
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <rosbag_path> [extra orchestrator args...]" >&2
    echo "       $0 --coco-json <coco> --images-dir <dir> [extra args...]" >&2
    echo "e.g.: $0 20260821_Centen_Clio-n-Metacam_Data/metacam_data/2026-08-20_22-06-52 --camera both" >&2
    exit 1
fi

ROSBAG="$1"
shift

# Path checking: the user must enter an existing rosbag folder with the
# expected layout (camera/<cam> + info/calibration.json).
if [[ ! -d "${ROSBAG}" ]]; then
    echo "Error: rosbag folder not found: ${ROSBAG}" >&2
    echo "       (check the path you passed as <rosbag_path>)" >&2
    exit 1
fi
if [[ ! -d "${ROSBAG}/camera" ]]; then
    echo "Error: not a valid rosbag folder, missing 'camera/' inside: ${ROSBAG}" >&2
    echo "       (expected camera/<cam> + info/calibration.json)" >&2
    exit 1
fi
if [[ ! -f "${ROSBAG}/info/calibration.json" ]]; then
    echo "Error: calibration file not found: ${ROSBAG}/info/calibration.json" >&2
    echo "       (expected info/calibration.json inside the rosbag folder)" >&2
    exit 1
fi

exec python3 "${SCRIPT_DIR}/orchestrate_pipeline.py" \
    --rosbag "${ROSBAG}" \
    --llamacpp-url "${LLAMACPP_URL}" \
    --qwen-model "${QWEN_MODEL}" \
    --qwen-mmproj "${QWEN_MMPROJ}" \
    "$@"
