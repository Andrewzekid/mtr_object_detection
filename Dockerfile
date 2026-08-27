# syntax=docker/dockerfile:1.6
#
# Dockerfile for the Object Detection Application — full pipeline image.
#
#   • label-review GUI (PyQt6, forwarded to the host X server)
#   • SAM3 / SAM3.1 segmentation + autolabel backends
#   • YOLO training / evaluation / tracking
#   • llama.cpp server for the Qwen VLM seed-labelling stage
#
# Build (from the repo root):
#   docker build -t object-detection-app .
#
# Run the GUI (needs X11 + GPU; on Linux):
#   xhost +local:docker
#   docker run --gpus all --rm -it --net=host \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -v "$PWD":/work -w /work \
#     -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
#     -p 8089:8089 \
#     object-detection-app
#
# Mount your own Qwen GGUF weights at /models/qwen/ and start the VLM
# server with:
#   docker run ... -v "$HOME/code/llama/llama.cpp":/llama.cpp \
#     object-detection-app \
#     /llama.cpp/build/bin/llama-server -m /models/qwen/Qwen3.8-27B-Q4_K_M.gguf \
#       --mmproj /models/qwen/Qwen3.8-mmproj-F16.gguf --port 8089 --host 0.0.0.0
#
# See README → Installation → Option B for the full walkthrough.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

# ---------------------------------------------------------------------------
# System: Python + Qt/X11 libs + ffmpeg + build tools for llama.cpp
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip \
      libgl1 libegl1 libxkbcommon0 libdbus-1-3 \
      libglib2.0-0 libfontconfig1 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
      libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 \
      libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 \
      ffmpeg wget ca-certificates git cmake build-essential \
      libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python deps: install CUDA torch from the PyTorch index first, then the
# rest of requirements.txt so torch's CUDA build wins over the CPU default.
# ---------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip wheel && \
    python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install "git+https://github.com/ultralytics/CLIP.git"

# ---------------------------------------------------------------------------
# App source (rest of the repo).  .dockerignore excludes datasets / runs /
# build artefacts so the layer stays small.
# ---------------------------------------------------------------------------
COPY . /app

# ---------------------------------------------------------------------------
# SAM3 + SAM3.1 weights.  The repo `facebook/sam3` is gated — pass
#   docker build --build-arg HF_TOKEN=hf_xxx .
# to fetch them at build time.  Without it the build skips the weights and
# you can drop them into the running container via a volume mount instead.
# ---------------------------------------------------------------------------
ARG HF_TOKEN=""
ARG SAM3_DIR=core/sam3/models/sam3-model
ARG SAM31_DIR=core/sam3/models/sam3.1-model
RUN mkdir -p "$SAM3_DIR" "$SAM31_DIR"
RUN if [ -n "$HF_TOKEN" ]; then \
      huggingface-cli download facebook/sam3 sam3.pt \
        --local-dir "$SAM3_DIR" --token "$HF_TOKEN" && \
      huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt \
        --local-dir "$SAM31_DIR" --token "$HF_TOKEN" ; \
    else echo "No HF_TOKEN — SAM3 weights not fetched; mount them at runtime." ; fi

# ---------------------------------------------------------------------------
# Optional llama.cpp build for the Qwen VLM server (controlled by
# --build-arg BUILD_LLAMACPP=1).  Off by default to keep the image small;
# most users mount a prebuilt llama-server binary instead.
# ---------------------------------------------------------------------------
ARG BUILD_LLAMACPP=0
RUN if [ "$BUILD_LLAMACPP" = "1" ]; then \
      git clone https://github.com/ggerganov/llama.cpp /llama.cpp && \
      cd /llama.cpp && mkdir build && cd build && \
      cmake .. -DLLAMA_CUDA=on -DLLAMA_BUILD_SERVER=on && \
      cmake --build . --config Release -j"$(nproc)" ; \
    fi

# ---------------------------------------------------------------------------
# Entrypoint: the label-review GUI by default; override to run the
# orchestrator or the llama.cpp server (see comments at the top).
# ---------------------------------------------------------------------------
EXPOSE 8089
ENV PYTHONPATH=/app
ENTRYPOINT ["python3", "-m", "gui.label_review.main"]
CMD ["--help"]