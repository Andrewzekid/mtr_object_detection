# object_detection_tracking_node

ROS2 node that consumes a live image topic (e.g., from a playing rosbag), runs YOLO segmentation, applies BoT-SORT tracking, and publishes the annotated image, detections, and segmentation masks.

## What it does

- Subscribes to `sensor_msgs/Image` (default: `/camera/image_raw`).
- Runs YOLO segmentation with `runs/segment/output/training/yolo_training/weights/best.pt`.
- Runs BoT-SORT tracking with `persist=True`.
- Publishes:
  - `/detection/tracked_image` — annotated `sensor_msgs/Image`.
  - `/detection/tracked_boxes` — `vision_msgs/Detection2DArray` with track IDs.
  - `/detection/tracked_masks_json` — `std_msgs/String` containing COCO-style polygon masks.
- Writes to disk:
  - `output_dir/results.json` — COCO-style tracking results.
  - `output_dir/tracking_result.mp4` — summary video of annotated frames.
  - `output_dir/tracked_frame_*.png` — per-frame annotated images.

## Prerequisites

Install ROS2 dependencies (replace `<distro>` with your ROS2 distribution, e.g., `humble`, `jazzy`):

```bash
sudo apt update
sudo apt install ros-<distro>-rclpy ros-<distro>-sensor-msgs \
    ros-<distro>-vision-msgs ros-<distro>-std-msgs ros-<distro>-cv-bridge \
    ros-<distro>-launch-ros ros-<distro>-ros2launch
```

Install Python dependencies:

```bash
pip install ultralytics opencv-python-headless numpy PyYAML
```

## Build

Place the package in a colcon workspace. The easiest layout is to use the project root as the workspace root so the default `model_path` and config resolve correctly:

```bash
cd /path/to/object_detection_app
mkdir -p src
cp -r ros2/object_detection_tracking_node src/
colcon build --packages-select object_detection_tracking_node
source install/setup.bash
```

If you prefer a separate workspace, pass absolute paths for `model_path` and `config` when launching.

## Run with a rosbag

### Option A: Launch file plays the bag for you

```bash
ros2 launch object_detection_tracking_node tracking_node.launch.py \
    rosbag_path:=/path/to/your/rosbag \
    image_topic:=/camera/image_raw \
    model_path:=runs/segment/output/training/yolo_training/weights/best.pt
```

### Option B: Play the bag yourself in another terminal

```bash
# Terminal 1
ros2 bag play /path/to/your/rosbag

# Terminal 2
ros2 launch object_detection_tracking_node tracking_node.launch.py \
    image_topic:=/camera/image_raw
```

### Run just the node

```bash
ros2 run object_detection_tracking_node tracking_node --ros-args \
    -p image_topic:=/camera/image_raw \
    -p model_path:=runs/segment/output/training/yolo_training/weights/best.pt \
    -p output_dir:=output/ros2_tracking
```

## Configuration

Edit `config/default.yaml` (or pass parameters via launch/CLI). Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_topic` | `/camera/image_raw` | Image topic to subscribe to. |
| `image_qos_profile` | `sensor_data` | Use `sensor_data` (best-effort) for rosbag2/camera topics, or `default` (reliable). |
| `model_path` | `runs/segment/output/training/yolo_training/weights/best.pt` | YOLO segmentation weights. |
| `conf` | `0.4` | Detection confidence threshold. |
| `iou` | `0.45` | NMS IoU threshold. |
| `imgsz` | `640` | Inference size. |
| `device` | `auto` | `cuda`, `cpu`, or `auto`. |
| `tracker_type` | `botsort` | Tracker name. |
| `with_cmc` | `true` | Camera-motion compensation. |
| `cmc_method` | `sparseOptFlow` | GMC method. |
| `track_buffer` | `30` | Lost-track buffer. |
| `track_high_thresh` | `0.5` | First-stage association threshold. |
| `with_reid` | `false` | Appearance ReID association. |
| `output_dir` | `output/ros2_tracking` | Where results are written. |
| `fps` | `30` | Summary video FPS. |
| `max_frames` | `-1` | Stop after N frames (`-1` = unlimited). |
| `publish_annotated_image` | `true` | Publish annotated image. |
| `publish_detections` | `true` | Publish detection array. |
| `publish_masks_json` | `true` | Publish mask JSON. |

## Offline test (no ROS2 required)

You can test the YOLO + tracking pipeline on a folder of images without installing ROS2:

```bash
cd /path/to/object_detection_app
python ros2/object_detection_tracking_node/object_detection_tracking_node/tracking_node.py \
    --offline \
    --data MTR_metacam_right \
    --output output/ros2_offline_test \
    --conf 0.4 \
    --device auto \
    --max-frames 50
```

This writes `output/ros2_offline_test/results.json` and `tracking_result.mp4`.

## Output format

`results.json` follows the COCO format with extra fields:

- `annotations[i].track_id` — consistent track ID from BoT-SORT.
- `annotations[i].confidence` — detection confidence.
- `images[i].ros_timestamp_sec` / `ros_timestamp_nanosec` — original ROS header timestamp.

## Notes

- The default image QoS is `sensor_data` because `ros2 bag` typically records image topics with best-effort reliability.
- The tracker YAML is generated at runtime under `output_dir/_runtime/` so CLI/launch overrides take effect.
- If the node shuts down normally or receives Ctrl-C, it flushes `results.json` and the summary video once.
