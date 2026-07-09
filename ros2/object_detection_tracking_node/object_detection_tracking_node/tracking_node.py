#!/usr/bin/env python3
from __future__ import annotations

"""
ROS2 node that subscribes to a sensor_msgs/Image topic, runs YOLO segmentation,
performs BoT-SORT tracking, and publishes the results.

PUBLISHERS
- /detection/tracked_image       sensor_msgs/Image (annotated frame)
- /detection/tracked_boxes       vision_msgs/Detection2DArray
- /detection/tracked_masks_json  std_msgs/String (COCO-style mask payload)

SUBSCRIBERS
- configurable image_topic (default /camera/image_raw)

DISK OUTPUT
- output_dir/results.json       COCO-style tracking results
- output_dir/tracking_result.mp4 summary video (if frames saved)

USAGE (ROS2 workspace):
    ros2 launch object_detection_tracking_node tracking_node.launch.py \
        image_topic:=/camera/image_raw \
        model_path:=runs/segment/output/training/yolo_training/weights/best.pt

USAGE (offline image folder — for testing without ROS2):
    python ros2/object_detection_tracking_node/object_detection_tracking_node/tracking_node.py \
        --offline --data MTR_metacam_right --output output/ros2_offline_test
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Resolve project root regardless of where this file is executed from.
FILE_DIR = Path(__file__).resolve().parent
# This file lives at <project_root>/ros2/object_detection_tracking_node/object_detection_tracking_node/tracking_node.py
PROJECT_ROOT = FILE_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ROS2 imports are optional so the file can be syntax/import-checked offline.
try:
    import rclpy
    from rclpy.node import Node as _NodeBase
    from rclpy.qos import qos_profile_sensor_data, qos_profile_default
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    ROS2_AVAILABLE = False
    ROS2_IMPORT_ERROR = str(exc)

    # Dummy base so the class body can be parsed when ROS2 is not installed.
    class _NodeBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ROS2 is not installed; this class is not usable.")

import yaml
from ultralytics import YOLO


def _resolve_device(device: str) -> str:
    """Map 'auto' to a concrete Ultralytics device string."""
    if device == "auto":
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    return device


DEFAULT_MODEL = str(
    PROJECT_ROOT / "runs" / "segment" / "output" / "training" / "yolo_training" / "weights" / "best.pt"
)
DEFAULT_TRACKER_BASE = "botsort.yaml"


def _find_ultralytics_trackers_dir() -> Path:
    """Locate ultralytics/cfg/trackers inside the active Python env."""
    import ultralytics

    return Path(ultralytics.__file__).parent / "cfg" / "trackers"


def build_runtime_tracker_yaml(
    base_yaml: Path,
    tracker_type: str,
    with_cmc: bool,
    cmc_method: str,
    track_buffer: int,
    track_high_thresh: float,
    with_reid: bool,
    output_dir: Path,
) -> Path:
    """Merge CLI/node overrides into a copy of the base tracker YAML for this run."""
    runtime_dir = output_dir / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / f"{tracker_type}_runtime.yaml"

    with open(base_yaml, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg["tracker_type"] = tracker_type
    cfg["track_high_thresh"] = float(track_high_thresh)
    cfg["track_buffer"] = int(track_buffer)
    cfg["with_reid"] = bool(with_reid)
    cfg["gmc_method"] = cmc_method if with_cmc else "none"

    with open(runtime_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return runtime_path


def mask_to_polygons(mask: np.ndarray) -> list:
    """Convert a binary mask to COCO-style polygons (list of flat [x,y,...] lists)."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            eps = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, eps, True)
            poly = approx.reshape(-1, 2).astype(float)
            polygons.append([float(c) for pt in poly for c in pt])
    return polygons


class DetectionTrackingNode(_NodeBase):
    """ROS2 node: YOLO segmentation + BoT-SORT tracking on image topics."""

    def __init__(self):
        super().__init__("object_detection_tracking_node")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("image_topic", "/camera/image_raw"),
                ("image_qos_profile", "sensor_data"),
                ("model_path", DEFAULT_MODEL),
                ("conf", 0.4),
                ("iou", 0.45),
                ("imgsz", 640),
                ("device", "auto"),
                ("tracker_type", "botsort"),
                ("with_cmc", True),
                ("cmc_method", "sparseOptFlow"),
                ("track_buffer", 30),
                ("track_high_thresh", 0.5),
                ("with_reid", False),
                ("output_dir", str(PROJECT_ROOT / "output" / "ros2_tracking")),
                ("fps", 30),
                ("max_frames", -1),
                ("publish_annotated_image", True),
                ("publish_detections", True),
                ("publish_masks_json", True),
            ],
        )

        self.image_topic = self.get_parameter("image_topic").value
        self.image_qos_profile = self.get_parameter("image_qos_profile").value
        self.model_path = Path(self.get_parameter("model_path").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.device = _resolve_device(self.get_parameter("device").value)
        self.tracker_type = self.get_parameter("tracker_type").value
        self.with_cmc = bool(self.get_parameter("with_cmc").value)
        self.cmc_method = self.get_parameter("cmc_method").value
        self.track_buffer = int(self.get_parameter("track_buffer").value)
        self.track_high_thresh = float(self.get_parameter("track_high_thresh").value)
        self.with_reid = bool(self.get_parameter("with_reid").value)
        self.output_dir = Path(self.get_parameter("output_dir").value)
        self.fps = int(self.get_parameter("fps").value)
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.publish_annotated_image = bool(self.get_parameter("publish_annotated_image").value)
        self.publish_detections = bool(self.get_parameter("publish_detections").value)
        self.publish_masks_json = bool(self.get_parameter("publish_masks_json").value)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build runtime tracker YAML.
        trackers_dir = _find_ultralytics_trackers_dir()
        base_yaml = trackers_dir / DEFAULT_TRACKER_BASE
        if not base_yaml.exists():
            self.get_logger().error(f"Base tracker YAML not found: {base_yaml}")
            raise FileNotFoundError(f"Base tracker YAML not found: {base_yaml}")

        self.tracker_yaml = build_runtime_tracker_yaml(
            base_yaml,
            self.tracker_type,
            self.with_cmc,
            self.cmc_method,
            self.track_buffer,
            self.track_high_thresh,
            self.with_reid,
            self.output_dir,
        )
        self.get_logger().info(f"Runtime tracker config: {self.tracker_yaml}")

        # Load YOLO segmentation model.
        if not self.model_path.exists():
            self.get_logger().error(f"YOLO model not found at {self.model_path}")
            raise FileNotFoundError(f"YOLO model not found at {self.model_path}")
        self.get_logger().info(f"Loading YOLO segmentation model from {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names if hasattr(self.model, "names") else {}

        # ROS interfaces.
        self.bridge = CvBridge()
        qos = (
            qos_profile_sensor_data()
            if self.image_qos_profile == "sensor_data"
            else qos_profile_default()
        )
        self.image_sub = self.create_subscription(Image, self.image_topic, self._image_callback, qos)

        self.annotated_pub = None
        if self.publish_annotated_image:
            self.annotated_pub = self.create_publisher(Image, "/detection/tracked_image", 10)

        self.detections_pub = None
        if self.publish_detections:
            self.detections_pub = self.create_publisher(Detection2DArray, "/detection/tracked_boxes", 10)

        self.masks_pub = None
        if self.publish_masks_json:
            self.masks_pub = self.create_publisher(String, "/detection/tracked_masks_json", 10)

        # Output state.
        self.frame_count = 0
        self.annotation_id = 1
        self.coco_images = []
        self.coco_annotations = []
        self.categories = [
            {"id": int(cid), "name": name} for cid, name in self.class_names.items()
        ]
        self.image_files = []  # Only used for offline video summary.
        self._results_saved = False

        self.get_logger().info(
            f"Node ready. Subscribed to {self.image_topic}, model={self.model_path}, "
            f"conf={self.conf}, iou={self.iou}, imgsz={self.imgsz}, device={self.device}"
        )

    def _image_callback(self, msg: Image):
        """Process one ROS image message."""
        if self.max_frames >= 0 and self.frame_count >= self.max_frames:
            self.get_logger().info("Max frames reached; stopping subscription.")
            self.destroy_subscription(self.image_sub)
            self._save_results()
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return

        self.frame_count += 1
        img_h, img_w = frame.shape[:2]
        timestamp = datetime.now().isoformat()
        image_id = self.frame_count

        self.coco_images.append({
            "id": image_id,
            "file_name": f"frame_{image_id:08d}.png",
            "width": img_w,
            "height": img_h,
            "ros_timestamp_sec": float(msg.header.stamp.sec),
            "ros_timestamp_nanosec": float(msg.header.stamp.nanosec),
        })

        # Run YOLO segmentation + tracking.
        try:
            results = self.model.track(
                source=frame,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                tracker=str(self.tracker_yaml),
                verbose=False,
                device=self.device,
            )
            result = results[0] if results and len(results) > 0 else None
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"YOLO/BoT-SORT inference failed on frame {image_id}: {exc}")
            return

        annotated_frame = result.plot() if result is not None else frame.copy()
        detections_msg = None
        masks_payload = None

        if (
            result is not None
            and result.boxes is not None
            and hasattr(result.boxes, "id")
            and result.boxes.id is not None
        ):
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            has_masks = (
                hasattr(result, "masks")
                and result.masks is not None
                and len(result.masks) > 0
            )

            if self.publish_detections:
                detections_msg = Detection2DArray()
                detections_msg.header = msg.header

            mask_records = []

            for i in range(len(track_ids)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                area = float((x2 - x1) * (y2 - y1))
                class_id = int(class_ids[i])
                track_id = int(track_ids[i])
                confidence = float(confidences[i])

                segmentation = []
                if has_masks and i < len(result.masks.data):
                    mask = result.masks.data[i].cpu().numpy()
                    if mask.shape != (img_h, img_w):
                        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    polygons = mask_to_polygons(mask_binary)
                    if polygons:
                        segmentation = polygons
                        area = float(np.sum(mask_binary))

                    if self.publish_masks_json:
                        mask_records.append({
                            "track_id": track_id,
                            "class_id": class_id,
                            "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                            "height": img_h,
                            "width": img_w,
                            "polygons": polygons,
                        })

                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox_coco,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                    "track_id": track_id,
                    "confidence": confidence,
                }
                self.coco_annotations.append(annotation)
                self.annotation_id += 1

                if detections_msg is not None:
                    detection = Detection2D()
                    detection.header = msg.header
                    detection.bbox.center.position.x = float((x1 + x2) / 2.0)
                    detection.bbox.center.position.y = float((y1 + y2) / 2.0)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(class_id)
                    hypothesis.hypothesis.score = confidence
                    detection.results.append(hypothesis)
                    detection.id = str(track_id)
                    detections_msg.detections.append(detection)

            unique_ids = np.unique(track_ids)
            info_text = f"Objects: {len(unique_ids)} | Tracks: {len(track_ids)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if self.publish_masks_json and mask_records:
                masks_payload = String()
                masks_payload.data = json.dumps({
                    "header": {
                        "stamp": {
                            "sec": msg.header.stamp.sec,
                            "nanosec": msg.header.stamp.nanosec,
                        },
                        "frame_id": msg.header.frame_id,
                    },
                    "image_id": image_id,
                    "masks": mask_records,
                })

        # Save annotated frame to disk for video summary.
        output_path = self.output_dir / f"tracked_frame_{image_id:08d}.png"
        cv2.imwrite(str(output_path), annotated_frame)
        self.image_files.append(output_path)

        # Publish results.
        if self.annotated_pub is not None:
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"Failed to publish annotated image: {exc}")

        if self.detections_pub is not None and detections_msg is not None:
            self.detections_pub.publish(detections_msg)

        if self.masks_pub is not None and masks_payload is not None:
            self.masks_pub.publish(masks_payload)

        self.get_logger().info(
            f"Frame {image_id}: {len(self.coco_annotations)} total annotations"
        )

    def _save_results(self):
        """Write COCO JSON and optionally build a summary video."""
        if self._results_saved:
            return
        self._results_saved = True
        coco_output = {
            "info": {
                "description": "ROS2 YOLO segmentation + BoT-SORT tracking results",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "images": self.coco_images,
            "annotations": self.coco_annotations,
            "categories": self.categories,
        }
        json_path = self.output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(coco_output, f, indent=2)
        self.get_logger().info(f"Tracking JSON saved to: {json_path}")

        self._create_video()

    def _create_video(self):
        """Create summary video from saved annotated frames."""
        if not self.image_files:
            return
        first_img = cv2.imread(str(self.image_files[0]))
        if first_img is None:
            return
        height, width = first_img.shape[:2]
        video_path = self.output_dir / "tracking_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
        for path in self.image_files:
            frame = cv2.imread(str(path))
            if frame is not None:
                out.write(frame)
        out.release()
        self.get_logger().info(f"Tracking video saved to: {video_path}")

    def destroy_node(self):
        """Ensure results are flushed when the node is shut down."""
        self._save_results()
        super().destroy_node()


def run_ros2_node(args=None):
    """Spin the ROS2 node."""
    if not ROS2_AVAILABLE:
        print(f"ROS2 dependencies are not installed: {ROS2_IMPORT_ERROR}")
        print("Install: sudo apt install ros-<distro>-vision-msgs ros-<distro>-cv-bridge")
        print("Then: pip install rclpy")
        sys.exit(1)

    node = None
    try:
        rclpy.init(args=args)
        node = DetectionTrackingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("Keyboard interrupt; shutting down.")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run_offline(args):
    """Offline mode for testing the pipeline on a folder of images without ROS2."""
    model_path = Path(args.model_path)
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    if not data_path.exists():
        print(f"Error: Data directory not found at {data_path}")
        sys.exit(1)

    trackers_dir = _find_ultralytics_trackers_dir()
    base_yaml = trackers_dir / DEFAULT_TRACKER_BASE
    tracker_yaml = build_runtime_tracker_yaml(
        base_yaml,
        args.tracker_type,
        args.with_cmc,
        args.cmc_method,
        args.track_buffer,
        args.track_high_thresh,
        args.with_reid,
        output_dir,
    )

    print(f"Loading model from {model_path}")
    model = YOLO(str(model_path))
    class_names = model.names if hasattr(model, "names") else {}

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([f for f in data_path.iterdir() if f.suffix.lower() in image_extensions])
    if args.max_frames >= 0:
        image_files = image_files[: args.max_frames]
    print(f"Found {len(image_files)} images")

    coco_images = []
    coco_annotations = []
    annotation_id = 1
    tracked_paths = []

    for idx, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  Warning: could not read {image_path}")
            continue
        img_h, img_w = frame.shape[:2]
        image_id = idx + 1
        coco_images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": img_w,
            "height": img_h,
        })

        results = model.track(
            source=frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            tracker=str(tracker_yaml),
            verbose=False,
            device=_resolve_device(args.device),
        )
        result = results[0] if results and len(results) > 0 else None
        annotated = result.plot() if result is not None else frame.copy()

        if (
            result is not None
            and result.boxes is not None
            and hasattr(result.boxes, "id")
            and result.boxes.id is not None
        ):
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            has_masks = (
                hasattr(result, "masks") and result.masks is not None and len(result.masks) > 0
            )

            for i in range(len(track_ids)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                area = float((x2 - x1) * (y2 - y1))
                class_id = int(class_ids[i])
                segmentation = []

                if has_masks and i < len(result.masks.data):
                    mask = result.masks.data[i].cpu().numpy()
                    if mask.shape != (img_h, img_w):
                        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    polygons = mask_to_polygons(mask_binary)
                    if polygons:
                        segmentation = polygons
                        area = float(np.sum(mask_binary))

                coco_annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox_coco,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                    "track_id": int(track_ids[i]),
                    "confidence": float(confidences[i]),
                })
                annotation_id += 1

            unique_ids = np.unique(track_ids)
            info_text = f"Objects: {len(unique_ids)} | Tracks: {len(track_ids)}"
            cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out_path = output_dir / f"tracked_{image_path.name}"
        cv2.imwrite(str(out_path), annotated)
        tracked_paths.append(out_path)
        print(f"  Frame {image_id}/{len(image_files)} -> {out_path.name}")

    categories = [{"id": int(cid), "name": name} for cid, name in class_names.items()]
    coco_output = {
        "info": {
            "description": "Offline YOLO segmentation + BoT-SORT tracking results",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(coco_output, f, indent=2)
    print(f"Results saved to {json_path}")

    create_tracking_video(output_dir, tracked_paths, args.fps)


def create_tracking_video(output_dir, image_files, fps):
    """Create a summary video from saved annotated frames."""
    if not image_files:
        return
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        return
    height, width = first_img.shape[:2]
    video_path = output_dir / "tracking_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for path in image_files:
        frame = cv2.imread(str(path))
        if frame is not None:
            out.write(frame)
    out.release()
    print(f"Tracking video saved to {video_path}")


def _build_offline_parser():
    parser = argparse.ArgumentParser(
        description="Offline YOLO segmentation + tracking runner (no ROS2 required)."
    )
    parser.add_argument("--offline", action="store_true", help="Run in offline image-folder mode.")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "MTR_metacam_right"))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "output" / "ros2_offline_test"))
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tracker-type", type=str, default="botsort")
    parser.add_argument("--with-cmc", action="store_true", default=True)
    parser.add_argument("--no-cmc", dest="with_cmc", action="store_false")
    parser.add_argument("--cmc-method", type=str, default="sparseOptFlow")
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--track-high-thresh", type=float, default=0.5)
    parser.add_argument("--with-reid", action="store_true", default=False)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=-1)
    return parser


def main():
    """Entry point: ROS2 when available, offline mode otherwise."""
    parser = _build_offline_parser()
    args, unknown = parser.parse_known_args()

    if args.offline or not ROS2_AVAILABLE:
        run_offline(args)
    else:
        run_ros2_node(unknown)


if __name__ == "__main__":
    main()
