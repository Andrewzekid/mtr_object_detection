"""
Visualization utilities for model predictions.
Provides image annotation and visualization helpers.

OOP paradigm with Visualizer class.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict, Any


class Visualizer:
    """Class for visualizing model predictions and drawing detections on images."""

    def __init__(self, color_palette: Optional[List[tuple]] = None):
        """Initialize with optional custom color palette.

        Args:
            color_palette: List of (B, G, R) colors for class labels.
                          If None, uses default palette based on number of classes.
        """
        self.color_palette: List[tuple] = color_palette or self._generate_default_colors()
        self.class_names: Dict[int, str] = {}

    def _generate_default_colors(self, num_classes: int = 80) -> List[tuple]:
        """Generate default color palette.

        Args:
            num_classes: Number of classes for which to generate colors.

        Returns:
            List of (B, G, R) colors.
        """
        # Use HSV to RGB for more vibrant colors
        import colorsys

        colors = []
        for i in range(num_classes):
            h = i / num_classes
            s = 1.0
            v = 0.9
            r, g, b, _ = colorsys.hsv_to_rgb(h, s, v)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def set_class_names(self, names: Dict[int, str]):
        """Set class names for labeling.

        Args:
            names: Dict mapping class_id to class_name
        """
        self.class_names = names

    def _get_label_color(self, class_id: int) -> tuple:
        """Get color for a class ID.

        Args:
            class_id: Class identifier.

        Returns:
            (B, G, R) color tuple.
        """
        if self.class_names and class_id in self.class_names:
            return (255, 255, 255)  # White for unknown classes

        idx = class_id % len(self.color_palette)
        return self.color_palette[idx]

    def draw_detections(
        self,
        image: np.ndarray,
        boxes: List[Dict],
        conf_threshold: float = 0.5,
        box_thickness: int = 2,
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on an image.

        Args:
            image: Input image in RGB format (will be converted to BGR)
            boxes: List of dicts with keys:
                   - "bbox": [x1, y1, x2, y2]
                   - "class_id": class identifier
                   - "class_name": optional class name string
                   - "confidence": optional confidence score
            conf_threshold: Minimum confidence for drawing
            box_thickness: Line thickness for boxes
            font_scale: Font scale for labels

        Returns:
            Annotated image in RGB format
        """
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotations = image_bgr.copy()

        for box in boxes:
            if box.get("confidence", 1.0) < conf_threshold:
                continue

            bbox = box["bbox"]
            class_id = box["class_id"]
            class_name = box.get("class_name") or self.class_names.get(
                class_id, f"class_{class_id}"
            )

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Draw rectangle
            cv2.rectangle(annotations, (x1, y1), (x2, y2),
                          (self._get_label_color(class_id), 255, 255), box_thickness)

            # Draw label
            if class_name:
                label = f"{class_name}: {box.get('confidence', 1.0):.2f}"
            else:
                label = f"{class_id}: {box.get('confidence', 1.0):.2f}"

            # Draw background for label
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            cv2.rectangle(
                annotations,
                (x1, y1 - text_height - 5),
                (x1 + text_width + 5, y1),
                (self._get_label_color(class_id), 255, 255),
                -1
            )

            # Draw text
            cv2.putText(
                annotations, label,
                (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 2
            )

        # Convert back to RGB
        return cv2.cvtColor(annotations, cv2.COLOR_BGR2RGB)

    def draw_detections_rgb(
        self,
        image: np.ndarray,
        boxes: List[Dict],
        conf_threshold: float = 0.5,
        box_thickness: int = 2,
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """Draw bounding boxes on RGB image (directly).

        Args:
            image: Input image in RGB format
            boxes: List of detection dicts
            conf_threshold: Minimum confidence for drawing
            box_thickness: Line thickness for boxes
            font_scale: Font scale for labels

        Returns:
            Annotated image in RGB format
        """
        # Copy to avoid modifying original
        annotations = image.copy()

        for box in boxes:
            if box.get("confidence", 1.0) < conf_threshold:
                continue

            bbox = box["bbox"]
            class_id = box["class_id"]
            class_name = box.get("class_name") or self.class_names.get(
                class_id, f"class_{class_id}"
            )

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Draw rectangle
            cv2.rectangle(annotations, (x1, y1), (x2, y2),
                          (self._get_label_color(class_id), 255, 255), box_thickness)

            # Draw label
            if class_name:
                label = f"{class_name}: {box.get('confidence', 1.0):.2f}"
            else:
                label = f"{class_id}: {box.get('confidence', 1.0):.2f}"

            # Draw background for label
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            cv2.rectangle(
                annotations,
                (x1, y1 - text_height - 5),
                (x1 + text_width + 5, y1),
                (self._get_label_color(class_id), 255, 255),
                -1
            )

            # Draw text
            cv2.putText(
                annotations, label,
                (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 2
            )

        return annotations

    def draw_segmentation_masks(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        scores: List[float],
        conf_threshold: float = 0.5,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Draw segmentation masks on an image.

        Args:
            image: Input image in RGB format
            masks: List of mask arrays (boolean or 0/1)
            scores: List of scores for each mask
            conf_threshold: Minimum score for drawing
            alpha: Blend alpha for masks

        Returns:
            Annotated image in RGB format
        """
        annotations = image.copy()

        for mask, score in zip(masks, scores):
            if score < conf_threshold:
                continue

            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Apply alpha blend
            blended = cv2.addWeighted(
                mask_uint8, alpha, annotations, 1 - alpha, 0
            )

            # Apply to RGB
            annotations[mask > 0] = blended[mask > 0]

        return annotations

    def save_image(
        self,
        image: np.ndarray,
        output_path: str,
        format: str = "jpg",
    ):
        """Save annotated image to file.

        Args:
            image: Annotated image (RGB or BGR)
            output_path: Output file path
            format: Image format ("jpg", "png")
        """
        from pathlib import Path

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(output_path).suffix.lower()
        if ext == ".jpg":
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# Backward compatibility
def draw_detections(
    image: np.ndarray,
    boxes: List[Dict],
    conf_threshold: float = 0.5,
    box_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Module-level detection drawing."""
    vis = Visualizer()
    return vis.draw_detections(image, boxes, conf_threshold, box_thickness, font_scale)


def draw_segmentation_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    scores: List[float],
    conf_threshold: float = 0.5,
    alpha: float = 0.5,
) -> np.ndarray:
    """Module-level mask drawing."""
    vis = Visualizer()
    return vis.draw_segmentation_masks(image, masks, scores, conf_threshold, alpha)