import cv2
import numpy as np

from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.utils.plotting import Annotator, colors


def scale_bboxes(bboxes: list, image_path: str, coord_scale: float = 1000.0) -> list:
    """Scale normalized 0-1000 bounding boxes to pixel coordinates.

    Args:
        bboxes: List of bounding boxes in [x, y, w, h] format, normalized 0-1000.
        image_path: Path to the input image used to get width and height.
        coord_scale: Scale factor the coordinates are normalized against (default: 1000.0).

    Returns:
        List of bounding boxes in [x, y, w, h] format in pixel coordinates.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_h, img_w = img.shape[:2]
    scaled = []
    for bbox in bboxes:
        x, y, w, h = bbox
        x_px = int(x / coord_scale * img_w)
        y_px = int(y / coord_scale * img_h)
        w_px = int(w / coord_scale * img_w)
        h_px = int(h / coord_scale * img_h)
        scaled.append([x_px, y_px, w_px, h_px])
    return scaled


def visualize_bboxes(image_path: str, bboxes: list, output_path: str = None, labels: list = None):
    """Draw bounding boxes from x, y, w, h format onto an image.

    Args:
        image_path: Path to the input image.
        bboxes: List of bounding boxes in [x, y, w, h] format.
        output_path: Optional path to save the visualization. If None, the image is displayed.
        labels: Optional list of labels, one per bounding box.

    Returns:
        The annotated image as a numpy array (BGR).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    labels = labels or [""] * len(bboxes)
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x, y, w, h = [int(v) for v in bbox]
        x2, y2 = x + w, y + h
        color = colors(i, True)
        # colors returns a tuple of floats in [0, 255]; convert to int BGR
        color_bgr = tuple(int(c) for c in color)
        cv2.rectangle(img, (x, y), (x2, y2), color_bgr, 2)
        if label:
            cv2.putText(img, str(label), (x, max(y - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

    if output_path:
        cv2.imwrite(output_path, img)
    else:
        cv2.imshow("bboxes", img)
        cv2.waitKey(0)

    return img

# draw bbox for xyxy bboxes
def visualize_bboxes_xyxy(image_path: str, bboxes: list, output_path: str = None, labels: list = None):
    """Draw bounding boxes from x1, y1, x2, y2 format onto an image.

    Args:
        image_path: Path to the input image.
        bboxes: List of bounding boxes in [x1, y1, x2, y2] format.
        output_path: Optional path to save the visualization. If None, the image is displayed.
        labels: Optional list of labels, one per bounding box.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    labels = labels or [""] * len(bboxes)
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = colors(i, True)
        # colors returns a tuple of floats in [0, 255]; convert to int BGR
        color_bgr = tuple(int(c) for c in color)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
        if label:
            cv2.putText(img, str(label), (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

    if output_path:
        cv2.imwrite(output_path, img)
    else:
        cv2.imshow("bboxes", img)
        cv2.waitKey(0)

    return img

# Initialize predictors
overrides = dict(conf=0.50, task="segment", mode="predict", model="core/sam3/models/sam3-model/sam3.pt", verbose=False)
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor2 = SAM3SemanticPredictor(overrides=overrides)

# Extract features from the first predictor
source = "core/lightstest.jpg"
# bbox_xywh_lst = [[316, 148, 73, 295], [0, 218, 156, 146], [512, 368, 234, 77], [458, 418, 128, 45], [630, 469, 296, 24], [572, 480, 426, 53], [775, 531, 223, 32], [616, 515, 382, 56]]
bbox_xywh_lst = [[316, 148, 73, 295]]
bbox_xyxy_lst=[[
        0,
        216,
        152,
        360
      ],[
        319,
        145,
        387,
        425
      ],
        [
        508,
        366,
        750,
        448
      ]]
scaled_bbox=scale_bboxes(bboxes=bbox_xyxy_lst, image_path=source)
# visualize_bboxes(source,bboxes=scaled_bbox, labels=["Ceiling light"]*len(scaled_bbox))
print(scaled_bbox)
visualize_bboxes_xyxy(source, bboxes=scaled_bbox, labels=["Ceiling light"]*len(scaled_bbox))
predictor.set_image(source)
src_shape = cv2.imread(source).shape[:2]
print(src_shape)

# Setup second predictor and reuse features
predictor2.setup_model()

# Perform inference using shared features with text prompt
# masks, boxes = predictor2.inference_features(predictor.features, src_shape=src_shape, text=["person"])

# Perform inference using shared features with bounding box prompt
print(predictor.features)
masks, boxes = predictor2.inference_features(predictor.features, src_shape=src_shape, bboxes=scaled_bbox)
print(f"masks: {masks}, boxes: {boxes}")
print("Multiple object segmentation")
# Visualize results
if masks is not None:
    masks, boxes = masks.cpu().numpy(), boxes.cpu().numpy()
    im = cv2.imread(source)
    annotator = Annotator(im, pil=False)
    annotator.masks(masks, [colors(x, True) for x in range(len(masks))])

    cv2.imshow("result", annotator.result())
    cv2.waitKey(0)

from ultralytics import SAM
model = SAM("core/sam3/models/sam3-model/sam3.pt")
results = model.predict(source=source, bboxes=scaled_bbox, task="segment", verbose=False)
results[0].show()