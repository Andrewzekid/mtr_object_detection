from PIL import Image, ImageDraw, ImageFont

# 1. Load your image
image_name = "test9"
image_path = f"scripts/{image_name}.jpg"  # Replace with your image path
image = Image.open(image_path)
width, height = image.size

# 2. Bounding box data in xyxy format normalized 0-1000
data = [{"bbox_2d": [596, 483, 627, 1000], "label": "Exit Sign"}]

draw = ImageDraw.Draw(image)

# Optional: Try loading a TTF font, fallback to default
try:
    font = ImageFont.truetype("arial.ttf", size=16)
except IOError:
    font = ImageFont.load_default()

for item in data:
    # Unpack as xyxy: [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = item["bbox_2d"]
    label = item["label"]

    # Denormalize 0-1000 scale to pixel coordinates
    abs_xmin = int((xmin / 1000.0) * width)
    abs_ymin = int((ymin / 1000.0) * height)
    abs_xmax = int((xmax / 1000.0) * width)
    abs_ymax = int((ymax / 1000.0) * height)

    # Draw rectangle: [xmin, ymin, xmax, ymax]
    draw.rectangle(
        [abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline="red", width=3
    )

    # Draw label above the bounding box
    text_bbox = draw.textbbox((abs_xmin, abs_ymin), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Label background tag
    draw.rectangle(
        [
            abs_xmin,
            abs_ymin - text_height - 6,
            abs_xmin + text_width + 6,
            abs_ymin,
        ],
        fill="red",
    )

    # Text overlay
    draw.text(
        (abs_xmin + 3, abs_ymin - text_height - 4),
        label,
        fill="white",
        font=font,
    )

# 3. Save and display
image.save(f"{image_name}_output.jpg")
image.show()


