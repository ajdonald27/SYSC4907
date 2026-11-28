
import cv2
import random
from pathlib import Path

# Dataset directories
SPLIT = "train"
IMAGE_DIR = Path(f"dataset/images/{SPLIT}")
LABEL_DIR = Path(f"dataset/labels/{SPLIT}")

# Number of random images to inspect
NUM_SAMPLES = 25

# Get random images
images = list(IMAGE_DIR.glob("*.jpg"))
samples = random.sample(images, min(NUM_SAMPLES, len(images)))

for image_path in samples:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read {image_path}")
        continue

    height, width = image.shape[:2]

    # Corresponding YOLO annotation
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"Missing label: {label_path}")
        continue

    with open(label_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        class_id, x, y, w, h = map(float, line.split())

        # Convert normalized YOLO coordinates to pixels
        xmin = int((x - w / 2) * width)
        ymin = int((y - h / 2) * height)
        xmax = int((x + w / 2) * width)
        ymax = int((y + h / 2) * height)

        # Draw green rectangle
        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2
        )

        # Label the object
        cv2.putText(
            image,
            "UAV",
            (xmin, max(20, ymin - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display image
    cv2.imshow(f"Annotations: {image_path.name}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
