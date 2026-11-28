
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

RAW = Path("raw")
OUT = Path("dataset")

for split in ("train", "val", "test"):
    image_dir = RAW / split / "img"
    xml_dir = RAW / split / "xml"
    output_images = OUT / "images" / split
    output_labels = OUT / "labels" / split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    converted = 0

    for image_path in sorted(image_dir.glob("*.jpg")):
        xml_path = xml_dir / f"{image_path.stem}.xml"

        if not xml_path.exists():
            raise FileNotFoundError(f"Missing annotation: {xml_path}")

        root = ET.parse(xml_path).getroot()
        width = int(root.findtext("size/width"))
        height = int(root.findtext("size/height"))

        labels = []


        for obj in root.findall("object"):
            if obj.findtext("name", "").strip().upper() != "UAV":
                raise ValueError(f"Unexpected class in {xml_path}")

            box = obj.find("bndbox")

            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))

            # Clip bounding boxes to image boundaries
            xmin = max(0, min(xmin, width))
            xmax = max(0, min(xmax, width))
            ymin = max(0, min(ymin, height))
            ymax = max(0, min(ymax, height))

            # Skip invalid or zero-area boxes
            if xmin >= xmax or ymin >= ymax:
                print(f"Warning: Invalid box in {xml_path}")
                continue

            # Convert to normalized YOLO coordinates
            x = ((xmin + xmax) / 2) / width
            y = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            labels.append(
                f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            )


        shutil.copy2(image_path, output_images / image_path.name)

        (output_labels / f"{image_path.stem}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""),
            encoding="utf-8"
        )

        converted += 1

    print(f"{split}: converted {converted} images")

print("Dataset conversion complete.")
