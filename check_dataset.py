from pathlib import Path
from collections import Counter

root = Path("data/raw/ppe_dataset")
names = {0: "Helmet", 1: "NoHelmet", 2: "NoVest", 3: "Vest"}

for split in ["train", "valid"]:
    label_dir = root / split / "labels"
    counts = Counter()
    invalid = []
    empty = 0
    total_boxes = 0

    for file in label_dir.glob("*.txt"):
        lines = file.read_text(encoding="utf-8-sig").splitlines()

        if not lines:
            empty += 1
            continue

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            try:
                if len(parts) != 5:
                    raise ValueError("Expected 5 YOLO values")

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                if class_id not in names:
                    raise ValueError("Unknown class ID")
                if not all(0 <= x <= 1 for x in coords):
                    raise ValueError("Coordinates outside 0..1")
                if coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError("Invalid box width/height")

                counts[class_id] += 1
                total_boxes += 1

            except Exception as e:
                invalid.append(f"{file.name}:{line_no} - {e}")

    print(f"\n--- {split.upper()} ---")
    print("Total bounding boxes:", total_boxes)

    for class_id, name in names.items():
        print(f"{name}: {counts[class_id]}")

    print("Empty label files:", empty)
    print("Invalid annotation rows:", len(invalid))

    for issue in invalid[:10]:
        print("  ", issue)
