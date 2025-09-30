import os
import glob
import json

def labelme_to_yolo(json_path, labels, out_dir):
    """Convert one Labelme JSON file to YOLO/VOLO format .txt"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w, img_h = data["imageWidth"], data["imageHeight"]
    txt_lines = []

    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in labels:
            continue
        class_id = labels.index(label)

        points = shape["points"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Convert to YOLO normalized format
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        txt_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # Write output
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(json_path))[0]
    out_path = os.path.join(out_dir, base + ".txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    return out_path


def batch_labelme_to_yolo(json_dir, labels, out_dir):
    """Convert all Labelme JSON files in a folder to YOLO/VOLO .txt"""
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {json_dir}")

    for js in json_files:
        out_file = labelme_to_yolo(js, labels, out_dir)
        print(f"✅ Converted: {js} -> {out_file}")


if __name__ == "__main__":
    # Define your class list here
    labels = [
        "upper_locked",
        "upper_unlocked",
        "upper_nolock",
        "lower_locked",
        "lower_unlocked",
        "lower_nolock",
        "knife_unlocked",
        "knife_locked",
        "lid_loose",
        "lid_tight"
    ]

    # Input folder containing Labelme JSONs
    input_folder = "datasets/dataset_end"
    # Output folder for YOLO txt labels
    output_folder = "datasets/dataset_end/labels"

    batch_labelme_to_yolo(input_folder, labels, output_folder)
