import os
import random
import shutil
from glob import glob

def split_dataset(image_dir, label_dir, out_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split dataset into train/val/test folders.
    
    Args:
        image_dir (str): Path to images folder.
        label_dir (str): Path to YOLO txt labels folder.
        out_dir (str): Output dataset root folder.
        train_ratio, val_ratio, test_ratio (float): Ratios must sum to 1.
    """
    os.makedirs(out_dir, exist_ok=True)

    images = sorted(glob(os.path.join(image_dir, "*.*")))
    exts = (".jpg", ".jpeg", ".png", ".bmp")

    images = [im for im in images if im.lower().endswith(exts)]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in splits.items():
        img_out = os.path.join(out_dir, split, "images")
        lbl_out = os.path.join(out_dir, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_path in files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(label_dir, base + ".txt")

            shutil.copy(img_path, os.path.join(img_out, os.path.basename(img_path)))
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, os.path.join(lbl_out, os.path.basename(lbl_path)))

        print(f"✅ {split}: {len(files)} images")

    print("✅ Dataset split finished!")


if __name__ == "__main__":
    # Paths
    image_dir = "datasets/dataset_end"   # folder with images
    label_dir = "datasets/dataset_end/labels"   # folder with YOLO txt labels
    out_dir = "datasets/dataset_end/datasets"    # output root folder

    # Split ratios
    split_dataset(image_dir, label_dir, out_dir,
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
