import math
import os
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("../cropped_dataset")
OUTPUT_DIR = Path("../dataset_split")
SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(class_dir: Path):
    return sorted(
        file_path for file_path in class_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS
    )


def split_counts(total_count: int):
    train_count = math.floor(total_count * SPLITS["train"])
    val_count = math.floor(total_count * SPLITS["val"])
    test_count = total_count - train_count - val_count

    if train_count == 0 or val_count == 0 or test_count == 0:
        raise ValueError(
            f"Need at least 3 images per class to make train/val/test splits, got {total_count}."
        )

    return train_count, val_count, test_count


def reset_output_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split_name in SPLITS:
        (OUTPUT_DIR / split_name).mkdir(parents=True, exist_ok=True)


def copy_group(files, split_name, class_name):
    target_dir = OUTPUT_DIR / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        shutil.copy2(file_path, target_dir / file_path.name)


def main():
    if not SOURCE_DIR.exists():
        raise SystemExit(f"Source dataset not found: {SOURCE_DIR.resolve()}")

    class_dirs = sorted(path for path in SOURCE_DIR.iterdir() if path.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found in {SOURCE_DIR.resolve()}")

    rng = random.Random(SEED)
    reset_output_dir()

    print(f"Creating dataset split in {OUTPUT_DIR.resolve()}")
    print(f"Seed: {SEED}")

    totals = {split_name: 0 for split_name in SPLITS}

    for class_dir in class_dirs:
        files = list_images(class_dir)
        if len(files) < 3:
            raise SystemExit(f"Class '{class_dir.name}' needs at least 3 images, found {len(files)}.")

        shuffled = files[:]
        rng.shuffle(shuffled)

        train_count, val_count, _ = split_counts(len(shuffled))
        train_files = shuffled[:train_count]
        val_files = shuffled[train_count:train_count + val_count]
        test_files = shuffled[train_count + val_count:]

        split_map = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        print(f"\nClass: {class_dir.name} ({len(files)} images)")
        for split_name, split_files in split_map.items():
            copy_group(split_files, split_name, class_dir.name)
            totals[split_name] += len(split_files)
            print(f"  {split_name}: {len(split_files)}")

    print("\nDone.")
    print(
        "Totals: "
        + ", ".join(f"{split_name}={count}" for split_name, count in totals.items())
    )


if __name__ == "__main__":
    main()
