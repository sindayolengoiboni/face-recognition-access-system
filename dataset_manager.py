"""
Dataset Management for Face Recognition Access Control System
=============================================================
Creates folder structure (dataset/raw/, dataset/processed/),
and generates a CSV file mapping processed image paths to labels (person names).
"""

import os
import csv

# Folder structure used by capture_images.py and preprocess.py
RAW_DIR = "dataset/raw"
PROCESSED_DIR = "dataset/processed"
LABELS_CSV = "dataset/labels.csv"


def create_folder_structure():
    """
    Create the standard folder structure:
    - dataset/
    - dataset/raw/     (for raw captured images from webcam)
    - dataset/processed/ (for preprocessed face crops)
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"Created/verified: {RAW_DIR}")
    print(f"Created/verified: {PROCESSED_DIR}")


def generate_labels_csv(csv_path=LABELS_CSV):
    """
    Scan dataset/processed/ for all images, assume each subfolder name
    is the person's label. Write a CSV with columns: image_path, label.
    """
    create_folder_structure()

    if not os.path.isdir(PROCESSED_DIR):
        print(f"Error: Processed folder not found: {PROCESSED_DIR}")
        return

    rows = []
    for person_name in sorted(os.listdir(PROCESSED_DIR)):
        person_dir = os.path.join(PROCESSED_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # Store path relative to project root for portability
            rel_path = os.path.join(PROCESSED_DIR, person_name, filename)
            rows.append((rel_path, person_name))

    # Ensure dataset folder exists for CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)

    print(f"Generated {csv_path} with {len(rows)} entries.")


def list_dataset_summary():
    """
    Print a simple summary of the dataset: number of persons and
    number of images per person (from processed folder).
    """
    if not os.path.isdir(PROCESSED_DIR):
        print(f"No processed dataset found at {PROCESSED_DIR}. Run preprocessing first.")
        return

    print("Dataset summary (processed images):")
    print("-" * 40)
    total = 0
    for person_name in sorted(os.listdir(PROCESSED_DIR)):
        person_dir = os.path.join(PROCESSED_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        count = sum(
            1
            for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        total += count
        print(f"  {person_name}: {count} images")
    print("-" * 40)
    print(f"  Total: {total} images")


if __name__ == "__main__":
    import sys

    # Create folders and generate CSV by default
    create_folder_structure()

    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        list_dataset_summary()
    else:
        generate_labels_csv()
        print("")
        list_dataset_summary()
