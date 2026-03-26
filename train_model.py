"""
Train an OpenCV face recognition model (LBPH) from your images.
===============================================================
This script:
1) Reads images from `dataset/raw/<person_name>/`
2) Detects the face using Haar cascades
3) Crops to a fixed size (200x200) and converts to grayscale
4) Trains OpenCV's LBPH face recognizer
5) Saves:
   - `dataset/encodings/lbph_model.yml` (the trained model)
   - `dataset/encodings/label_map.json` (label id -> person name)

OpenCV LBPH does not store "128-d embeddings" like deep models,
but it DOES learn from the cropped face images and produces a model.
"""

import os
import json
import argparse

import cv2
import numpy as np


RAW_DIR_DEFAULT = "dataset/raw"
OUTPUT_DIR_DEFAULT = "dataset/encodings"
FACE_SIZE = (200, 200)  # Must match your preprocessing crop size


HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_face_detector():
    """Load Haar cascade classifier for face detection."""
    if not os.path.exists(HAAR_CASCADE_PATH):
        raise FileNotFoundError(f"Haar cascade file not found: {HAAR_CASCADE_PATH}")
    detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade classifier.")
    return detector


def extract_faces_and_labels(raw_dir, detector):
    """
    Walk `dataset/raw/<person_name>/`, detect faces, and build:
    - faces: list of grayscale cropped face images (uint8)
    - labels: list of integer labels aligned with `faces`
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    # Map folder name -> label id (integers required by recognizer)
    person_names = [d for d in sorted(os.listdir(raw_dir)) if os.path.isdir(os.path.join(raw_dir, d))]
    if not person_names:
        raise RuntimeError(f"No person folders found in {raw_dir}. Expected dataset/raw/<person_name>/")

    label_map = {name: i for i, name in enumerate(person_names)}

    faces = []
    labels = []

    for person_name in person_names:
        person_dir = os.path.join(raw_dir, person_name)
        image_files = [
            f
            for f in sorted(os.listdir(person_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            continue

        for filename in image_files:
            path = os.path.join(person_dir, filename)
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read image: {path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces_rect = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
            )

            if len(faces_rect) == 0:
                print(f"Warning: No face detected, skipping: {path}")
                continue

            # Pick the largest face rectangle (closest face)
            x, y, w, h = max(faces_rect, key=lambda r: r[2] * r[3])
            face_roi = gray[y : y + h, x : x + w]

            # Resize to consistent size for training
            face_resized = cv2.resize(face_roi, FACE_SIZE, interpolation=cv2.INTER_AREA)

            faces.append(face_resized)
            labels.append(label_map[person_name])

    return faces, labels, label_map


def train_lbph(faces, labels):
    """Train LBPH face recognizer and return the trained recognizer."""
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "OpenCV 'cv2.face' was not found. "
            "Install opencv-contrib-python and re-run pip install -r requirements.txt."
        )

    if len(faces) == 0:
        raise RuntimeError("No training faces found. Check your dataset and face detection settings.")

    # OpenCV expects lists/arrays of uint8 images and integer labels
    labels_np = np.array(labels, dtype=np.int32)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels_np)
    return recognizer


def save_outputs(recognizer, label_map, output_dir):
    """Save the trained model and label mapping."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "lbph_model.yml")
    recognizer.write(model_path)

    # Save inverse label map for easy prediction usage: id -> name
    inv_map = {str(v): k for k, v in label_map.items()}
    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(inv_map, f, indent=2)

    print(f"Saved trained model to: {model_path}")
    print(f"Saved label map to: {label_map_path}")


def main():
    parser = argparse.ArgumentParser(description="Train OpenCV LBPH face recognizer from dataset/raw/")
    parser.add_argument("--raw-dir", default=RAW_DIR_DEFAULT, help="Path to dataset/raw/")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Where to save model and label map")
    args = parser.parse_args()

    detector = load_face_detector()
    faces, labels, label_map = extract_faces_and_labels(args.raw_dir, detector)

    print(f"Collected training samples: {len(faces)} faces")
    print(f"Number of people: {len(label_map)}")

    recognizer = train_lbph(faces, labels)

    # Explicitly ensure dataset/encodings/ exists for the saved model file
    os.makedirs("dataset/encodings", exist_ok=True)

    # Ensure output directory exists before saving model
    os.makedirs(args.output_dir, exist_ok=True)
    save_outputs(recognizer, label_map, args.output_dir)
    print("[SUCCESS] Model saved to dataset/encodings/lbph_model.yml.")


if __name__ == "__main__":
    main()

