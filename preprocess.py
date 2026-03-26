"""
Preprocessing Script for Face Recognition Access Control System
===============================================================
Reads raw images from dataset/raw/, detects faces with Haar Cascades,
converts to grayscale, crops to fixed size, applies Gaussian blur,
and saves to dataset/processed/person_name/.
"""

import cv2
import os

# Configuration
RAW_DIR = "dataset/raw"
PROCESSED_DIR = "dataset/processed"
FACE_SIZE = (200, 200)  # Consistent size for all face crops
GAUSSIAN_KERNEL = (3, 3)  # Small kernel for light noise reduction

# Path to Haar Cascade for face detection (bundled with OpenCV)
# On Windows/Linux it's often in site-packages/cv2/data/
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_face_detector():
    """
    Load the Haar Cascade classifier for face detection.
    Returns the classifier or None if loading fails.
    """
    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"Error: Haar cascade file not found at {HAAR_CASCADE_PATH}")
        return None
    detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if detector.empty():
        print("Error: Failed to load Haar cascade classifier.")
        return None
    return detector


def preprocess_face(gray_face_roi):
    """
    Apply preprocessing to a cropped face region:
    - Resize to FACE_SIZE
    - Apply Gaussian blur for noise reduction
    """
    resized = cv2.resize(gray_face_roi, FACE_SIZE, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, GAUSSIAN_KERNEL, 0)
    return blurred


def process_image(image_path, detector, output_dir):
    """
    Process a single image: load, convert to grayscale, detect face,
    crop and preprocess, save. Returns True if at least one face was processed.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Warning: Could not read image: {image_path}")
        return False

    # Step 1: Convert to grayscale (required for Haar cascades and reduces data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Detect faces (returns list of (x, y, w, h) rectangles)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        print(f"  Warning: No face detected in {image_path} - skipping.")
        return False

    # Use the first (largest) face if multiple are detected
    (x, y, w, h) = faces[0]
    face_roi = gray[y : y + h, x : x + w]

    # Step 3 & 4: Preprocess (resize to fixed size + Gaussian blur)
    processed = preprocess_face(face_roi)

    # Save with same base filename in processed folder
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    out_path = os.path.join(output_dir, f"{name_without_ext}.jpg")
    cv2.imwrite(out_path, processed)
    return True


def process_person(person_name, detector):
    """
    Process all images in dataset/raw/person_name/ and save
    processed faces to dataset/processed/person_name/.
    """
    raw_person_dir = os.path.join(RAW_DIR, person_name)
    processed_person_dir = os.path.join(PROCESSED_DIR, person_name)

    if not os.path.isdir(raw_person_dir):
        print(f"Warning: No folder found for person '{person_name}' at {raw_person_dir}")
        return 0

    os.makedirs(processed_person_dir, exist_ok=True)

    count = 0
    for filename in sorted(os.listdir(raw_person_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(raw_person_dir, filename)
        if process_image(path, detector, processed_person_dir):
            count += 1

    return count


def run_preprocessing(person_name=None):
    """
    Run preprocessing on raw images.
    If person_name is None, process all persons in dataset/raw/.
    """
    detector = load_face_detector()
    if detector is None:
        return

    if person_name:
        total = process_person(person_name, detector)
        print(f"Processed {total} face images for '{person_name}'.")
    else:
        if not os.path.isdir(RAW_DIR):
            print(f"Error: Raw dataset folder not found: {RAW_DIR}")
            return
        total = 0
        for name in sorted(os.listdir(RAW_DIR)):
            path = os.path.join(RAW_DIR, name)
            if os.path.isdir(path):
                n = process_person(name, detector)
                total += n
                print(f"  {name}: {n} images")
        print(f"Total processed face images: {total}")


if __name__ == "__main__":
    import sys
    # Optional: pass a person name to process only that person
    # Example: python preprocess.py John_Doe
    person = sys.argv[1] if len(sys.argv) > 1 else None
    run_preprocessing(person)
