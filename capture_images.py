"""
Image Acquisition Script for Face Recognition Access Control System
==================================================================
Captures multiple images from webcam for each person and saves them
to a structured folder (dataset/raw/person_name/).
Uses OpenCV VideoCapture - suitable for introductory AI class.
"""

import cv2
import os

# Configuration - easy to change for your project
DATASET_RAW_DIR = "dataset/raw"
NUM_IMAGES_TO_CAPTURE = 25  # 20-30 range; adjust as needed
DELAY_BETWEEN_CAPTURES_MS = 200  # Pause AFTER capturing so images have natural variation

# If your webcam acts like a "selfie" camera, mirroring makes it feel natural.
# This also affects what gets saved (so training data matches what you saw).
MIRROR_IMAGE = True

# Resize preview window for better visibility on smaller screens.
# Saving uses the original frame size.
PREVIEW_WIDTH = 900

# Optional: draw face detection rectangle on the live preview
SHOW_FACE_BOX = True

# Haar cascade used for live preview face detection
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

QUIT_KEYS = {ord("q"), ord("Q")}
KEY_POLL_MS = 10  # How often we poll for key presses during pauses


def get_person_name():
    """
    Ask user to enter their name or ID.
    Sanitizes the input for use as a folder name (no spaces/special chars).
    """
    name = input("Enter your name or ID (e.g., John_Doe or 001): ").strip()
    if not name:
        raise ValueError("Name/ID cannot be empty.")
    # Replace spaces and problematic characters for folder names
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return safe_name


def ensure_output_dir(person_folder):
    """Create the output folder for this person if it doesn't exist."""
    os.makedirs(person_folder, exist_ok=True)
    print(f"Images will be saved to: {person_folder}")


def capture_images():
    """
    Main function: open webcam, get person name, capture NUM_IMAGES_TO_CAPTURE
    images with slight variations, and save to dataset/raw/person_name/.
    """
    # Get person identifier before opening camera
    try:
        person_name = get_person_name()
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Build path: dataset/raw/PersonName/
    person_folder = os.path.join(DATASET_RAW_DIR, person_name)
    ensure_output_dir(person_folder)

    # Open webcam using DirectShow backend (often more reliable on Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Request a standard resolution to make capture more consistent
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(
            "Error: Could not open webcam. "
            "The camera may be busy (close other apps using your webcam) or not connected."
        )
        return

    face_detector = None
    if SHOW_FACE_BOX:
        if not os.path.exists(HAAR_CASCADE_PATH):
            print(f"Warning: Haar cascade file not found at {HAAR_CASCADE_PATH}. Face box disabled.")
        else:
            face_detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            if face_detector.empty():
                print("Warning: Failed to load Haar cascade classifier. Face box disabled.")
                face_detector = None

    print("\nInstructions:")
    print("  - Look at the camera and move slightly (turn head, smile) for variety.")
    print("  - Press SPACE to capture an image.")
    print("  - Press Q to quit when you have enough images.")
    print(f"  - Target: about {NUM_IMAGES_TO_CAPTURE} images.\n")

    count = 0

    def should_quit(key_code: int) -> bool:
        """Helper so quitting works reliably for both 'q' and 'Q'."""
        return key_code in QUIT_KEYS

    while True:
        # Read one frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Optionally mirror for a natural selfie-style preview
        frame_to_display = cv2.flip(frame, 1) if MIRROR_IMAGE else frame
        display = frame_to_display.copy()

        # Resize preview (better visibility). Keep aspect ratio.
        if PREVIEW_WIDTH and display.shape[1] > PREVIEW_WIDTH:
            scale = PREVIEW_WIDTH / float(display.shape[1])
            new_w = int(display.shape[1] * scale)
            new_h = int(display.shape[0] * scale)
            display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Draw face bounding box on the preview so you can see detection working
        if face_detector is not None:
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
            )
            if len(faces) > 0:
                # Pick the largest face for the box (usually closest to camera)
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    "Face detected",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display,
                    "No face detected - adjust position",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        cv2.putText(
            display,
            f"Captured: {count}/{NUM_IMAGES_TO_CAPTURE} - SPACE=Capture, q/Q=Quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Face Capture - Press SPACE to capture, Q to quit", display)

        # Keep preview smooth; only pause AFTER a capture.
        key = cv2.waitKey(1) & 0xFF

        if should_quit(key):
            print("Quitting capture.")
            break

        if key == ord(" "):
            # Save this frame with a sequential number
            count += 1
            filename = os.path.join(person_folder, f"{person_name}_{count:03d}.jpg")
            # Save the same image you saw in the preview (mirrored if enabled)
            frame_to_save = frame_to_display
            cv2.imwrite(filename, frame_to_save)
            print(f"  Saved: {filename}")

            if count >= NUM_IMAGES_TO_CAPTURE:
                print(f"\nCaptured {count} images. You can press Q to quit or keep capturing.")
                # Optional: uncomment below to auto-stop at target count
                # break

            # Brief pause so you don't accidentally capture multiple very-similar frames.
            # During the pause we still poll for q/Q so quitting feels instant.
            remaining = DELAY_BETWEEN_CAPTURES_MS
            while remaining > 0:
                key2 = cv2.waitKey(min(KEY_POLL_MS, remaining)) & 0xFF
                if should_quit(key2):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Quitting capture.")
                    return
                remaining -= KEY_POLL_MS

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Total images saved for '{person_name}': {count}")


if __name__ == "__main__":
    capture_images()
