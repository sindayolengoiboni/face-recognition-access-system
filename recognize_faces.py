"""
Real-time Face Recognition Access Control Demo
==============================================
Features:
- Uses OpenCV LBPH model trained by train_model.py
- Green box + "ACCESS GRANTED" for known users
- Red box + "ACCESS DENIED" for unknown users
- Logs known access events to access_log.csv
- Prevents repeated logs for same user within 10 seconds
- Prints a hardware simulation message when access is granted
- Uses cv2.CAP_DSHOW to avoid MSMF grab-frame issues on Windows
"""

import csv
import json
import os
import time
from datetime import datetime

import cv2


# Paths
MODEL_PATH = "dataset/encodings/lbph_model.yml"
LABEL_MAP_PATH = "dataset/encodings/label_map.json"
ACCESS_LOG_PATH = "access_log.csv"

# Recognition settings
FACE_SIZE = (200, 200)
UNKNOWN_THRESHOLD = 70.0  # Lower = stricter recognition, tune if needed
LOG_COOLDOWN_SECONDS = 10

# Haar cascade for detection
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def ensure_access_log_exists(log_path):
    """Create CSV log file with header if missing."""
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])


def log_access(name, log_path):
    """Append one access event row to CSV."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])


def load_resources():
    """Load detector, recognizer, and label map. Raise clear errors if missing."""
    print("Looking for model at:", os.path.abspath(MODEL_PATH))

    if not os.path.exists(HAAR_CASCADE_PATH):
        raise FileNotFoundError(f"Haar cascade not found: {HAAR_CASCADE_PATH}")

    detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade classifier.")

    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "OpenCV face module not found. Install opencv-contrib-python "
            "and run pip install -r requirements.txt."
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model file not found: {MODEL_PATH}")
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map file not found: {LABEL_MAP_PATH}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        id_to_name = json.load(f)

    # Convert label keys back to integers
    id_to_name = {int(k): v for k, v in id_to_name.items()}
    return detector, recognizer, id_to_name


def main():
    detector, recognizer, id_to_name = load_resources()
    ensure_access_log_exists(ACCESS_LOG_PATH)

    # Keep track of last log time per person to avoid spam
    last_logged_time = {}

    # Use DirectShow backend on Windows for stable webcam capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(
            "Error: Could not open webcam. "
            "Camera may be busy (close Zoom/Teams/Discord/browser camera tabs)."
        )
        return

    print("Recognition started.")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi, FACE_SIZE, interpolation=cv2.INTER_AREA)

            predicted_id, confidence = recognizer.predict(face_resized)

            # LBPH: lower confidence value means better match
            is_known = confidence < UNKNOWN_THRESHOLD and predicted_id in id_to_name

            if is_known:
                name = id_to_name[predicted_id]
                color = (0, 255, 0)  # Green
                status_text = "ACCESS GRANTED"
                label_text = f"{name} ({confidence:.1f})"

                now = time.time()
                last_time = last_logged_time.get(name, 0.0)
                if now - last_time >= LOG_COOLDOWN_SECONDS:
                    log_access(name, ACCESS_LOG_PATH)
                    last_logged_time[name] = now
                    print(f"[SYSTEM] Sending signal to unlock door for: {name}")
            else:
                color = (0, 0, 255)  # Red
                status_text = "ACCESS DENIED"
                label_text = "Unknown"

            # Draw bounding box and labels
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label_text,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # Large access status text
            cv2.putText(
                frame,
                status_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3,
            )

        cv2.imshow("Face Recognition Access Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped.")


if __name__ == "__main__":
    main()

