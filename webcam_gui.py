import cv2
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ==========================================
# CONFIG
# ==========================================

model = load_model("models/emotion_mobilenet_final.h5")

labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

METADATA_FILE = "dataset_metadata.csv"

# create folders
for emotion in labels:
    os.makedirs(
        os.path.join("custom_dataset", emotion),
        exist_ok=True
    )

# ==========================================
# FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

print("\n")
print("Press C to capture image")
print("Press Q to quit")
print("\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    current_emotion = None
    current_confidence = None

    for (x, y, w, h) in faces:

        roi = frame[
            y:y+h,
            x:x+w
        ]

        roi_resized = cv2.resize(
            roi,
            (96, 96)
        )

        roi_input = roi_resized / 255.0

        roi_input = roi_input.reshape(
            1,
            96,
            96,
            3
        )

        pred = model.predict(
            roi_input,
            verbose=0
        )

        idx = np.argmax(pred)

        current_emotion = labels[idx]

        current_confidence = (
            float(np.max(pred)) * 100
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        text = (
            f"{current_emotion}"
            f" ({current_confidence:.1f}%)"
        )

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Emotion AI",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ======================================
    # QUIT
    # ======================================

    if key == ord('q'):
        break

    # ======================================
    # CAPTURE
    # ======================================

    if key == ord('c'):

        if current_emotion is None:
            print("No face detected.")
            continue

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        unique_id = str(uuid.uuid4())[:8]

        filename = (
            f"capture_"
            f"{timestamp}_"
            f"{unique_id}.jpg"
        )

        save_path = os.path.join(
            "custom_dataset",
            current_emotion,
            filename
        )

        cv2.imwrite(
            save_path,
            frame
        )

        print("\nPrediction")
        print("-------------------")
        print(
            f"Emotion    : {current_emotion}"
        )

        print(
            f"Confidence : "
            f"{current_confidence:.2f}%"
        )

        answer = input(
            "\nWas prediction correct? (y/n): "
        )

        # ==========================
        # CORRECT
        # ==========================

        if answer.lower() == "y":

            actual_emotion = current_emotion

            correct = True

        # ==========================
        # WRONG
        # ==========================

        else:

            print("\nAvailable labels:")

            for emotion in labels:
                print(emotion)

            actual_emotion = input(
                "\nCorrect emotion: "
            )

            correct = False

            new_path = os.path.join(
                "custom_dataset",
                actual_emotion,
                filename
            )

            os.rename(
                save_path,
                new_path
            )

        # ==========================
        # METADATA
        # ==========================

        row = {

            "timestamp":
                datetime.now(),

            "filename":
                filename,

            "source":
                "webcam",

            "predicted":
                current_emotion,

            "actual":
                actual_emotion,

            "confidence":
                current_confidence,

            "correct":
                correct
        }

        df = pd.DataFrame([row])

        if os.path.exists(
                METADATA_FILE
        ):

            df.to_csv(
                METADATA_FILE,
                mode="a",
                header=False,
                index=False
            )

        else:

            df.to_csv(
                METADATA_FILE,
                index=False
            )

        print(
            "\nSample saved successfully."
        )

cap.release()

cv2.destroyAllWindows()