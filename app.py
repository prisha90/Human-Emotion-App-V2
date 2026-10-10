from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import base64
import os
import shutil
import uuid
import csv
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ==================================================
# MODEL CONFIGURATION
# ==================================================

MODEL_VERSION = "v1"

MODEL_PATH = "models/emotion_mobilenet_final.h5"

model = load_model(MODEL_PATH)

# with open(
#         "production_model.txt",
#         "r"
# ) as f:

#     current_model = f.read().strip()

# model = load_model(
#     f"models/{current_model}.h5"
# )

labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# ==================================================
# CREATE REQUIRED FOLDERS
# ==================================================

os.makedirs("static/temp", exist_ok=True)

os.makedirs("training_logs", exist_ok=True)

for emotion in labels:

    os.makedirs(
        os.path.join("custom_dataset", emotion),
        exist_ok=True
    )


# ==================================================
# DATASET METADATA FILE
# ==================================================

METADATA_FILE = "dataset_metadata.csv"

if not os.path.exists(METADATA_FILE):

    with open(METADATA_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "predicted",
            "actual",
            "confidence",
            "correct",
            "timestamp",
            "source",
            "model_version"
        ])

# ==================================================
# GENERATE UNIQUE TEMP IMAGE NAME
# ==================================================

def generate_temp_filename(extension=".jpg"):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    unique_id = uuid.uuid4().hex[:6]

    filename = f"capture_{timestamp}_{unique_id}{extension}"

    return os.path.join(
        "static",
        "temp",
        filename
    )


# ==================================================
# SAVE IMAGE TO CUSTOM DATASET
# ==================================================

def save_to_dataset(image_path, emotion):

    folder = os.path.join(
        "custom_dataset",
        emotion
    )

    count = len(os.listdir(folder)) + 1

    filename = f"{emotion.lower()}_{count}.jpg"

    destination = os.path.join(
        folder,
        filename
    )

    shutil.copy(
        image_path,
        destination
    )

    return destination


# ==================================================
# LOG DATASET METADATA
# ==================================================

def log_metadata(
        image_path,
        predicted_emotion,
        actual_emotion,
        confidence,
        is_correct,
        source):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
            METADATA_FILE,
            "a",
            newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            os.path.basename(image_path),
            predicted_emotion,
            actual_emotion,
            confidence,
            is_correct,
            timestamp,
            source,
            MODEL_VERSION
        ])


# ==================================================
# DELETE TEMP IMAGE
# ==================================================

def cleanup_temp_image(image_path):

    if os.path.exists(image_path):

        try:

            os.remove(image_path)

        except Exception as e:

            print(
                "Could not delete temp image:",
                e
            )
            
            
# ==================================================
# HOME PAGE
# ==================================================

@app.route("/", methods=["GET", "POST"])
def index():

    emotion = ""
    confidence = ""
    user_image = ""
    image_path = ""
    source = ""

    if request.method == "POST":

        mode = request.form.get("mode")

        # ======================================
        # IMAGE UPLOAD
        # ======================================

        if mode == "upload":

            file = request.files["image"]

            image_path = generate_temp_filename(".jpg")

            file.save(image_path)

            user_image = image_path

            source = "upload"

            img = cv2.imread(image_path)

        # ======================================
        # WEBCAM CAPTURE
        # ======================================

        elif mode == "webcam":

            data = request.form["imageData"]

            encoded_data = data.split(",")[1]

            nparr = np.frombuffer(
                base64.b64decode(encoded_data),
                np.uint8
            )

            img = cv2.imdecode(
                nparr,
                cv2.IMREAD_COLOR
            )

            image_path = generate_temp_filename(".jpg")

            cv2.imwrite(
                image_path,
                img
            )

            user_image = image_path

            source = "webcam"

        else:

            return redirect(
                url_for("index")
            )

        # ======================================
        # PREPROCESS IMAGE
        # ======================================

        img = cv2.resize(
            img,
            (96, 96)
        )

        img = img.astype("float32") / 255.0

        img = img.reshape(
            1,
            96,
            96,
            3
        )

        # ======================================
        # PREDICTION
        # ======================================

        prediction = model.predict(
            img,
            verbose=0
        )

        index_prediction = np.argmax(
            prediction
        )

        emotion = labels[
            index_prediction
        ]

        confidence = round(
            float(
                np.max(prediction)
            ) * 100,
            2
        )

        return render_template(
            "index.html",
            emotion=emotion,
            confidence=confidence,
            user_image=user_image,
            image_path=image_path,
            source=source
        )
        

    return render_template(
        "index.html"
    )
    
    
# ==================================================
# CORRECT PREDICTION
# ==================================================

@app.route("/correct", methods=["POST"])
def correct():

    image_path = request.form["image_path"]

    predicted_emotion = request.form["predicted_emotion"]

    confidence = float(
        request.form["confidence"]
    )

    source = request.form.get(
    "source",
    "upload"
)

    # Save image into dataset
    save_to_dataset(
        image_path,
        predicted_emotion
    )

    # Log metadata
    log_metadata(
        image_path=image_path,
        predicted_emotion=predicted_emotion,
        actual_emotion=predicted_emotion,
        confidence=confidence,
        is_correct=True,
        source=source
    )

    # Remove temp image
    cleanup_temp_image(
        image_path
    )

    return redirect(
        url_for("index")
    )
    
# ==================================================
# WRONG PREDICTION PAGE
# ==================================================

@app.route("/wrong", methods=["POST"])
def wrong():

    image_path = request.form["image_path"]

    predicted_emotion = request.form["predicted_emotion"]

    confidence = float(
        request.form["confidence"]
    )

    source = request.form["source"]

    return render_template(
        "correction.html",
        image_path=image_path,
        predicted_emotion=predicted_emotion,
        confidence=confidence,
        source=source,
        labels=labels
    )

# ==================================================
# SAVE CORRECTION
# ==================================================

@app.route("/save_correction", methods=["POST"])
def save_correction():

    image_path = request.form["image_path"]

    predicted_emotion = request.form["predicted_emotion"]

    actual_emotion = request.form["actual_emotion"]

    confidence = float(
        request.form["confidence"]
    )

    source = request.form.get(
    "source",
    "upload"
)

    # Save image under the ACTUAL emotion
    save_to_dataset(
        image_path,
        actual_emotion
    )

    # Store metadata
    log_metadata(
        image_path=image_path,
        predicted_emotion=predicted_emotion,
        actual_emotion=actual_emotion,
        confidence=confidence,
        is_correct=False,
        source=source
    )

    # Remove temp image
    cleanup_temp_image(
        image_path
    )

    return redirect(
        url_for("index")
    )

# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )