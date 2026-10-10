import os
import json
from datetime import datetime

import glob

import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

TRAINING_LOG_DIR = "training_logs"

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

METRICS_FILE = os.path.join(
    TRAINING_LOG_DIR,
    f"metrics_{timestamp}.json"
)


# ============================================
# GET BEST PREVIOUS MODEL ACCURACY
# ============================================

def get_best_previous_accuracy():

    if not os.path.exists(TRAINING_LOG_DIR):
        return 0.6021

    json_files = glob.glob(
        os.path.join(
            TRAINING_LOG_DIR,
            "*_metrics.json"
        )
    )

    if len(json_files) == 0:
        return 0.6021

    best_accuracy = 0

    for file in json_files:

        try:

            with open(file, "r") as f:

                data = json.load(f)

                accuracy = data.get(
                    "test_accuracy",
                    0
                )

                best_accuracy = max(
                    best_accuracy,
                    accuracy
                )

        except Exception as e:

            print(
                f"Error reading {file}: {e}"
            )

    return best_accuracy


CURRENT_MODEL_ACCURACY = (
    get_best_previous_accuracy()
)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ============================================
# CONFIGURATION
# ============================================

IMG_SIZE = (96, 96)

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-5

BASE_MODEL_PATH = "models/emotion_mobilenet_final.h5"

NEW_MODEL_PATH = "models/emotion_v2.h5"

BEST_MODEL_PATH = "models/emotion_v2_best.h5"

TRAIN_DIR = "balanced_dataset/train"

TEST_DIR = "balanced_dataset/test"

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# ============================================
# DATA AUGMENTATION
# ============================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,

    rotation_range=15,

    width_shift_range=0.1,
    height_shift_range=0.1,

    zoom_range=0.1,

    horizontal_flip=True,

    brightness_range=[0.8, 1.2]
)


test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# ============================================
# TRAIN GENERATOR
# ============================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True
)


# ============================================
# VALIDATION GENERATOR
# ============================================

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)


# ============================================
# LOAD PREVIOUS MODEL
# ============================================

print("\nLoading previous model...\n")

model = load_model(BASE_MODEL_PATH)

print("Model loaded successfully.")


# ============================================
# FREEZE MOST LAYERS
# ============================================

# Freeze all layers first
for layer in model.layers:
    layer.trainable = False


# Unfreeze the last 60 layers
for layer in model.layers[-60:]:
    layer.trainable = True


# ============================================
# COMPILE MODEL
# ============================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================
# MODEL SUMMARY
# ============================================

print("\nTrainable Layers:\n")

trainable_count = 0

for layer in model.layers:

    if layer.trainable:

        trainable_count += 1

        print(layer.name)

print(
    f"\nTotal Trainable Layers: {trainable_count}"
)

# ============================================
# CALLBACKS
# ============================================

early_stopping = EarlyStopping(
    monitor="val_loss",

    patience=7,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",

    factor=0.3,

    patience=3,

    min_lr=1e-7,

    verbose=1
)


checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)


callbacks = [
    early_stopping,
    reduce_lr,
    checkpoint
]



history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


# ============================================
# EVALUATE MODEL
# ============================================

print("\nEvaluating model...\n")

test_loss, test_accuracy = model.evaluate(
    validation_generator,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")


# ============================================
# PREDICTIONS
# ============================================

validation_generator.reset()

predictions = model.predict(
    validation_generator,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = validation_generator.classes


# ============================================
# ADDITIONAL METRICS
# ============================================

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================
# CLASSIFICATION REPORT
# ============================================

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True
)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
)

print("\nClassification Report:\n")

print(report)


# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:\n")

print(cm)


# ============================================
# CREATE TRAINING LOG DIRECTORY
# ============================================

os.makedirs(
    TRAINING_LOG_DIR,
    exist_ok=True
)

# ============================================
# SAVE FINAL MODEL
# ============================================

print("\nSaving model...\n")

model.save(
    NEW_MODEL_PATH
)

print(
    f"Model saved to {NEW_MODEL_PATH}"
)

# ============================================
# COUNT CUSTOM IMAGES
# ============================================

custom_image_count = 0

for emotion in CLASS_NAMES:

    folder = os.path.join(
        "custom_dataset",
        emotion
    )

    if os.path.exists(folder):

        custom_image_count += len(
            os.listdir(folder)
        )


# ============================================
# SAVE METRICS
# ============================================

metrics = {

    "base_model": "emotion_mobilenet_final",

    "new_model": "emotion_v2",

    "date": datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),

    "epochs_run": len(
        history.history["loss"]
    ),

    "test_loss": float(test_loss),

    "test_accuracy": float(
        test_accuracy
    ),

    "precision": float(
        precision
    ),

    "recall": float(
        recall
    ),

    "f1_score": float(
        f1
    ),

    "custom_images": custom_image_count,

    "confusion_matrix": cm.tolist(),
 
    "classification_report": report_dict
}

with open(
        METRICS_FILE,
        "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print(
    f"Metrics saved to {METRICS_FILE}"
)


# ============================================
# MODEL PROMOTION
# ============================================

new_accuracy = float(test_accuracy)

print(
    f"Current best accuracy : "
    f"{CURRENT_MODEL_ACCURACY*100:.2f}%"
)

print(
    f"New accuracy          : "
    f"{new_accuracy*100:.2f}%"
)


if new_accuracy > CURRENT_MODEL_ACCURACY:

    print("\nNew model is better.")

    production_model_path = (
        "models/emotion_mobilenet_final.h5"
    )

best_model = load_model(BEST_MODEL_PATH)

best_model.save(production_model_path)

    print(
        "Production model updated."
    )

else:

    print(
        "\nNew model rejected."
    )

    print(
        "Keeping current production model."
    )




