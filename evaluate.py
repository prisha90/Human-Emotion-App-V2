import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ============================================
# CONFIG
# ============================================

MODEL_PATH = "models/emotion_mobilenet_final.h5"

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
# LOAD MODEL
# ============================================

print("\nLoading model...\n")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================
# TEST DATA
# ============================================

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_gen = test_datagen.flow_from_directory(
    "data/test",
    target_size=(96, 96),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)

# ============================================
# EVALUATION
# ============================================

print("\nEvaluating model...\n")

loss, accuracy = model.evaluate(
    test_gen,
    verbose=1
)

print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy*100:.2f}%")

# ============================================
# PREDICTIONS
# ============================================

test_gen.reset()

predictions = model.predict(
    test_gen,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_gen.classes

# ============================================
# METRICS
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

print("\nAdditional Metrics\n")

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ============================================
# CLASSIFICATION REPORT
# ============================================

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