# Emotion Detection AI - Self-Improving Emotion Recognition System

## Overview

Emotion Detection AI is a deep learning-based facial emotion recognition system capable of predicting seven human emotions from uploaded images and webcam captures.

Unlike traditional static classifiers, this project incorporates **active learning and user feedback**, allowing the system to continuously improve over time by collecting real-world examples and periodically retraining itself.

The goal is to transform the model from a fixed classifier into a **self-improving emotion recognition platform**.

---

## Features

### Emotion Prediction

Predicts the following seven emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

Supports:

- Image Upload
- Webcam Capture

---

### User Feedback System

After every prediction, users can provide feedback:

#### Correct Prediction

If the prediction is correct:

- Image is stored in the corresponding emotion folder.
- Metadata is recorded for future retraining.

#### Wrong Prediction

If the prediction is incorrect:

- User selects the correct emotion from a dropdown menu.
- Image is saved under the correct class.
- Metadata is updated accordingly.

This allows the model to learn from its mistakes.

---

## Active Learning Pipeline

```text
User Image
      ↓
Prediction
      ↓
User Feedback
      ↓
custom_dataset/
      ↓
dataset_metadata.csv
      ↓
dataset_builder.py
      ↓
balanced_dataset/
      ↓
retrain.py
      ↓
Improved Model
```

---

## Dataset Structure

### Original Dataset

```
data/

train/
    Angry/
    Disgust/
    Fear/
    Happy/
    Sad/
    Surprise/
    Neutral/

test/
    Angry/
    Disgust/
    Fear/
    Happy/
    Sad/
    Surprise/
    Neutral/
```

---

### Custom Dataset

User-generated samples are stored in:

```
custom_dataset/

Angry/
Disgust/
Fear/
Happy/
Sad/
Surprise/
Neutral/
```

These images represent real-world data and gradually improve model robustness.

---

## Metadata Logging

Each collected sample stores:

- Timestamp
- Filename
- Source (Upload/Webcam)
- Predicted Emotion
- Actual Emotion
- Confidence Score
- Whether prediction was correct

Stored in:

```
dataset_metadata.csv
```

---

## Weighted Sample Strategy

Not all images contribute equally.

The system assigns higher importance to:

### Wrong Predictions

Misclassified samples are considered highly valuable.

Weight:

```
+2
```

---

### Low Confidence Predictions

Predictions with confidence below 60% receive additional weight.

Weight:

```
+1
```

---

### Webcam Images

Webcam samples are generally noisier and more realistic.

Weight:

```
+1
```

---

## Dataset Builder

The script:

```
dataset_builder.py
```

creates:

```
balanced_dataset/

train/
test/
```

by:

1. Copying the original FER dataset.
2. Adding user-generated samples.
3. Applying weighted duplication based on metadata.

This enables the model to focus more on difficult examples.

---

## Model Architecture

### Backbone

Pretrained MobileNetV2

- ImageNet weights
- Partial fine-tuning of last 60 layers

---

### Feature Extraction

- Global Average Pooling

---

### Classification Head

```
Dense(2048, ReLU)
Dropout(0.5)

Dense(1024, ReLU)
Dropout(0.4)

Dense(512, ReLU)
Dropout(0.3)

Dense(256, ReLU)
Dropout(0.2)

Dense(7, Softmax)
```

---

## Training Configuration

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Input Size    | 96×96                    |
| Batch Size    | 32                       |
| Optimizer     | Adam                     |
| Learning Rate | 0.00005                  |
| Epochs        | 40                       |
| Loss Function | Categorical Crossentropy |
| Metric        | Accuracy                 |

---

## Fine-Tuning Strategy

For continual retraining:

- Previous model weights are loaded.
- All layers are frozen.
- Only the last 60 layers are unfrozen.
to prevent catastrophic forgetting.

---

## Data Augmentation

During retraining:

- Rotation
- Width Shift
- Height Shift
- Zoom
- Horizontal Flip
- Brightness Adjustment

This improves generalization and robustness.

---

## Training Callbacks

### EarlyStopping

Stops training when validation loss stops improving.

Patience:

```
7 epochs
```

---

### ReduceLROnPlateau

Automatically reduces learning rate.

Factor:

```
0.3
```

---

### ModelCheckpoint

Stores the best model:

```
emotion_mobilenet_best.h5
```

---

## Evaluation Metrics

The system evaluates:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

---

## Automatic Model Promotion

After retraining:

```
New Model Accuracy
          ↓
Compare With Current Model
          ↓
If Better
          ↓
Replace Production Model
```

This prevents poor retraining runs from degrading performance.

---

## Training Logs

Metrics are stored in:

```
training_logs/
```

including:

- Accuracy
- Precision
- Recall
- F1 Score
- Number of custom images
- Date and time
- Epoch count

This creates a history of model evolution.

---

## Current Performance

### Production Model

```
emotion_mobilenet_final.h5
```

Accuracy:

```
60.21%
```

The system is designed to continuously improve through user feedback and periodic retraining.

---

## Project Structure

```
EmotionDetectionAI/

app.py

templates/
    index.html
    correction.html

static/

models/
    emotion_mobilenet_final.h5
    emotion_v2.h5
    emotion_v2_best.h5

data/
    train/
    test/

custom_dataset/
    Angry/
    Disgust/
    Fear/
    Happy/
    Sad/
    Surprise/
    Neutral/

balanced_dataset/
    train/
    test/

training_logs/

dataset_metadata.csv

dataset_builder.py

retrain.py

requirements.txt
```

---

## Future Improvements

- Automatic retraining after collecting sufficient samples.
- Model registry and versioning.
- Production model tracking.
- Advanced active learning.
- Hard example mining.
- Confidence calibration.
- TensorBoard visualization.
- Ensemble models.
- Explainable AI using Grad-CAM.
- Real-time webcam emotion tracking.
- Multi-face emotion recognition.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- Flask
- NumPy
- Pandas
- Scikit-learn
- MobileNetV2

---

## Goal

To build a continuously learning facial emotion recognition system that becomes more accurate over time by leveraging user feedback and real-world data.
:::
