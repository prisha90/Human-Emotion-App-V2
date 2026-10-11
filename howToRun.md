# How to Run — Human Emotion Detection App V2

This guide separates four different things:

1. **Running the app**: using the already-trained model to predict emotions.
2. **Evaluating a model**: checking accuracy and other metrics.
3. **Training from scratch**: building a new model from the original dataset.
4. **Improving a model**: fine-tuning the existing model with feedback images.

Anyone cloning this repository can run the app right away without retraining. Steps 3 to 6 are optional and are for reproducing or extending the workflow.

---

## Quick Summary

| Step | Goal                         | Command                                             | Required? |
| ---- | ---------------------------- | --------------------------------------------------- | --------- |
| 1    | Clone and install            | `git clone ...` / `pip install -r requirements.txt` | Yes       |
| 2    | Run the existing model       | `python app.py` or `python webcam_gui.py`           | Yes       |
| 3    | Evaluate the existing model  | `python evaluate.py`                                | Optional  |
| 4    | Train a new model            | `python train_mobile.py`                            | Optional  |
| 5    | Build the feedback dataset   | `python dataset_builder.py`                         | Optional  |
| 6    | Fine-tune the improved model | `python retrain.py`                                 | Optional  |

---

## Step 1 — Clone and Install Dependencies

```bash
git clone https://github.com/prisha90/Human-Emotion-App-V2.git
cd Human-Emotion-App-V2
pip install -r requirements.txt
```

## Step 2 — Run the Existing Model

First confirm that `models/emotion_mobilenet_final.h5` is present.

**Web app (Flask):**

```bash
python app.py
```

Open <http://127.0.0.1:5000> in your browser.

**Desktop webcam interface:**

Stop Flask with `Ctrl+C`, then run:

```bash
python webcam_gui.py
```

## Step 3 — Evaluate the Existing Model (Optional)

```bash
python evaluate.py
```

This evaluates the saved production model against `data/test`. The latest result is **60.21% accuracy**. No retraining is needed to reproduce this evaluation, provided the model file and the test dataset are available.

## Step 4 — Train a New Model (Optional)

Only needed to reproduce training or experiment with the architecture.

```bash
python train_mobile.py
```

Notes:

- Requires the original dataset under `data/train` and `data/test`.
- Results may differ from the saved model's accuracy, because training is not guaranteed to be reproducible.
- The script may overwrite an existing model if it saves to the same path. **Check its output filenames and back up your model first.**

## Step 5 — Build the Feedback Dataset (Optional)

After collecting user corrections through the app:

```bash
python dataset_builder.py
```

This builds `balanced_dataset/train` and `balanced_dataset/test` from the original dataset plus the custom feedback images and metadata.

## Step 6 — Fine-Tune and Evaluate the Improved Model (Optional)

```bash
python retrain.py
```

This uses the saved base model and the balanced dataset to fine-tune, evaluate, and potentially promote the updated model. **Check the script's promotion logic and back up your existing production model before running it.**

---

## Model Files

| File                         | Purpose                                                              |
| ---------------------------- | -------------------------------------------------------------------- |
| `emotion_mobilenet_final.h5` | Current production model; loaded by the Flask app and the webcam GUI |
| `emotion_v2_best.h5`         | Best checkpoint saved during the V2 fine-tuning run                  |
| `emotion_v2.h5`              | Final saved model from that run                                      |

Having three model files in the repository does not mean all three are trained, evaluated, or used by the app. The app currently loads only `emotion_mobilenet_final.h5`. The others are optional alternatives unless model switching is added.
