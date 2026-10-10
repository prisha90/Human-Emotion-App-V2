import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from sklearn.utils.class_weight import compute_class_weight


# =====================================================
# RANDOM SEED
# =====================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================
# CONFIGURATION
# =====================================================

IMG_SIZE = 96

BATCH_SIZE = 32

EPOCHS = 40

LEARNING_RATE = 5e-5


# =====================================================
# DATA AUGMENTATION
# =====================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,

    rotation_range=25,

    zoom_range=0.2,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# =====================================================
# DATA GENERATORS
# =====================================================

train_gen = train_datagen.flow_from_directory(
    "data/train",

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode="rgb",

    class_mode="categorical",

    batch_size=BATCH_SIZE,

    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    "data/test",

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode="rgb",

    class_mode="categorical",

    batch_size=BATCH_SIZE,

    shuffle=False
)


# =====================================================
# CLASS WEIGHTS
# =====================================================

labels = train_gen.classes

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(
    enumerate(class_weights)
)

print("\nClass Weights:\n")

for idx, weight in class_weights.items():

    print(
        f"Class {idx}: {weight:.4f}"
    )


# =====================================================
# BASE MODEL
# =====================================================

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)


# =====================================================
# FINE TUNING
# =====================================================

for layer in base_model.layers[:-60]:

    layer.trainable = False


for layer in base_model.layers[-60:]:

    layer.trainable = True


# =====================================================
# CLASSIFICATION HEAD
# =====================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)


# ------------------------
# Dense 2048
# ------------------------

x = Dense(
    2048,
    activation="relu"
)(x)

x = Dropout(
    0.5
)(x)


# ------------------------
# Dense 1024
# ------------------------

x = Dense(
    1024,
    activation="relu"
)(x)

x = Dropout(
    0.4
)(x)


# ------------------------
# Dense 512
# ------------------------

x = Dense(
    512,
    activation="relu"
)(x)

x = BatchNormalization()(x)

x = Dropout(
    0.3
)(x)


# ------------------------
# Dense 256
# ------------------------

x = Dense(
    256,
    activation="relu"
)(x)

x = BatchNormalization()(x)

x = Dropout(
    0.2
)(x)


# =====================================================
# OUTPUT LAYER
# =====================================================

output = Dense(
    7,
    activation="softmax"
)(x)


model = Model(
    inputs=base_model.input,
    outputs=output
)


# =====================================================
# COMPILE MODEL
# =====================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =====================================================
# CALLBACKS
# =====================================================

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
    "models/emotion_mobilenet_best_v2.h5",

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)

callbacks = [
    early_stopping,
    reduce_lr,
    checkpoint
]


# =====================================================
# TRAIN MODEL
# =====================================================

history = model.fit(
    train_gen,

    validation_data=test_gen,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


# =====================================================
# SAVE FINAL MODEL
# =====================================================

model.save(
    "models/emotion_mobilenet_final_v2.h5"
)

print(
    "\nFinal model saved successfully."
)