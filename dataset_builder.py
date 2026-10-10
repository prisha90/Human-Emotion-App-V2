import os
import shutil
import pandas as pd


# ==================================================
# EMOTION LABELS
# ==================================================

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
# PATHS
# ==================================================

ORIGINAL_TRAIN = "data/train"

ORIGINAL_TEST = "data/test"

CUSTOM_DATASET = "custom_dataset"

METADATA_FILE = "dataset_metadata.csv"

BALANCED_DATASET = "balanced_dataset"

BALANCED_TRAIN = os.path.join(
    BALANCED_DATASET,
    "train"
)

BALANCED_TEST = os.path.join(
    BALANCED_DATASET,
    "test"
)

# ==================================================
# DELETE OLD BALANCED DATASET
# ==================================================

def clear_balanced_dataset():

    if os.path.exists(BALANCED_DATASET):

        print("Deleting old balanced_dataset...")

        shutil.rmtree(BALANCED_DATASET)


# ==================================================
# CREATE NEW BALANCED DATASET FOLDERS
# ==================================================

def create_balanced_folders():

    # Train folders
    for emotion in labels:

        os.makedirs(
            os.path.join(
                BALANCED_TRAIN,
                emotion
            ),
            exist_ok=True
        )

    # Test folders
    for emotion in labels:

        os.makedirs(
            os.path.join(
                BALANCED_TEST,
                emotion
            ),
            exist_ok=True
        )

    print("Created balanced dataset folders.")

# ==================================================
# COPY ORIGINAL TRAIN DATASET
# ==================================================

def copy_original_train():

    print("Copying original training dataset...")

    for emotion in labels:

        source_folder = os.path.join(
            ORIGINAL_TRAIN,
            emotion
        )

        destination_folder = os.path.join(
            BALANCED_TRAIN,
            emotion
        )

        for filename in os.listdir(source_folder):

            source_path = os.path.join(
                source_folder,
                filename
            )

            destination_path = os.path.join(
                destination_folder,
                filename
            )

            shutil.copy(
                source_path,
                destination_path
            )

    print("Training dataset copied.")


# ==================================================
# COPY ORIGINAL TEST DATASET
# ==================================================

def copy_original_test():

    print("Copying original test dataset...")

    for emotion in labels:

        source_folder = os.path.join(
            ORIGINAL_TEST,
            emotion
        )

        destination_folder = os.path.join(
            BALANCED_TEST,
            emotion
        )

        for filename in os.listdir(source_folder):

            source_path = os.path.join(
                source_folder,
                filename
            )

            destination_path = os.path.join(
                destination_folder,
                filename
            )

            shutil.copy(
                source_path,
                destination_path
            )

    print("Test dataset copied.")

# ==================================================
# CALCULATE SAMPLE WEIGHT
# ==================================================

def calculate_weight(row):

    weight = 1

    # Wrong prediction → very important
    if not row["correct"]:
        weight += 2

    # Low confidence prediction
    if row["confidence"] < 60:
        weight += 1

    # Webcam images are harder and more realistic
    if row["source"] == "webcam":
        weight += 1

    return weight

# ==================================================
# LOAD DATASET METADATA
# ==================================================

def load_metadata():

    if not os.path.exists(METADATA_FILE):

        print("Metadata file not found.")

        return pd.DataFrame()

    metadata = pd.read_csv(METADATA_FILE)

    print(
        f"Loaded {len(metadata)} metadata entries."
    )

    return metadata

# ==================================================
# COPY CUSTOM IMAGES WITH WEIGHTS
# ==================================================

def copy_custom_images():

    metadata = load_metadata()

    if metadata.empty:

        print("No metadata available.")
        return

    print("Copying custom images...")

    for _, row in metadata.iterrows():

        actual_emotion = row["actual"]

        filename = row["filename"]

        source_path = None

        # Find image inside custom_dataset/<emotion>
        emotion_folder = os.path.join(
            CUSTOM_DATASET,
            actual_emotion
        )

        for file in os.listdir(emotion_folder):

            if file.startswith(
                    filename.split(".")[0]
            ):

                source_path = os.path.join(
                    emotion_folder,
                    file
                )

                break

        if source_path is None:

            print(
                f"Could not find image for {filename}"
            )

            continue

        # Calculate importance weight
        weight = calculate_weight(row)

        destination_folder = os.path.join(
            BALANCED_TRAIN,
            actual_emotion
        )

        base_name, extension = os.path.splitext(
            os.path.basename(source_path)
        )

        # Create weighted copies
        for i in range(weight):

            new_filename = (
                f"{base_name}_custom_{i+1}"
                f"{extension}"
            )

            destination_path = os.path.join(
                destination_folder,
                new_filename
            )

            shutil.copy(
                source_path,
                destination_path
            )

    print("Custom images copied.")

# ==================================================
# BUILD COMPLETE BALANCED DATASET
# ==================================================

def create_balanced_dataset():

    print("\nBuilding balanced dataset...\n")

    # Step 1
    clear_balanced_dataset()

    # Step 2
    create_balanced_folders()

    # Step 3
    copy_original_train()

    # Step 4
    copy_original_test()

    # Step 5
    copy_custom_images()

    print("\nBalanced dataset created successfully.\n")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    create_balanced_dataset()