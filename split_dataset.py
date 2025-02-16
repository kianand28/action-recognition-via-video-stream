import os
import shutil
import random

# Define paths
DATASET_PATH = "C:\\Users\\Newtons\\Downloads\\ML Engineering Assignment\\action-recognition-via-video-stream\\datasets\\UCF-101"
OUTPUT_PATH = "datasets/UCF101_small"

# Ensure output directories exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(OUTPUT_PATH, split), exist_ok=True)

# Select categories with 2+ human activities
selected_categories = ["Basketball", "BlowingCandles", "Billiards", 
                        "BasketballDunk", "BaseballPitch", "BandMarching"]

# Process each category
for category in selected_categories:
    category_path = os.path.join(DATASET_PATH, category)
    videos = os.listdir(category_path)
    random.shuffle(videos)

    # Split dataset: 70% train, 15% val, 15% test
    train_split = int(0.7 * len(videos))
    val_split = int(0.85 * len(videos))

    for idx, video in enumerate(videos):
        src = os.path.join(category_path, video)

        if idx < train_split:
            dest = os.path.join(OUTPUT_PATH, "train", category)
        elif idx < val_split:
            dest = os.path.join(OUTPUT_PATH, "val", category)
        else:
            dest = os.path.join(OUTPUT_PATH, "test", category)

        os.makedirs(dest, exist_ok=True)
        shutil.copy(src, dest)

print("Dataset successfully split into train, val, and test sets!")
