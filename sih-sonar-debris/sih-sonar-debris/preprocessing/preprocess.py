import os
import cv2
import numpy as np

input_dir = "../data/raw"
output_dir = "../data/processed"
os.makedirs(output_dir, exist_ok=True)

print("Starting preprocessing pipeline & syncing directories...")

# SYNC: Remove any processed files that no longer exist in raw
raw_files = set(f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
for p_file in os.listdir(output_dir):
    # Strip prefix to find original name
    orig_name = p_file.replace("processed_", "").replace("annotated_", "")
    if orig_name not in raw_files:
        stale_path = os.path.join(output_dir, p_file)
        os.remove(stale_path)
        print(f"Removed stale file: {p_file}")

# Loop through current raw images
for filename in raw_files:
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Warning: Could not load image {filename}. Skipping.")
        continue

    # Step A: Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Step B: Resolution Normalization
    resized = cv2.resize(enhanced, (256, 256))

    # Step C: Save processed output
    out_path = os.path.join(output_dir, f"processed_{filename}")
    cv2.imwrite(out_path, resized)
    print(f"Processed and saved: {out_path}")

print("Preprocessing & sync complete!")