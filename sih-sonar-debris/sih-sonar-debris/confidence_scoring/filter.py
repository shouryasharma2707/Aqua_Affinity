import os
import cv2
import numpy as np

print("Initializing Confidence Scoring & False-Positive Filter...")

def score_detections(mask_image_path):
    # Load binary mask
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # Threshold to ensure binary image
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find connected components (individual predicted blobs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    detections = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # Heuristic rules to score debris vs. natural rock/noise
        # Rule 1: Filter out tiny speckle noise or massive terrain patches
        if area < 10 or area > 10000:
            confidence = 15.0  # Very low confidence (likely noise)
        else:
            # Rule 2: Check aspect ratio regularity (compact shapes score higher)
            aspect_ratio = max(width, height) / (min(width, height) + 1e-5)
            if aspect_ratio < 2.5: 
                confidence = 88.5  # High confidence debris profile
            else:
                confidence = 45.0  # Elongated ridge/rock profile

        detections.append({
            "blob_id": i,
            "bbox": (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], width, height),
            "area": int(area),
            "confidence_score": confidence
        })

    return detections

# Run test across our processed dataset folder
processed_dir = "data/processed"
if os.path.exists(processed_dir):
    for filename in os.listdir(processed_dir):
        if filename.endswith(".png"):
            path = os.path.join(processed_dir, filename)
            results = score_detections(path)
            print(f"File: {filename} -> Filtered {len(results)} potential objects. Confidence Scores: {[d['confidence_score'] for d in results]}")
else:
    print("Processed folder not found. Run preprocessing first!")

print("Confidence scoring module executed successfully!")