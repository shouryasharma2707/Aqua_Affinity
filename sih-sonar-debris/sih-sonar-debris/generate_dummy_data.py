import os
import numpy as np
import cv2

# Create directories if they don't exist
os.makedirs("data/raw", exist_ok=True)

print("Generating dummy side-scan sonar images...")

for i in range(5):
    # Create a grayscale waterfall-style strip (e.g., height 512, width 256)
    # Base background noise mimicking water floor
    sonar_strip = np.random.randint(40, 80, (512, 256), dtype=np.uint8)

    # Add a random "debris" object (bright white ellipse) and its acoustic shadow (dark patch)
    cx, cy = np.random.randint(100, 400), np.random.randint(80, 180)
    cv2.ellipse(sonar_strip, (cx, cy), (15, 8), 0, 0, 360, 220, -1)
    cv2.ellipse(sonar_strip, (cx + 20, cy), (15, 8), 0, 0, 360, 10, -1)  # shadow effect

    # Save the image
    file_path = f"data/raw/sonar_pass_{i+1}.png"
    cv2.imwrite(file_path, sonar_strip)
    print(f"Saved: {file_path}")

print("Dummy data generation complete!")