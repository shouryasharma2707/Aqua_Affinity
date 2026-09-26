import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np


# ==========================================
# 1. MODEL ARCHITECTURE DEFINITION
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightweightUNet(nn.Module):
    def __init__(self):
        super(LightweightUNet, self).__init__()
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = ConvBlock(64, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ConvBlock(128 + 64, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ConvBlock(64 + 32, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return torch.sigmoid(self.final(d1))


# ==========================================
# 2. BOX MERGING (NON-MAX SUPPRESSION)
# ==========================================
def merge_overlapping_boxes(boxes, iou_thresh=0.2):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: -b["confidence"])
    merged = []
    used = [False] * len(boxes)

    def iou(a, b):
        xa1, ya1, xa2, ya2 = a["xmin"], a["ymin"], a["xmax"], a["ymax"]
        xb1, yb1, xb2, yb2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]

        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)

        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih

        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)

        return inter / max(area_a + area_b - inter, 1e-6)

    for i in range(len(boxes)):
        if used[i]:
            continue

        group = [boxes[i]]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if not used[j] and iou(boxes[i], boxes[j]) > iou_thresh:
                group.append(boxes[j])
                used[j] = True

        xmin = min(b["xmin"] for b in group)
        ymin = min(b["ymin"] for b in group)
        xmax = max(b["xmax"] for b in group)
        ymax = max(b["ymax"] for b in group)
        conf = max(b["confidence"] for b in group)

        merged.append(
            {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "confidence": conf,
            }
        )

    return merged


# ==========================================
# 3. CONFIGURATION & INFERENCE SETUP
# ==========================================
PROCESSED_DIR = "data/processed"
WEIGHTS_PATH = "model/sonar_model.pth"

# Tuning Parameters: Lower THRESHOLD if boxes are not showing up
THRESHOLD = 0.5  # Confidence threshold for binary mask (e.g., 0.2 to 0.5)
MIN_CONTOUR_AREA = 40  # Minimum pixel area required to form a bounding box
MAX_DETECTIONS_PER_IMAGE = 5

# Load model weights
model = LightweightUNet()
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    print(f"Successfully loaded model weights from: {WEIGHTS_PATH}")
else:
    print(f"Warning: Model weights file not found at '{WEIGHTS_PATH}'!")

model.eval()

predictions = []

# Fetch clean processed images (excluding previous annotated/predicted outputs)
image_files = [
    f
    for f in os.listdir(PROCESSED_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
    and not f.startswith(("annotated_", "predicted_"))
]

kernel = np.ones((5, 5), np.uint8)

# ==========================================
# 4. INFERENCE LOOP
# ==========================================
for filename in image_files:
    img_path = os.path.join(PROCESSED_DIR, filename)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Could not read image: {filename}")
        continue

    # Prepare input tensor shape: (1, 1, H, W)
    inp = gray.astype(np.float32) / 255.0
    inp = torch.tensor(inp).unsqueeze(0).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        pred_mask = model(inp).squeeze().numpy()

    # Thresholding and morphological cleaning
    binary_mask = (pred_mask > THRESHOLD).astype(np.uint8) * 255
    binary_mask = cv2.morphologyEx(
        binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    binary_mask = cv2.morphologyEx(
        binary_mask, cv2.MORPH_OPEN, kernel, iterations=1
    )

    # Find contour blobs
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    raw_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # Average probability across pixels inside the bounding box
        region_conf = float(pred_mask[y : y + h, x : x + w].mean())
        raw_boxes.append(
            {
                "xmin": x,
                "ymin": y,
                "xmax": x + w,
                "ymax": y + h,
                "confidence": region_conf,
            }
        )

    # Merge overlapping bounding boxes and cap max detections
    merged_boxes = merge_overlapping_boxes(raw_boxes, iou_thresh=0.05)
    merged_boxes = sorted(merged_boxes, key=lambda b: -b["confidence"])[
        :MAX_DETECTIONS_PER_IMAGE
    ]

    # Convert grayscale image to BGR for drawing green bounding boxes
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, box in enumerate(merged_boxes, start=1):
        x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        conf = box["confidence"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"Debris #{i} ({conf * 100:.0f}%)",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

        predictions.append(
            {
                "image": filename,
                "anomaly_id": i,
                "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                "confidence": round(conf * 100, 2),
            }
        )

    # Save output annotated prediction image
    out_path = os.path.join(PROCESSED_DIR, f"predicted_{filename}")
    cv2.imwrite(out_path, annotated)
    print(
        f"{filename}: {len(merged_boxes)} clean detection(s) -> saved {out_path}"
    )

# ==========================================
# 5. EXPORT REPORT
# ==========================================
with open(os.path.join(PROCESSED_DIR, "predicted_report.json"), "w") as jf:
    json.dump(predictions, jf, indent=4)

print(
    f"Done. Inference complete on {len(image_files)} images. Report: predicted_report.json"
)