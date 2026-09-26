import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
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
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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


def yolo_txt_to_mask(txt_path, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.float32)
    if txt_path and os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f.readlines():
                vals = line.strip().split()
                if len(vals) >= 5:
                    _, xc, yc, bw, bh = map(float, vals[:5])
                    xmin = int((xc - bw / 2) * img_w)
                    ymin = int((yc - bh / 2) * img_h)
                    xmax = int((xc + bw / 2) * img_w)
                    ymax = int((yc + bh / 2) * img_h)
                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(img_w, xmax), min(img_h, ymax)
                    mask[ymin:ymax, xmin:xmax] = 1.0
    return mask


PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"
WEIGHTS_PATH = "model/sonar_model.pth"
TEST_SPLIT_FILE = "model/test_split.txt"
THRESHOLD = 0.5

model = LightweightUNet()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

label_map = {}
if os.path.exists(RAW_DIR):
    for file in os.listdir(RAW_DIR):
        if file.lower().endswith(".txt"):
            base_name = os.path.splitext(file)[0]
            label_map[base_name] = os.path.join(RAW_DIR, file)

# Evaluate only on the held-out test split if it exists (honest, unseen-data metrics).
# Falls back to all images if train_unet.py hasn't produced a split file yet.
if os.path.exists(TEST_SPLIT_FILE):
    with open(TEST_SPLIT_FILE, "r") as f:
        image_files = [line.strip() for line in f if line.strip()]
    print(f"Evaluating on held-out TEST split ({len(image_files)} images) from {TEST_SPLIT_FILE}")
else:
    image_files = [
        f for f in os.listdir(PROCESSED_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith(("annotated_", "predicted_"))
    ]
    print(f"No test_split.txt found — evaluating on ALL {len(image_files)} images (not a clean unseen-data test).")

eps = 1e-6
totals = {"dice": [], "iou": [], "precision": [], "recall": []}
per_image_results = []

for filename in image_files:
    img_path = os.path.join(PROCESSED_DIR, filename)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue

    raw_base = filename.replace("processed_", "").rsplit(".", 1)[0]
    txt_path = label_map.get(raw_base)
    gt_mask = yolo_txt_to_mask(txt_path, gray.shape[0], gray.shape[1])

    inp = gray.astype(np.float32) / 255.0
    inp = torch.tensor(inp).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred_prob = model(inp).squeeze().numpy()
    pred_mask = (pred_prob > THRESHOLD).astype(np.float32)

    tp = float((pred_mask * gt_mask).sum())
    fp = float((pred_mask * (1 - gt_mask)).sum())
    fn = float(((1 - pred_mask) * gt_mask).sum())

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    totals["dice"].append(dice)
    totals["iou"].append(iou)
    totals["precision"].append(precision)
    totals["recall"].append(recall)

    per_image_results.append({
        "image": filename,
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    })

    print(f"{filename} | Dice: {dice:.4f} | IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

summary = {metric: round(float(np.mean(vals)), 4) if vals else 0.0 for metric, vals in totals.items()}

print("\n=== Overall Evaluation Summary ===")
for metric, val in summary.items():
    print(f"Mean {metric.capitalize()}: {val}")

with open(os.path.join(PROCESSED_DIR, "evaluation_report.json"), "w") as jf:
    json.dump({"per_image": per_image_results, "summary": summary}, jf, indent=4)

print(f"\nSaved detailed report to {os.path.join(PROCESSED_DIR, 'evaluation_report.json')}")