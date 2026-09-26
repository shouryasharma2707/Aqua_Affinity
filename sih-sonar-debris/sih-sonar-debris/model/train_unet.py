import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

print("Initializing U-Net training script...")

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

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


class SonarDataset(Dataset):
    def __init__(self, processed_dir, raw_dir, augment=True):
        self.image_files = [
            f for f in os.listdir(processed_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith(("annotated_", "predicted_"))
        ]
        self.label_map = {}
        if os.path.exists(raw_dir):
            for file in os.listdir(raw_dir):
                if file.lower().endswith(".txt"):
                    base_name = os.path.splitext(file)[0]
                    self.label_map[base_name] = os.path.join(raw_dir, file)
        self.processed_dir = processed_dir
        self.augment = augment

        matched = sum(
            1 for f in self.image_files
            if f.replace("processed_", "").rsplit(".", 1)[0] in self.label_map
        )
        print(f"Dataset: {len(self.image_files)} images found, {matched} have matching label files.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.processed_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        raw_base = filename.replace("processed_", "").rsplit(".", 1)[0]
        txt_path = self.label_map.get(raw_base)
        mask = yolo_txt_to_mask(txt_path, image.shape[0], image.shape[1])

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() < 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


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
    """Deeper 3-level U-Net (still lightweight enough for edge deployment)."""
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
        e1 = self.enc1(x)          # 32, H, W
        p1 = self.pool(e1)         # 32, H/2, W/2
        e2 = self.enc2(p1)         # 64, H/2, W/2
        p2 = self.pool(e2)         # 64, H/4, W/4

        b = self.bottleneck(p2)    # 128, H/4, W/4

        u2 = self.up2(b)           # 128, H/2, W/2
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # 64, H/2, W/2

        u1 = self.up1(d2)          # 64, H, W
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # 32, H, W

        return torch.sigmoid(self.final(d1))


def dice_loss(pred, target, eps=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + eps) / (pred.sum() + target.sum() + eps)


print("Model architecture ready. Loading dataset...")
dataset = SonarDataset(PROCESSED_DIR, RAW_DIR, augment=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = LightweightUNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Debris pixels are a small minority -> weight positive pixels much higher in BCE
pos_weight = torch.tensor([15.0])
bce = nn.BCELoss(reduction='none')

EPOCHS = 60

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for images, masks in dataloader:
        optimizer.zero_grad()
        outputs = model(images)

        raw_bce = bce(outputs, masks)
        weight_map = 1.0 + masks * (pos_weight - 1.0)  # upweight debris pixels
        weighted_bce = (raw_bce * weight_map).mean()

        loss = weighted_bce + dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / max(len(dataloader), 1)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/sonar_model.pth")
print("Training complete. Weights saved to model/sonar_model.pth")