import torch
import torch.nn as nn
import os

print("Initializing Edge Deployment Exporter (ONNX)...")

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

WEIGHTS_PATH = "model/sonar_model.pth"
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"{WEIGHTS_PATH} not found. Run train_unet.py first to produce trained weights."
    )

model = LightweightUNet()
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()
print(f"Loaded trained weights from {WEIGHTS_PATH}")

dummy_input = torch.randn(1, 1, 256, 256)
onnx_output_path = "model/sonar_model_edge.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['sonar_input'],
    output_names=['debris_mask_output'],
    dynamic_axes={'sonar_input': {0: 'batch_size'}, 'debris_mask_output': {0: 'batch_size'}}
)

print(f"Trained model successfully exported for edge deployment at: {onnx_output_path}")