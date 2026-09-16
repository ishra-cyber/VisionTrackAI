"""Model definitions: U-Net (disc + cup) and EfficientNet-B0 classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = DoubleConv(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Classic U-Net. Output channel 0 = optic disc, channel 1 = optic cup (logits)."""

    def __init__(self, in_ch=3, out_ch=2, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc1 = DoubleConv(in_ch, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3], dropout=0.1)
        self.bott = DoubleConv(c[3], c[4], dropout=0.2)
        self.pool = nn.MaxPool2d(2)
        self.up4 = Up(c[4], c[3], c[3])
        self.up3 = Up(c[3], c[2], c[2])
        self.up2 = Up(c[2], c[1], c[1])
        self.up1 = Up(c[1], c[0], c[0])
        self.head = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bott(self.pool(e4))
        d = self.up4(b, e4)
        d = self.up3(d, e3)
        d = self.up2(d, e2)
        d = self.up1(d, e1)
        return self.head(d)


def build_classifier(pretrained=True, dropout=0.3):
    """EfficientNet-B0 with a single-logit head (glaucoma vs non-glaucoma)."""
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
    return model


class SegLoss(nn.Module):
    """BCE + soft Dice, cup channel weighted higher (harder, smaller target)."""

    def __init__(self, channel_weights=(1.0, 1.5)):
        super().__init__()
        self.register_buffer("w", torch.tensor(channel_weights, dtype=torch.float32))
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, target):
        bce = self.bce(logits, target).mean(dim=(0, 2, 3))
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(0, 2, 3))
        denom = prob.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        dice_loss = 1 - (2 * inter + 1.0) / (denom + 1.0)
        return ((bce + dice_loss) * self.w).sum() / self.w.sum()


def load_unet(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = UNet(base=cfg.get("base_ch", 32))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


def load_classifier(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_classifier(pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg
