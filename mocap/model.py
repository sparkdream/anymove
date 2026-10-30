"""The joint reader: DINOv2 (ViT-B/14 with registers) plus a small upsampling head that draws one heatmap per joint.

Both views go through the same network one at a time. A view is padded to a square with the backdrop grey, centred,
and resized to 448 x 448 (32 x 32 patches); heatmaps come out at 128 x 128.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

BACKBONE = "facebook/dinov2-with-registers-base"
SIZE, HM = 448, 128
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def device():
    """The GPU to run on: CUDA, else an Intel GPU (xpu), else the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


class JointReader(nn.Module):
    def __init__(self, n_joints, backbone=BACKBONE):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.n_reg = getattr(self.backbone.config, "num_register_tokens", 0)
        c = self.backbone.config.hidden_size

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(16, o), nn.GELU())

        self.neck = block(c, 256)
        self.up1 = block(256, 128)
        self.up2 = block(128, 128)
        self.out = nn.Conv2d(128, n_joints, 1)

    def forward(self, x):
        b = x.shape[0]
        tokens = self.backbone(pixel_values=(x - MEAN.to(x)) / STD.to(x)).last_hidden_state
        g = SIZE // 14
        f = tokens[:, 1 + self.n_reg:].transpose(1, 2).reshape(b, -1, g, g)
        f = self.neck(f)
        f = self.up1(F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False))
        f = self.up2(F.interpolate(f, size=(HM, HM), mode="bilinear", align_corners=False))
        return self.out(f)


def square(img, fill):
    """Pad an H x W x 3 view to a centred square; return it and the (offset_x, offset_y) of the view inside."""
    h, w = img.shape[:2]
    s = max(h, w)
    out = np.empty((s, s, 3), img.dtype)
    out[:] = fill
    ox, oy = (s - w) // 2, (s - h) // 2
    out[oy:oy + h, ox:ox + w] = img
    return out, ox, oy, s


def decode(hm):
    """Heatmaps (B, J, 128, 128) to sub-pixel peaks in heatmap units and peak values."""
    b, j, h, w = hm.shape
    flat = hm.reshape(b, j, -1)
    conf, idx = flat.max(-1)
    y, x = (idx // w).float(), (idx % w).float()
    # quadratic refinement from the neighbours of the peak
    xi, yi = x.long().clamp(1, w - 2), y.long().clamp(1, h - 2)
    gather = lambda dy, dx: hm.gather(-1, (xi + dx).view(b, j, 1, 1).expand(b, j, h, 1)).squeeze(-1) \
        .gather(-1, (yi + dy).view(b, j, 1)).squeeze(-1)
    c, l, r, u, d = gather(0, 0), gather(0, -1), gather(0, 1), gather(-1, 0), gather(1, 0)
    dx = 0.5 * (r - l) / (2 * c - l - r).clamp(min=1e-6)
    dy = 0.5 * (d - u) / (2 * c - u - d).clamp(min=1e-6)
    x = x + dx.clamp(-0.5, 0.5)
    y = y + dy.clamp(-0.5, 0.5)
    return torch.stack([x, y], -1), conf
