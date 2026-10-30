"""Teach the joint reader one creature, from the rendered dataset.

python -m mocap.train DATASET_DIR rig_info.json OUT.pt [epochs] [batch]
Every sample gives two training images (front and side). The renders are transparent, so each one is laid on a
fresh random backdrop, then jittered in framing, colour, sharpness and compression, the ways a video model's frames
differ from a clean render. Holds out 4% of samples and reports their mean joint error in layout pixels.
"""
import json, math, os, pathlib, sys, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mocap import layout
from mocap.model import HM, SIZE, JointReader, decode, device, square


def backdrop(h, w, ground, rng):
    g = rng.uniform(95, 175)
    tint = rng.uniform(-6, 6, 3)
    img = np.empty((h, w, 3), np.float32)
    img[:] = g + tint
    img += np.linspace(-1, 1, h)[:, None, None] * rng.uniform(-10, 10)
    img[ground:] += rng.uniform(5, 35) * (1 if rng.random() < 0.85 else -1)
    return img


def photometric(img, rng):
    img = img.astype(np.float32)
    if rng.random() < 0.8:  # brightness, contrast, saturation, hue
        img = img * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
        m = img.mean()
        img = (img - m) * rng.uniform(0.75, 1.25) + m
        hsv = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180
        hsv[..., 1] *= rng.uniform(0.7, 1.3)
        img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.3:
        f = rng.uniform(0.4, 0.8)
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.2:
        img = img + rng.normal(0, rng.uniform(2, 8), img.shape)
    img = img.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return img


class Views(Dataset):
    def __init__(self, metas, train):
        self.metas, self.train = metas, train

    def __len__(self):
        return 2 * len(self.metas)

    def __getitem__(self, i):
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        m = self.metas[i // 2]
        view = ("front", "side")[i % 2]
        meta = json.loads(m.read_text())
        s = meta["scale"]
        rgba = cv2.imread(str(m).replace("_meta.json", f"_{view}.png"), cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]].astype(np.float32)
        h, w = rgba.shape[:2]
        kp = np.array(meta[f"{view}_uv"], np.float32) * s
        if self.train:
            bg = backdrop(h, w, round(layout.GROUND_V * s), rng)
        else:
            bg = np.empty((h, w, 3), np.float32); bg[:] = layout.BG; bg[round(layout.GROUND_V * s):] = layout.FLOOR
        a = rgba[..., 3:] / 255
        img = rgba[..., :3] * a + bg * (1 - a)
        if self.train and rng.random() < 0.7:  # the divider at the shared edge
            img[:, -2:] = 34 if view == "front" else img[:, -2:]
            img[:, :2] = 34 if view == "side" else img[:, :2]
        _, ox, oy, side = square(img, bg[0, 0])
        k = SIZE / side
        A = np.array([[k, 0, ox * k], [0, k, oy * k]], np.float32)
        if self.train:  # framing jitter: scale, small rotation, shift
            sc, rot = rng.uniform(0.85, 1.15), math.radians(rng.uniform(-4, 4))
            c = SIZE / 2
            R = cv2.getRotationMatrix2D((c, c), math.degrees(rot), sc)
            R[:, 2] += rng.uniform(-0.05, 0.05, 2) * SIZE
            A = R @ np.vstack([A, [0, 0, 1]])
        out = cv2.warpAffine(img, A.astype(np.float32), (SIZE, SIZE), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(x) for x in bg[0, 0]))
        kp = kp @ A[:, :2].T + A[:, 2]
        if self.train:
            out = photometric(out, rng)
        x = torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).float() / 255
        return x, torch.from_numpy(kp.astype(np.float32)), torch.tensor([ox, oy, side, s], dtype=torch.float32)


def heatmaps(kp, sigma=2.0):
    """Gaussian targets at HM resolution for keypoints given in SIZE pixels."""
    b, j, _ = kp.shape
    g = (kp * HM / SIZE).view(b, j, 1, 1, 2)
    ys, xs = torch.meshgrid(torch.arange(HM, device=kp.device), torch.arange(HM, device=kp.device), indexing="ij")
    d2 = (xs - g[..., 0]) ** 2 + (ys - g[..., 1]) ** 2
    return torch.exp(-d2 / (2 * sigma ** 2))


def main():
    data, info_path, out = sys.argv[1:4]
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    bs = int(sys.argv[5]) if len(sys.argv) > 5 else 24
    info = json.load(open(info_path))
    names = [k["name"] for k in info["keypoints"]]
    def split():
        metas = sorted(pathlib.Path(data).glob("*_meta.json"))
        val = [m for m in metas if int(m.name[:5]) % 25 == 0]
        return [m for m in metas if int(m.name[:5]) % 25 != 0], val

    train, val = split()
    print(f"{len(train)} train / {len(val)} val samples so far, {len(names)} joints", flush=True)
    cpus = len(os.sched_getaffinity(0))  # no more loader workers than the CPUs this process may run on
    vl = DataLoader(Views(val, False), batch_size=bs, num_workers=min(4, cpus))
    dev = device()
    model = JointReader(len(names)).to(dev)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": 2e-5},
                             {"params": head, "lr": 1e-3}], weight_decay=0.05)
    base = [g["lr"] for g in opt.param_groups]
    step, t0, warm = 0, time.time(), 300
    for ep in range(epochs):
        train, _ = split()
        dl = DataLoader(Views(train, True), batch_size=bs, shuffle=True, num_workers=min(10, cpus), drop_last=True)
        print(f"epoch {ep}: {len(train)} train samples", flush=True)
        model.train()
        for bi_, (x, kp, _) in enumerate(dl):
            f = 0.5 * (1 + math.cos(math.pi * (ep + bi_ / len(dl)) / epochs)) * min(1, (step + 1) / warm)
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f
            x, kp = x.to(dev, non_blocking=True), kp.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.bfloat16):
                pred = model(x)
            loss = F.mse_loss(pred.float(), heatmaps(kp)) * 1000
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"ep {ep} step {step} loss {loss.item():.4f} lr {f:.3f} {time.time() - t0:.0f}s", flush=True)
        model.eval()
        errs = {"front": [], "side": []}
        with torch.no_grad():
            for bi, (x, kp, geo) in enumerate(vl):
                with torch.autocast(dev, dtype=torch.bfloat16):
                    pred = model(x.to(dev)).float()
                uv, _ = decode(pred)
                uv = uv.cpu() * SIZE / HM
                # back to layout pixels: undo resize, padding and the render scale
                lay = lambda p: (p * geo[:, None, 2:3] / SIZE - geo[:, None, :2]) / geo[:, None, 3:4]
                e = (lay(uv) - lay(kp)).norm(dim=-1)
                for r in range(len(x)):
                    errs[("front", "side")[(bi * bs + r) % 2]].append(e[r].numpy())
        msg = {v: float(np.mean(errs[v])) for v in errs}
        print(f"epoch {ep} val mean px error (layout pixels) front {msg['front']:.2f} side {msg['side']:.2f}", flush=True)
        torch.save({"state_dict": model.state_dict(), "names": names, "epoch": ep, "val": msg}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
