"""Read every joint in both halves of a two-camera video with a trained joint reader.

python -m mocap.detect VIDEO.mp4 MODEL.pt REF.png OUT.npz [overlay.mp4]
The clip is first aligned to the reference frame (the video model may resize or crop the canvas), then each frame is
split into its front and side views. Writes pixel positions in layout coordinates and peak confidences per view.
"""
import sys

import cv2
import numpy as np
import torch

from mocap import layout
from mocap.model import HM, SIZE, JointReader, decode, device, square


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[..., ::-1])
    return np.array(frames), fps


def calibrate(first, ref):
    """Affine map from video pixels to layout pixels, found by aligning the first frame to the reference image."""
    h, w = first.shape[:2]
    S = np.array([[layout.W / w, 0, 0], [0, layout.H / h, 0]], np.float32)  # plain stretch to start
    a = cv2.cvtColor(cv2.resize(ref, (640, 360)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(cv2.warpAffine(first, S, (layout.W, layout.H)), (640, 360)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        print("ECC did not converge; using a plain stretch")
    # warp maps reference (half-size) coords to frame coords; we want frame -> layout
    W2 = np.vstack([warp, [0, 0, 1]])
    W2[:2, 2] *= 2  # half-size -> full-size translation
    A = np.linalg.inv(W2) @ np.vstack([S, [0, 0, 1]])
    return A[:2].astype(np.float32)


def prep(view):
    _, ox, oy, side = square(view, view[2, 2])
    k = SIZE / side
    A = np.array([[k, 0, ox * k], [0, k, oy * k]], np.float32)
    fill = tuple(float(x) for x in view[2, 2])
    out = cv2.warpAffine(view, A, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return out, (ox, oy, side)


@torch.no_grad()
def run(model, views, bs=32):
    uv_all, conf_all = [], []
    for i in range(0, len(views), bs):
        batch = [prep(v) for v in views[i:i + bs]]
        dev = next(model.parameters()).device
        x = torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2).float().to(dev) / 255
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            hm = model(x).float()
        uv, conf = decode(hm)
        uv = uv.cpu().numpy() * SIZE / HM
        for (_, (ox, oy, side)), u in zip(batch, uv):
            uv_all.append(u * side / SIZE - np.array([ox, oy]))
        conf_all.append(conf.cpu().numpy())
    return np.array(uv_all), np.concatenate(conf_all)


def load(model_path):
    ck = torch.load(model_path, map_location="cpu", weights_only=True)  # tensors, names and numbers only: safe for downloaded readers
    model = JointReader(len(ck["names"])).to(device()).eval()
    model.load_state_dict(ck["state_dict"])
    return model, ck["names"]


def overlay(frames, fuv, suv, fconf, names, path, fps):
    h, w = layout.H, layout.W
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    col = lambda n: (209, 60, 245) if n.endswith("_L") else (255, 107, 47) if n.endswith("_R") else (255, 255, 255)  # BGR
    for t, f in enumerate(frames):
        img = np.ascontiguousarray(f[..., ::-1])
        for j, n in enumerate(names):
            for (u, v), x0 in ((fuv[t, j], 0), (suv[t, j], layout.W_FRONT)):
                cv2.circle(img, (int(u + x0), int(v)), 4, col(n), -1, cv2.LINE_AA)
        vw.write(img)
    vw.release()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    video, model_path, ref_path, out = args[:4]
    frames, fps = read_frames(video)
    ref = cv2.imread(ref_path)[..., ::-1]
    if "--no-calib" in sys.argv:  # already a layout frame (the exam)
        A = np.array([[layout.W / frames.shape[2], 0, 0], [0, layout.H / frames.shape[1], 0]], np.float32)
    else:
        A = calibrate(frames[0], ref)
    print("calibration (video px -> layout px):", A.round(4).tolist())
    frames = np.array([cv2.warpAffine(f, A, (layout.W, layout.H), flags=cv2.INTER_AREA) for f in frames])
    model, names = load(model_path)
    fuv, fconf = run(model, [f[:, :layout.W_FRONT] for f in frames])
    suv, sconf = run(model, [f[:, layout.W_FRONT:] for f in frames])
    np.savez(out, front_uv=fuv, side_uv=suv, front_conf=fconf, side_conf=sconf, fps=fps, names=np.array(names), calib=A)
    print(f"{len(frames)} frames at {fps:.2f} fps; mean confidence front {fconf.mean():.2f} side {sconf.mean():.2f}")
    if len(args) > 4:
        overlay(frames, fuv, suv, fconf, names, args[4], fps)


if __name__ == "__main__":
    main()
