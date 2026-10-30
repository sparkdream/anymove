"""The exam: a made-up motion with known joints, filmed in the same studio, read back and scored bone by bone.

python -m mocap.exam make rig_info.json motion.json [seed] [frames]   random smooth motion inside the joint limits
python -m mocap.exam video RENDER_DIR out.mp4 [fps]                   front/ + side/ renders -> one layout video
python -m mocap.exam score truth_joints.json solved_joints.json       per-joint error in centimetres
"""
import json, math, pathlib, sys

import numpy as np


def make(info_path, out, seed=0, T=120, fps=24):
    info = json.load(open(info_path))
    rng = np.random.default_rng(seed)
    bones = [b["name"] for b in info["bones"]]
    t = np.arange(T) / fps
    eul = np.zeros((T, len(bones), 3))

    def wave(amp):
        f = rng.uniform(0.4, 1.8, 3)
        ph = rng.uniform(0, 2 * math.pi, 3)
        w = rng.dirichlet(np.ones(3))
        return amp * (w[:, None] * np.sin(2 * math.pi * f[:, None] * t + ph[:, None])).sum(0)

    for i, b in enumerate(info["bones"]):
        if i == 0:
            continue
        lim = np.radians(np.array(b["limits_deg"], float))
        for k in range(3):
            lo, hi = lim[k]
            c = np.clip(0.0, lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo))
            amp = (0.45 if k == 0 else 0.25) * (hi - lo) / 2
            eul[:, i, k] = np.clip(c + wave(amp), lo, hi)
    yaw, pitch, roll = wave(math.radians(20)), wave(math.radians(8)), wave(math.radians(5))
    quat = []
    for y, p, r in zip(yaw, pitch, roll):
        # pelvis basis rotation: local X = pitch, local Y (forward bone axis) = roll, local Z (down) = yaw
        cx, sx, cy, sy, cz, sz = math.cos(p / 2), math.sin(p / 2), math.cos(r / 2), math.sin(r / 2), math.cos(y / 2), math.sin(y / 2)
        qx, qy, qz = np.array([cx, sx, 0, 0]), np.array([cy, 0, sy, 0]), np.array([cz, 0, 0, sz])
        quat.append(qmul(qz, qmul(qy, qx)).tolist())
    trans = np.stack([wave(0.05), wave(0.12), 0.03 * np.abs(np.sin(2 * math.pi * 1.2 * t))], 1)
    json.dump({"fps": fps, "bones": bones, "trans": trans.round(5).tolist(), "root_quat": quat,
               "euler": eul.round(5).tolist()}, open(out, "w"))
    print("exam motion", out, T, "frames")


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def video(render_dir, out, fps=24):
    import cv2
    from mocap import layout
    d = pathlib.Path(render_dir)
    fr = sorted((d / "front").glob("*.png"))
    sd = sorted((d / "side").glob("*.png"))
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (layout.W, layout.H))
    for a, b in zip(fr, sd):
        f = cv2.imread(str(a), cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]]
        s = cv2.imread(str(b), cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]]
        if f.shape[0] != layout.H:  # rendered below layout size
            f = cv2.resize(f, (layout.W_FRONT, layout.H), interpolation=cv2.INTER_CUBIC)
            s = cv2.resize(s, (layout.W_SIDE, layout.H), interpolation=cv2.INTER_CUBIC)
        vw.write(np.ascontiguousarray(layout.compose(f, s)[..., ::-1]))
    vw.release()
    print("video", out, len(fr), "frames")


GROUPS = {"paws": ("fpaw", "ftoe", "hpaw", "htoe", "ankle", "ball", "toetip", "handtip"),
          "legs": ("elbow", "wrist", "knee", "hock", "shoulder", "hip_", "scap", "clav"),
          "spine": ("hips", "spine", "chest"), "head": ("neck", "head", "nose"), "tail": ("tail",)}


def score(truth_path, solved_path):
    a = json.load(open(truth_path))
    b = json.load(open(solved_path))
    names = a["names"]
    A, B = np.array(a["joints"]), np.array(b["joints"])
    T = min(len(A), len(B))
    e = np.linalg.norm(A[:T] - B[:T], axis=-1) * 100
    out = {"mean_cm": float(e.mean()), "median_cm": float(np.median(e)), "p95_cm": float(np.percentile(e, 95))}
    for g, keys in GROUPS.items():
        idx = [i for i, n in enumerate(names) if any(n.startswith(k) for k in keys)]
        if idx:
            out[f"{g}_cm"] = float(e[:, idx].mean())
    worst = np.argsort(-e.mean(0))[:5]
    out["worst_joints"] = {names[i]: round(float(e[:, i].mean()), 2) for i in worst}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0, int(sys.argv[5]) if len(sys.argv) > 5 else 120)
    elif cmd == "video":
        video(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 24)
    elif cmd == "score":
        score(sys.argv[2], sys.argv[3])
