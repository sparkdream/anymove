"""Depth from one view: SAM 3D Body reads a 3D human in the front view, and its depths stand in for the side view.

python -m mocap.body3d VIDEO.mp4 DET.npz OUT.npz [--keep-side]
DET.npz is mocap.detect's reading of the clip. Each frame's front view goes through SAM 3D Body (Meta, SAM License,
commercial use allowed), cropped to the box around the joints the reader found. Its joints are mapped onto the rig's,
scaled to the rig's size, and their depths (distance from the front camera) become the side view's across-the-image
positions; heights come from the front reading. OUT.npz has DET.npz's layout, so mocap.solve fits it unchanged: the
video's own side half is no longer needed, and a video model that turns it into a second front view does no harm.
A limb SAM reads folded for a single frame while it points at the camera (a slash swinging through) is put back
straight when the front view allows it (unfold). A prop's tip (a blade) counts at PROP of its reading's confidence: the
least sure reading, and video models change the grip mid-move (the wrist has to follow it).
--keep-side: keep the side reading where both views agree on a joint's height, and use the depths only elsewhere.
--no-lengths: keep the readings as they are. By default each arm and leg is rebuilt from its root along the directions
read, with the rig's own bone lengths: video models stretch a limb in a fast move (a slashing arm a third longer than
the body allows), and the fit then leaves the arm hanging short of a point it cannot reach instead of pointing it there.

Needs a Python with sam_3d_body importable (SAM3D_BODY_DIR, default ~/src/sam-3d-body) and access to the gated
weights on Hugging Face (SAM3D_BODY_REPO, default facebook/sam-3d-body-dinov3).
"""
import contextlib, io, json, os, sys

import cv2
import numpy as np
import torch

from mocap import layout
from mocap.detect import read_frames

# rig keypoint -> MHR70 keypoints averaged for it, and how much its depth is trusted (1: read directly; less: implied)
MHR = {
    "hips": (["left-hip", "right-hip"], 1.0), "neck": (["neck"], 1.0), "nose": (["nose"], 1.0),
    "head": (["left-ear", "right-ear"], 1.0), "head_top": (["left-ear", "right-ear"], 0.5),
}
for s, side in (("_L", "left"), ("_R", "right")):
    MHR |= {f"shoulder{s}": ([f"{side}-shoulder"], 1.0), f"elbow{s}": ([f"{side}-elbow"], 1.0),
            f"wrist{s}": ([f"{side}-wrist"], 1.0), f"hip{s}": ([f"{side}-hip"], 1.0), f"knee{s}": ([f"{side}-knee"], 1.0),
            f"ankle{s}": ([f"{side}-ankle"], 1.0), f"toetip{s}": ([f"{side}-big-toe-tip"], 1.0),
            f"ball{s}": ([f"{side}-big-toe-tip", f"{side}-small-toe-tip", f"{side}-heel"], 0.8),
            f"handtip{s}": ([f"{side}-{f}-first-joint" for f in ("index", "middle", "ring", "pinky")], 0.8),
            f"clav{s}": ([f"{side}-shoulder", "neck"], 0.5)}
SPINE = ("spine1", "spine2", "chest")
SAM_CONF = 0.6  # the least a joint SAM reads counts for (times its trust); below it SAM's position takes over
PROP = 0.6  # a prop's tip (a blade) against the body's joints, in the confidence the solver weighs readings by  # depth between the hips and the neck, at their rest heights
# bones whose rest lengths set the scale from SAM's metres to the rig's
SCALE = [("hip_L", "knee_L"), ("knee_L", "ankle_L"), ("hip_R", "knee_R"), ("knee_R", "ankle_R"),
         ("shoulder_L", "elbow_L"), ("elbow_L", "wrist_L"), ("shoulder_R", "elbow_R"), ("elbow_R", "wrist_R"), ("hips", "neck")]


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")


def load_estimator(dev):
    sys.path.insert(0, os.path.expanduser(os.environ.get("SAM3D_BODY_DIR", "~/src/sam-3d-body")))
    with contextlib.redirect_stderr(io.StringIO()):
        import sam_3d_body.sam_3d_body_estimator as est
        from sam_3d_body.build_models import _hf_download, load_sam_3d_body
    if dev.type != "cuda":  # the estimator moves its batches to "cuda" by name
        to = est.recursive_to
        est.recursive_to = lambda x, t: to(x, dev if t == "cuda" else t)
        torch.cuda.empty_cache = lambda: None
        torch.Tensor.cuda = lambda self, *a, **k: self.to(dev)
    ckpt, mhr = _hf_download(os.environ.get("SAM3D_BODY_REPO", "facebook/sam-3d-body-dinov3"))
    model, cfg = load_sam_3d_body(ckpt, device=dev, mhr_path=mhr)
    return est.SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)


def box(uv, conf, w, h, pad=0.15, ok=0.3):
    """The box around the joints read with confidence, grown by pad of its size."""
    p = uv[conf > ok] if (conf > ok).sum() >= 4 else uv
    (x0, y0), (x1, y1) = p.min(0), p.max(0)
    g = pad * max(x1 - x0, y1 - y0)
    return np.array([max(0, x0 - g), max(0, y0 - g), min(w, x1 + g), min(h, y1 + g)], np.float32)


def read_bodies(estimator, views, boxes):
    """SAM 3D Body on each front view: (T, 70, 3) joints in camera axes (x right, y down, z away), metres, the
    body's centre at the origin; (T, 70, 2) the same joints in the view's pixels."""
    from sam_3d_body.metadata.mhr70 import mhr_names
    k3, k2 = [], []
    for i, (v, b) in enumerate(zip(views, boxes)):
        with contextlib.redirect_stdout(io.StringIO()):
            out = estimator.process_one_image(np.ascontiguousarray(v), bboxes=b[None], inference_type="body")
        k3.append(out[0]["pred_keypoints_3d"][:70])
        k2.append(out[0]["pred_keypoints_2d"][:70])
        print(f"\r  SAM 3D Body {i + 1}/{len(views)}", end="", flush=True)
    print()
    return np.array(k3), np.array(k2), mhr_names


def rig_points(k, mhr_names, names, rest):
    """MHR joints (T, 70, D) to the rig's keypoints (T, K, D) and the trust in each; unmapped joints get NaN."""
    T, K = len(k), len(names)
    out, trust = np.full((T, K, k.shape[-1]), np.nan), np.zeros(K)
    for j, n in enumerate(names):
        if n in MHR:
            src, w = MHR[n]
            out[:, j] = k[:, [mhr_names.index(s) for s in src]].mean(1)
            trust[j] = w
    h, nk = names.index("hips"), names.index("neck")
    for n in SPINE:
        if n in names:
            j = names.index(n)
            a = (rest[j, 2] - rest[h, 2]) / (rest[nk, 2] - rest[h, 2])
            out[:, j] = (1 - a) * out[:, h] + a * out[:, nk]
            trust[j] = 0.5
    return out, trust


def scale(k3, rest, names):
    """Rig metres per SAM metre: the rig's rest bone lengths over SAM's, median over the clip."""
    r = sum(np.linalg.norm(rest[names.index(a)] - rest[names.index(b)]) for a, b in SCALE)
    s = sum(np.linalg.norm(k3[:, names.index(a)] - k3[:, names.index(b)], axis=-1) for a, b in SCALE)
    return float(r / np.median(s))


def limbs(info, names):
    """Each arm and leg as its keypoints from the root out (shoulder to hand tip, hip to toe tip), and the keypoints
    that ride on its tip (a blade's tip)."""
    out = []
    for root in ("shoulder", "hip"):
        for s in ("_L", "_R"):
            chain = [root + s]
            while True:
                nxt = [b["tail_joint"] for b in info["bones"] if b["head_joint"] == chain[-1] and b["tail_joint"] in MHR]
                if not nxt:
                    break
                chain.append(nxt[0])
            rides = [b["tail_joint"] for b in info["bones"] if b["head_joint"] == chain[-1] and b["tail_joint"] not in MHR]
            out.append(([names.index(n) for n in chain], [names.index(n) for n in rides if n in names]))
    return out


def unfold(P, info, names, rest, bend=90, jump=50, line=0.15):
    """P (T, K, 3) world points. Seen from the front, a straight arm pointing at the camera and an arm folded at the
    elbow look alike, and SAM sometimes reads the fold for one frame in the middle of a slash. Where a limb is bent
    more than `bend` degrees, and `jump` more than in the frames either side, and its three joints line up in the front view
    (the middle one within `line` of the limb's length of the line), it is put back straight: depths along the line at the rig's lengths,
    toward the camera or away as the frames around it go. Returns the new points and {tip joint: frames fixed}."""
    P = P.copy()
    fixed = {}
    for chain, rides in limbs(info, names):
        a, b, c = chain[:3]
        u, l = np.linalg.norm(rest[b] - rest[a]), np.linalg.norm(rest[c] - rest[b])
        s1, s2 = P[:, b] - P[:, a], P[:, c] - P[:, b]
        cos = (s1 * s2).sum(-1) / (np.linalg.norm(s1, axis=-1) * np.linalg.norm(s2, axis=-1)).clip(1e-9)
        ang = np.degrees(np.arccos(cos.clip(-1, 1)))
        for t in range(1, len(P) - 1):
            if not (ang[t] > bend and ang[t] - max(ang[t - 1], ang[t + 1]) > jump):
                continue
            ab, ac = P[t, b, [0, 2]] - P[t, a, [0, 2]], P[t, c, [0, 2]] - P[t, a, [0, 2]]
            r = np.linalg.norm(ac)
            off = abs(ab[0] * ac[1] - ab[1] * ac[0]) / max(r, 1e-9)
            if off > line * (u + l) or r >= u + l:
                continue
            around = (P[t - 1, c, 1] - P[t - 1, a, 1] + P[t + 1, c, 1] - P[t + 1, a, 1]) / 2
            dy = (1.0 if around > 0 else -1.0) * np.sqrt((u + l) ** 2 - r ** 2)
            shift = P[t, a, 1] + dy - P[t, c, 1]
            P[t, b, 1] = P[t, a, 1] + dy * u / (u + l)
            for j in [c] + chain[3:] + rides:  # the hand and what it holds move with the wrist
                P[t, j, 1] += shift
            fixed.setdefault(names[c], []).append(t)
    return P, fixed


def prop_depth(P, info, names, rest):
    """P (T, K, 3) world points. A prop's tip (a blade) has no SAM point, so it sat at its hand's depth: flat to the
    front camera even where the arm holding it points at the camera (a torso turned side on), and the blade came out
    across the arm. Where it lines up with the forearm from the front (a dagger held along the arm), it carries on
    along the forearm in depth too; otherwise its depth comes from its length in the rig and how long it looks from
    the front, toward or away from the camera as the forearm goes. Either way it is put at the prop's length in the rig
    along that direction (video models draw a blade longer, as they do a swung arm). Returns the points and the tips."""
    P = P.copy()
    tips = []
    for chain, rides in limbs(info, names):
        if len(chain) < 3 or not rides:
            continue
        hand, base = chain[-1], chain[-3]  # the hand tip, and the elbow (forearm then hand)
        fore = P[:, hand] - P[:, base]
        fore_flat = np.linalg.norm(fore[:, [0, 2]], axis=-1).clip(1e-6)
        for r in rides:
            L = np.linalg.norm(rest[r] - rest[hand])
            v = P[:, r][:, [0, 2]] - P[:, hand][:, [0, 2]]
            flat = np.linalg.norm(v, axis=-1)
            by_length = np.sign(fore[:, 1]) * np.sqrt(np.clip(L ** 2 - flat ** 2, 0, None))
            along = fore[:, 1] / fore_flat * flat  # the forearm's slope into depth, carried on
            cos = (v * fore[:, [0, 2]]).sum(-1) / (flat * fore_flat).clip(1e-6)
            w = np.clip((np.degrees(np.arccos(cos.clip(-1, 1))) - 25) / -10, 0, 1)  # in line within 15-25 degrees
            d = np.concatenate([v[:, :1], (w * along + (1 - w) * by_length)[:, None], v[:, 1:]], -1)
            P[:, r] = P[:, hand] + d / np.linalg.norm(d, axis=-1, keepdims=True).clip(1e-6) * L  # at the rig's length
            tips.append(r)
    return P, tips


def rig_lengths(P, info, names, rest):
    """P (T, K, 3) world points: every limb rebuilt from its root along its read directions with the rig's bone lengths.
    Returns the new points and how much each limb's reading was stretched (largest length over the rig's, per frame)."""
    P = P.copy()
    stretch = {}
    for chain, rides in limbs(info, names):
        old = P[:, chain].copy()
        ratio = []
        for a, b in zip(chain, chain[1:]):
            seg = old[:, chain.index(b)] - old[:, chain.index(a)]  # the direction read; P[:, a] is already rebuilt
            n = np.linalg.norm(seg, axis=-1, keepdims=True).clip(1e-6)
            length = np.linalg.norm(rest[b] - rest[a])
            P[:, b] = P[:, a] + seg / n * length
            ratio.append(n[:, 0] / length)
        tip = chain[-1]
        for r in rides:
            P[:, r] += P[:, tip] - old[:, -1]
        stretch[names[chain[-1]]] = np.max(ratio, 0)
    return P, stretch


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    video, det_path, out = args[:3]
    det = dict(np.load(det_path))
    names = [str(n) for n in det["names"]]
    info = json.load(open(os.path.join(os.environ["MOCAP_CREATURE"], "rigged", "rig_info.json")))
    rest = np.array([[k["rest"] for k in info["keypoints"] if k["name"] == n][0] for n in names])
    frames, _ = read_frames(video)
    views = [cv2.warpAffine(f, det["calib"], (layout.W, layout.H), flags=cv2.INTER_AREA)[:, :layout.W_FRONT] for f in frames]
    boxes = [box(u, c, layout.W_FRONT, layout.H) for u, c in zip(det["front_uv"], det["front_conf"])]
    dev = device()
    print("SAM 3D Body on", dev)
    k3, k2, mhr_names = read_bodies(load_estimator(dev), views, boxes)
    p3, trust = rig_points(k3, mhr_names, names, rest)
    p2, _ = rig_points(k2, mhr_names, names, rest)
    # left/right sanity: SAM's 2D joints should land on the reader's, not on the mirror-image limb
    fc = det["front_conf"]
    near = np.nanmedian(np.where(fc > 0.5, np.linalg.norm(p2 - det["front_uv"], axis=-1), np.nan))
    print(f"SAM's joints vs the reader's in the front view: median {near:.1f} px apart")
    s = scale(p3, rest, names)
    h = names.index("hips")
    depth = (p3[..., 2] - p3[:, h:h + 1, 2]) * s + rest[h, 1]  # world y: the front camera looks along +y
    # joints SAM has no point for (the blade tips): their parent's depth, so the fit starts sane, and no weight
    for j, n in enumerate(names):
        if np.isnan(depth[:, j]).all():
            bone = info["bones"][info["keypoints"][j]["bone"]]
            depth[:, j] = depth[:, names.index(bone["head_joint"])]
    # where the reader is unsure of a joint (an elbow in a fast swing, at 0.3), SAM's own position takes over: it reads
    # the whole body at once and holds a straight arm the reader loses in the blur. Joints SAM reads count at least
    # SAM_CONF of their trust.
    seen = (trust >= 0.8) & ~np.isnan(p2[..., 0]).all(0)
    take = np.clip((SAM_CONF - fc) / 0.3, 0, 1) * seen
    uv = det["front_uv"] * (1 - take[..., None]) + np.nan_to_num(p2) * take[..., None]
    conf = np.where(seen, np.maximum(fc, SAM_CONF * trust), fc)
    print(f"joint-frames placed by SAM where the reader was unsure: {(take > 0.5).mean():.0%}")
    xz = layout.unproject_front(uv)
    P = np.stack([xz[..., 0], depth, xz[..., 1]], -1)
    front_uv = uv
    P, fixed = unfold(P, info, names, rest)
    print("limbs read folded for a frame, put back straight (frames):", fixed or "none")
    P, props = prop_depth(P, info, names, rest)
    if "--no-lengths" not in sys.argv:
        P, stretch = rig_lengths(P, info, names, rest)
        print("limbs read longer than the rig's (frames over 15%):",
              {n: int((r > 1.15).sum()) for n, r in stretch.items()}, "| longest", {n: round(float(r.max()), 2) for n, r in stretch.items()})
    front_uv, side_uv = layout.project(P)
    # a prop's tip (a blade): SAM has no point for it, and video models change the grip mid-move (a dagger held across
    # the hand at rest swung in line with the arm). The rig turns the prop in the fist to follow (its own joint), but
    # the reading is the least sure of all, so it counts for less than the body's joints
    front_conf = np.where(trust > 0, conf, conf * PROP)
    side_conf = front_conf * trust
    side_conf[:, props] = front_conf[:, props] * 0.5  # a prop's depth is a guess from its length (prop_depth)
    if "--keep-side" in sys.argv:
        agree = (np.abs(det["side_uv"][..., 1] - det["front_uv"][..., 1]) * layout.M_PER_PX < 0.06) & (det["side_conf"] > 0.7)
        side_uv = np.where(agree[..., None], det["side_uv"], side_uv)
        side_conf = np.where(agree, det["side_conf"], side_conf)
        print(f"kept the side reading for {agree.mean():.0%} of joint-frames")
    np.savez(out, **{**det, "front_uv": front_uv, "front_conf": front_conf, "front_uv_read": det["front_uv"], "side_uv": side_uv,
                     "side_conf": side_conf, "body3d": p3 * s, "body3d_scale": s, "body3d_px": near})
    print(f"rig scale {s:.3f}; depth range per joint (m): median {np.median(np.ptp(depth, 0)):.3f}, "
          f"largest {np.ptp(depth, 0).max():.3f} ({names[int(np.ptp(depth, 0).argmax())]}); saved {out}")


if __name__ == "__main__":
    main()
