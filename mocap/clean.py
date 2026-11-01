"""Clean up a solved motion the way an animator would: hold limbs the reader lost, hold the pose once a move has
settled, and smooth what is left with a strength that follows the speed.

python -m mocap.clean DETECTIONS.npz rig_info.json RAW.json OUT.json [--no-occlusion] [--no-settle] [--no-smooth]
    [--off] [--still BODY_PER_S] [--occluded RATIO]
--off writes RAW unchanged (the same layout either way, so the rest of the pipeline does not care).

The rig splits into chains: from a branch point (the pelvis, the chest) out to a tip (a hand, a foot, the head). The
bones before the first branch point are the trunk, which carries the root.

1. Occlusion. A chain whose far half (elbow to tip) is read in either view with a confidence well below its own usual
   level for a while (an arm hidden behind the other one swinging in front) is not followed: the solver would trust
   the other view, which confidently guesses at a limb it cannot see either. The chain holds its last pose read
   clearly and, if it comes back, eases over to where it is read again.
2. Settle. When a chain's joints barely move for a few frames, the move is over: the chain holds that pose instead of
   following what is left, which is mostly reader noise on the least observed turns (a hand's twist about the
   forearm). The chain's last bone (a blade, a toe) is left out of the test: its turning is exactly that noise. The
   trunk and the root hold only where every chain holds: a few millimetres of sway at the chest swing a blade tip by
   a centimetre a frame.
3. Smoothing, only where the reader was unsure. Each bone's track is smoothed with a Gaussian whose width grows as
   the confidence on the keypoints that place it drops below their usual level, and shrinks in fast moves (a slash
   keeps its snap); neighbours count by how well they were read. A clearly read track is left as solved: the solver
   already smooths, and blurring a clean track only loses motion (on the exam, plain smoothing tripled the error).
"""
import json, sys

import numpy as np
import torch

from mocap.solve import Rig, euler_xyz, project

SIGMA_MAX = 2.0  # frames, for a bone the reader could hardly see
CLEAR, UNSURE = 0.85, 0.4  # reader confidence, as a share of its usual level, between which smoothing fades in
SPEED_SLOW, SPEED_FAST = 1.5, 6.0  # rad/s between which the width shrinks to a third (fast moves keep their snap)
MIN_SETTLE, MIN_OCCLUDED, BLEND = 4, 6, 4  # frames


def chains(info):
    """Bone chains from a branch point out to a tip, and the trunk (bones up to the first branch point)."""
    parent = [b["parent"] for b in info["bones"]]
    kids = [[i for i, p in enumerate(parent) if p == j] for j in range(len(parent))]
    out, trunk = [], []
    for leaf in (i for i in range(len(parent)) if not kids[i]):
        c = [leaf]
        while parent[c[-1]] >= 0 and len(kids[parent[c[-1]]]) == 1:
            c.append(parent[c[-1]])
        if parent[c[-1]] < 0:  # an unbranched rig: the whole thing is trunk
            trunk += c
            continue
        out.append(c[::-1])
    branched = {i for c in out for i in c}
    trunk += [i for i in range(len(parent)) if i not in branched and i not in trunk]
    return out, sorted(trunk)


def runs(mask, min_len):
    """(start, end) of every run of True at least min_len long, end exclusive."""
    out, t, T = [], 0, len(mask)
    while t < T:
        if mask[t]:
            s = t
            while t < T and mask[t]:
                t += 1
            if t - s >= min_len:
                out.append((s, t))
        else:
            t += 1
    return out


def hold(track, s, e, at):
    """Hold track[s:e] at track[at]; if the track goes on after e, ease from the hold to track[e] over the last frames."""
    track[s:e] = track[at]
    if e < len(track):
        n = min(BLEND, e - s)
        w = (np.arange(1, n + 1) / (n + 1)).reshape((-1,) + (1,) * (track.ndim - 1))
        track[e - n:e] = track[at] * (1 - w) + track[e] * w


def doubt_smooth(x, sig, conf):
    """Gaussian smoothing along time, frame t with width sig[t], each neighbour weighted by how well it was read: an
    unsure frame is filled in from the sure ones around it, and a sure frame (width ~0) keeps its value."""
    t = np.arange(len(x))
    w = np.exp(-0.5 * ((t[None] - t[:, None]) / np.maximum(sig, 0.05)[:, None]) ** 2) * (conf[None] + 1e-3)
    w /= w.sum(1, keepdims=True)
    return np.tensordot(w, x, axes=(1, 0))


def turn_speed(R, fps):
    """Per-frame turning speed (rad/s) of a (T,3,3) rotation track, lightly smoothed."""
    rel = np.einsum("tji,tjk->tik", R[:-1], R[1:])
    ang = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)) * fps
    v = np.r_[ang[:1], (ang[1:] + ang[:-1]) / 2, ang[-1:]]
    return np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, mode="valid")


def clean(d, info, m, occlusion=True, settle=True, smooth=True, still=0.045, occluded=0.6):
    rig = Rig(info)
    fps = m["fps"]
    E = np.unwrap(np.array(m["euler"]), axis=0)
    trans, q = np.array(m["trans"]), np.array(m["root_quat"])
    for t in range(1, len(q)):  # one hemisphere, so averaging quaternions is meaningful
        if (q[t] * q[t - 1]).sum() < 0:
            q[t] = -q[t]
    T = len(E)
    J = np.array(m["joints"])
    size = np.ptp(rig.rest_kp.numpy(), 0).max()
    cs, trunk = chains(info)
    kp_bone = np.array([b for b, _ in rig.kp])
    report = {"occluded": {}, "settled": {}, "trunk_held": []}

    if occlusion:
        fc, sc = d["front_conf"].clip(0, 1), d["side_conf"].clip(0, 1)
        for c in cs:
            kps = np.isin(kp_bone, c[len(c) // 2:])  # the far half (elbow to tip) is what gets hidden
            ratio = np.min([(x := conf[:, kps].mean(1)) / max(np.percentile(x, 90), 1e-6) for conf in (fc, sc)], 0)
            low, on = np.zeros(T, bool), False
            for t in range(T):  # hysteresis: hidden below `occluded`, seen again only well above it
                on = ratio[t] < (0.8 if on else occluded)
                low[t] = on
            for s, e in runs(low, MIN_OCCLUDED):
                if s == 0 and e == T:
                    continue  # never read with confidence: nothing better to hold
                seen = np.nonzero(ratio[:s] >= 0.8)[0]  # the last frame read clearly, before the drop began
                at = seen[-1] if len(seen) else e
                sub = E[:, c].copy()  # fancy indexing copies, so edit a copy and put it back
                hold(sub, min(at, s), e, at)
                E[:, c] = sub
                report["occluded"].setdefault(f"{rig.names[c[0]]}..{rig.names[c[-1]]}", []).append([s, e])

    held = np.zeros((len(cs), T), bool)
    if settle:
        thr = still * size / fps  # metres per frame
        for ci, c in enumerate(cs):
            kps = [i for i, (b, e) in enumerate(rig.kp) if b in c[:-1]]
            v = np.linalg.norm(np.diff(J[:, kps], axis=0), axis=-1).max(1)
            v = np.r_[v[:1], np.maximum(v[1:], v[:-1]), v[-1:]]  # a frame is still when both its steps are
            for s, e in runs(v < thr, MIN_SETTLE):
                sub = E[:, c].copy()
                hold(sub, s, e, s)
                E[:, c] = sub
                held[ci, s:e] = True
                report["settled"].setdefault(f"{rig.names[c[0]]}..{rig.names[c[-1]]}", []).append([s, e])
        for s, e in runs(held.all(0), MIN_SETTLE):
            sub = E[:, trunk].copy()
            hold(sub, s, e, s)
            E[:, trunk] = sub
            hold(trans, s, e, s)
            hold(q, s, e, s)
            report["trunk_held"].append([s, e])

    if smooth:
        # how well each keypoint was read, as a share of its own usual level, in its weaker view
        rel = np.min([c.clip(0, 1) / np.maximum(np.percentile(c.clip(0, 1), 90, axis=0), 1e-6)
                      for c in (d["front_conf"], d["side_conf"])], 0).clip(0, 1)
        R = euler_xyz(torch.tensor(E)).numpy()
        Rq = np.array([quat_matrix(x) for x in q / np.linalg.norm(q, axis=1, keepdims=True)])
        for b in range(E.shape[1]):
            # a bone's turn is read from the keypoints on it and on its children
            kps = [i for i, (kb, _) in enumerate(rig.kp) if kb == b or rig.parent[kb] == b]
            if not kps:
                continue
            c = rel[:, kps].mean(1)
            sp = turn_speed(R[:, b] if b else Rq, fps)
            sig = SIGMA_MAX * np.clip((CLEAR - c) / (CLEAR - UNSURE), 0, 1) * \
                (1 - 2 / 3 * np.clip((sp - SPEED_SLOW) / (SPEED_FAST - SPEED_SLOW), 0, 1))
            if b:
                E[:, b] = doubt_smooth(E[:, b], sig, c)
            else:  # the root bone: its turn and travel
                q, trans = doubt_smooth(q, sig, c), doubt_smooth(trans, sig, c)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    E = np.clip(E, rig.lo.numpy(), rig.hi.numpy())

    with torch.no_grad():
        Jn, _ = rig.forward(torch.tensor(trans), torch.tensor(np.array([quat_matrix(x) for x in q])), torch.tensor(E))
    Jn = Jn.numpy()
    out = dict(m, euler=E.round(5).tolist(), trans=trans.round(5).tolist(), root_quat=q.round(6).tolist(),
               joints=Jn.round(4).tolist())
    pf, ps = project(torch.tensor(Jn))
    out["reproj_px"] = {"front": float(np.linalg.norm(pf.numpy() - d["front_uv"], axis=-1).mean()),
                        "front_x_only": float(np.abs(pf.numpy()[..., 0] - d["front_uv"][..., 0]).mean()),
                        "side": float(np.linalg.norm(ps.numpy() - d["side_uv"], axis=-1).mean())}
    out["reproj_px_raw"] = m["reproj_px"]
    out["clean"] = report
    return out


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    det_path, info_path, raw_path, out_path = sys.argv[1:5]
    a = sys.argv[5:]
    opt = lambda k, dflt: float(a[a.index(k) + 1]) if k in a else dflt
    m = json.load(open(raw_path))
    if "--off" in a:
        json.dump(m, open(out_path, "w"))
        print("clean: off, saved", out_path)
        return
    raw = np.load(det_path)
    d = {k: raw[k].copy() for k in ("front_uv", "side_uv", "front_conf", "side_conf")}
    shift = (m.get("sync") or {}).get("front_shift_frames")
    if shift is not None:  # the solver re-timed the front view onto the side view: follow it
        T, t = len(shift), np.arange(len(shift))
        for k in ("front_uv", "front_conf"):
            x = d[k].reshape(T, -1)
            d[k] = np.stack([np.interp(t + np.array(shift), t, x[:, c]) for c in range(x.shape[1])], 1).reshape(d[k].shape)
    out = clean(d, json.load(open(info_path)), m, occlusion="--no-occlusion" not in a, settle="--no-settle" not in a,
                smooth="--no-smooth" not in a, still=opt("--still", 0.045), occluded=opt("--occluded", 0.6))
    r = out["clean"]
    print("clean: occluded (held)", r["occluded"] or "none", "| settled", r["settled"] or "none",
          "| trunk held", r["trunk_held"] or "none")
    print("reprojection error px after clean:", {k: round(v, 2) for k, v in out["reproj_px"].items()})
    json.dump(out, open(out_path, "w"))
    print("saved", out_path)


if __name__ == "__main__":
    main()
