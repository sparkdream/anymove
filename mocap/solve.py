"""Fit the rig to the joints read from both views: one pose per frame, smooth over time, inside the joint limits.

python -m mocap.solve DETECTIONS.npz rig_info.json OUT.json [--no-align] [--fix-swaps] [--acc WEIGHT] [--vel WEIGHT]
    [--acc-arms WEIGHT] [--vel-arms WEIGHT] [--trust-front]
Before fitting, the side view's left and right legs are checked against the front view: a joint has the same height
in both halves, so if swapping a pair of legs in the side view makes the heights agree better, the side reader
mixed up the near and far leg and they are swapped back. The same height test gives the sync score: how far apart
the two halves disagree, and the time lag between them that agrees best.

The fit runs Blender's forward kinematics in torch (pose = parent pose x rest offset x Euler XYZ), with the pelvis
free to turn and move, every other bone limited to its range (and to its twist_deg about its own axis), and minimises
the confidence-weighted distance between the rig's projected joints and the readings in both views, plus smoothness
and a small pull toward the rest pose.

--trust-front: where the two halves show a limb at clearly different heights (the video model played a different
move in each), the side view's reading of that limb is dropped and the front view alone places it, heights included.
Its depth then comes only from the bone lengths, the joint limits and smoothness: right from the front, a guess from
the side.
"""
import json, sys

import numpy as np
import torch

from mocap import layout

# a humanoid's arm bones (rig.json names, before _L/_R): --acc-arms and --vel-arms smooth them apart from the body
ARMS = ("clavicle", "upperarm", "forearm", "hand", "blade")
LEGS = {"F": ["scap", "shoulder", "elbow", "wrist", "fpaw", "ftoe"], "H": ["hip", "knee", "hock", "hpaw", "htoe"]}


def leg_pairs(names):
    """Index pairs (left, right) of every leg joint, grouped per front/hind pair."""
    out = {}
    for k, stems in LEGS.items():
        pairs = []
        for s in stems:
            if f"{s}_L" in names:
                pairs.append((names.index(f"{s}_L"), names.index(f"{s}_R")))
        out[k] = pairs
    return out


def fix_swaps(d, names, margin=4.0):
    """Swap near/far legs in the side view where the heights say the reader mixed them up."""
    fv, sv = d["front_uv"][..., 1], d["side_uv"][..., 1]
    c = np.minimum(d["front_conf"], d["side_conf"]).clip(0, 1)
    swapped = {}
    for k, pairs in leg_pairs(names).items():
        L, R = np.array(pairs).T
        same = (c[:, L] * np.abs(sv[:, L] - fv[:, L]) + c[:, R] * np.abs(sv[:, R] - fv[:, R])).sum(1)
        swap = (c[:, L] * np.abs(sv[:, R] - fv[:, L]) + c[:, R] * np.abs(sv[:, L] - fv[:, R])).sum(1)
        gain = (same - swap) / len(pairs)
        gain = np.array([np.median(gain[max(0, t - 2):t + 3]) for t in range(len(gain))])
        do = gain > margin
        for t in np.where(do)[0]:
            d["side_uv"][t, L], d["side_uv"][t, R] = d["side_uv"][t, R].copy(), d["side_uv"][t, L].copy()
            d["side_conf"][t, L], d["side_conf"][t, R] = d["side_conf"][t, R].copy(), d["side_conf"][t, L].copy()
        swapped[k] = int(do.sum())
    return swapped


def limb_bones(info):
    """Bone indices of each limb: the bones from a branch point out to a tip (an arm, a leg, the head)."""
    parent = [b["parent"] for b in info["bones"]]
    kids = [[i for i, p in enumerate(parent) if p == j] for j in range(len(parent))]
    out = []
    for leaf in (i for i in range(len(parent)) if not kids[i]):
        c = [leaf]
        while parent[c[-1]] >= 0 and len(kids[parent[c[-1]]]) == 1:
            c.append(parent[c[-1]])
        if parent[c[-1]] >= 0:
            out.append(c)
    return out


def limb_chains(info):
    """Keypoint indices of each limb (limb_bones)."""
    kp_bone = [k["bone"] for k in info["keypoints"]]
    out = [[k for k, b in enumerate(kp_bone) if b in c] for c in limb_bones(info)]
    return [c for c in out if c]


def trust_front(d, info, gap_m=0.12, ok=0.7, pad=2, bridge=5, min_len=3):
    """Drop the side view's reading of a limb wherever the two views put it at clearly different heights; the front
    view then places it alone, so its heights count fully. The side reading is replaced (height from the front,
    across-the-image position carried over from the frames around) only so the starting pose is sane.
    Returns {keypoint name: [[first, end), ...]} frames dropped, and a (T, K) mask."""
    T, K = d["front_conf"].shape
    gap = np.abs(d["front_uv"][..., 1] - d["side_uv"][..., 1]) * layout.M_PER_PX
    both = (d["front_conf"] > ok) & (d["side_conf"] > ok)
    mask = np.zeros((T, K), bool)
    for kps in limb_chains(info):
        bad = np.nonzero(((gap[:, kps] > gap_m) & both[:, kps]).any(1))[0]
        spans = []
        for t in bad:
            if spans and t - spans[-1][1] <= bridge:
                spans[-1][1] = t + 1
            else:
                spans.append([t, t + 1])
        for a, b in spans:
            if b - a >= min_len:
                mask[max(0, a - pad):min(T, b + pad), kps] = True
    frames = np.arange(T)
    for k in range(K):
        m = mask[:, k]
        if not m.any():
            continue
        d["side_conf"][m, k] = 0
        d["side_uv"][m, k, 1] = d["front_uv"][m, k, 1]
        if (~m).any():
            d["side_uv"][m, k, 0] = np.interp(frames[m], frames[~m], d["side_uv"][~m, k, 0])
    names = [k["name"] for k in info["keypoints"]]
    report = {}
    for k in range(K):
        idx = np.nonzero(mask[:, k])[0]
        if len(idx):
            cuts = np.nonzero(np.diff(idx) > 1)[0]
            report[names[k]] = [[int(idx[a]), int(idx[b]) + 1] for a, b in zip(np.r_[0, cuts + 1], np.r_[cuts, len(idx) - 1])]
    return report, mask


def height_features(uv, conf, names):
    """Per-frame heights that do not care about left/right labels: middle joints as they are, each leg pair as its
    mean height and its height difference. Returns (T, K) features and (T, K) weights."""
    v = uv[..., 1]
    feats, w = [], []
    for i, n in enumerate(names):
        if not n.endswith(("_L", "_R")):
            feats.append(v[:, i]); w.append(conf[:, i])
        elif n.endswith("_L"):
            a, b = i, names.index(n[:-2] + "_R")
            c = np.minimum(conf[:, a], conf[:, b])
            feats += [(v[:, a] + v[:, b]) / 2, np.abs(v[:, a] - v[:, b])]
            w += [c, c]
    return np.stack(feats, 1), np.stack(w, 1).clip(0, 1)


def align_views(d, names, band=48, step_penalty=1.0):
    """Video models often play the two halves out of step (one view runs ahead). Match the halves frame to frame
    with dynamic time warping on joint heights, which both views share, and resample the front view onto the side
    view's timeline. Returns how far each side frame's match was moved, in frames."""
    F, Fw = height_features(d["front_uv"], d["front_conf"], names)
    S, Sw = height_features(d["side_uv"], d["side_conf"], names)
    T = len(F)
    C = np.full((T, T), np.inf)
    for i in range(T):
        lo, hi = max(0, i - band), min(T, i + band + 1)
        w = np.minimum(Fw[i][None], Sw[lo:hi])
        C[i, lo:hi] = (w * np.abs(F[i][None] - S[lo:hi])).sum(1) / w.sum(1).clip(1e-6)
    D = np.full((T, T), np.inf)
    D[0, 0] = C[0, 0]
    for i in range(T):
        for j in range(max(0, i - band), min(T, i + band + 1)):
            if i == j == 0:
                continue
            best = min(D[i - 1, j - 1] if i and j else np.inf, D[i - 1, j] + step_penalty if i else np.inf,
                       D[i, j - 1] + step_penalty if j else np.inf)
            D[i, j] = C[i, j] + best
    i, j, path = T - 1, T - 1, [(T - 1, T - 1)]
    while i or j:
        cands = [(D[i - 1, j - 1], i - 1, j - 1) if i and j else (np.inf, 0, 0),
                 (D[i - 1, j], i - 1, j) if i else (np.inf, 0, 0), (D[i, j - 1], i, j - 1) if j else (np.inf, 0, 0)]
        _, i, j = min(cands)
        path.append((i, j))
    path = np.array(path[::-1])
    match = np.array([path[path[:, 1] == j, 0].mean() for j in range(T)])  # front frame for each side frame
    match = np.convolve(np.pad(match, 3, mode="edge"), np.ones(7) / 7, mode="valid")  # no stair steps
    t = np.arange(T)
    for k in ("front_uv", "front_conf"):
        a = d[k].reshape(T, -1)
        d[k] = np.stack([np.interp(match, t, a[:, c]) for c in range(a.shape[1])], 1).reshape(d[k].shape)
    return match - t


def feet():
    """The joints whose heights make the sync score, and that the fit keeps above the ground: creature.json "feet" when
    MOCAP_CREATURE is set, else the wolf's."""
    import os, pathlib
    cfg = pathlib.Path(os.environ.get("MOCAP_CREATURE", "")) / "creature.json"
    if os.environ.get("MOCAP_CREATURE") and cfg.exists():
        return json.load(open(cfg)).get("feet", [])
    return ["fpaw_L", "fpaw_R", "hpaw_L", "hpaw_R"]


def paw_agreement(d, names):
    """Correlation of each paw's height between the two views over the clip: +1 when both halves show the same
    footfalls, near 0 or negative when the video model animated the views differently."""
    out = {}
    for k in feet():
        if k in names:
            j = names.index(k)
            a, b = d["front_uv"][:, j, 1], d["side_uv"][:, j, 1]
            out[k] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0.5 and b.std() > 0.5 else None
    vals = [v for v in out.values() if v is not None]
    out["score"] = float(np.mean(vals)) if vals else None
    return out


def sync_report(d):
    """Mean height disagreement between the halves (layout px) and the lag (frames) that minimises it."""
    fv, sv = d["front_uv"][..., 1], d["side_uv"][..., 1]
    c = np.minimum(d["front_conf"], d["side_conf"]).clip(0, 1)
    err = (c * np.abs(fv - sv)).sum(1) / c.sum(1).clip(1e-6)
    best = None
    T = len(fv)
    for lag in range(-12, 13):
        a = fv[max(0, lag):T + min(0, lag)]
        b = sv[max(0, -lag):T + min(0, -lag)]
        e = np.abs(a - b).mean()
        if best is None or e < best[1]:
            best = (lag, e)
    return {"mean_height_gap_px": float(err.mean()), "worst_frame_gap_px": float(err.max()),
            "per_frame_gap_px": err.round(2).tolist(), "best_lag_frames": best[0], "gap_at_best_lag_px": float(best[1])}


def euler_xyz(a):
    """Blender Euler XYZ (applied X, then Y, then Z) to rotation matrices: Rz @ Ry @ Rx."""
    x, y, z = a.unbind(-1)
    cx, sx, cy, sy, cz, sz = x.cos(), x.sin(), y.cos(), y.sin(), z.cos(), z.sin()
    o, l = torch.zeros_like(x), torch.ones_like(x)
    Rx = torch.stack([l, o, o, o, cx, -sx, o, sx, cx], -1).view(*x.shape, 3, 3)
    Ry = torch.stack([cy, o, sy, o, l, o, -sy, o, cy], -1).view(*x.shape, 3, 3)
    Rz = torch.stack([cz, -sz, o, sz, cz, o, o, o, l], -1).view(*x.shape, 3, 3)
    return Rz @ Ry @ Rx


def twist(R):
    """How far each rotation turns about its bone's own axis (local Y), from its swing-twist split. Euler Y is not
    this: with the bone swung far, Euler angles trade twist for swing, so their limits don't bound it."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return 2 * torch.atan2(R[..., 0, 2] - R[..., 2, 0], 1 + tr + 1e-9)


def axis_angle(v):
    th = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = v / th
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return torch.eye(3, dtype=v.dtype, device=v.device) + th.sin() * K + (1 - th.cos()) * K @ K


class Rig:
    def __init__(self, info, dev="cpu", roll_weight=1.0):
        self.names = [b["name"] for b in info["bones"]]
        self.parent = [b["parent"] for b in info["bones"]]
        B = torch.tensor([b["matrix_local"] for b in info["bones"]], dtype=torch.float64, device=dev)
        self.B = B
        self.L = torch.tensor([b["length"] for b in info["bones"]], dtype=torch.float64, device=dev)
        # rest offset of each bone in its parent's frame
        self.off = torch.stack([B[i] if p < 0 else torch.linalg.inv(B[p]) @ B[i] for i, p in enumerate(self.parent)])
        lim = np.radians(np.array([b["limits_deg"] for b in info["bones"]], float))
        self.lo = torch.tensor(lim[..., 0], device=dev)
        self.hi = torch.tensor(lim[..., 1], device=dev)
        # twist_deg: the skin pinches where a bone turns far about its own axis (a ribbon past ~90 degrees)
        self.twist = torch.tensor(np.radians([b.get("twist_deg", 360) for b in info["bones"]]), device=dev)
        # and nothing in a video shows that turn once the bone's joint is straight (no bend plane to read), so it is
        # pulled gently toward the rest roll; otherwise it wanders within twist_deg and the skin visibly wrings
        self.rolls = (self.twist < np.pi).to(self.twist.dtype) * roll_weight
        self.kp = [(k["bone"], k["end"]) for k in info["keypoints"]]
        self.kp_names = [k["name"] for k in info["keypoints"]]
        self.rest_kp = torch.tensor([k["rest"] for k in info["keypoints"]], dtype=torch.float64, device=dev)

    def forward(self, trans, root_rot, eul):
        """trans (T,3) world offset, root_rot (T,3,3) pelvis basis, eul (T,nb,3) Euler for every bone (root ignored)."""
        T, nb = eul.shape[:2]
        R = euler_xyz(eul)
        P = [None] * nb
        for i, p in enumerate(self.parent):
            basis = torch.zeros(T, 4, 4, dtype=eul.dtype, device=eul.device)
            basis[:, :3, :3] = root_rot if p < 0 else R[:, i]
            basis[:, 3, 3] = 1
            if p < 0:
                M = self.off[i].expand(T, 4, 4) @ basis
                M = M.clone()
                M[:, :3, 3] += trans
            else:
                M = P[p] @ self.off[i] @ basis
            P[i] = M
        P = torch.stack(P, 1)
        joints = []
        for b, end in self.kp:
            if end == "head":
                joints.append(P[:, b, :3, 3])
            else:
                joints.append(P[:, b, :3, 1] * self.L[b] + P[:, b, :3, 3])
        return torch.stack(joints, 1), P


def project(J):
    v = layout.H / 2 - (J[..., 2] - layout.Z_CENTER) / layout.M_PER_PX
    f = torch.stack([layout.W_FRONT / 2 + J[..., 0] / layout.M_PER_PX, v], -1)
    s = torch.stack([layout.W_SIDE / 2 - J[..., 1] / layout.M_PER_PX, v], -1)
    return f, s


def triangulate(d, front_height_weight=0.25):
    """Joints in metres: x from the front view, y from the side view, height mostly from the side view."""
    fx = (d["front_uv"][..., 0] - layout.W_FRONT / 2) * layout.M_PER_PX
    sy = -(d["side_uv"][..., 0] - layout.W_SIDE / 2) * layout.M_PER_PX
    fz = layout.Z_CENTER + (layout.H / 2 - d["front_uv"][..., 1]) * layout.M_PER_PX
    sz = layout.Z_CENTER + (layout.H / 2 - d["side_uv"][..., 1]) * layout.M_PER_PX
    w1, w2 = front_height_weight * d["front_conf"].clip(1e-3), d["side_conf"].clip(1e-3)  # weight: scalar or (T, K)
    return np.stack([fx, sy, (fz * w1 + sz * w2) / (w1 + w2)], -1)


def root_from_triangulation(tri, conf, rig, names, info):
    """Per-frame root rotation (as the basis of the root bone, axis-angle) from the points that ride rigidly on it
    (its two ends and the heads of the bones hung straight off it), by weighted Kabsch against the rest pose;
    smoothed over time."""
    bones = info["bones"]
    ride = {bones[0]["head_joint"], bones[0]["tail_joint"]} | {b["head_joint"] for b in bones if b["parent"] == 0}
    body = sorted(names.index(n) for n in ride if n in names)
    h = names.index(bones[0]["head_joint"])
    rest = rig.rest_kp.cpu().numpy()
    a = rest[body] - rest[h]
    Brot = rig.B[0, :3, :3].cpu().numpy()
    out = []
    for t in range(len(tri)):
        b = tri[t, body] - tri[t, h]
        w = conf[t, body]
        U, _, Vt = np.linalg.svd((a * w[:, None]).T @ b)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        W = Vt.T @ D @ U.T
        R = Brot.T @ W @ Brot
        ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        out.append(axis / max(np.linalg.norm(axis), 1e-8) * ang if ang > 1e-6 else np.zeros(3))
    out = np.array(out)
    k = np.ones(5) / 5
    return np.stack([np.convolve(np.pad(out[:, i], 2, mode="edge"), k, mode="valid") for i in range(3)], 1)


def euler_from_matrix(R):
    """Inverse of euler_xyz for one 3x3 matrix (Blender XYZ: R = Rz Ry Rx)."""
    b = -np.arcsin(np.clip(R[2, 0], -1, 1))
    return np.array([np.arctan2(R[2, 1], R[2, 2]), b, np.arctan2(R[1, 0], R[0, 0])])


def rot_between(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def ik_init(tri, root_aa, info, rig):
    """Point every bone at its triangulated child joint, down the hierarchy, with no twist and inside its limits:
    a starting pose close to the answer, so the fit does not have to find a sit or a rear from a standing pose."""
    names = rig.kp_names
    bones = info["bones"]
    off = rig.off.cpu().numpy()
    Brot = rig.B[0, :3, :3].cpu().numpy()
    lo, hi = rig.lo.cpu().numpy(), rig.hi.cpu().numpy()
    T, nb = len(tri), len(bones)
    eul = np.zeros((T, nb, 3))
    y = np.array([0.0, 1.0, 0.0])
    for t in range(T):
        aa = root_aa[t]
        ang = np.linalg.norm(aa)
        k = aa / ang if ang > 1e-9 else np.array([1.0, 0, 0])
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        W = [None] * nb
        W[0] = Brot @ (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K)
        for i in range(1, nb):
            p = bones[i]["parent"]
            base = W[p] @ off[i][:3, :3]
            if bones[i]["head_joint"] not in names or bones[i]["tail_joint"] not in names:
                W[i] = base  # e.g. fur bones, whose root is not a tracked joint: start at rest
                continue
            d = tri[t, names.index(bones[i]["tail_joint"])] - tri[t, names.index(bones[i]["head_joint"])]
            if np.linalg.norm(d) < 1e-6:
                W[i] = base
                continue
            R = rot_between(y, base.T @ (d / np.linalg.norm(d)))
            e = np.clip(euler_from_matrix(R), lo[i], hi[i])
            eul[t, i] = e
            W[i] = base @ euler_xyz(torch.tensor(e)).numpy()
    return eul


def refit_limb(rig, t, bones, kps, trans, root, phi, f_obs, f_c, s_obs, s_c, f_axis, err, starts=24, iters=300, better=0.7):
    """One limb in frame t fitted alone to that frame's readings, from the neighbouring frames' poses and random ones,
    the rest of the body held. Returns the limb's new phi when it fits better than err (px) by the factor `better`."""
    T = len(phi)
    with torch.no_grad():
        p0 = phi[t].detach().clone()
        seeds = [phi[u, bones].detach() for u in (t - 1, t + 1) if 0 <= u < T]
        g = torch.Generator(device="cpu").manual_seed(t)
        seeds += [torch.randn(len(bones), 3, generator=g, dtype=phi.dtype).to(phi.device) * 1.5
                  for _ in range(starts - len(seeds))]
    z = torch.stack(seeds).clone().requires_grad_(True)
    n = len(z)
    tr, R = trans[t].detach().expand(n, 3), axis_angle(root[t].detach()).expand(n, 3, 3)
    fa = f_axis[t, kps] if f_axis.dim() == 3 else f_axis
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(iters):
        p = p0.expand(n, *p0.shape).clone()
        p[:, bones] = z
        eul = rig.lo + (rig.hi - rig.lo) * torch.sigmoid(p)
        J, _ = rig.forward(tr, R, eul)
        pf, ps = project(J)
        rf = ((pf[:, kps] - f_obs[t, kps]) ** 2 * fa).sum(-1)
        rs = ((ps[:, kps] - s_obs[t, kps]) ** 2).sum(-1)
        data = (f_c[t, kps] * torch.log1p(rf / 36) + s_c[t, kps] * torch.log1p(rs / 36)).sum(-1)  # the fit's own, at 6 px
        tw = twist(euler_xyz(eul[:, bones]))
        turn = (torch.relu(tw.abs() - rig.twist[bones]) ** 2).sum(1) * 1000
        roll = (tw ** 2 * rig.rolls[bones]).sum(1)
        opt.zero_grad()
        (data + turn + roll).sum().backward()
        opt.step()
    with torch.no_grad():
        ok = f_c[t, kps] + s_c[t, kps] > 0.1
        e = ((f_c[t, kps] * rf.sqrt() + s_c[t, kps] * rs.sqrt()) / (f_c[t, kps] + s_c[t, kps]).clamp(min=1e-6))[:, ok]
        e = e.median(1).values
        i = int((data + turn + roll).argmin())
    return z[i].detach() if float(e[i]) < better * err and float(turn[i]) < 1 else None


def fit(d, info, iters=1500, dev="cpu", log=True, front_height_weight=0.25, acc_weight=0.02, vel_weight=2.0,
        acc_arms=None, vel_arms=None, reseed_worse=3.0, reseed_floor=20.0, roll_weight=1.0, twist_vel=20.0):
    """acc_weight: the joint-acceleration penalty. Lower it for fast moves with long levers (a blade tip in a slash
    moves faster than anything else and gets held back first): 0.002 follows them far better.
    vel_weight: the penalty on every joint turning from one frame to the next. It holds back a whole-arm swing that
    crosses the body in a few frames even with a low acc_weight: 0.2 lets it through.
    acc_arms, vel_arms: the same for the arms alone (ARMS), so a slash can be followed loosely while the body stays
    smooth: video models swing an arm across the body between two frames, and a smoothness low enough to follow
    that everywhere passes the reader's frame-to-frame noise into the hips and head as a bounce.
    reseed_worse, reseed_floor: a limb fitting a frame reseed_worse times worse than its median over the clip, and
    worse than reseed_floor px, is fitted again in that frame from new starts (refit_limb).
    roll_weight, twist_vel: the pull of twist-limited bones (twist_deg) toward their rest roll, and the penalty on that
    roll changing from one frame to the next. A loose vel_arms lets a slash through but would let the arm wring too."""
    rig = Rig(info, dev, roll_weight)
    names = rig.kp_names
    T = len(d["front_uv"])
    f_obs = torch.tensor(d["front_uv"], dtype=torch.float64, device=dev)
    s_obs = torch.tensor(d["side_uv"], dtype=torch.float64, device=dev)
    f_c = torch.tensor(d["front_conf"], dtype=torch.float64, device=dev).clamp(0, 1) ** 2
    s_c = torch.tensor(d["side_conf"], dtype=torch.float64, device=dev).clamp(0, 1) ** 2
    tri_np = triangulate(d, front_height_weight)
    tri = torch.tensor(tri_np, dtype=torch.float64, device=dev)
    hips = names.index(info["bones"][0]["head_joint"])
    trans = (tri[:, hips] - rig.rest_kp[hips]).clone().requires_grad_(True)
    conf = np.minimum(d["front_conf"], d["side_conf"]).clip(1e-3, 1)
    root0 = root_from_triangulation(tri_np, conf, rig, names, info) + 1e-4
    root = torch.tensor(root0, dtype=torch.float64, device=dev).requires_grad_(True)
    nb = len(rig.names)
    # Euler = lo + (hi - lo) * sigmoid(phi), starting from the analytic IK pose
    e0 = torch.tensor(ik_init(tri_np, root0, info, rig), dtype=torch.float64, device=dev)
    z = ((e0 - rig.lo) / (rig.hi - rig.lo)).clamp(1e-3, 1 - 1e-3)
    phi = torch.logit(z).clone().requires_grad_(True)
    foot = set(feet())
    paws = [i for i, n in enumerate(names) if n in foot or any(n.startswith(p) for p in ("fpaw", "ftoe", "hpaw", "htoe"))]
    arm_b = [i for i, n in enumerate(rig.names) if n.rsplit("_", 1)[0] in ARMS]
    # the elbow out: the shoulder joint sits on the torso, and following it loosely would shake the body
    arm_k = [i for i, (b, e) in enumerate(rig.kp) if b in arm_b and not (e == "head" and rig.names[b].rsplit("_", 1)[0] in ARMS[:2])]
    acc_w = torch.full((len(names),), acc_weight, dtype=torch.float64, device=dev)
    acc_w[arm_k] = acc_weight if acc_arms is None else acc_arms
    vel_w = torch.full((nb,), vel_weight, dtype=torch.float64, device=dev)
    vel_w[arm_b] = vel_weight if vel_arms is None else vel_arms
    vel_w = vel_w.expand(max(T - 1, 0), nb).clone()  # per step between frames, so a step can be let go
    sigma = 6.0  # px, robust scale (annealed down from 40 px so far-off joints still pull early on)
    # The side view sees legs swing and the hips clearly; the front view foreshortens leg lifts and hides the hips
    # when the creature sits, and video models animate it less faithfully. So heights come mostly from the side
    # view and the front view mainly supplies left-right position.
    fhw = torch.as_tensor(front_height_weight, dtype=torch.float64, device=dev)
    f_axis = torch.stack(torch.broadcast_tensors(torch.ones_like(fhw), fhw), -1)  # (2,) or (T, K, 2)

    def loss_fn(stage):
        nonlocal sigma
        eul = rig.lo + (rig.hi - rig.lo) * torch.sigmoid(phi)
        J, _ = rig.forward(trans, axis_angle(root), eul)
        pf, ps = project(J)
        rf = ((pf - f_obs) ** 2 * f_axis).sum(-1)
        rs = ((ps - s_obs) ** 2).sum(-1)
        data = (f_c * torch.log1p(rf / sigma ** 2)).sum() + (s_c * torch.log1p(rs / sigma ** 2)).sum()
        acc = (((J[2:] - 2 * J[1:-1] + J[:-2]) ** 2).sum((0, 2)) * acc_w).sum() / layout.M_PER_PX ** 2
        vel = (((eul[1:] - eul[:-1]) ** 2).sum(2) * vel_w).sum() + ((root[1:] - root[:-1]) ** 2).sum() * vel_weight
        prior = (eul ** 2).sum() * 0.05
        ground = (torch.relu(-J[:, paws, 2]) ** 2).sum() / layout.M_PER_PX ** 2 * 0.5
        tw = twist(euler_xyz(eul))
        turn = (torch.relu(tw.abs() - rig.twist) ** 2).sum() * 1000 + (tw ** 2 * rig.rolls).sum() \
            + (((tw[1:] - tw[:-1]) ** 2) * (vel_w > 0) * rig.rolls).sum() * twist_vel
        total = data + acc + vel + prior + ground + turn
        return total, {"data": data.item(), "acc": acc.item(), "vel": vel.item(), "prior": prior.item(), "ground": ground.item(),
                       "twist": turn.item()}, J, pf, ps, eul

    opt = torch.optim.Adam([{"params": [trans], "lr": 0.01}, {"params": [root], "lr": 0.03}, {"params": [phi], "lr": 0.05}])
    for it in range(iters):
        sigma = 6.0 + 34.0 * max(0.0, 1 - it / (0.6 * iters))
        if it == int(iters * 0.7):
            for g in opt.param_groups:
                g["lr"] *= 0.3
        total, parts, J, pf, ps, eul = loss_fn(it)
        opt.zero_grad()
        total.backward()
        opt.step()
        if log and it % 250 == 0:
            print(f"  iter {it} loss {total.item():.1f}", {k: round(v, 1) for k, v in parts.items()}, flush=True)
    sigma = 6.0
    # A limb can settle on the wrong one of two poses (an arm swung through pointing at the camera, fitted folded) and
    # smoothness keeps it there. Where a limb fits a frame far worse than it does the rest of the clip, it is fitted
    # again in that frame alone from several starts (the frames either side, random), and the whole clip refined
    # around it with the new pose held (left free, smoothness would pull it back to the old one) and the angle
    # smoothness to the frames either side let go (the same arm can sit at far-apart joint angles a frame apart).
    with torch.no_grad():
        _, _, J, pf, ps, _ = loss_fn(iters)
        err = (f_c * (pf - f_obs).norm(dim=-1) + s_c * (ps - s_obs).norm(dim=-1)) / (f_c + s_c).clamp(min=1e-6)
    redone = []
    hold = torch.zeros(phi.shape, dtype=torch.bool, device=dev)
    again = []
    for bones in limb_bones(info):
        kps = [k for k, (b, _) in enumerate(rig.kp) if b in bones]
        if not kps:
            continue
        # the median over the limb's joints: one bad reading (a blade tip lost in the blur) doesn't make a bad fit
        e = torch.where(f_c[:, kps] + s_c[:, kps] > 0.1, err[:, kps], torch.nan).nanmedian(1).values.nan_to_num(0)
        for t in torch.nonzero(e > max(reseed_worse * float(e.median()), reseed_floor)).flatten().tolist():
            got = refit_limb(rig, t, bones, kps, trans, root, phi, f_obs, f_c, s_obs, s_c, f_axis, float(e[t]))
            if got is not None:
                with torch.no_grad():
                    phi[t, bones] = got
                hold[t, bones] = True
                vel_w[max(t - 1, 0):t + 1, bones] = 0  # its joint angles may jump: the pose is a different branch
                again.append((bones, kps, t))
                redone.append((f"{rig.names[bones[-1]]}..{rig.names[bones[0]]}", t))
    if redone:
        if log:
            print("  limbs fitted again from new starts (limb, frame):", redone, flush=True)
        for g in opt.param_groups:
            g["lr"] *= 0.5
        held = phi.detach()[hold].clone()
        for it in range(iters // 4):
            total, parts, J, pf, ps, eul = loss_fn(it)
            opt.zero_grad()
            total.backward()
            opt.step()
            with torch.no_grad():
                phi[hold] = held
        # the body has moved under the held limbs: fit them once more to it
        with torch.no_grad():
            _, _, J, pf, ps, _ = loss_fn(iters)
            err = (f_c * (pf - f_obs).norm(dim=-1) + s_c * (ps - s_obs).norm(dim=-1)) / (f_c + s_c).clamp(min=1e-6)
        for bones, kps, t in again:
            e = err[t, kps][f_c[t, kps] + s_c[t, kps] > 0.1].median()
            got = refit_limb(rig, t, bones, kps, trans, root, phi, f_obs, f_c, s_obs, s_c, f_axis, float(e), better=1.0)
            if got is not None:
                with torch.no_grad():
                    phi[t, bones] = got
    with torch.no_grad():
        total, parts, J, pf, ps, eul = loss_fn(iters)
        err_f = (pf - f_obs).norm(dim=-1)
        err_fx = (pf - f_obs)[..., 0].abs()
        err_s = (ps - s_obs).norm(dim=-1)
    return {"trans": trans.detach().cpu().numpy(), "root_rotmat": axis_angle(root).detach().cpu().numpy(),
            "euler": eul.detach().cpu().numpy(), "joints": J.cpu().numpy(),
            "reproj_px": {"front": float(err_f.mean()), "front_x_only": float(err_fx.mean()), "side": float(err_s.mean())},
            "bones": rig.names, "joint_names": names}


def to_quat(Rm):
    q = []
    for R in Rm:
        w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        x = np.copysign(np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
        y = np.copysign(np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
        z = np.copysign(np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
        q.append([w, x, y, z])
    return np.array(q)


def main():
    det_path, info_path, out = sys.argv[1:4]
    raw = np.load(det_path)
    d = {k: raw[k].copy() for k in ("front_uv", "side_uv", "front_conf", "side_conf")}
    info = json.load(open(info_path))
    names = [k["name"] for k in info["keypoints"]]
    before = sync_report(d)
    paws = paw_agreement(d, names)
    print("paw height agreement between views:", {k: (None if v is None else round(v, 2)) for k, v in paws.items()})
    print(f"sync as generated: mean height gap {before['mean_height_gap_px']:.1f} px, worst {before['worst_frame_gap_px']:.1f} px, "
          f"best constant lag {before['best_lag_frames']} frames")
    shift = None
    if "--no-align" not in sys.argv:
        trial = {k: v.copy() for k, v in d.items()}
        s = align_views(trial, names)
        gain = 1 - sync_report(trial)["mean_height_gap_px"] / max(before["mean_height_gap_px"], 1e-6)
        if gain > 0.15:  # only re-time when the halves really were out of step
            d.update(trial)
            shift = s
            print(f"time-aligned the front view onto the side view: mean shift {np.abs(shift).mean():.1f} frames, "
                  f"largest {np.abs(shift).max():.1f} frames (height gap down {gain:.0%})")
        else:
            print(f"views already in step (re-timing would only cut the height gap by {gain:.0%}); left as generated")
    # painted legs make the reader's left/right labels trustworthy, so swaps are only fixed on request
    swapped = fix_swaps(d, names) if "--fix-swaps" in sys.argv else {}
    sync = sync_report(d)
    print("side-view leg swaps fixed (frames):", swapped)
    print(f"sync after: mean height gap {sync['mean_height_gap_px']:.1f} px, worst {sync['worst_frame_gap_px']:.1f} px")
    sync["before"] = {k: v for k, v in before.items() if k != "per_frame_gap_px"}
    sync["paw_agreement"] = paws
    sync["front_shift_frames"] = None if shift is None else shift.round(2).tolist()
    acc = float(sys.argv[sys.argv.index("--acc") + 1]) if "--acc" in sys.argv else 0.02
    opt = lambda k, default: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else default
    vel = opt("--vel", 2.0)
    trusted, fhw = {}, 0.25
    if "--trust-front" in sys.argv:
        trusted, mask = trust_front(d, info)
        fhw = np.where(mask, 1.0, 0.25)
        print(f"front view trusted alone for {len(trusted)} joints, {int(mask.any(1).sum())} frames:",
              {n: s for n, s in trusted.items()})
    res = fit(d, info, dev="cuda" if torch.cuda.is_available() else "cpu", acc_weight=acc, vel_weight=vel,
              acc_arms=opt("--acc-arms", None), vel_arms=opt("--vel-arms", None),
              roll_weight=opt("--roll", 1.0), twist_vel=opt("--twist-vel", 20.0), front_height_weight=fhw)
    print("reprojection error px:", {k: round(v, 2) for k, v in res["reproj_px"].items()})
    motion = {"fps": float(raw["fps"]), "bones": res["bones"], "trans": res["trans"].round(5).tolist(),
              "root_quat": to_quat(res["root_rotmat"]).round(6).tolist(), "euler": res["euler"].round(5).tolist(),
              "joints": res["joints"].round(4).tolist(), "joint_names": res["joint_names"],
              "reproj_px": res["reproj_px"], "sync": sync, "swaps": swapped, "trusted_front": trusted}
    json.dump(motion, open(out, "w"))
    print("saved", out)


if __name__ == "__main__":
    main()
