"""Build the creature's armature from rig.json, skin it with Blender's automatic weights, paint the legs and save.

blender -b -P blender/build_rig.py -- static.blend rig.json out_dir [preview]
"preview" only renders the joints over an x-ray of the mesh (side, front, top) so they can be checked.
Otherwise writes rigged.blend, rig_info.json (rest matrices, parents, keypoints for the solver) and test renders.
"""
import json, math, pathlib, sys

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
static_blend, rig_json, out = args[:3]
preview = len(args) > 3 and args[3] == "preview"
out = pathlib.Path(out)
out.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=static_blend)
body = bpy.data.objects["body"]
joints, bones, limits = common.load_rig(rig_json)

if preview:
    common.stick_figure(joints, bones)
    common.xray_views(out, "joints")
    raise SystemExit

# Tripo's GLB splits vertices along every texture seam (185 pieces). Merge them back into one surface first, or the
# automatic weights stop at each seam and neighbouring patches pull apart when a joint bends. UVs live on face
# corners, so the texture is unaffected.
bpy.context.view_layer.objects.active = body
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
n0 = len(body.data.vertices)
bpy.ops.mesh.remove_doubles(threshold=1e-5)
bpy.ops.object.mode_set(mode="OBJECT")
print("merged seam vertices:", n0, "->", len(body.data.vertices))

# A held prop can be re-gripped (rig.json grip_deg: {"blade": 45}): the generator put a dagger in a hammer grip,
# square to the forearm, and no wrist bends far enough to lay it along the arm. Its blade, guard and pommel (whatever
# lies on its axis outside the fist) are turned about the middle of the fist until the prop sits that many degrees
# off the hand bone, and cut from the fingers they touched, which stay. The joint at its tip turns with it. Later the
# whole prop is weighted to the hand alone: it can't turn in the grip, so the wrist bends instead.
grip = {}
for prop, h, tip, hand in bones:
    deg = limits[prop].get("grip_deg")
    if deg is None or hand is None:
        continue
    _, wr, ht, _ = next(b for b in bones if b[0] == hand)
    co = np.array([v.co[:] for v in body.data.vertices])
    a, b, w = np.array(joints[h]), np.array(joints[tip]), np.array(joints[wr])
    u = (b - a) / np.linalg.norm(b - a)
    t = (co - a) @ u
    d = np.linalg.norm(co - a - np.outer(t, u), axis=1)
    blade = (t > 0.4 * np.linalg.norm(b - a)) & (t < 0.8 * np.linalg.norm(b - a)) & (d < 0.04)
    c = co[blade].mean(0)  # the prop's own axis: the joints sit a few cm off it
    u = np.linalg.svd(co[blade] - c)[2][0]
    u *= np.sign(u @ (b - a))
    s = (co - c) @ u
    d = np.linalg.norm(co - c - np.outer(s, u), axis=1)
    hd = np.array(joints[ht]) - w
    hl = np.linalg.norm(hd)
    _, el, _, _ = next(x for x in bones if x[0] == next(x for x in bones if x[0] == hand)[3])
    th = np.clip(((co - w) @ hd) / hl ** 2, 0, 1)
    fe = np.array(joints[el]) - w
    tf = np.clip(((co - w) @ fe) / (fe @ fe), 0, 1)
    dh = np.minimum(np.linalg.norm(w + np.outer(th, hd) - co, axis=1), np.linalg.norm(w + np.outer(tf, fe) - co, axis=1))
    near = d < 0.6 * hl
    fist = near & (dh < 0.6 * hl)  # the fingers round the grip: everything near the hand and forearm bones
    sp = s[fist].mean()
    sel = near & ~fist & (s < (b - c) @ u + 0.05) & (s > s[fist].min() - 2 * hl)
    n = np.cross(u, hd)
    n /= np.linalg.norm(n)
    ang = math.degrees(math.acos(np.clip(u @ hd / np.linalg.norm(hd), -1, 1))) - deg
    rot = np.array(Matrix.Rotation(math.radians(ang), 3, Vector(n)))
    piv = c + sp * u
    co[sel] = (co[sel] - piv) @ rot.T + piv
    joints[tip] = list((np.array(joints[tip]) - piv) @ rot.T + piv)
    for v in np.nonzero(sel)[0]:
        body.data.vertices[v].co = co[v]
    grip[prop] = sel
    print(f"{prop}: re-gripped {ang:.0f} deg toward {hand}, {int(sel.sum())} vertices")
if grip:
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(body.data)
    mark = bm.verts.layers.int.new("grip")
    bm.verts.ensure_lookup_table()
    for k, sel in enumerate(grip.values(), 1):
        for v in np.nonzero(sel)[0]:
            bm.verts[v][mark] = k
    cut = [f for f in bm.faces if len({v[mark] for v in f.verts}) > 1]
    bmesh.ops.delete(bm, geom=cut, context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    grip_mark = np.array([v[mark] for v in bm.verts])
    bm.to_mesh(body.data)
    bm.free()
    grip = {p: k for k, p in enumerate(grip, 1)}
    print(f"props cut from the fingers: {len(cut)} faces")

arm_data = bpy.data.armatures.new("rig")
arm = bpy.data.objects.new("rig", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = {}
for name, h, t, parent in bones:
    b = arm_data.edit_bones.new(name)
    b.head, b.tail = Vector(joints[h]), Vector(joints[t])
    d = (b.tail - b.head).normalized()
    z = Vector((1, 0, 0)).cross(d)
    if z.length < 0.5:  # runs mostly along X: local X horizontal and across the bone
        xl = Vector((0, 0, 1)).cross(d)
        z = xl.normalized().cross(d) if xl.length > 1e-4 else Vector((0, 0, 1))
    b.align_roll(z.normalized())  # local X = world X for bones in the body's plane: flexion is about X
    eb[name] = b
for name, h, t, parent in bones:
    if parent:
        eb[name].parent = eb[parent]
        eb[name].use_connect = False
bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_AUTO")

# vertices the heat solve missed go fully to the nearest bone segment
names = [b[0] for b in bones]
segs = np.array([[joints[h], joints[t]] for _, h, t, _ in bones])
groups = {g.index: g.name for g in body.vertex_groups}
missing = 0
for v in body.data.vertices:
    if sum(g.weight for g in v.groups) < 1e-3:
        p = np.array(v.co[:])
        a, b = segs[:, 0], segs[:, 1]
        t = np.clip(((p - a) * (b - a)).sum(1) / ((b - a) ** 2).sum(1), 0, 1)
        d = np.linalg.norm(a + (b - a) * t[:, None] - p, axis=1)
        body.vertex_groups[names[int(d.argmin())]].add([v.index], 1.0, "REPLACE")
        missing += 1
print("vertices without weights fixed:", missing)

# soften the weight borders (a few Laplacian passes over mesh edges), then keep at most 4 bones per vertex (what
# game engines skin with) and normalise
nv, ng = len(body.data.vertices), len(body.vertex_groups)
W = np.zeros((nv, ng))
for v in body.data.vertices:
    for g in v.groups:
        W[v.index, g.group] = g.weight
E = np.array([e.vertices[:] for e in body.data.edges])
deg = np.bincount(E.ravel(), minlength=nv)[:, None].clip(1)
for _ in range(3):
    acc = np.zeros_like(W)
    np.add.at(acc, E[:, 0], W[E[:, 1]])
    np.add.at(acc, E[:, 1], W[E[:, 0]])
    W = 0.5 * W + 0.5 * acc / deg

# The heat solve can spill a limb's weights far onto the body (an arm hanging against a long coat drags the coat's
# front with it when it swings across). Fade each limb bone's weight out with the distance from that bone, 5 to 10%
# of the creature's length, and hand it to the vertex's other bones; a vertex left with nothing goes to the nearest
# body bone.
co = np.array([v.co[:] for v in body.data.vertices])
size = np.ptp(co, axis=0).max()
parents = {b[0]: b[3] for b in bones}
limb = {n for chain in common.LEG_BONES.values() for n in chain if n in names}
for n in names:  # and everything hanging off a limb (a blade, a hoof)
    p = parents[n]
    while p and n not in limb:
        if p in limb:
            limb.add(n)
        p = parents[p]


def seg_dist(i):
    a, b = segs[i]
    t = np.clip(((co - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12), 0, 1)
    return np.linalg.norm(a + (b - a) * t[:, None] - co, axis=1)


gi = {g.name: g.index for g in body.vertex_groups}
before = W.copy()
for n in limb:
    if n in gi:
        W[:, gi[n]] *= np.clip((0.10 * size - seg_dist(names.index(n))) / (0.05 * size), 0, 1)
empty = W.sum(1) < 1e-3
if empty.any():
    body_bones = [i for i, n in enumerate(names) if n not in limb and n in gi]
    d = np.stack([seg_dist(i)[empty] for i in body_bones], 1)
    W[np.nonzero(empty)[0], [gi[names[body_bones[j]]] for j in d.argmin(1)]] = 1.0
moved = (before - W).clip(0).sum(1) > 0.05
print(f"limb weight pulled back from {int(moved.sum())} vertices far from their limb ({int(empty.sum())} reassigned)")

# Where two limbs nearly touch at rest (feet set close), the heat solve bleeds one into the other: the right boot's
# heel carried a third of the left toe's weight and stretched into a spike once the feet moved apart. Each vertex
# belongs to the limb it is part of, found across the mesh surface from vertices clearly on one side or the other, and
# hands the other side's weights below the limb's first bone to the same bones of its own (both thighs may share a
# coat, both feet may not).
for chain in common.LEG_BONES.values():
    other = [common.mirror(n) for n in chain]
    if not chain[0].endswith("_L") or not all(n in gi for n in chain[1:] + other[1:]):
        continue
    sides = [W[:, [gi[n] for n in c[1:]]].sum(1) for c in (chain, other)]
    label = np.full(nv, -1)
    for k in (0, 1):
        label[(sides[k] > 0.5) & (sides[1 - k] < 0.01)] = k
    while True:  # breadth-first over mesh edges: a vertex takes the label that reaches it first
        a, b = label[E[:, 0]], label[E[:, 1]]
        grow = np.concatenate([E[(a >= 0) & (b < 0)][:, [1, 0]], E[(b >= 0) & (a < 0)]])
        if not len(grow):
            break
        label[grow[:, 0]] = label[grow[:, 1]]
    fixed = 0
    for k, c in ((0, other), (1, chain)):
        bad = (label == k) & (W[:, [gi[n] for n in c[1:]]].sum(1) > 0)
        for n in c[1:]:  # onto the same bone of the vertex's own limb
            W[bad, gi[common.mirror_any(n)]] += W[bad, gi[n]]
            W[bad, gi[n]] = 0
        fixed += int(bad.sum())
    print(f"{chain[0]}/{other[0]}: the other side's weights moved to their own on {fixed} vertices")

# Along each limb, the heat solve and the smoothing blend neighbouring bones over half a bone's length (the hand moved
# vertices 8 to 14 cm up the forearm): a bent wrist then curves the forearm away from its bone and pinches it thin. So
# a limb's share of each vertex's weight is split again by where the vertex sits along the limb, blending only within
# a fifth of the shorter bone either side of each joint. Its share against the body's bones is kept. A limb hung
# from a side bone of its own (an arm from its collarbone) takes that bone in too: the heat solve gave the collarbone
# a third of the upper arm's first 10 cm, which stayed behind and flattened the arm when it swung across the chest.
for chain in common.LEG_BONES.values():
    chain = [n for n in chain if n in gi]
    side = bool(chain) and bool(parents.get(chain[0])) and parents[chain[0]][-2:] in ("_L", "_R") and parents[chain[0]] in gi
    if side:
        chain = [parents[chain[0]]] + chain
    if len(chain) < 2:
        continue
    ci = [gi[n] for n in chain]
    share = W[:, ci].sum(1)
    on = share > 1e-4
    ends = np.array([segs[names.index(chain[0])][0]] + [segs[names.index(n)][1] for n in chain])
    lens = np.linalg.norm(np.diff(ends, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(lens)])
    best, s = np.full(nv, np.inf), np.zeros(nv)
    for k in range(len(chain)):  # how far along the limb each vertex sits: on its nearest bone
        a, b = ends[k], ends[k + 1]
        t = np.clip(((co - a) @ (b - a)) / (lens[k] ** 2), 0, 1)
        d = np.linalg.norm(a + (b - a) * t[:, None] - co, axis=1)
        near = d < best
        best[near], s[near] = d[near], L[k] + t[near] * lens[k]
    # r[j]: how far past joint j the vertex is (0 before its window, 1 after); bone k gets r[k] - r[k + 1]
    r = [np.ones(nv)]
    for j in range(1, len(chain)):
        h = 0.2 * min(lens[j - 1], lens[j])
        x = np.clip((s - L[j] + h) / (2 * h), 0, 1)
        r.append(x * x * (3 - 2 * x))
    r.append(np.zeros(nv))
    for k, g in enumerate(ci):
        W[on, g] = share[on] * (r[k] - r[k + 1])[on]
    print(f"{chain[0]}..{chain[-1]}: weights split along the limb for {int(on.sum())} vertices")
    # A prop held in the limb's last bone (a dagger: a bone off the hand that is in no limb) is rigid and turns in the
    # fist: what lies along it past the fist is all prop, and the fist none of it (the heat solve gave the fist a
    # third of the blade's weight, and a dagger turned in the grip would have twisted the hand with it).
    for prop in [n for n, _, _, p in bones if p == chain[-1] and n in gi]:
        a, b = segs[names.index(prop)]
        t = ((co - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12)
        d = np.linalg.norm(a + np.outer(np.clip(t, 0, 1), b - a) - co, axis=1)
        own = np.clip((t - 0.02) / 0.06, 0, 1) * (d < seg_dist(names.index(chain[-1]))) \
            * (t < 1.15) * (d < 0.3 * np.linalg.norm(b - a))
        pair = W[:, gi[prop]] + W[:, gi[chain[-1]]]
        W[:, gi[prop]], W[:, gi[chain[-1]]] = pair * own, pair * (1 - own)
        print(f"   {prop}: rigid in {chain[-1]}, {int((own > 0.99).sum())} vertices")
keep = np.argsort(-W, axis=1)[:, :4]
mask = np.zeros_like(W, bool)
np.put_along_axis(mask, keep, True, axis=1)
W = np.where(mask, W, 0)
W /= W.sum(1, keepdims=True).clip(1e-9)

# Loose pieces (a buckle, a button the generator left floating off the surface) get weights from whatever bones the heat
# solve reached. Give every vertex of a small piece the weights of the closest vertex of the main body instead, so
# the piece moves rigidly with the surface it sits on.
parent = list(range(nv))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for a, b in E:
    ra, rb = find(int(a)), find(int(b))
    if ra != rb:
        parent[ra] = rb
piece = np.array([find(i) for i in range(nv)])
ids, counts = np.unique(piece, return_counts=True)
co = np.array([v.co[:] for v in body.data.vertices])
main = np.nonzero(piece == ids[counts.argmax()])[0]
loose = 0
for pid, n in zip(ids, counts):
    if n < 0.02 * nv:
        idx = np.nonzero(piece == pid)[0]
        sub = idx[:: max(1, len(idx) // 64)]
        d = ((co[sub, None] - co[None, main]) ** 2).sum(-1)
        W[idx] = W[main[d.min(0).argmin()]]
        loose += 1
print("loose pieces moving with the nearest surface:", loose)
for prop, k in grip.items():  # a re-gripped prop is fixed in its hand: its own bone only carries the tip's reading
    hand = gi[parents[prop]]
    W[:, hand] += W[:, gi[prop]]
    W[:, gi[prop]] = 0
    W[grip_mark == k] = 0
    W[grip_mark == k, hand] = 1
    print(f"{prop}: fixed in {parents[prop]}, {int((grip_mark == k).sum())} vertices")

# A shoulder blends the collarbone into the upper arm, and an arm raised forward turns 120 degrees between them: blended
# straight across, the shoulder collapses into a flat band. A helper at the upper arm's root turns half as far
# (Copy Rotation at half influence, so Blender poses it wherever the arm is posed; the solver never sees it), and the
# blend runs collarbone -> helper -> upper arm, 60 degrees a step. Where a vertex sat between the two (p, its upper
# arm's share of the pair), it now sits the same way along the three.
helpers = []
for chain in common.LEG_BONES.values():  # a limb's first bone hung from a side bone of its own (an arm's collarbone)
    name = chain[0]
    parent = parents.get(name)
    if name in gi and parent and parent[-2:] in ("_L", "_R") and parent in gi:
        helpers.append((name[:-2] + "_half" + name[-2:], name, parent))
if helpers:
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for half, name, parent in helpers:
        b = arm_data.edit_bones.new(half)
        src = arm_data.edit_bones[name]
        b.head, b.tail, b.roll = src.head, src.head + (src.tail - src.head) * 0.5, src.roll
        b.parent = arm_data.edit_bones[parent]
        b.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for half, name, parent in helpers:
        c = arm.pose.bones[half].constraints.new("COPY_ROTATION")
        c.target, c.subtarget = arm, name
        c.owner_space = c.target_space = "LOCAL"
        c.mix_mode = "REPLACE"
        c.influence = 0.5
        body.vertex_groups.new(name=half)
        W = np.hstack([W, np.zeros((nv, 1))])
        u, cl, hf = gi[name], gi[parent], W.shape[1] - 1
        pair = W[:, u] + W[:, cl]
        p = np.where(pair > 1e-6, W[:, u] / np.maximum(pair, 1e-6), 0)
        both = (W[:, u] > 1e-4) & (W[:, cl] > 1e-4)
        W[both, cl] = pair[both] * np.clip(1 - 2 * p[both], 0, 1)
        W[both, hf] = pair[both] * (1 - np.abs(2 * p[both] - 1))
        W[both, u] = pair[both] * np.clip(2 * p[both] - 1, 0, 1)
        print(f"{half}: shoulder helper turning half of {name}, for {int(both.sum())} vertices")
for gi, vg in enumerate(body.vertex_groups):
    vg.remove(list(range(nv)))
    idx = np.nonzero(W[:, gi] > 1e-4)[0]
    for i in idx:
        vg.add([int(i)], float(W[i, gi]), "REPLACE")
print("weights smoothed; max bones per vertex:", int((W > 1e-4).sum(1).max()))

common.paint_legs(body)
common.write_rig_info(arm, joints, bones, limits, out / "rig_info.json")
bpy.ops.wm.save_as_mainfile(filepath=str(out / "rigged.blend"))
print("saved", out / "rigged.blend")

# a test pose: front-left leg folded, hind-right stepping, head down, tail up
test = {"upperarm_L": (-30, 0, 0), "forearm_L": (-70, 0, 0), "metacarpus_L": (90, 0, 0),
        "thigh_R": (-35, 0, 0), "shin_R": (60, 0, 0), "metatarsus_R": (-60, 0, 0),
        "neck1": (25, 0, 0), "head": (20, 0, 15), "tail1": (-30, 0, 0), "tail2": (-20, 0, 10)}
for name, deg in [(n, d) for n, d in test.items() if n in arm.pose.bones]:
    pb = arm.pose.bones[name]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = Euler([math.radians(a) for a in deg], "XYZ")
common.studio(bpy.context.scene)
common.render_views(out, "testpose", res=(640, 480))
