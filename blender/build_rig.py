"""Build the creature's armature from rig.json, skin it with Blender's automatic weights, paint the legs and save.

blender -b -P blender/build_rig.py -- static.blend rig.json out_dir [preview]
"preview" only renders the joints over an x-ray of the mesh (side, front, top) so they can be checked.
Otherwise writes rigged.blend, rig_info.json (rest matrices, parents, keypoints for the solver) and test renders.
"""
import json, math, pathlib, sys

import bpy
import numpy as np
from mathutils import Euler, Vector

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
