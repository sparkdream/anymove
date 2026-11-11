"""Give a character's long hair its own mesh and bones: chains of spring bones hung beside the head, which
blender/apply_motion.py --springs simulates from the solved motion (the solver never sees them).

blender -b -P blender/add_hair.py -- rigged.blend [creature.json]
Rewrites rigged.blend in place and writes springs.json beside it; rig_info.json is unchanged.

A generated mesh is one shell, hair included, and the automatic weights hand the hair to whatever bone is nearest: a
third of it followed the collarbones, so raising an arm dragged the hair up with it and tore it. Here the hair is found
by its texture colour (dark and unsaturated by default; creature.json "hair" can change any of HAIR's settings),
cleaned up over the mesh, and split off as its own object, "hair". The shell has no scalp or back under the hair, so
where the hair lies on the head, shoulders and back it moves as that surface does (the head, neck and chest), but
never with an arm; toward its free ends, the strands it hangs in take over (clusters of where the hair falls, one
chain of bones each). The body beside the hair lets go of the arms too, or the arm drags it up through the hair.
"""
import json, pathlib, sys

import bpy
import numpy as np
from mathutils import Vector

HAIR = {
    "max_saturation": 0.14, "max_value": 0.38, "min_hue": 200, "grey_saturation": 0.05,  # dark grey, or tinted blue-violet
    "min_height": 0.95,  # metres: nothing below is hair
    "start": "head",  # the strands start at this bone's head's height
    "strand_parent": "chest",  # and hang from this bone (most of long hair lies on the back and shoulders)
    "free": 0.35, "tip": 0.8,  # the strands move nothing above `free` of their length, and `tip` of their ends
    "strands": 6, "segments": 3,
    "seam_rings": 3,  # rings of hair vertices from the seam over which the strands take over
    "body_rings": 0,  # rings of body vertices beside the hair that let go of the arms (0: the sleeves must follow them)
    "freq": [2.5, 2.0, 1.6], "damping": 0.3, "max_deg": [10, 15, 20], "gravity": 0.5,  # per segment, root first
}

args = sys.argv[sys.argv.index("--") + 1:]
blend = pathlib.Path(args[0])
if len(args) > 1:
    HAIR.update(json.load(open(args[1])).get("hair", {}))
bpy.ops.wm.open_mainfile(filepath=str(blend))
rig, body = bpy.data.objects["rig"], bpy.data.objects["body"]
if "hair" in bpy.data.objects:
    raise SystemExit("rigged.blend already has its hair split off: rebuild it (blender/build_rig.py) first")
me = body.data
nv = len(me.vertices)
co = np.array([v.co[:] for v in me.vertices])
E = np.array([e.vertices[:] for e in me.edges])
gn = [g.name for g in body.vertex_groups]
W = np.zeros((nv, len(gn)))
for v in me.vertices:
    for g in v.groups:
        W[v.index, g.group] = g.weight


def smooth01(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def vertex_colours():
    """Each vertex's base colour, averaged over its face corners, from the texture behind the material's colour."""
    img = None
    for slot in body.material_slots:
        for n in slot.material.node_tree.nodes:
            if n.type == "TEX_IMAGE" and any(l.to_node.type in ("MIX", "MIX_RGB", "BSDF_PRINCIPLED") for l in n.outputs[0].links):
                img = img or n.image
    w, h = img.size
    px = np.array(img.pixels[:]).reshape(h, w, 4)[..., :3]
    uv = np.zeros((len(me.loops), 2))
    me.uv_layers.active.data.foreach_get("uv", uv.ravel())
    vi = np.zeros(len(me.loops), int)
    me.loops.foreach_get("vertex_index", vi)
    c = px[(uv[:, 1] % 1 * (h - 1)).astype(int), (uv[:, 0] % 1 * (w - 1)).astype(int)]
    out, n = np.zeros((nv, 3)), np.zeros(nv)
    np.add.at(out, vi, c)
    np.add.at(n, vi, 1)
    return out / n.clip(1)[:, None]


# --- which vertices are hair
rgb = vertex_colours()
mx, mn = rgb.max(1), rgb.min(1)
val, sat = mx, np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
d = np.maximum(mx - mn, 1e-9)
hue = np.where(mx == rgb[:, 0], (rgb[:, 1] - rgb[:, 2]) / d % 6,
               np.where(mx == rgb[:, 1], (rgb[:, 2] - rgb[:, 0]) / d + 2, (rgb[:, 0] - rgb[:, 1]) / d + 4)) * 60
mask = blend.parent / HAIR.get("mask", "hair_region.npy")
if mask.exists() and len(np.load(mask)) == nv:  # what a segmentation model saw (rigging/blender/region_views.py): any
    x = (np.load(mask) > 0.5).astype(float)     # colour of hair or fur; the colour rule below is the fallback
    print(f"hair from {mask.name}: {int(x.sum())} vertices before the vote")
else:
    x = ((sat < HAIR["max_saturation"]) & (val < HAIR["max_value"]) & ((hue > HAIR["min_hue"]) | (sat < HAIR["grey_saturation"]))
         & (co[:, 2] > HAIR["min_height"])).astype(float)
deg = np.bincount(E.ravel(), minlength=nv).clip(1)
for _ in range(6):  # a vote over each vertex's neighbours: texture specks and stray dark patches drop out
    acc = np.zeros(nv)
    np.add.at(acc, E[:, 0], x[E[:, 1]])
    np.add.at(acc, E[:, 1], x[E[:, 0]])
    x = 0.5 * x + 0.5 * acc / deg
hair = x > 0.5  # whatever bone it follows now: the locks lying on a shoulder went to the upper arm
parent = np.arange(nv)


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for a, b in E[hair[E[:, 0]] & hair[E[:, 1]]]:
    ra, rb = find(int(a)), find(int(b))
    if ra != rb:
        parent[ra] = rb
root = np.array([find(i) for i in range(nv)])
ids, counts = np.unique(root[hair], return_counts=True)
hair &= np.isin(root, ids[counts > max(500, 0.05 * counts.max())])
print(f"hair: {int(hair.sum())} of {nv} vertices, {int((counts > max(500, 0.05 * counts.max())).sum())} piece(s)")

# --- strands: where the hair hangs below the head, in clusters around it, one chain of bones down each
hb = rig.data.bones[HAIR["start"]]
z_root = hb.head_local.z
centre = np.array(hb.head_local)
hang = hair & (co[:, 2] < z_root)
P = co[hang]
K = HAIR["strands"]
cen = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(-np.pi, np.pi, K, endpoint=False)]) * 0.1 + centre[:2]
for _ in range(30):  # k-means on where the hair hangs, seen from above
    lab = np.argmin(((P[:, None, :2] - cen[None]) ** 2).sum(-1), 1)
    cen = np.array([P[lab == k, :2].mean(0) if (lab == k).any() else cen[k] for k in range(K)])
chains = []
for k in range(K):
    Q = P[lab == k]
    if len(Q) < 200:
        continue
    z_end = np.percentile(Q[:, 2], 3)
    if z_root - z_end < 0.1:
        continue
    pts = []
    for z in np.linspace(z_root, z_end, HAIR["segments"] + 1):
        near = np.abs(Q[:, 2] - z) < 0.03
        xy = Q[near, :2].mean(0) if near.sum() > 10 else Q[np.argsort(np.abs(Q[:, 2] - z))[:20], :2].mean(0)
        pts.append([xy[0], xy[1], z])
    chains.append(np.array(pts))
print("strands:", len(chains), "| lengths (m):", [round(float(c[0, 2] - c[-1, 2]), 2) for c in chains])

bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
bone_names = []
for k, pts in enumerate(chains):
    names = []
    for j in range(len(pts) - 1):
        n = f"hair_{k}_{j}"
        b = rig.data.edit_bones.new(n)
        b.head, b.tail = Vector(pts[j].tolist()), Vector(pts[j + 1].tolist())
        zv = Vector((1, 0, 0)).cross((b.tail - b.head).normalized())
        b.align_roll(zv.normalized() if zv.length > 1e-4 else Vector((0, 0, 1)))
        b.parent = rig.data.edit_bones[names[-1] if names else HAIR["strand_parent"]]
        names.append(n)
    bone_names.append(names)
bpy.ops.object.mode_set(mode="OBJECT")

# --- the hair's weights. Where it lies on the head, shoulders and back it is that surface (the shell has nothing under
# it), so it keeps the balance of head, neck and chest the automatic weights gave it, but no limb: a side bone's share
# (a collarbone's, an arm's) goes to the body bone that limb hangs from, and a raised arm no longer drags the hair.
# The strands take over toward the free ends only: from `free` of the way down each strand, rising to `tip` at its end.
cols = {n: i for i, n in enumerate(gn)}
for names in bone_names:
    for n in names:
        cols[n] = len(cols)
base = np.hstack([W, np.zeros((nv, len(cols) - len(gn)))])
for n in gn:
    if n[-2:] in ("_L", "_R"):
        up = rig.data.bones[n].parent
        while up and up.name[-2:] in ("_L", "_R"):
            up = up.parent
        if up and up.name in cols:
            base[:, cols[up.name]] += base[:, cols[n]]
            base[:, cols[n]] = 0
dist, along = [], []
for pts in chains:
    best, s = np.full(nv, np.inf), np.zeros(nv)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    for j in range(len(pts) - 1):
        a, b = pts[j], pts[j + 1]
        t = np.clip(((co - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12), 0, 1)
        dd = np.linalg.norm(a + (b - a) * t[:, None] - co, axis=1)
        m = dd < best
        best[m], s[m] = dd[m], L[j] + t[m] * (L[j + 1] - L[j])
    dist.append(best)
    along.append((s, L))
dist = np.array(dist)
member = np.exp(-(dist - dist.min(0)) / 0.03)
member[np.argsort(dist, 0)[2:], np.arange(nv)] = 0  # the two nearest strands at most
member /= member.sum(0)
strand = np.zeros_like(base)
free = np.zeros(nv)
for k, names in enumerate(bone_names):
    s, L = along[k]
    f = member[k] * HAIR["tip"] * smooth01((s / L[-1] - HAIR["free"]) / (1 - HAIR["free"]))
    free += f
    r = [np.ones(nv)]
    for j in range(1, len(names)):
        h = 0.25 * min(L[j] - L[j - 1], L[j + 1] - L[j])
        r.append(smooth01((s - L[j] + h) / (2 * h)))
    r.append(np.zeros(nv))
    for j, n in enumerate(names):
        strand[:, cols[n]] += f * (r[j] - r[j + 1])
# Along the seam the strands give way over seam_rings rings of hair, and the body beside the hair (the back of the
# shoulders, the upper back between locks) lets go of its limbs over body_rings rings: left on the collarbone it was
# dragged up through the hair when an arm rose. The seam's own vertices go into both objects with the same weights.


def rings(start, inside, n):
    """Edge steps from the `start` vertices, up to n, walking only over `inside` vertices; -1 beyond."""
    r = np.where(start, 0, -1)
    for i in range(1, n + 1):
        a, b = r[E[:, 0]], r[E[:, 1]]
        grow = np.concatenate([E[(a == i - 1) & (b < 0), 1], E[(b == i - 1) & (a < 0), 0]])
        grow = grow[inside[grow]]
        r[grow] = i
    return r


edge = np.zeros(nv, bool)
edge[E[hair[E[:, 0]] != hair[E[:, 1]]].ravel()] = True
r_hair = rings(edge & hair, hair, HAIR["seam_rings"])
r_body = rings(edge & hair, ~hair, HAIR["body_rings"])
fade = np.where(r_hair < 0, 0.0, smooth01(1 - r_hair / (HAIR["seam_rings"] + 1)))
let_go = np.where(r_body < 0, 0.0, smooth01(1 - r_body / (HAIR["body_rings"] + 1)))
old = np.hstack([W, np.zeros((nv, base.shape[1] - W.shape[1]))])
Wn = np.where(hair[:, None], base * (1 - free[:, None] * (1 - fade[:, None])) + strand * (1 - fade[:, None]),
              let_go[:, None] * base + (1 - let_go[:, None]) * old)
keep = np.argsort(-Wn, axis=1)[:, :4]
mask = np.zeros_like(Wn, bool)
np.put_along_axis(mask, keep, True, axis=1)
Wn = np.where(mask, Wn, 0)
Wn /= Wn.sum(1, keepdims=True).clip(1e-9)
for names in bone_names:
    for n in names:
        body.vertex_groups.new(name=n)
for vg in body.vertex_groups:
    j = cols[vg.name]
    vg.remove(list(range(nv)))
    idx = np.nonzero(Wn[:, j] > 1e-4)[0]
    for i in idx:
        vg.add([int(i)], float(Wn[i, j]), "REPLACE")
arm_like = [cols[n] for n in gn if n.rsplit("_", 1)[0] in ("clavicle", "upperarm", "forearm", "hand")]
print(f"hair vertices still moved more than 10% by a collarbone or arm: "
      f"{int((Wn[hair][:, arm_like].sum(1) > 0.1).sum())} (were {int((W[hair][:, arm_like].sum(1) > 0.1).sum())})")

# --- split the hair off: the faces whose corners are all hair
sel = np.zeros(len(me.polygons), bool)
fv = [p.vertices[:] for p in me.polygons]
for i, f in enumerate(fv):
    sel[i] = all(hair[v] for v in f)
me.polygons.foreach_set("select", sel)
me.vertices.foreach_set("select", np.zeros(nv, bool))
me.edges.foreach_set("select", np.zeros(len(me.edges), bool))
for o in bpy.context.view_layer.objects:
    o.select_set(False)
bpy.context.view_layer.objects.active = body
body.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_mode(type="FACE")
bpy.ops.mesh.separate(type="SELECTED")
bpy.ops.object.mode_set(mode="OBJECT")
obj = next(o for o in bpy.context.selected_objects if o != body)
obj.name = obj.data.name = "hair"
# Where the hair lay against an arm (an arm hanging back into it at rest, fused with it in the shell), the seam is cut:
# its vertices follow the arm on the body's side and the hair on the hair's, so a raised arm comes away from the hair
# instead of stretching the sleeve back to it in a sheet.
from mathutils import kdtree  # noqa: E402

arm_bones = [i for i, n in enumerate(gn) if n[-2:] in ("_L", "_R") and rig.data.bones[n].parent
             and rig.data.bones[n].parent.name[-2:] in ("_L", "_R")]  # past the collarbone: upper arm, forearm, ...
on_arm = hair & (W[:, arm_bones].sum(1) > 0.3)
kd = kdtree.KDTree(nv)
for i, c in enumerate(co):
    kd.insert(c, i)
kd.balance()
groups = body.vertex_groups
cut = 0
for v in body.data.vertices:
    i = kd.find(v.co)[1]
    if on_arm[i]:
        groups_v = [g.group for g in v.groups]
        for g in groups_v:
            groups[g].remove([v.index])
        for j in np.nonzero(W[i] > 1e-4)[0]:
            groups[gn[j]].add([v.index], float(W[i, j] / W[i].sum()), "REPLACE")
        cut += 1
print(f"seam cut where the hair lay against an arm: {cut} vertices")
print(f"split off: hair {len(obj.data.vertices)} vertices, body {len(body.data.vertices)}")

springs = {}
for names in bone_names:
    for j, n in enumerate(names):
        pick = lambda v: v[min(j, len(v) - 1)] if isinstance(v, list) else v
        springs[n] = {"parent": names[j - 1] if j else HAIR["strand_parent"], "freq": pick(HAIR["freq"]),
                      "damping": pick(HAIR["damping"]), "max_deg": pick(HAIR["max_deg"]), "gravity": pick(HAIR["gravity"])}
json.dump(springs, open(blend.parent / "springs.json", "w"), indent=1)
bpy.ops.wm.save_as_mainfile(filepath=str(blend))
print("saved", blend, "and", blend.parent / "springs.json")
