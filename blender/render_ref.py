"""Render the rest pose (or a starting pose) from both layout cameras (RGBA), for the video model's first frame.

blender -b -P blender/render_ref.py -- rigged.blend out_dir [samples] [--pose POSE.json] [--clean-paint]
  --pose         {"bone": [x, y, z], ...}: point each listed bone along that direction in the rig's space (-y is the
                 front, +z up), parents first; the other bones keep their rest pose. Both views then start the clip
                 from the same pose, e.g. a guard stance instead of the rest pose.
  --clean-paint  repaint the limbs for this frame only, with creature.json's per-limb thresholds and loose pieces
                 (floating bits the rig moves with the nearest hand) left unpainted, so stray specks don't read as
                 parts of a limb. The dataset renders keep the rig's own paint.
"""
import json, pathlib, sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
opt = lambda k: args[args.index(k) + 1] if k in args else None
pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--pose")]
out = pathlib.Path(pos[1])
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=pos[0])
rig = bpy.data.objects["rig"]
for pb in rig.pose.bones:
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (0, 0, 0)
    pb.location = (0, 0, 0)
if opt("--pose"):
    aims = json.load(open(opt("--pose")))
    for pb in rig.pose.bones:  # parents come before their children
        if pb.name not in aims:
            continue
        bpy.context.view_layer.update()
        par = pb.parent
        rest = par.matrix @ (par.bone.matrix_local.inverted() @ pb.bone.matrix_local) if par else pb.bone.matrix_local
        local = rest.to_3x3().inverted() @ Vector(aims[pb.name]).normalized()
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Vector((0, 1, 0)).rotation_difference(local)
    bpy.context.view_layer.update()
if "--clean-paint" in args:
    body = next(o for o in bpy.data.objects if o.type == "MESH" and "legpaint" in o.data.color_attributes)
    rgba = common.limb_paint(body, skip_loose=True)
    body.data.color_attributes["legpaint"].data.foreach_set("color", rgba.ravel())
common.studio(bpy.context.scene, samples=int(pos[2]) if len(pos) > 2 else 128)
common.render_views(out, "ref")
