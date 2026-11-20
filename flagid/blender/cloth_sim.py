"""Cloth-simulated flag *shapes* for the training generator (runs inside Blender).

We never render actual flags here. For each random setup (attachment, size, fabric, wind,
pole angle, lighting) we simulate a blank cloth, grab a few settled frames, and render each
from a few cameras into a multilayer EXR with:
    Combined     white cloth under the HDRI (denoised)  -> lighting/shading
    GlossDir/Ind/Col  specular = (Dir + Ind) * Col      -> sheen (subtracted to get diffuse light)
    AOV uv       (u, v) flag texture coordinate per pixel; back faces see the mirrored design
    AOV back     1 where we see the back of the cloth
    AOV cloth    cloth coverage, AOV pole: pole coverage
Any flag is mapped onto any shape later, in Python, with one remap (src/cloth_shapes.py).

Usage:
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
      --python blender/cloth_sim.py -- --out data/cloth_raw --n-sims 6 --start 0
Each sim appends one JSON line per render to <out>/params.jsonl. Re-running with the same
--start/--seed reproduces the same setups; existing EXRs are skipped (resumable).
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import bpy
from mathutils import Vector

# ----------------------------------------------------------------------------- sampling config
CFG = dict(
    # Every flag is simulated at one nominal size: the drape only depends on the ratios of wind,
    # weight and stiffness, so "flag size" is covered by sampling those (avoids per-size retuning).
    width_m=(1.5, 1.5),
    aspect=(1.0, 2.0),               # width / height
    grid_long=48,                    # cloth vertices along the longer side
    # fabric: 0 = light/silky polyester, 1 = heavy cotton/nylon (interpolated, see fabric())
    stiffness=(0.0, 1.0),
    attach_probs={"sleeve": 0.65, "grommets": 0.20, "corners": 0.15},
    # pole angle from vertical (deg): standard pole / wall bracket / horizontal (hanging banner)
    pole_angle_bins=[((0, 10), 0.60), ((30, 60), 0.25), ((80, 90), 0.15)],
    wind_zero_p=0.2,                 # limp flags
    wind_strength=(300.0, 20000.0),  # Blender wind-field strength, log-uniform. Calibrated at 1.5 m:
                                     # 600-1500 half-lifted, 4000+ flying, 10000 flying with ripples
    wind_yaw_deg=60.0,               # wind direction jitter around "away from the pole"
    # tilted/horizontal poles (wall brackets, banners): v1 test showed strong wind wraps the cloth
    # around the pole, which looks wrong -> only calm to light wind, so the flag drapes off the pole
    tilted_pole_deg=20.0,            # poles tilted more than this count as tilted
    tilted_wind_zero_p=0.5,
    tilted_wind_strength=(300.0, 1500.0),
    turbulence_frac=(0.2, 0.8),      # turbulence strength relative to wind
    wind_flow=1.0,                   # Blender wind 'flow' (0 = constant force, 1 = drag-like air flow)
    corner_slack=(0.05, 0.25),       # 'corners' mode: cloth expands this much -> sags between pins
    sim_quality=8,                   # cloth solver steps per frame
    self_collision=True,
    # The pole is NOT a collider: the pinned hoist sits inside its collision margin, which made the
    # solver push/wrap/freeze the cloth (v1's "crumpled around the pole" shapes)
    pole_collision=False,
    min_wind_field=1.0,              # wind field strength used for "no wind" (see build())
    sim_frames=110,                  # frames simulated (24 fps)
    settle_frame=50,                 # grab frames after this
    # camera
    # (v2: narrower than v1; steep views from below/above or edge-on along the pole looked unrealistic)
    back_p=0.2,                      # camera behind the flag (sees the mirrored design)
    elevation_deg=(-30.0, 12.0),     # negative = camera below, looking up (most common)
    azimuth_jitter_deg=45.0,         # around the front (or back) normal
    focal_mm=(24.0, 85.0),
    framing_margin=(1.02, 1.35),     # >1 = some space around the cloth
    # lighting
    hdri_strength=(0.4, 2.0),
    # render
    filter_width=1.0,
)


def U(r, lo_hi):
    return r.uniform(*lo_hi)


def logU(r, lo_hi):
    return math.exp(r.uniform(math.log(lo_hi[0]), math.log(lo_hi[1])))


def choice_w(r, items):
    """items: dict name->p or list of (value, p)."""
    pairs = list(items.items()) if isinstance(items, dict) else items
    x, acc = r.random() * sum(p for _, p in pairs), 0.0
    for v, p in pairs:
        acc += p
        if x <= acc:
            return v
    return pairs[-1][0]


def sample_setup(r, hdris):
    s = dict(
        width=logU(r, CFG["width_m"]),
        aspect=U(r, CFG["aspect"]),
        stiffness=U(r, CFG["stiffness"]),
        attach=choice_w(r, CFG["attach_probs"]),
        pole_angle=U(r, choice_w(r, CFG["pole_angle_bins"])),
        wind=0.0 if r.random() < CFG["wind_zero_p"] else logU(r, CFG["wind_strength"]),
        wind_yaw=U(r, (-CFG["wind_yaw_deg"], CFG["wind_yaw_deg"])),
        turbulence=U(r, CFG["turbulence_frac"]),
        corner_slack=U(r, CFG["corner_slack"]),
        hdri=r.choice(hdris),
        hdri_rot=U(r, (0, 360)),
        hdri_strength=U(r, CFG["hdri_strength"]),
        seed=r.randrange(1 << 30),
    )
    if s["attach"] == "corners":
        s["pole_angle"] = 0.0  # held/draped between two points, no pole
    # tilted poles and hand-held (corners) flags: calm to light wind only (strong wind looked wrong)
    if s["pole_angle"] > CFG["tilted_pole_deg"] or s["attach"] == "corners":
        s["wind"] = 0.0 if r.random() < CFG["tilted_wind_zero_p"] else logU(r, CFG["tilted_wind_strength"])
    return s


def fabric(stiffness):
    """Interpolate Blender cloth parameters between a silk-like and a heavy-cotton-like preset."""
    lerp = lambda a, b: a + (b - a) * stiffness  # noqa: E731
    loglerp = lambda a, b: math.exp(lerp(math.log(a), math.log(b)))  # noqa: E731
    # v2: stiffer than v1 (bending 0.02-1.5, tension 5-25 crinkled like tissue paper); real flag
    # fabric resists compression and forms a few large folds
    return dict(mass=lerp(0.15, 0.4), tension=loglerp(15, 40), shear=loglerp(10, 30),
                bending=loglerp(0.3, 5.0), air_damping=lerp(1.0, 2.0))


# ----------------------------------------------------------------------------- scene building
def reset_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
    except Exception as e:  # noqa: BLE001
        print("GPU unavailable, using CPU:", e)
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces = 4
    sc.cycles.filter_width = CFG["filter_width"]
    sc.render.resolution_x = sc.render.resolution_y = args.res
    sc.render.film_transparent = True
    sc.render.fps = 24
    sc.frame_start, sc.frame_end = 1, CFG["sim_frames"]
    im = sc.render.image_settings
    im.media_type = "MULTI_LAYER_IMAGE"  # Blender 5: choose the media type first
    im.file_format = "OPEN_EXR_MULTILAYER"
    im.color_depth = "16"
    im.exr_codec = "ZIP"
    vl = sc.view_layers[0]
    vl.use_pass_glossy_direct = True     # glossy *light* passes (not multiplied by the gloss colour)
    vl.use_pass_glossy_indirect = True
    vl.use_pass_glossy_color = True      # specular contribution = (direct + indirect) * colour
    for name, typ in (("uv", "COLOR"), ("back", "VALUE"), ("cloth", "VALUE"), ("pole", "VALUE")):
        a = vl.aovs.add()
        a.name, a.type = name, typ
    return sc


def aov_node(nt, name, socket, is_value):
    n = nt.nodes.new("ShaderNodeOutputAOV")
    n.aov_name = name
    nt.links.new(socket, n.inputs["Value" if is_value else "Color"])
    return n


def make_cloth_material():
    m = bpy.data.materials.new("cloth")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
    bsdf.inputs["Roughness"].default_value = 0.65
    if "Sheen Weight" in bsdf.inputs:
        bsdf.inputs["Sheen Weight"].default_value = 0.3
    tc = nt.nodes.new("ShaderNodeTexCoord")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    one = nt.nodes.new("ShaderNodeValue")
    one.outputs[0].default_value = 1.0
    aov_node(nt, "uv", tc.outputs["UV"], False)
    aov_node(nt, "back", geo.outputs["Backfacing"], True)
    aov_node(nt, "cloth", one.outputs[0], True)
    return m


def make_pole_material(r):
    m = bpy.data.materials.new("pole")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    g = r.uniform(0.25, 0.85)
    bsdf.inputs["Base Color"].default_value = (g, g * r.uniform(0.9, 1.0), g * r.uniform(0.8, 1.0), 1)
    bsdf.inputs["Metallic"].default_value = r.choice([0.0, 1.0])
    bsdf.inputs["Roughness"].default_value = r.uniform(0.2, 0.7)
    one = nt.nodes.new("ShaderNodeValue")
    one.outputs[0].default_value = 1.0
    aov_node(nt, "pole", one.outputs[0], True)
    return m


def rot_y(v, deg):
    """Rotate a vector about world Y (the flag plane is XZ; this tilts the pole toward +X)."""
    a = math.radians(deg)
    return Vector((v.x * math.cos(a) + v.z * math.sin(a), v.y, -v.x * math.sin(a) + v.z * math.cos(a)))


def build(s, r):
    W, H = s["width"], s["width"] / s["aspect"]
    nl = CFG["grid_long"]
    nx, nz = (nl, max(8, round(nl / s["aspect"]))) if W >= H else (max(8, round(nl * s["aspect"])), nl)
    # cloth in local frame: hoist edge along +Z at x=0, fly end at x=W; front faces -Y.
    # world = rot_y(local, pole_angle): 0 deg -> vertical pole, 90 deg -> horizontal pole, cloth hangs down
    verts, faces, uvs = [], [], []
    for j in range(nz):
        for i in range(nx):
            u, v = i / (nx - 1), j / (nz - 1)
            verts.append(rot_y(Vector((u * W, 0.0, v * H - H)), s["pole_angle"]))
    for j in range(nz - 1):
        for i in range(nx - 1):
            a = j * nx + i
            faces.append((a, a + 1, a + nx + 1, a + nx))
    me = bpy.data.meshes.new("cloth")
    me.from_pydata([tuple(v) for v in verts], [], faces)
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            uvl.data[li].uv = ((vi % nx) / (nx - 1), (vi // nx) / (nz - 1))
    me.update()
    cloth = bpy.data.objects.new("cloth", me)
    bpy.context.scene.collection.objects.link(cloth)
    cloth.data.materials.append(make_cloth_material())
    for p in me.polygons:
        p.use_smooth = True

    # pinned vertices
    hoist = [j * nx for j in range(nz)]
    if s["attach"] == "sleeve":
        pins = hoist
    elif s["attach"] == "grommets":
        k = max(1, nz // 12)
        pins = hoist[:k] + hoist[-k:]
    else:  # corners: top-left and top-right
        pins = [(nz - 1) * nx, (nz - 1) * nx + nx - 1]
    vg = cloth.vertex_groups.new(name="pin")
    vg.add(pins, 1.0, "REPLACE")

    mod = cloth.modifiers.new("Cloth", "CLOTH")
    cs = mod.settings
    f = fabric(s["stiffness"])
    cs.quality = s.get("quality", CFG["sim_quality"])
    cs.mass = f["mass"]
    cs.tension_stiffness = cs.compression_stiffness = f["tension"]
    cs.shear_stiffness = f["shear"]
    cs.bending_stiffness = f["bending"]
    cs.air_damping = f["air_damping"]
    cs.vertex_group_mass = "pin"
    if s["attach"] == "corners":
        cs.shrink_min = -s["corner_slack"]  # negative shrink = cloth grows -> sags between the pins
    cc = mod.collision_settings
    cc.use_self_collision = s.get("self_collision", CFG["self_collision"])
    cc.self_distance_min = 0.004 * W
    cc.distance_min = 0.005 * W
    mod.point_cache.frame_start, mod.point_cache.frame_end = 1, CFG["sim_frames"]
    sub = cloth.modifiers.new("Smooth", "SUBSURF")
    sub.levels = sub.render_levels = 1

    # pole (collider), along the hoist edge, extending above the top and well below the cloth
    if s["attach"] != "corners":
        radius = r.uniform(0.012, 0.03) * max(W, 1.0)
        top_ext, bottom_ext = r.uniform(0.02, 0.3) * H, 3.0 * H
        length = H + top_ext + bottom_ext
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=length)
        pole = bpy.context.active_object
        center_local = Vector((-radius, 0, (top_ext - (H + bottom_ext)) / 2))
        pole.location = rot_y(center_local, s["pole_angle"])
        pole.rotation_euler = (0, math.radians(s["pole_angle"]), 0)
        pole.data.materials.append(make_pole_material(r))
        if s.get("pole_collision", CFG["pole_collision"]):
            pole.modifiers.new("Collision", "COLLISION")
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius * 1.8,
                                             location=rot_y(Vector((-radius, 0, top_ext)), s["pole_angle"]))
        bpy.context.active_object.data.materials.append(pole.data.materials[0])

    # wind: blows roughly away from the hoist (+X in world, rotated by yaw), plus turbulence.
    # The wind field is ALWAYS added (near-zero strength for "no wind"): in Blender 5.3 a cloth with
    # no collider/effector in the scene barely responds to gravity (sags ~5 cm instead of hanging).
    yaw = math.radians(s["wind_yaw"])
    d = Vector((math.cos(yaw), math.sin(yaw), 0.0))
    bpy.ops.object.effector_add(type="WIND", location=(-2 * W, 0, 0))
    wind = bpy.context.active_object
    wind.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    wind.field.strength = max(s["wind"], CFG["min_wind_field"])
    if s["wind"] > 0:
        wind.field.flow = s.get("wind_flow", CFG["wind_flow"])
        wind.field.noise = 1.0
        wind.field.seed = s["seed"] % 128 + 1
        bpy.ops.object.effector_add(type="TURBULENCE", location=(W / 2, 0, -H / 2))
        turb = bpy.context.active_object
        turb.field.strength = s["wind"] * s["turbulence"]
        turb.field.size = r.uniform(0.3, 1.0) * W
        turb.field.flow = 0.5
        turb.field.seed = (s["seed"] // 7) % 128 + 1
    return cloth


def setup_world(s):
    w = bpy.data.worlds.new("world")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(s["hdri"])
    mp = nt.nodes.new("ShaderNodeMapping")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp.inputs["Rotation"].default_value[2] = math.radians(s["hdri_rot"])
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    bg = nt.nodes["Background"]
    bg.inputs["Strength"].default_value = s["hdri_strength"]
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])


def place_camera(r, cloth, force_back=None, force_el=None, force_az=None):
    """Aim at the evaluated cloth's bounding box from a random view; returns camera params."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = cloth.evaluated_get(dg)
    pts = [ev.matrix_world @ v.co for v in ev.data.vertices]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center, radius = (lo + hi) / 2, max((p - (lo + hi) / 2).length for p in pts)
    back = r.random() < CFG["back_p"] if force_back is None else force_back
    jitter = r.uniform(-CFG["azimuth_jitter_deg"], CFG["azimuth_jitter_deg"]) if force_az is None else force_az
    az = math.radians((180.0 if back else 0.0) + jitter)
    el = math.radians(U(r, CFG["elevation_deg"]) if force_el is None else force_el)
    # front of the flag faces -Y
    direction = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    focal = U(r, CFG["focal_mm"])
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = focal
    cam_data.sensor_width = 36.0
    fov = 2 * math.atan(18.0 / focal)
    dist = radius * U(r, CFG["framing_margin"]) / math.sin(fov / 2)
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = center + direction * dist
    cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam, dict(back=back, azimuth=math.degrees(az), elevation=math.degrees(el), focal=focal)


# ----------------------------------------------------------------------------- main loop
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sims", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames-per-sim", type=int, default=6)
    ap.add_argument("--cams-per-frame", type=int, default=3)
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--hdri-dir", default=None, help="folder of .exr/.hdr; default: Blender's bundled ones")
    ap.add_argument("--override", default=None,
                    help='JSON list, one dict per sim, overriding sampled params (calibration sweeps), '
                         'e.g. \'[{"wind": 500, "attach": "sleeve"}]\'; also "back": false fixes the camera side')
    args = ap.parse_args(argv)
    overrides = json.loads(args.override) if args.override else []
    if overrides:
        args.n_sims = len(overrides)

    hdri_dir = args.hdri_dir or bpy.utils.system_resource("DATAFILES", path="studiolights/world")
    hdris = sorted(glob.glob(os.path.join(hdri_dir, "*.exr")) + glob.glob(os.path.join(hdri_dir, "*.hdr")))
    assert hdris, f"no HDRIs in {hdri_dir}"
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "params.jsonl"), "a")

    for sim in range(args.start, args.start + args.n_sims):
        r = random.Random(f"{args.seed}-{sim}")
        s = sample_setup(r, hdris)
        if overrides:
            s.update(overrides[sim - args.start])
        n_frames = 1 if s["wind"] == 0 else args.frames_per_sim  # limp flags barely change
        frames = sorted(r.sample(range(CFG["settle_frame"], CFG["sim_frames"] + 1), n_frames))
        if "frames" in s:  # override (debugging): render exactly these frames
            frames = list(s["frames"])
        names = [f"s{sim:05d}_f{f:03d}_c{c}" for f in frames for c in range(args.cams_per_frame)]
        if all(os.path.exists(os.path.join(args.out, n + ".exr")) for n in names):
            continue
        t0 = time.time()
        sc = reset_scene(args)
        cloth = build(s, r)
        setup_world(s)
        sim_t = 0.0
        for fr in range(1, CFG["sim_frames"] + 1):
            ts = time.time()
            sc.frame_set(fr)
            sim_t += time.time() - ts
            if fr not in frames:
                continue
            for c in range(args.cams_per_frame):
                name = f"s{sim:05d}_f{fr:03d}_c{c}"
                cam, cp = place_camera(r, cloth, force_back=s.get("back"), force_el=s.get("elevation"),
                                       force_az=s.get("azimuth"))
                sc.render.filepath = os.path.join(args.out, name + ".exr")
                bpy.ops.render.render(write_still=True)
                bpy.data.objects.remove(cam)
                log.write(json.dumps({"name": name, "sim": sim, "frame": fr, **s, **cp,
                                      "hdri_name": os.path.basename(s["hdri"])}) + "\n")
                log.flush()
        print(f"SIM {sim} attach={s['attach']} pole={s['pole_angle']:.0f} wind={s['wind']:.0f} "
              f"width={s['width']:.2f} stiff={s['stiffness']:.2f} frames={len(frames)} "
              f"sim={sim_t:.1f}s total={time.time() - t0:.1f}s", flush=True)


main()
