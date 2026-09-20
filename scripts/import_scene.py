"""Turn a scene package into a demo scene folder: convert the model to plain glTF (convert_scene.mjs +
convert_scene_textures.py), copy its environment, and write the demo's view.json.

  python scripts/import_scene.py <source dir> <build/scenes/id> --modules <npm modules dir> [--name "Label"]
         [--model file.glb] [--environment file.hdr] [--drop-material name]... [--no-occlusion]

The source directory may carry a web-style `view.json` ({camera: {position, target, fov, near, far}, exposure,
environmentIntensity, background, backgroundEnvironment, sun: {direction, intensity}, model, instances,
groom}); its fields are carried over. Otherwise pass the model / environment and edit the written view.json
(camera etc.) afterwards. Instance transforms (instances.json + instances.bin) and hair grooms (groom.json) next
to the model are converted when the view.json names them.

The demo reads from view.json: name, model, environment (relative), environmentRotation (radians),
camera {position, target, fov, near, far} (or top-level position/target), moveSpeed, exposure,
environmentIntensity (IBL scale, 0..1), sun {direction, intensity}, background ("#rrggbb"),
backgroundEnvironment (bool).
"""
import json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]
    positional = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    if len(positional) < 2:
        raise SystemExit(__doc__)
    src, dst = os.path.abspath(positional[0]), os.path.abspath(positional[1])
    opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default   # noqa: E731
    drops = [args[i + 1] for i, a in enumerate(args) if a == "--drop-material"]
    modules = opt("--modules", os.getcwd())
    view = {}
    if os.path.exists(os.path.join(src, "view.json")):
        view = json.load(open(os.path.join(src, "view.json"), encoding="utf-8"))
    model = opt("--model", view.get("model"))
    if not model:
        candidates = [f for f in os.listdir(src) if f.lower().endswith((".glb", ".gltf"))]
        if len(candidates) != 1:
            raise SystemExit("pass --model: candidates %s" % candidates)
        model = candidates[0]
    environment = opt("--environment", view.get("environment", "lighting.hdr" if os.path.exists(os.path.join(src, "lighting.hdr")) else None))
    os.makedirs(dst, exist_ok=True)
    name = os.path.basename(dst.rstrip("\\/"))
    out_model = os.path.join(dst, name + ".gltf")
    cmd = ["node", os.path.join(HERE, "convert_scene.mjs"), os.path.join(src, model), out_model, "--modules", modules]
    for d in drops:
        cmd += ["--drop-material", d]
    if "--no-occlusion" in args:
        cmd.append("--no-occlusion")
    if view.get("instances") or (os.path.exists(os.path.join(src, "instances.json")) and os.path.exists(os.path.join(src, "instances.bin"))):
        cmd += ["--instances", os.path.join(src, "instances.json"), os.path.join(src, "instances.bin")]
    groom = view.get("groom")
    if groom:
        cmd += ["--groom", os.path.join(src, groom if isinstance(groom, str) else "groom.json")]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, os.path.join(HERE, "convert_scene_textures.py"), out_model], check=True)
    out = {"name": opt("--name", view.get("name", name)), "model": name + ".gltf"}
    if environment:
        shutil.copyfile(os.path.join(src, environment), os.path.join(dst, os.path.basename(environment)))
        out["environment"] = os.path.basename(environment)
    if groom:
        out["msaa"] = 4   # sub-pixel strands: the scene pass needs multisampling (see demo/README.md)
    for key in ("camera", "exposure", "environmentIntensity", "environmentRotation", "background", "backgroundEnvironment", "sun", "moveSpeed", "msaa"):
        if key in view:
            out[key] = view[key]
    if "camera" not in out and "position" in view:
        out["camera"] = {k: view[k] for k in ("position", "target", "fov", "near", "far") if k in view}
    if "sun" in out:
        out["sun"] = {k: v for k, v in out["sun"].items() if k in ("direction", "intensity", "color")}
    json.dump(out, open(os.path.join(dst, "view.json"), "w", encoding="utf-8"), indent=2)
    print("wrote", os.path.join(dst, "view.json"), json.dumps(out)[:300])


if __name__ == "__main__":
    main()
