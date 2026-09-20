"""Second half of the scene conversion: WebP images referenced by a .gltf -> PNG (Pillow), thin transmission
   (KHR_materials_transmission without a volume: a glass cornea, a tear film) -> alpha blending, JSON patched in place.
   python scripts/convert_scene_textures.py <scene.gltf>"""
import json, os, sys
from PIL import Image, ImageChops

path = sys.argv[1]
base = os.path.dirname(path)
doc = json.load(open(path, encoding="utf-8"))
converted = 0
for image in doc.get("images", []):
    uri = image.get("uri", "")
    if not uri or not uri.lower().endswith(".webp"):
        continue
    src = os.path.join(base, uri)
    dst = os.path.splitext(src)[0] + ".png"
    Image.open(src).save(dst)
    os.remove(src)
    image["uri"] = os.path.splitext(uri)[0] + ".png"
    image["mimeType"] = "image/png"
    converted += 1
for texture in doc.get("textures", []):
    ext = texture.get("extensions", {})
    if "EXT_texture_webp" in ext:
        texture["source"] = ext.pop("EXT_texture_webp")["source"]
        if not ext:
            texture.pop("extensions")


def image_of(texture_index):
    texture = doc["textures"][texture_index]
    source = texture.get("source", texture.get("extensions", {}).get("KHR_texture_basisu", {}).get("source"))
    return doc["images"][source] if source is not None else None


# Thin transmission -> alpha: the renderer's screen-space refraction cannot show an object's inside through its own
# shell (an eye's iris under the cornea), so a transmissive surface without a volume becomes a blended one with
# alpha = 1 - transmission, the transmission texture baked into the base color's alpha channel (same uv set).
baked = 0
for material in doc.get("materials", []):
    ext = material.get("extensions", {})
    transmission = ext.get("KHR_materials_transmission")
    if transmission is None or "KHR_materials_volume" in ext:
        continue
    pbr = material.setdefault("pbrMetallicRoughness", {})
    factor = float(transmission.get("transmissionFactor", 0.0))
    color = list(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]))
    textured = "transmissionTexture" in transmission and "baseColorTexture" in pbr
    if not textured:
        color[3] = color[3] * (1.0 - factor)
    pbr["baseColorFactor"] = color
    if textured:
        base_texture = pbr["baseColorTexture"]["index"]
        base_image = image_of(base_texture)
        trans_image = image_of(transmission["transmissionTexture"]["index"])
        if base_image and trans_image and base_image.get("uri", "").lower().endswith(".png") and trans_image.get("uri", "").lower().endswith(".png"):
            rgba = Image.open(os.path.join(base, base_image["uri"])).convert("RGBA")
            t = Image.open(os.path.join(base, trans_image["uri"])).convert("L").resize(rgba.size)
            keep = t.point(lambda v: int(round(255 - factor * v)))   # 1 - factor x transmission
            rgba.putalpha(ImageChops.multiply(rgba.getchannel("A"), keep))
            out = os.path.splitext(base_image["uri"])[0] + "_alpha.png"
            rgba.save(os.path.join(base, out))
            doc["images"].append({"uri": out, "mimeType": "image/png"})
            texture = {"source": len(doc["images"]) - 1}
            if "sampler" in doc["textures"][base_texture]:
                texture["sampler"] = doc["textures"][base_texture]["sampler"]
            doc["textures"].append(texture)
            pbr["baseColorTexture"] = dict(pbr["baseColorTexture"], index=len(doc["textures"]) - 1)
        else:
            color[3] = color[3] * (1.0 - factor)
    material["alphaMode"] = "BLEND"
    ext.pop("KHR_materials_transmission")
    ext.pop("KHR_materials_ior", None)   # the ior only mattered for the refraction that is gone
    if not ext:
        material.pop("extensions")
    baked += 1
if baked:
    used = {e for m in doc.get("materials", []) for e in m.get("extensions", {})}
    for key in ("extensionsUsed", "extensionsRequired"):
        if key in doc:
            doc[key] = [e for e in doc[key] if not e.startswith("KHR_materials_") or e in used]

for key in ("extensionsUsed", "extensionsRequired"):
    if key in doc:
        doc[key] = [e for e in doc[key] if e != "EXT_texture_webp"]
        if not doc[key]:
            doc.pop(key)
json.dump(doc, open(path, "w", encoding="utf-8"))
print(f"{path}: {converted} WebP images converted to PNG, {baked} thin transmission materials -> alpha; extensions {doc.get('extensionsUsed')}")
