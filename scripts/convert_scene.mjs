// Convert a web-optimized scene (.glb with Draco / meshopt compression, WebP textures, GPU instancing) into a
// plain .gltf (+ .bin + image files) the Vulkan-glTF-PBR renderer loads: meshes decoded, instances expanded into
// nodes; run convert_scene_textures.py afterwards for the WebP -> PNG step. Needs @gltf-transform/core,
// @gltf-transform/extensions, @gltf-transform/functions, meshoptimizer and draco3dgltf installed in one directory
// (npm install ... --prefix <dir>), passed with --modules (default: the current directory).
//   node scripts/convert_scene.mjs <in.glb> <out.gltf> [--modules <dir with node_modules>] [--drop-material <name>]...
//        [--instances <instances.json> <instances.bin>]   scattered copies of named nodes (16 floats per instance,
//                                                          column-major, applied on top of the node's world matrix)
//        [--groom <groom.json>]                             compact hair grooms (gzipped "rings-v2" or plain strips
//                                                          next to the manifest) as vertex-colored meshes
//        [--no-occlusion]                                   drop occlusion textures (exports whose AO channel is empty)
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import zlib from 'node:zlib';

const args = process.argv.slice(2);
const input = args[0];
const output = args[1];
let modulesDir = process.cwd();
const dropMaterials = [];
let instancesJson = null, instancesBin = null, groomJson = null, noOcclusion = false;
for (let i = 2; i < args.length; i++) {
  if (args[i] === '--modules' || args[i] === '--draco-dir') modulesDir = args[++i];
  if (args[i] === '--drop-material') dropMaterials.push(args[++i]);   // e.g. the scene's baked sky dome
  if (args[i] === '--instances') { instancesJson = args[++i]; instancesBin = args[++i]; }
  if (args[i] === '--groom') groomJson = args[++i];
  if (args[i] === '--no-occlusion') noOcclusion = true;
}
if (!input || !output) {
  console.error('usage: node convert_scene.mjs <in.glb> <out.gltf> [--modules <dir>] [--drop-material <name>]...');
  process.exit(2);
}
const dracoDir = modulesDir;
const requireFromModules = createRequire(path.join(path.resolve(modulesDir), 'package.json'));
const importFromModules = (name) => import(pathToFileURL(requireFromModules.resolve(name)).href);

const { NodeIO, Document } = await importFromModules('@gltf-transform/core');
const { ALL_EXTENSIONS, EXTMeshGPUInstancing } = await importFromModules('@gltf-transform/extensions');
const { dequantize, unpartition, prune } = await importFromModules('@gltf-transform/functions');
const { MeshoptDecoder } = await importFromModules('meshoptimizer');

const dependencies = { 'meshopt.decoder': MeshoptDecoder };
if (dracoDir) {
  const requireDraco = createRequire(path.join(path.resolve(dracoDir), 'package.json'));
  const draco3d = requireDraco('draco3dgltf');
  dependencies['draco3d.decoder'] = await draco3d.createDecoderModule();
}

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies(dependencies);
const document = await io.read(input);
const root = document.getRoot();
console.log(`read ${input}: ${root.listMeshes().length} meshes, ${root.listTextures().length} textures, extensions ${root.listExtensionsUsed().map((e) => e.extensionName).join(', ')}`);

// 1. GPU instancing -> one child node per instance (the renderer draws nodes, not instance batches)
let expanded = 0;
for (const node of root.listNodes()) {
  const batch = node.getExtension('EXT_mesh_gpu_instancing');
  if (!batch) continue;
  const mesh = node.getMesh();
  const translation = batch.getAttribute('TRANSLATION');
  const rotation = batch.getAttribute('ROTATION');
  const scale = batch.getAttribute('SCALE');
  const count = (translation || rotation || scale).getCount();
  for (let i = 0; i < count; i++) {
    const child = document.createNode(`${node.getName()}_instance${i}`).setMesh(mesh);
    if (translation) child.setTranslation(Array.from(translation.getElement(i, [])));
    if (rotation) child.setRotation(Array.from(rotation.getElement(i, [])));
    if (scale) child.setScale(Array.from(scale.getElement(i, [])));
    node.addChild(child);
  }
  node.setMesh(null);
  node.setExtension('EXT_mesh_gpu_instancing', null);
  expanded += count;
}
if (expanded) console.log(`expanded ${expanded} instances into nodes`);
for (const ext of root.listExtensionsUsed()) {
  if (ext.extensionName === 'EXT_mesh_gpu_instancing') ext.dispose();
}

// 1a. external instance transforms: every mesh node under the named source node becomes `count` scene nodes with
//     matrix = instance * world(source mesh node); the source subtree is removed (it was the template)
if (instancesJson) {
  const scene = root.getDefaultScene() || root.listScenes()[0];
  const sanitize = (name) => name.replace(/\s/g, '_').replace(/[\[\]\.:\/]/g, '');
  const entries = JSON.parse(fs.readFileSync(instancesJson, 'utf8'));
  const binary = fs.readFileSync(instancesBin);
  const matrices = new Float32Array(binary.buffer, binary.byteOffset, binary.byteLength / 4);
  let created = 0;
  for (const entry of entries) {
    const source = root.listNodes().find((n) => sanitize(n.getName()) === sanitize(entry.name));
    if (!source) throw new Error(`instance source node not found: ${entry.name}`);
    const meshNodes = [];
    source.traverse((n) => { if (n.getMesh()) meshNodes.push(n); });
    const worlds = meshNodes.map((n) => n.getWorldMatrix());
    for (let i = 0; i < entry.count; i++) {
      const m = Array.from(matrices.subarray(entry.offset / 4 + i * 16, entry.offset / 4 + i * 16 + 16));
      for (let k = 0; k < meshNodes.length; k++) {
        const node = document.createNode(`${entry.name}_${i}`).setMesh(meshNodes[k].getMesh()).setMatrix(multiply(m, worlds[k]));
        scene.addChild(node);
        created++;
      }
    }
    const parent = source.getParentNode();
    if (parent) parent.removeChild(source); else scene.removeChild(source);
    source.dispose();
  }
  console.log(`expanded ${created} instance nodes from ${entries.length} sources`);
}
function multiply(a, b) {   // column-major 4x4: a * b
  const out = new Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let v = 0;
    for (let k = 0; k < 4; k++) v += a[k * 4 + r] * b[c * 4 + k];
    out[c * 4 + r] = v;
  }
  return out;
}

// 1c. compact hair grooms: triangular tubes rebuilt from quantized ring records ("rings-v2") or plain strips,
//     one vertex-colored mesh per part (the alpha channel carries the strand opacity: alpha-blended)
if (groomJson) {
  const scene = root.getDefaultScene() || root.listScenes()[0];
  const base = path.dirname(groomJson);
  const parts = JSON.parse(fs.readFileSync(groomJson, 'utf8'));
  const buffer = root.listBuffers()[0] || document.createBuffer();
  let vertices = 0;
  for (const part of parts) {
    let bytes = fs.readFileSync(path.join(base, part.file));
    if (part.file.endsWith('.gz')) bytes = zlib.gunzipSync(bytes);
    const decoded = part.encoding === 'rings-v2' ? decodeGroomRings(bytes, part) : decodeGroomStrips(bytes, part);
    const color = new Float32Array(decoded.color.length);
    for (let i = 0; i < color.length; i++) color[i] = decoded.color[i] / 255;   // the renderer reads float colors
    const position = document.createAccessor().setType('VEC3').setArray(decoded.position).setBuffer(buffer);
    const normal = document.createAccessor().setType('VEC3').setArray(decoded.normal).setBuffer(buffer);
    const colors = document.createAccessor().setType('VEC4').setArray(color).setBuffer(buffer);
    const indices = document.createAccessor().setType('SCALAR').setArray(decoded.index).setBuffer(buffer);
    const baseColor = part.colorLinear ? [...part.colorLinear, part.rasterOpacity ?? 1] : [0.02, 0.015, 0.01, part.rasterOpacity ?? 1];
    const material = document.createMaterial(part.name)
      .setBaseColorFactor(baseColor).setRoughnessFactor(0.48).setMetallicFactor(0).setDoubleSided(true)
      .setAlphaMode(part.vertexOpacity || part.rasterOpacity !== undefined ? 'BLEND' : 'OPAQUE');   // strand opacity: vertex alpha x rasterOpacity
    const primitive = document.createPrimitive().setAttribute('POSITION', position).setAttribute('NORMAL', normal)
      .setAttribute('COLOR_0', colors).setIndices(indices).setMaterial(material);
    const mesh = document.createMesh(part.name).addPrimitive(primitive);
    scene.addChild(document.createNode(part.name).setMesh(mesh).setExtras({ groom: true, part: part.name }));   // the demo draws grooms with depth writes
    vertices += decoded.position.length / 3;
  }
  console.log(`added ${parts.length} groom parts, ${vertices} vertices`);
}
function decodeGroomStrips(bytes, part) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const f32 = (offset, count) => { const a = new Float32Array(count); for (let i = 0; i < count; i++) a[i] = view.getFloat32(offset + i * 4, true); return a; };
  const position = f32(0, part.vertices * 3), normal = f32(part.vertices * 12, part.vertices * 3);
  const index = new Uint32Array(part.indices);
  for (let i = 0; i < part.indices; i++) index[i] = view.getUint32(part.vertices * 24 + i * 4, true);
  const color = new Uint8Array(part.vertices * 4).fill(255);
  if (part.vertexOpacity) color.set(new Uint8Array(bytes.buffer, bytes.byteOffset + part.vertices * 24 + part.indices * 4, part.vertices * 4));
  return { position, normal, color, index };
}
function decodeGroomRings(buffer, part) {
  const { rings, strands, vertices, indices, centerStep, radiusStep, frameScale } = part;
  if (vertices !== rings * 3 || indices !== (rings - strands) * 18 || buffer.byteLength !== strands * 2 + rings * 22) throw new Error(`invalid groom ${part.name}`);
  const bytes = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const centerOffset = strands * 2;
  const frameOffset = centerOffset + rings * 12;
  const omittedOffset = frameOffset + rings * 6;
  const radiusOffset = omittedOffset + rings;
  const opacityOffset = radiusOffset + rings * 2;
  const position = new Float32Array(vertices * 3), normal = new Float32Array(vertices * 3), color = new Uint8Array(vertices * 4);
  const index = new Uint32Array(indices);
  const read16 = (base, i, count) => bytes[base + i] | bytes[base + count + i] << 8;
  const read32 = (base, i) => bytes[base + i] | bytes[base + rings + i] << 8 | bytes[base + rings * 2 + i] << 16 | bytes[base + rings * 3 + i] << 24;
  let cx = 0, cy = 0, cz = 0;
  const q = new Float64Array(4);
  for (let i = 0; i < rings; i++) {
    cx += read32(centerOffset, i); cy += read32(centerOffset + rings * 4, i); cz += read32(centerOffset + rings * 8, i);
    const omitted = bytes[omittedOffset + i];
    let component = 0, square = 0;
    for (let k = 0; k < 4; k++) {
      if (k === omitted) continue;
      const packed = read16(frameOffset + component++ * rings * 2, i, rings);
      q[k] = (packed << 16 >> 16) / frameScale;
      square += q[k] * q[k];
    }
    q[omitted] = Math.sqrt(Math.max(0, 1 - square));
    const [x, y, z, w] = q;
    const ux = 1 - 2 * (y * y + z * z), uy = 2 * (x * y + z * w), uz = 2 * (x * z - y * w);
    const vx = 2 * (x * y - z * w), vy = 1 - 2 * (x * x + z * z), vz = 2 * (y * z + x * w);
    const radius = read16(radiusOffset, i, rings) * radiusStep;
    const opacity = bytes[opacityOffset + i];
    for (let side = 0; side < 3; side++) {
      const u = side === 0 ? 1 : -0.5, v = side === 0 ? 0 : side === 1 ? 0.8660254 : -0.8660254;
      const nx = u * ux + v * vx, ny = u * uy + v * vy, nz = u * uz + v * vz;
      const vertex = i * 3 + side, j = vertex * 3;
      position[j] = cx * centerStep + nx * radius; position[j + 1] = cy * centerStep + ny * radius; position[j + 2] = cz * centerStep + nz * radius;
      normal[j] = nx; normal[j + 1] = ny; normal[j + 2] = nz;
      color[vertex * 4] = color[vertex * 4 + 1] = color[vertex * 4 + 2] = 255;
      color[vertex * 4 + 3] = opacity;
    }
  }
  let ring = 0, cursor = 0;
  for (let strand = 0; strand < strands; strand++) {
    const length = read16(0, strand, strands);
    for (let j = 0; j < length - 1; j++) {
      const base = (ring + j) * 3;
      for (let side = 0; side < 3; side++) {
        const a = base + side, b = base + (side + 1) % 3;
        index[cursor++] = a; index[cursor++] = b; index[cursor++] = a + 3;
        index[cursor++] = b; index[cursor++] = b + 3; index[cursor++] = a + 3;
      }
    }
    ring += length;
  }
  return { position, normal, color, index };
}

// 1d. occlusion textures whose channel is empty would multiply every lit surface by zero
if (noOcclusion) {
  let dropped = 0;
  for (const material of root.listMaterials()) if (material.getOcclusionTexture()) { material.setOcclusionTexture(null); dropped++; }
  console.log(`dropped ${dropped} occlusion textures`);
}

// 1b. primitives of unwanted materials (a baked sky dome that would hide the environment)
if (dropMaterials.length) {
  let dropped = 0;
  for (const mesh of root.listMeshes()) {
    for (const primitive of mesh.listPrimitives()) {
      const material = primitive.getMaterial();
      if (material && dropMaterials.includes(material.getName())) { mesh.removePrimitive(primitive); primitive.dispose(); dropped++; }
    }
  }
  // a mesh left without primitives goes too (gltfio reads past the end of an empty primitive list)
  let emptied = 0;
  for (const mesh of root.listMeshes()) {
    if (mesh.listPrimitives().length) continue;
    for (const node of root.listNodes()) if (node.getMesh() === mesh) node.setMesh(null);
    mesh.dispose(); emptied++;
  }
  console.log(`dropped ${dropped} primitives with materials ${dropMaterials.join(', ')} (${emptied} meshes emptied)`);
}

// 1c0. a clearcoat with a zero factor does nothing, but keeps Filament's ubershaders from combining the material's
//      transmission / IOR / sheen with it (an eye's tear meniscus would render as an opaque plane)
{
  let dropped = 0;
  for (const material of root.listMaterials()) {
    const clearcoat = material.getExtension('KHR_materials_clearcoat');
    if (clearcoat && clearcoat.getClearcoatFactor() === 0) { material.setExtension('KHR_materials_clearcoat', null); clearcoat.dispose(); dropped++; }
  }
  if (dropped) console.log(`dropped ${dropped} zero clearcoats`);
}

// 1c. vertex attributes the renderer has no use for (Blender exports carry spare color / uv sets, and their
//     materials may sample TEXCOORD_2 / _3): Filament reads COLOR_0 and TEXCOORD_0 / _1 and allows eight vertex
//     buffers per primitive, so the uv sets a material samples are renumbered to 0 / 1 and the rest dropped
{
  let dropped = 0, remapped = 0;
  const textureInfos = (material) => {
    const infos = [material.getBaseColorTextureInfo(), material.getMetallicRoughnessTextureInfo(), material.getNormalTextureInfo(),
      material.getOcclusionTextureInfo(), material.getEmissiveTextureInfo()];
    for (const ext of material.listExtensions()) {
      for (const key of Object.getOwnPropertyNames(Object.getPrototypeOf(ext))) {
        if (/^get\w+TextureInfo$/.test(key)) infos.push(ext[key]());
      }
    }
    return infos.filter((info) => info);
  };
  for (const material of root.listMaterials()) {
    const infos = textureInfos(material);
    const used = [...new Set(infos.map((info) => info.getTexCoord()))].sort((a, b) => a - b);
    if (used.length > 2) console.warn(`${material.getName()}: samples ${used.length} uv sets, only two are kept`);
    const mapping = new Map(used.slice(0, 2).map((old, i) => [old, i]));
    for (const info of infos) info.setTexCoord(mapping.get(info.getTexCoord()) ?? 0);
    for (const mesh of root.listMeshes()) {
      for (const primitive of mesh.listPrimitives()) {
        if (primitive.getMaterial() !== material) continue;
        const uvs = new Map();
        for (const semantic of primitive.listSemantics()) {
          const m = /^TEXCOORD_(\d+)$/.exec(semantic);
          if (m) uvs.set(+m[1], primitive.getAttribute(semantic));
        }
        for (const old of uvs.keys()) primitive.setAttribute(`TEXCOORD_${old}`, null);   // clear first: 2 -> 0 must not be undone by dropping 0
        for (const [old, accessor] of uvs) {
          const to = mapping.get(old);
          if (to === undefined) { dropped++; continue; }
          if (to !== old) remapped++;
          primitive.setAttribute(`TEXCOORD_${to}`, accessor);
        }
      }
    }
  }
  for (const mesh of root.listMeshes()) {
    for (const primitive of mesh.listPrimitives()) {
      for (const semantic of primitive.listSemantics()) {
        const m = /^COLOR_(\d+)$/.exec(semantic);
        if (m && +m[1] > 0) { primitive.setAttribute(semantic, null); dropped++; }
      }
    }
  }
  if (dropped || remapped) console.log(`dropped ${dropped} spare vertex attribute sets, renumbered ${remapped} uv sets`);
}

// 2. quantized attributes back to float, one buffer
await document.transform(dequantize(), unpartition(), prune());

// 3. textures are written as separate files next to a .gltf; scripts/convert_scene_textures.py turns the WebP
//    ones into PNG (sharp's libvips build on this machine cannot decode them) and patches the JSON
// 4. drop the compression extensions on write (the data is decoded in memory)
for (const ext of root.listExtensionsUsed()) {
  const name = ext.extensionName;
  if (name === 'KHR_draco_mesh_compression' || name === 'EXT_meshopt_compression' || name === 'EXT_texture_webp' || name === 'KHR_mesh_quantization') ext.dispose();
}
await io.write(output, document);
console.log(`wrote ${output}: ${root.listNodes().length} nodes, extensions ${root.listExtensionsUsed().map((e) => e.extensionName).join(', ') || 'none'}`);
