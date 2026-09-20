(() => {
  var __create = Object.create;
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getProtoOf = Object.getPrototypeOf;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
    get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
  }) : x)(function(x) {
    if (typeof require !== "undefined") return require.apply(this, arguments);
    throw Error('Dynamic require of "' + x + '" is not supported');
  });
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
    // If the importer is in node compatibility mode or this is not an ESM
    // file that has been converted to a CommonJS file using a Babel-
    // compatible transform (i.e. "__esModule" has not been set), then set
    // "default" to the CommonJS "module.exports" for node compatibility.
    isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
    mod
  ));
  var __decorateClass = (decorators, target, key, kind) => {
    var result = kind > 1 ? void 0 : kind ? __getOwnPropDesc(target, key) : target;
    for (var i = decorators.length - 1, decorator; i >= 0; i--)
      if (decorator = decorators[i])
        result = (kind ? decorator(target, key, result) : decorator(result)) || result;
    if (kind && result) __defProp(target, key, result);
    return result;
  };

  // src/webgi-global.ts
  var api = globalThis;
  if (!api.ViewerApp || !api.GenericFilterPlugin || !api.Mesh) {
    throw new Error("WebGI 0.22.1 CDN bundle must load before the DLSS add-on");
  }
  var {
    ACESFilmicToneMapping,
    AgXToneMapping,
    BackSide,
    BaseGroundPlugin,
    BasicShadowMap,
    Box3,
    BufferAttribute,
    BufferGeometry,
    Camera,
    CanvasTexture,
    Color,
    ContactShadowGroundPlugin,
    DataTexture,
    DebugPlugin,
    DirectionalLight,
    DirectionalLight2,
    DiamondPlugin,
    DoubleSide,
    DynamicDrawUsage,
    Float32BufferAttribute,
    FloatType,
    FogExp2,
    FrontSide,
    Frustum,
    GBufferPlugin,
    GenericFilterPlugin,
    GemRefractionPlugin,
    GLSL3,
    GLTFMeshOptPlugin,
    Group,
    HalfFloatType,
    InstancedMesh,
    KTX2LoadPlugin,
    Light,
    LinearFilter,
    LinearMipmapLinearFilter,
    Material,
    MathUtils,
    Matrix3,
    Matrix4,
    Mesh,
    MeshDepthMaterial,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    MeshStandardMaterial2,
    NearestFilter,
    NeutralToneMapping,
    NoBlending,
    NoColorSpace,
    Object3D,
    onChange,
    OrthographicCamera,
    PCFShadowMap,
    PCFSoftShadowMap,
    PerspectiveCamera,
    Plane,
    PropertyBinding,
    ProgressivePlugin,
    RedFormat,
    Reflector2,
    RGBAFormat,
    RGFormat,
    Scene,
    serialize,
    setThreeRendererMode,
    shaderReplaceString,
    ShadowMapBaker,
    ShaderMaterial,
    Side,
    Sphere,
    SpotLight,
    SRGBColorSpace,
    SSAOPlugin,
    SSBevelPlugin,
    SSContactShadows,
    TemporalAAPlugin,
    Texture,
    TonemapPlugin,
    uiFolder,
    uiToggle,
    UnsignedByteType,
    UnsignedIntType,
    Vector2,
    Vector3,
    Vector4,
    ViewerApp,
    VSMShadowMap,
    WebGLRenderer,
    WebGLRenderTarget,
    addBasePlugins,
    mergeGeometries
  } = api;

  // src/webgi-addon/passes/Pass.ts
  var Pass = class {
    constructor() {
      this.isPass = true;
      this.enabled = true;
      this.needsSwap = true;
      this.clear = false;
      this.renderToScreen = false;
    }
    setSize(_width, _height) {
    }
    render(..._args) {
      throw new Error("Pass.render must be implemented");
    }
    dispose() {
    }
  };
  var camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  var geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute([-1, 3, 0, -1, -1, 0, 3, -1, 0], 3));
  geometry.setAttribute("uv", new Float32BufferAttribute([0, 2, 0, 0, 2, 0], 2));
  var FullScreenQuad = class {
    constructor(material) {
      this.mesh = new Mesh(geometry, material);
    }
    dispose() {
      this.mesh.geometry.dispose();
    }
    render(renderer) {
      renderer.render(this.mesh, camera);
    }
    get material() {
      return this.mesh.material;
    }
    set material(value) {
      this.mesh.material = value;
    }
  };

  // src/webgi-addon/passes/RenderPass.ts
  var RenderPass = class extends Pass {
    constructor(scene, camera2, overrideMaterial = null, clearColor = null, clearAlpha = null) {
      super();
      this.clearDepth = false;
      this.oldClearColor = new Color();
      this.scene = scene;
      this.camera = camera2;
      this.overrideMaterial = overrideMaterial;
      this.clearColor = clearColor;
      this.clearAlpha = clearAlpha;
      this.clear = true;
      this.needsSwap = false;
    }
    render(renderer, _writeBuffer, readBuffer, _deltaTime, _maskActive, depthRenderBuffer) {
      if (!this.scene || !this.camera) return;
      const oldAutoClear = renderer.autoClear;
      renderer.autoClear = false;
      let oldClearAlpha;
      let oldOverrideMaterial;
      if (this.overrideMaterial !== null) {
        oldOverrideMaterial = this.scene.overrideMaterial;
        this.scene.overrideMaterial = this.overrideMaterial;
      }
      if (this.clearColor !== null) {
        renderer.getClearColor(this.oldClearColor);
        renderer.setClearColor(this.clearColor, renderer.getClearAlpha());
      }
      if (this.clearAlpha !== null) {
        oldClearAlpha = renderer.getClearAlpha();
        renderer.setClearAlpha(this.clearAlpha);
      }
      if (this.clearDepth) renderer.clearDepth();
      renderer.setRenderTarget(this.renderToScreen ? null : readBuffer);
      if (depthRenderBuffer) {
        const gl = renderer.getContext();
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRenderBuffer);
      }
      if (this.clear) renderer.clear(renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil);
      renderer.render(this.scene, this.camera);
      if (depthRenderBuffer) {
        const gl = renderer.getContext();
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, null);
      }
      if (this.clearColor !== null) renderer.setClearColor(this.oldClearColor);
      if (this.clearAlpha !== null) renderer.setClearAlpha(oldClearAlpha);
      if (this.overrideMaterial !== null) this.scene.overrideMaterial = oldOverrideMaterial;
      renderer.autoClear = oldAutoClear;
    }
  };

  // src/webgi-addon/shaders/ssVelocityVert.glsl
  var ssVelocityVert_default = "#ifdef USE_ALPHAMAP\r\n#define USE_UV\r\n#endif\r\n#include <batching_pars_vertex>\r\n#include <uv_pars_vertex>\r\n#include <morphtarget_pars_vertex>\r\n#include <skinning_pars_vertex>\r\n#include <logdepthbuf_pars_vertex>\r\n#include <clipping_planes_pars_vertex>\r\n\r\n//varying vec3 vViewPosition;\r\n\r\nvarying vec3 vWorldPosition;\r\nvarying vec3 vWorldPositionPrevious;\r\n\r\nuniform mat4 modelMatrixPrevious;\r\n\r\nvoid main() {\r\n\r\n    #include <uv_vertex>\r\n    #include <batching_vertex>\r\n    #include <skinbase_vertex>\r\n\r\n    #include <begin_vertex>\r\n    #include <morphtarget_vertex>\r\n    #include <skinning_vertex>\r\n    #include <displacementmap_vertex>\r\n\r\n    // project_vertex\r\n\r\n    vec4 mvPosition = vec4( transformed, 1.0 );\r\n\r\n    #ifdef USE_INSTANCING\r\n\r\n    mvPosition = instanceMatrix * mvPosition;\r\n\r\n    #endif\r\n\r\n    vWorldPosition = (modelMatrix * mvPosition).xyz;\r\n    vWorldPositionPrevious = (modelMatrixPrevious * mvPosition).xyz;\r\n\r\n    mvPosition = modelViewMatrix * mvPosition;\r\n\r\n    gl_Position = projectionMatrix * mvPosition;\r\n\r\n    #include <logdepthbuf_vertex>\r\n    #include <clipping_planes_vertex>\r\n\r\n//    vViewPosition = - mvPosition.xyz;\r\n\r\n}\r\n";

  // src/webgi-addon/shaders/ssVelocityFrag.glsl
  var ssVelocityFrag_default = "varying vec3 vWorldPosition;\r\nvarying vec3 vWorldPositionPrevious;\r\nuniform mat4 currentProjectionViewMatrix;\r\nuniform mat4 lastProjectionViewMatrix;\r\n\r\nvec2 computeScreenSpaceVelocity2() {\r\n    vec4 currentPositionClip = currentProjectionViewMatrix * vec4(vWorldPosition, 1.0);\r\n    vec4 prevPositionClip = lastProjectionViewMatrix * vec4(vWorldPositionPrevious, 1.0);\r\n\r\n    vec2 currentPositionNDC = currentPositionClip.xy / currentPositionClip.w;\r\n    vec2 prevPositionNDC = prevPositionClip.xy / prevPositionClip.w;\r\n\r\n    if(prevPositionNDC.x >= 1.0 || prevPositionNDC.x <= -1.0 || prevPositionNDC.y >= 1.0 || prevPositionNDC.y <= -1.0) {\n        return vec2(0.0);\n    }\n    // DLSS expects current-to-previous motion: from the current pixel to\n    // where that surface point was in the previous frame.\n    return 0.5 * (prevPositionNDC - currentPositionNDC);\n}\n\r\nvoid main() {\n    vec2 velocity = clamp(computeScreenSpaceVelocity2(), -1.0, 1.0);\n#ifdef VELOCITY_RAW\n    gl_FragColor = vec4(velocity, 0.0, 1.0);\n#else\n    velocity = sign(velocity) * pow(abs(velocity), vec2(1./4.));\n    velocity = velocity * 0.5 + 0.5;\n    gl_FragColor = vec4(velocity.x, velocity.y, 1., 1.);\n#endif\n\n//    float speed = length(computeScreenSpaceVelocity2());\n//    gl_FragColor = vec4(speed, speed, speed, 1.);\r\n}\r\n";

  // src/webgi-addon/passes/DlssVelocityPass.ts
  var DlssVelocityPass = class extends RenderPass {
    constructor(scene, camera2, overrideMaterial, rawOutput = false) {
      super(
        scene,
        camera2,
        overrideMaterial ?? new SSVelocityMaterial(rawOutput),
        rawOutput ? new Color(0, 0, 0) : new Color(0.5, 0.5, 0.5),
        1
      );
      this.enabled = true;
      this._firstCall = true;
    }
    render(renderer, writeBuffer, readBuffer, deltaTime, maskActive) {
      if (!this.enabled || !this.camera) return;
      const mat = this.overrideMaterial;
      this.camera.updateMatrixWorld(true);
      const projection = this.camera.userData.dlssUnjitteredProjectionMatrix ?? this.camera.projectionMatrix;
      mat.uniforms.currentProjectionViewMatrix.value.copy(projection).multiply(this.camera.matrixWorldInverse);
      if (this._firstCall) {
        mat.uniforms.lastProjectionViewMatrix.value.copy(mat.uniforms.currentProjectionViewMatrix.value);
        this._firstCall = false;
      }
      mat.uniformsNeedUpdate = true;
      super.render(renderer, writeBuffer, readBuffer, deltaTime, maskActive);
      mat.uniforms.lastProjectionViewMatrix.value.copy(mat.uniforms.currentProjectionViewMatrix.value);
    }
  };
  __decorateClass([
    uiToggle("Enabled")
  ], DlssVelocityPass.prototype, "enabled", 2);
  DlssVelocityPass = __decorateClass([
    uiFolder("Velocity Buffer (TAA)")
  ], DlssVelocityPass);
  var SSVelocityMaterial = class extends ShaderMaterial {
    constructor(rawOutput = false) {
      super({
        vertexShader: ssVelocityVert_default,
        fragmentShader: ssVelocityFrag_default,
        uniforms: {
          cameraNearFar: { value: new Vector2(0.1, 1e3) },
          alphaMap: { value: null },
          alphaTest: { value: null },
          alphaMapTransform: { value: /* @__PURE__ */ new Matrix3() },
          currentProjectionViewMatrix: { value: new Matrix4() },
          lastProjectionViewMatrix: { value: new Matrix4() }
        },
        defines: rawOutput ? { VELOCITY_RAW: 1 } : {}
      });
      this.extraUniformsToUpload = {
        modelMatrixPrevious: { value: new Matrix4().identity() }
      };
      this._previousWorldMatrices = {};
    }
    // this gets called for each object.
    onBeforeRender(renderer, s, c, geometry2, object, group) {
      super.onBeforeRender(renderer, s, c, geometry2, object, group);
      const prevMatrix = this._previousWorldMatrices[object.uuid];
      this.extraUniformsToUpload.modelMatrixPrevious.value.copy(prevMatrix ?? object.matrixWorld);
      if (prevMatrix) {
        prevMatrix.copy(object.matrixWorld);
      } else {
        this._previousWorldMatrices[object.uuid] = object.matrixWorld.clone();
      }
      let mat = object.material;
      if (Array.isArray(mat)) {
        mat = mat[0];
      }
      this.uniforms.alphaMap.value = mat?.alphaMap ?? null;
      this.uniforms.alphaTest.value = !mat || !mat.alphaTest || mat.alphaTest < 1e-7 ? 1e-3 : mat.alphaTest;
      let x = this.uniforms.alphaMap.value ? 1 : void 0;
      if (x !== this.defines.USE_ALPHAMAP) {
        if (x === void 0) {
          delete this.defines.USE_ALPHAMAP;
          delete this.defines.ALPHAMAP_UV;
        } else {
          this.defines.USE_ALPHAMAP = x;
          this.defines.ALPHAMAP_UV = "uv";
        }
        this.needsUpdate = true;
      }
      x = mat.userData.ALPHA_I_RGBA_PACKING ? 1 : void 0;
      if (x !== this.defines.ALPHA_I_RGBA_PACKING) {
        if (x === void 0) delete this.defines.ALPHA_I_RGBA_PACKING;
        else this.defines.ALPHA_I_RGBA_PACKING = x;
        this.needsUpdate = true;
      }
      this.side = mat.side ?? DoubleSide;
    }
  };

  // src/webgi-addon/DlssVelocityBufferPlugin.ts
  var DlssVelocityBufferPlugin = class _DlssVelocityBufferPlugin extends GenericFilterPlugin {
    constructor(enabled = true, includeTransparent = false, continuous = false, rawOutput = false, exposeToPostProcessing = true) {
      super();
      this.includeTransparent = includeTransparent;
      this.continuous = continuous;
      this.rawOutput = rawOutput;
      this.exposeToPostProcessing = exposeToPostProcessing;
      this.passId = "velocityBuffer";
      this._beforeFilters = ["render"];
      this._afterFilters = [];
      this._requiredFilters = ["render"];
      this._velocityBuffers = [];
      this.enabled = enabled;
    }
    static {
      this.PluginType = "VelocityBuffer";
    }
    // private _velocityBufferPass?: IFilter<SSVelocityPass>
    passCtor(v) {
      const rawOutput = this.rawOutput;
      const target = v.renderer.createTarget({
        depthBuffer: true,
        type: rawOutput ? HalfFloatType : UnsignedByteType,
        ...rawOutput ? { format: RGFormat } : {}
      });
      target.texture.name = "velocityBuffer";
      this._velocityBuffers.push(target);
      const debug = v.getPluginByType("debug");
      if (debug) {
      }
      const transparentMats = /* @__PURE__ */ new Set();
      const transmissiveMats = /* @__PURE__ */ new Set();
      const includeTransparent = this.includeTransparent;
      const continuous = this.continuous;
      return new class DlssVelocityPass2 extends DlssVelocityPass {
        render(renderer, writeBuffer, readBuffer, deltaTime, maskActive) {
          if (!continuous && v.renderer.frameCount > 0) return;
          const t = renderer.getRenderTarget();
          const activeCubeFace = renderer.getActiveCubeFace();
          const activeMipLevel = renderer.getActiveMipmapLevel();
          this.scene?.traverse(({ material }) => {
            if (!material) return;
            const forceRender = includeTransparent || material.userData.renderToDepth && !material.userData.pluginsDisabled && !material.userData[_DlssVelocityBufferPlugin.PluginType]?.disabled;
            const doNotRender = material.userData.renderToDepth === false || material.userData.pluginsDisabled || material.userData[_DlssVelocityBufferPlugin.PluginType]?.disabled;
            if (material.transparent && forceRender || // transparent and render to depth
            !material.transparent && !material.transmission && doNotRender) {
              transparentMats.add(material);
              material.transparent = !material.transparent;
            }
            if (Math.abs(material.transmission || 0) > 0 && forceRender) {
              transmissiveMats.add([material, material.transmission]);
              material.transmission = 0;
            }
          });
          setThreeRendererMode(renderer, {
            shadowMapRender: false,
            backgroundRender: false,
            opaqueRender: true,
            transparentRender: false,
            transmissionRender: false,
            mainRenderPass: false
          }, () => super.render(renderer, writeBuffer, target, deltaTime, maskActive));
          transparentMats.forEach((m) => m.transparent = !m.transparent);
          transparentMats.clear();
          transmissiveMats.forEach(([m, tr]) => m.transmission = tr);
          transmissiveMats.clear();
          renderer.setRenderTarget(t, activeCubeFace, activeMipLevel);
        }
      }(void 0, void 0, void 0, rawOutput);
    }
    _update(v) {
      if (!super._update(v)) return false;
      if (!this.continuous && v.renderer.frameCount > 0) return false;
      const pass = this.pass.passObject;
      pass.scene = v.scene.modelObject;
      v.scene.renderCamera.updateShaderProperties(pass.overrideMaterial);
      pass.camera = v.scene.renderCamera.cameraObject;
      return true;
    }
    getVelocityBuffer() {
      return this._velocityBuffers.length > 0 ? this._velocityBuffers[0] : void 0;
    }
    async onDispose(viewer) {
      return;
    }
    async onRemove(viewer) {
      this._velocityBuffers.forEach((value) => viewer.renderer.disposeTarget(value?.dispose?.()));
      return super.onRemove(viewer);
    }
    updateShaderProperties(material) {
      if (!this.exposeToPostProcessing) return this;
      if (material.uniforms.tVelocity) material.uniforms.tVelocity.value = this.enabled ? this.getVelocityBuffer()?.texture ?? null : null;
      else console.warn("BaseRenderer: no uniform: tVelocity");
      return this;
    }
    get uiConfig() {
      return this.pass?.passObject.uiConfig;
    }
  };

  // src/webgi-addon/passes/CopyShader.ts
  var CopyShader = {
    vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `
  };

  // src/webgi-addon/mathutil.ts
  function updateBit(value, bit, enabled) {
    const mask = 1 << bit;
    return enabled ? value | mask : value & ~mask;
  }

  // src/webgi-addon/DlssBridgePlugin.ts
  var DlssBridgePlugin = class extends GenericFilterPlugin {
    constructor(renderScale = 2 / 3, jitterEnabled = true, preserveWebGiTemporal = false, displayScale = renderScale, colorOnly = false) {
      super();
      this.renderScale = renderScale;
      this.jitterEnabled = jitterEnabled;
      this.preserveWebGiTemporal = preserveWebGiTemporal;
      this.displayScale = displayScale;
      this.colorOnly = colorOnly;
      this.passId = "dlssBridgeExport";
      this._beforeFilters = ["combinedPost"];
      this._afterFilters = ["render", "taa", "progressive", "bloom"];
      this._requiredFilters = ["render", "combinedPost"];
      this._frameIndex = 0;
      this._unjitteredProjection = new Matrix4();
      this._jitter = new Vector2();
      this._previousJitter = new Vector2();
      this._serializationIgnoresAdded = [];
      /** Frame generation: publish full-scale depth and motion export targets each frame (the presented
       *  colour is taken from the canvas by the FG runtime); no SR scaling, no NR temporal history. */
      this.fgMode = false;
      /** Browser Performance/J consumes color, depth and motion only. */
      this.srOnly = false;
      /** Feature 18 consumes color and motion; its default pre-kernel skips depth. */
      this.nrTemporal = false;
      this._updateGBuffer = (object, data) => {
        if (!(object instanceof Mesh) || !object.material) return;
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        const isDiamond = materials.some((material) => material?.isDiamondMaterial);
        data.w = updateBit(data.w, 4, isDiamond ? 0 : 1);
      };
      this._captureNrProjection = () => {
        if (!this.nrTemporal || !this._viewer) return;
        const camera2 = this._viewer.scene.activeCamera.cameraObject;
        if (!(camera2 instanceof PerspectiveCamera)) return;
        this._nrCamera = camera2;
        this._unjitteredProjection.copy(camera2.projectionMatrix);
        camera2.userData.dlssUnjitteredProjectionMatrix = this._unjitteredProjection;
      };
      this._applyCameraJitter = () => {
        if (!this.jitterEnabled || !this._viewer) return;
        const camera2 = this._viewer.scene.activeCamera.cameraObject;
        if (!(camera2 instanceof PerspectiveCamera)) return;
        const width = Math.max(1, Math.floor(this._viewer.renderer.renderSize.width * this.displayScale));
        const height = Math.max(1, Math.floor(this._viewer.renderer.renderSize.height * this.displayScale));
        this._camera = camera2;
        this._unjitteredProjection.copy(camera2.projectionMatrix);
        camera2.userData.dlssUnjitteredProjectionMatrix = this._unjitteredProjection;
        const sampleIndex = this._frameIndex++ % 32 + 1;
        this._previousJitter.copy(this._jitter);
        this._jitter.set(halton(sampleIndex, 2) - 0.5, halton(sampleIndex, 3) - 0.5);
        camera2.setViewOffset(width, height, this._jitter.x, this._jitter.y, width, height);
      };
      this._clearCameraJitter = () => {
        if (this._nrCamera) {
          delete this._nrCamera.userData.dlssUnjitteredProjectionMatrix;
          this._nrCamera = void 0;
        }
        if (!this._camera) return;
        this._camera.clearViewOffset();
        delete this._camera.userData.dlssUnjitteredProjectionMatrix;
        this._camera = void 0;
      };
    }
    static {
      this.PluginType = "DlssBridge";
    }
    passCtor(viewer) {
      return new DlssBridgeExportPass(viewer);
    }
    _update(viewer) {
      const velocity = viewer.getPlugin(DlssVelocityBufferPlugin);
      if (!this.colorOnly && velocity && !velocity.enabled) {
        velocity.enabled = true;
        console.info("[dlss-bridge] restored velocity buffer after scene import");
      }
      const progressive = viewer.getPlugin(ProgressivePlugin);
      const taa = viewer.getPlugin(TemporalAAPlugin);
      if (this.preserveWebGiTemporal) {
        if (progressive && !progressive.enabled) progressive.enabled = true;
        if (taa && !taa.enabled) taa.enabled = true;
      } else {
        if (progressive?.enabled) {
          progressive.enabled = false;
          console.info("[dlss-bridge] disabled progressive history restored by scene import");
        }
        if (taa?.enabled) {
          taa.enabled = false;
          console.info("[dlss-bridge] disabled WebGI TAA restored by scene import");
        }
      }
      return super._update(viewer);
    }
    async onAdded(viewer) {
      await super.onAdded(viewer);
      if (this.pass) {
        Object.defineProperty(this.pass, "dirty", {
          configurable: true,
          get: () => this.enabled,
          set: () => void 0
        });
      }
      if (viewer.useRgbm) throw new Error("DlssBridgePlugin requires useRgbm=false for a linear RGBA16F source");
      const progressive = viewer.getPlugin(ProgressivePlugin);
      if (progressive) {
        progressive.jitter = this.preserveWebGiTemporal;
        progressive.enabled = this.preserveWebGiTemporal;
      }
      const taa = viewer.getPlugin(TemporalAAPlugin);
      if (taa) taa.enabled = this.preserveWebGiTemporal;
      const velocity = viewer.getPlugin(DlssVelocityBufferPlugin);
      if (velocity && !this.colorOnly) velocity.enabled = true;
      const gbuffer = viewer.getPlugin(GBufferPlugin);
      gbuffer?.registerGBufferUpdater(this._updateGBuffer);
      const flags = gbuffer?.getFlagsTexture();
      if (flags) {
        flags.minFilter = NearestFilter;
        flags.magFilter = NearestFilter;
        flags.needsUpdate = true;
      }
      const ownedPluginTypes = this.preserveWebGiTemporal ? [DlssVelocityBufferPlugin.PluginType] : [DlssVelocityBufferPlugin.PluginType, ProgressivePlugin.PluginType, TemporalAAPlugin.PluginType];
      for (const type of ownedPluginTypes) {
        if (!viewer.serializePluginsIgnored.includes(type)) {
          viewer.serializePluginsIgnored.push(type);
          this._serializationIgnoresAdded.push(type);
        }
      }
      viewer.renderer.displayCanvasScaling = this.displayScale;
      viewer.addEventListener("preRender", this._applyCameraJitter);
      viewer.addEventListener("preFrame", this._captureNrProjection);
      viewer.addEventListener("postRender", this._clearCameraJitter);
    }
    async onRemove(viewer) {
      viewer.removeEventListener("preRender", this._applyCameraJitter);
      viewer.removeEventListener("preFrame", this._captureNrProjection);
      viewer.removeEventListener("postRender", this._clearCameraJitter);
      viewer.getPlugin(GBufferPlugin)?.unregisterGBufferUpdater?.(this._updateGBuffer);
      for (const type of this._serializationIgnoresAdded) {
        const index = viewer.serializePluginsIgnored.indexOf(type);
        if (index >= 0) viewer.serializePluginsIgnored.splice(index, 1);
      }
      this._serializationIgnoresAdded = [];
      this._clearCameraJitter();
      return super.onRemove(viewer);
    }
    setRenderScale(scale) {
      const next = Math.max(0.25, Math.min(1, scale));
      if (Math.abs(next - this.renderScale) < 1e-4) return;
      this.renderScale = next;
      this._frameIndex = 0;
      if (this._viewer) {
        if (!this.preserveWebGiTemporal) this.displayScale = next;
        this._viewer.renderer.displayCanvasScaling = this.displayScale;
        this._viewer.setDirty(this);
      }
    }
    get jitter() {
      return this._jitter;
    }
    get previousJitter() {
      return this._previousJitter;
    }
    /**
     * Installs an alternate consumer for the exact resources exported for the
     * native feeder. The Electron path does not install one and is unchanged.
     */
    setFrameConsumer(consumer) {
      this._frameConsumer = consumer;
    }
    /** Discard the preceding scene's temporal inputs after an in-place load. */
    resetHistory() {
      this.pass?.passObject.resetHistory();
      this._frameIndex = 0;
      this._jitter.set(0, 0);
      this._previousJitter.set(0, 0);
    }
    publishFrame(frame) {
      if (!this._frameConsumer) return;
      try {
        this._frameConsumer(frame);
      } catch (error) {
        console.error("[dlss-bridge] frame consumer failed", error);
      }
    }
  };
  function halton(index, base) {
    let fraction = 1;
    let result = 0;
    while (index > 0) {
      fraction /= base;
      result += fraction * (index % base);
      index = Math.floor(index / base);
    }
    return result;
  }
  var DlssBridgeExportPass = class extends Pass {
    constructor(viewer) {
      super();
      this._width = 0;
      this._height = 0;
      this._outputWidth = 0;
      this._outputHeight = 0;
      this._historyValid = false;
      this._viewer = viewer;
      this.needsSwap = false;
      this.clear = false;
      this._colorMaterial = makeMaterial(`
            uniform sampler2D tInput;
            varying vec2 vUv;
            void main() { gl_FragColor = vec4(texture2D(tInput, vUv).rgb, 1.0); }
        `, { tInput: { value: null } });
      this._depthMaterial = makeMaterial(`
            uniform sampler2D tDepth;
            varying vec2 vUv;
            void main() { gl_FragColor = vec4(texture2D(tDepth, vUv).r, 0.0, 0.0, 1.0); }
        `, { tDepth: { value: null } });
      this._motionMaterial = makeMaterial(`
            uniform sampler2D tVelocity;
            varying vec2 vUv;
            void main() {
                gl_FragColor = vec4(texture2D(tVelocity, vUv).xy, 0.0, 1.0);
            }
        `, { tVelocity: { value: null } });
      this._reactiveMaterial = makeMaterial(`
            uniform sampler2D tInput;
            uniform sampler2D tPrevious;
            uniform sampler2D tDepth;
            uniform sampler2D tFlags;
            uniform sampler2D tVelocity;
            uniform vec2 jitterDeltaUv;
            uniform float historyValid;
            varying vec2 vUv;

            vec3 compressHdr(vec3 value) {
                value = max(value, vec3(0.0));
                return value / (vec3(1.0) + value);
            }

            void main() {
                vec4 color = texture2D(tInput, vUv);
                float depth = texture2D(tDepth, vUv).r;
                float flagByte = floor(texture2D(tFlags, vUv).a * 255.0 + 0.5);
                // GemRefractionPlugin stores an inverted diamond flag in bit 4:
                // zero for diamond geometry, one for ordinary geometry/background.
                float diamond = 1.0 - mod(floor(flagByte / 16.0), 2.0);
                float covered = depth < 0.999999 ? 1.0 : 0.0;
                float transparency = covered * clamp(1.0 - color.a, 0.0, 1.0);

                // A diamond is view-dependent, but that does not make every
                // diamond pixel invalid every frame. Reproject the previous
                // pre-tonemap colour with the real motion vector and only bias
                // DLSS toward the current frame where the refracted/reflected
                // result actually changed. The explicit jitter delta aligns the
                // two jittered colour buffers; velocity itself is jitter-free.
                vec2 previousUv = vUv + texture2D(tVelocity, vUv).xy + jitterDeltaUv;
                float outside = step(previousUv.x, 0.0) + step(1.0, previousUv.x)
                    + step(previousUv.y, 0.0) + step(1.0, previousUv.y);
                float diamondBias = diamond;
                if (historyValid > 0.5) {
                    vec3 previous = texture2D(tPrevious, clamp(previousUv, 0.0, 1.0)).rgb;
                    vec3 delta = abs(compressHdr(color.rgb) - compressHdr(previous));
                    float change = max(delta.r, max(delta.g, delta.b));
                    diamondBias *= max(step(0.5, outside), smoothstep(0.035, 0.22, change));
                }

                gl_FragColor = vec4(max(transparency, diamondBias), 0.0, 0.0, 1.0);
            }
        `, {
        tInput: { value: null },
        tPrevious: { value: null },
        tDepth: { value: null },
        tFlags: { value: null },
        tVelocity: { value: null },
        jitterDeltaUv: { value: new Vector2() },
        historyValid: { value: 0 }
      });
      this._controlMaterial = makeMaterial(`
            uniform sampler2D tDepth;
            varying vec2 vUv;
            void main() {
                float depth = texture2D(tDepth, vUv).r;
                float preserveOriginal = step(0.999999, depth);
                gl_FragColor = vec4(preserveOriginal, 0.0, 0.0, 1.0);
            }
        `, { tDepth: { value: null } });
      this._quad = new FullScreenQuad(this._colorMaterial);
    }
    resetHistory() {
      this._historyValid = false;
    }
    render(renderer, _writeBuffer, readBuffer) {
      const colorBridge = this._viewer.getPlugin(DlssBridgePlugin);
      if (colorBridge?.nrTemporal) {
        const motion = this._viewer.getPlugin(DlssVelocityBufferPlugin)?.getVelocityBuffer();
        if (!motion) throw new Error("Temporal NR requires the velocity buffer");
        colorBridge.publishFrame({
          renderer,
          color: readBuffer,
          motion,
          renderWidth: readBuffer.width,
          renderHeight: readBuffer.height,
          outputWidth: readBuffer.width,
          outputHeight: readBuffer.height,
          jitter: { x: 0, y: 0 },
          previousJitter: { x: 0, y: 0 },
          reset: !this._historyValid
        });
        this._historyValid = true;
        return;
      }
      if (colorBridge?.colorOnly) {
        colorBridge.publishFrame({
          renderer,
          color: readBuffer,
          renderWidth: readBuffer.width,
          renderHeight: readBuffer.height,
          outputWidth: readBuffer.width,
          outputHeight: readBuffer.height,
          jitter: { x: 0, y: 0 },
          previousJitter: { x: 0, y: 0 },
          reset: false
        });
        return;
      }
      const gbuffer = this._viewer.getPlugin(GBufferPlugin);
      const velocity = this._viewer.getPlugin(DlssVelocityBufferPlugin);
      const depthTexture = gbuffer?.getDepthTexture();
      const flagsTexture = gbuffer?.getFlagsTexture();
      const velocityTexture = velocity?.getVelocityBuffer()?.texture;
      if (!depthTexture || !flagsTexture || !velocityTexture) return;
      const bridge = this._viewer.getPlugin(DlssBridgePlugin);
      const outputWidth = bridge?.preserveWebGiTemporal ? readBuffer.width : Math.max(1, Math.floor(this._viewer.renderer.renderSize.width));
      const outputHeight = bridge?.preserveWebGiTemporal ? readBuffer.height : Math.max(1, Math.floor(this._viewer.renderer.renderSize.height));
      this._ensureTargets(readBuffer.width, readBuffer.height, outputWidth, outputHeight);
      if ((bridge?.srOnly || bridge?.fgMode) && this._colorTarget && this._depthTarget && this._motionTarget) {
        this._colorMaterial.uniforms.tInput.value = readBuffer.texture;
        this._depthMaterial.uniforms.tDepth.value = depthTexture;
        this._motionMaterial.uniforms.tVelocity.value = velocityTexture;
        const oldTarget2 = renderer.getRenderTarget();
        this._renderTarget(renderer, this._colorTarget, this._colorMaterial);
        this._renderTarget(renderer, this._depthTarget, this._depthMaterial);
        this._renderTarget(renderer, this._motionTarget, this._motionMaterial);
        renderer.setRenderTarget(oldTarget2);
        bridge.publishFrame({
          renderer,
          color: this._colorTarget,
          depth: this._depthTarget,
          motion: this._motionTarget,
          renderWidth: this._width,
          renderHeight: this._height,
          outputWidth,
          outputHeight,
          jitter: { x: bridge.jitter.x, y: bridge.jitter.y },
          previousJitter: { x: bridge.previousJitter.x, y: bridge.previousJitter.y },
          reset: !this._historyValid
        });
        this._historyValid = true;
        return;
      }
      if (!this._colorTarget || !this._depthTarget || !this._motionTarget || !this._reactiveTarget || !this._controlTarget) return;
      this._colorMaterial.uniforms.tInput.value = readBuffer.texture;
      this._depthMaterial.uniforms.tDepth.value = depthTexture;
      this._motionMaterial.uniforms.tVelocity.value = velocityTexture;
      this._reactiveMaterial.uniforms.tInput.value = readBuffer.texture;
      this._reactiveMaterial.uniforms.tPrevious.value = this._colorTarget.texture;
      this._reactiveMaterial.uniforms.tDepth.value = depthTexture;
      this._reactiveMaterial.uniforms.tFlags.value = flagsTexture;
      this._reactiveMaterial.uniforms.tVelocity.value = velocityTexture;
      this._controlMaterial.uniforms.tDepth.value = depthTexture;
      const jitter = bridge?.jitter;
      const previousJitter = bridge?.previousJitter;
      this._reactiveMaterial.uniforms.jitterDeltaUv.value.set(
        ((jitter?.x ?? 0) - (previousJitter?.x ?? 0)) / this._width,
        -((jitter?.y ?? 0) - (previousJitter?.y ?? 0)) / this._height
      );
      this._reactiveMaterial.uniforms.historyValid.value = this._historyValid ? 1 : 0;
      const oldTarget = renderer.getRenderTarget();
      const oldClearColor = renderer.getClearColor(new Color());
      const oldClearAlpha = renderer.getClearAlpha();
      renderer.setClearColor(0, 0);
      const reset = !this._historyValid;
      if (!this._historyValid) this._renderTarget(renderer, this._colorTarget, this._colorMaterial);
      this._renderTarget(renderer, this._depthTarget, this._depthMaterial);
      this._renderTarget(renderer, this._motionTarget, this._motionMaterial);
      this._renderTarget(renderer, this._reactiveTarget, this._reactiveMaterial);
      this._renderTarget(renderer, this._controlTarget, this._controlMaterial, this._outputWidth, this._outputHeight);
      if (this._historyValid) this._renderTarget(renderer, this._colorTarget, this._colorMaterial);
      this._historyValid = true;
      const exportedFrame = {
        renderer,
        color: this._colorTarget,
        depth: this._depthTarget,
        motion: this._motionTarget,
        reactive: this._reactiveTarget,
        control: this._controlTarget,
        renderWidth: this._width,
        renderHeight: this._height,
        outputWidth: this._outputWidth,
        outputHeight: this._outputHeight,
        jitter: { x: jitter?.x ?? 0, y: jitter?.y ?? 0 },
        previousJitter: { x: previousJitter?.x ?? 0, y: previousJitter?.y ?? 0 },
        reset
      };
      renderer.setRenderTarget(oldTarget);
      renderer.setClearColor(oldClearColor, oldClearAlpha);
      bridge?.publishFrame(exportedFrame);
    }
    _renderTarget(renderer, target, material, usefulWidth = this._width, usefulHeight = this._height) {
      target.scissorTest = false;
      target.viewport.set(0, 0, target.width, target.height);
      renderer.setRenderTarget(target);
      renderer.clear(true, false, false);
      target.viewport.set(0, 0, usefulWidth, usefulHeight);
      renderer.setRenderTarget(target);
      this._quad.material = material;
      this._quad.render(renderer);
    }
    _ensureTargets(width, height, outputWidth, outputHeight) {
      if (!this._colorTarget) {
        this._colorTarget = this._createTarget(width, height + 4, HalfFloatType, RGBAFormat, "WEBGI_DLSS_COLOR_V1");
        this._colorTarget.texture.minFilter = LinearFilter;
        this._colorTarget.texture.magFilter = LinearFilter;
        this._depthTarget = this._createTarget(width, height + 3, FloatType, RedFormat, "WEBGI_DLSS_DEPTH_V1");
        this._motionTarget = this._createTarget(width, height + 2, HalfFloatType, RGFormat, "WEBGI_DLSS_MOTION_V1");
        if (!this._viewer.getPlugin(DlssBridgePlugin)?.srOnly) {
          this._reactiveTarget = this._createTarget(width, height + 1, UnsignedByteType, RedFormat, "WEBGI_DLSS_REACTIVE_V1");
          this._controlTarget = this._createTarget(outputWidth, outputHeight + 5, UnsignedByteType, RedFormat, "WEBGI_DLSS_NR_CONTROL_V1");
        }
      } else if (width !== this._width || height !== this._height || outputWidth !== this._outputWidth || outputHeight !== this._outputHeight) {
        this._colorTarget.setSize(width, height + 4);
        this._depthTarget.setSize(width, height + 3);
        this._motionTarget.setSize(width, height + 2);
        this._reactiveTarget?.setSize(width, height + 1);
        this._controlTarget?.setSize(outputWidth, outputHeight + 5);
        this._historyValid = false;
      }
      this._width = width;
      this._height = height;
      this._outputWidth = outputWidth;
      this._outputHeight = outputHeight;
    }
    _createTarget(width, height, type, format, name) {
      const target = this._viewer.renderer.createTarget({
        size: { width, height },
        type,
        format,
        colorSpace: NoColorSpace,
        depthBuffer: false,
        generateMipmaps: false,
        minFilter: NearestFilter,
        magFilter: NearestFilter
      });
      target.texture.name = name;
      return target;
    }
    setSize() {
    }
    dispose() {
      for (const target of [
        this._colorTarget,
        this._depthTarget,
        this._motionTarget,
        this._reactiveTarget,
        this._controlTarget
      ]) {
        if (target) this._viewer.renderer.disposeTarget(target);
      }
      this._quad.dispose();
      this._colorMaterial.dispose();
      this._depthMaterial.dispose();
      this._motionMaterial.dispose();
      this._reactiveMaterial.dispose();
      this._controlMaterial.dispose();
      super.dispose();
    }
  };
  function makeMaterial(fragmentShader, uniforms) {
    return new ShaderMaterial({
      vertexShader: CopyShader.vertexShader,
      fragmentShader,
      uniforms,
      blending: NoBlending,
      depthTest: false,
      depthWrite: false,
      toneMapped: false
    });
  }

  // src/scenes/nr-bistro-lighting.ts
  async function configureBistroLighting(viewer) {
    const { scene } = viewer;
    const capabilities = viewer.renderer.rendererObject.capabilities;
    const materials = /* @__PURE__ */ new Set();
    const textures = /* @__PURE__ */ new Set();
    const cafeLamps = [];
    scene.modelRoot.traverse((object) => {
      const mesh = object;
      if (!mesh.isMesh) return;
      const meshMaterials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      meshMaterials.forEach((material) => materials.add(material));
      if (meshMaterials.some((material) => material.name === "Emissive_StreetLight")) {
        mesh.geometry.computeBoundingBox();
        const position = mesh.geometry.boundingBox.getCenter(new Vector3()).applyMatrix4(mesh.matrixWorld);
        if (position.y > 3 && position.y < 4.2 && position.x > -4 && position.x < 14 && position.z > -14 && position.z < 9) cafeLamps.push(position);
      }
    });
    for (const material of materials) {
      if (material.aoMap?.name.endsWith("_Specular")) {
        material.aoMap = null;
        material.needsUpdate = true;
      }
      if (material.transparent && material.name.startsWith("Foliage_")) {
        material.transparent = false;
        material.alphaTest = 0.5;
        material.depthWrite = true;
        material.needsUpdate = true;
      }
      for (const texture of [
        material.map,
        material.normalMap,
        material.roughnessMap,
        material.metalnessMap,
        material.aoMap
      ]) {
        if (texture) textures.add(texture);
      }
    }
    for (const texture of textures) {
      texture.anisotropy = Math.min(16, capabilities.getMaxAnisotropy());
      texture.needsUpdate = true;
    }
    const sun = scene.modelRoot.getObjectByName("directionalLight1");
    if (sun?.isDirectionalLight) {
      const bounds = scene.getModelBounds();
      const center = bounds.getCenter(new Vector3());
      const distance = bounds.getSize(new Vector3()).length();
      scene.attach(sun);
      scene.attach(sun.target);
      sun.target.position.copy(center);
      sun.position.copy(center).addScaledVector(new Vector3(-0.62, 0.85, -0.4).normalize(), distance);
      sun.color.set("#ffedd4");
      sun.intensity = 5.5;
      sun.castShadow = true;
      sun.updateMatrixWorld(true);
      sun.target.updateMatrixWorld(true);
      const { shadow } = sun;
      const camera2 = shadow.camera;
      camera2.position.copy(sun.position);
      camera2.lookAt(center);
      camera2.updateMatrixWorld(true);
      const lightBounds = bounds.clone().applyMatrix4(camera2.matrixWorldInverse);
      camera2.left = lightBounds.min.x - 2;
      camera2.right = lightBounds.max.x + 2;
      camera2.bottom = lightBounds.min.y - 2;
      camera2.top = lightBounds.max.y + 2;
      camera2.near = Math.max(0.1, -lightBounds.max.z - 2);
      camera2.far = -lightBounds.min.z + 2;
      camera2.updateProjectionMatrix();
      const shadowSize = Math.min(4096, capabilities.maxTextureSize);
      shadow.mapSize.set(shadowSize, shadowSize);
      shadow.bias = -1e-4;
      shadow.normalBias = 0.04;
      shadow.radius = 1.5;
    }
    scene.environmentIntensity = 0.8;
    for (const [index, position] of cafeLamps.entries()) {
      const lamp = new SpotLight("#ffd09a", 24, 5, Math.PI / 3, 0.75, 2);
      lamp.name = `Bistro cafe pendant ${index + 1}`;
      lamp.position.copy(position).add(new Vector3(0, -0.12, 0));
      lamp.target.position.copy(position).add(new Vector3(0, -3, 0));
      scene.add(lamp, lamp.target);
    }
    viewer.getPlugin(SSAOPlugin).passes.ssao.passObject.parameters.occlusionWorldRadius = 1.2;
    viewer.renderer.resetShadows();
    scene.setDirty();
  }

  // src/asset-url.js
  var ASSET_PATH = /^\/(?:scenes|environments|generated)(?:\/|$)/;
  function assetUrl(path) {
    const value = String(path);
    const base = String(globalThis.__DLSS5_ASSET_BASE__ ?? "").replace(/\/$/, "");
    return base && ASSET_PATH.test(value) ? `${base}${value}` : value;
  }

  // src/scenes/nr-lone-monk-scene.ts
  var convergenceFrames = 128;
  function createPageTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 512;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Cannot create the courtyard page texture");
    const image = context.createImageData(canvas.width, canvas.height);
    let seed = 7281;
    const random = () => {
      seed = Math.imul(1664525, seed) + 1013904223 >>> 0;
      return seed / 4294967296;
    };
    for (let y = 0; y < canvas.height; y++) {
      const line = 0.72 + random() * 0.28;
      for (let x = 0; x < canvas.width; x++) {
        const grain = 0.95 + random() * 0.1;
        const edge = 0.85 + 0.15 * Math.sin(Math.PI * x / canvas.width);
        const value = line * grain * edge;
        const i = (y * canvas.width + x) * 4;
        image.data.set([214 * value, 196 * value, 163 * value, 255], i);
      }
    }
    context.putImageData(image, 0, 0);
    const texture = new CanvasTexture(canvas);
    texture.name = "Lone Monk aged page edges";
    texture.colorSpace = SRGBColorSpace;
    texture.flipY = false;
    return texture;
  }
  function colorRoofTiles(mesh) {
    const geometry2 = mesh.geometry;
    const position = geometry2.getAttribute("position");
    const parent = new Int32Array(position.count);
    const welded = /* @__PURE__ */ new Map();
    const root = (i) => {
      while (parent[i] !== i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
      }
      return i;
    };
    for (let i = 0; i < position.count; i++) {
      parent[i] = i;
      const key = [position.getX(i), position.getY(i), position.getZ(i)].map((v) => Math.round(v * 1e4)).join(",");
      const previous = welded.get(key);
      if (previous === void 0) welded.set(key, i);
      else parent[i] = root(previous);
    }
    const index = geometry2.index;
    const count = index?.count ?? position.count;
    for (let i = 0; i < count; i += 3) {
      const a = index ? index.getX(i) : i;
      const b = index ? index.getX(i + 1) : i + 1;
      const c = index ? index.getX(i + 2) : i + 2;
      parent[root(b)] = root(a);
      parent[root(c)] = root(a);
    }
    const colors = new Float32Array(position.count * 3);
    const palette = ["#a6866b", "#c1a58a", "#c5b49b", "#8c7969", "#b8997a", "#b2a78c"].map((value) => new Color(value));
    for (let i = 0; i < position.count; i++) {
      const hash = Math.imul(root(i) + 17, 2654435761) >>> 0;
      palette[hash % palette.length].toArray(colors, i * 3);
    }
    geometry2.setAttribute("color", new Float32BufferAttribute(colors, 3));
  }
  function placeLight(light, direction, bounds) {
    const center = bounds.getCenter(new Vector3());
    light.position.copy(center).addScaledVector(direction, bounds.getSize(new Vector3()).length());
    light.target.position.copy(center).sub(light.position);
    light.updateMatrixWorld(true);
    const camera2 = light.shadow.camera;
    camera2.position.copy(light.position);
    camera2.lookAt(center);
    camera2.updateMatrixWorld(true);
    const lightBounds = bounds.clone().applyMatrix4(camera2.matrixWorldInverse);
    camera2.left = lightBounds.min.x - 1;
    camera2.right = lightBounds.max.x + 1;
    camera2.bottom = lightBounds.min.y - 1;
    camera2.top = lightBounds.max.y + 1;
    camera2.near = Math.max(0.1, -lightBounds.max.z - 1);
    camera2.far = -lightBounds.min.z + 1;
    camera2.updateProjectionMatrix();
  }
  async function configureLoneMonkScene(viewer, onCleanup) {
    const { scene } = viewer;
    const square = new URLSearchParams(location.search).get("sr") === "1";
    const renderer = viewer.renderer.rendererObject;
    const materials = /* @__PURE__ */ new Set();
    const textures = /* @__PURE__ */ new Set();
    const preparedGeometry = /* @__PURE__ */ new Set();
    const pageTexture = createPageTexture();
    scene.modelRoot.traverse((object) => {
      const mesh = object;
      if (!mesh.isMesh) return;
      const list = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      if (list.some((material) => material.name === "sky1")) {
        mesh.visible = false;
        mesh.userData.bboxVisible = false;
      }
      if (list.some((material) => material.name === "Greenleaf Treeline 003")) {
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        mesh.userData.bboxVisible = false;
      }
      if (!preparedGeometry.has(mesh.geometry)) {
        if (list.some((material) => material.name === "paper - book")) {
          const geometry2 = mesh.geometry;
          geometry2.computeBoundingBox();
          const bounds2 = geometry2.boundingBox;
          if (!bounds2) throw new Error("Courtyard page geometry has no bounds");
          const position = geometry2.getAttribute("position");
          const uv = new Float32Array(position.count * 2);
          for (let i = 0; i < position.count; i++) {
            uv[i * 2] = (position.getX(i) + position.getZ(i)) / 0.6 + 0.5;
            uv[i * 2 + 1] = (position.getY(i) - bounds2.min.y) / Math.max(1e-3, bounds2.max.y - bounds2.min.y);
          }
          geometry2.setAttribute("uv", new Float32BufferAttribute(uv, 2));
        }
        if (list.some((material) => material.name === "roof")) colorRoofTiles(mesh);
        preparedGeometry.add(mesh.geometry);
      }
      list.forEach((material) => materials.add(material));
    });
    for (const material of materials) {
      material.normalScale.multiplyScalar(0.5);
      if (material.name === "Greenleaf Treeline 003") material.emissiveIntensity = 1;
      if (material.name.startsWith("wood -")) {
        material.metalness = 0;
        material.color.set("#bba28b");
        material.roughnessMap = null;
        material.roughness = 0.82;
      }
      if (material.name.startsWith("leather - book - cover")) {
        const colors = ["#a78d72", "#887764", "#ac8e71"];
        material.color.set(colors[Number(material.name.slice(-1)) - 1]);
        material.roughnessMap = null;
        material.roughness = 0.88;
      }
      if (material.name === "paper - book") {
        material.color.set("#ffffff");
        material.map = pageTexture;
        material.roughness = 1;
      }
      if (["column marble", "brick marble", "brick marble smooth"].includes(material.name)) {
        material.color.set("#fff5e2");
        material.normalScale.multiplyScalar(0.3);
      }
      if (["stone pavement", "outdoor pavement", "grass - ground"].includes(material.name)) {
        material.normalScale.multiplyScalar(0.4);
      }
      if (material.name.startsWith("plaster")) material.color.multiply(new Color("#fff4e4"));
      if (material.name === "brick clay") material.color.set("#ead0b3");
      if (material.name === "roof") {
        material.color.set("#c6a17c");
        material.vertexColors = true;
        material.normalScale.multiplyScalar(0.4);
        material.roughnessMap = null;
        material.roughness = 0.95;
      }
      if (material.name === "iron_nail") {
        material.color.set("#8b897d");
        material.metalness = 0.4;
        material.roughness = 0.8;
      }
      if (material.name === "glass far") {
        material.color.set("#b3c5c9");
        material.metalness = 1;
        material.roughness = 0.08;
        material.envMapIntensity = 4;
      }
      for (const texture of [
        material.map,
        material.normalMap,
        material.roughnessMap,
        material.metalnessMap,
        material.aoMap,
        material.emissiveMap
      ]) {
        if (texture) textures.add(texture);
      }
      material.needsUpdate = true;
    }
    for (const texture of textures) {
      texture.anisotropy = Math.min(16, renderer.capabilities.getMaxAnisotropy());
      texture.needsUpdate = true;
    }
    await viewer.setEnvironmentMap(
      assetUrl("/scenes/lone-monk/kloofendal_48d_partly_cloudy_puresky_4k.hdr"),
      { setBackground: true }
    );
    scene.environmentIntensity = 0.4;
    scene.environmentRotation.y = 0.1;
    scene.backgroundRotation.copy(scene.environmentRotation);
    const tonemap = viewer.getPlugin(TonemapPlugin);
    if (tonemap) tonemap.exposure = 1.2;
    const bounds = scene.getModelBounds();
    const sunDirection = new Vector3(0.554743, 0.741466, 0.377476).normalize().applyAxisAngle(new Vector3(0, 1, 0), scene.environmentRotation.y);
    const sun = new DirectionalLight2("#ffe0b1", 9);
    sun.name = "Lone Monk sun";
    const sky = new DirectionalLight2("#c7dcff", 1.8);
    sky.name = "Lone Monk sky fill";
    for (const light of [sun, sky]) {
      light.castShadow = true;
      light.userData.bboxVisible = false;
      const size = Math.min(light === sun ? 4096 : 2048, renderer.capabilities.maxTextureSize);
      light.shadow.mapSize.set(size, size);
      light.shadow.bias = -5e-5;
      light.shadow.normalBias = 0.02;
      scene.addLight(light, { addToRoot: true });
      placeLight(light, sunDirection, bounds);
    }
    renderer.shadowMap.type = PCFSoftShadowMap;
    placeLight(sky, new Vector3(0, 1, 0), bounds);
    const tangent = new Vector3().crossVectors(sunDirection, new Vector3(0, 1, 0)).normalize();
    const bitangent = new Vector3().crossVectors(sunDirection, tangent);
    const direction = new Vector3();
    const updateShadows = () => {
      const frame = viewer.renderer.frameCount % convergenceFrames;
      let bits = frame, radical = 0, weight = 0.5;
      while (bits > 0) {
        radical += (bits & 1) * weight;
        bits >>= 1;
        weight *= 0.5;
      }
      const azimuth = frame * 2.399963229728653;
      const height = 0.04 + 0.92 * radical;
      const radius = Math.sqrt(1 - height * height);
      direction.set(Math.cos(azimuth) * radius, height, Math.sin(azimuth) * radius);
      placeLight(sky, direction, bounds);
      const sunRadius = 0.018 * Math.sqrt(radical);
      direction.copy(sunDirection).addScaledVector(tangent, Math.cos(azimuth) * sunRadius).addScaledVector(bitangent, Math.sin(azimuth) * sunRadius).normalize();
      placeLight(sun, direction, bounds);
      viewer.renderer.resetShadows();
    };
    if (!square) {
      viewer.addEventListener("preRender", updateShadows);
      onCleanup?.(() => viewer.removeEventListener("preRender", updateShadows));
    }
    viewer.getPlugin(SSAOPlugin).passes.ssao.passObject.parameters.occlusionWorldRadius = 0.8;
    const contact = viewer.getPlugin(SSContactShadows);
    if (contact) contact.enabled = false;
    const progressive = viewer.getPlugin(ProgressivePlugin);
    if (progressive) progressive.maxFrameCount = convergenceFrames;
    const camera2 = scene.activeCamera;
    Object.assign(camera2.cameraObject.userData, { autoNearFar: false, minNearPlane: 1.12, maxFarPlane: 200 });
    camera2.setCameraOptions({
      position: [20, 0.9, -7.512925],
      target: [20, 2.35, -20],
      fov: square ? 65 : 55.795,
      near: 1.12,
      far: 200
    });
    const controls = camera2.getControls();
    if (controls) {
      controls.minDistance = 2;
      controls.maxDistance = 45;
      controls.maxPolarAngle = Math.PI * 0.95;
    }
    viewer.renderer.resetShadows();
    scene.setDirty();
  }

  // src/webgi-addon/shaders/reflectorSample.glsl
  var reflectorSample_default = "\n#ifndef D_sceneBoundingRadius\n#define D_sceneBoundingRadius\nuniform float sceneBoundingRadius;\n#endif\n\nvarying vec4 vRefUv;\nuniform sampler2D tRefDiffuse;\nuniform vec2 tRefDiffuseSize;\n//uniform sampler2D tRefDepth;\nfloat getSpecularMIPLevel(const in float roughness, const in float maxMIPLevel) {\n    float sigma = PI * roughness * roughness / (1.0 + roughness);\n    float desiredMIPLevel = maxMIPLevel + log2(sigma);\n    // clamp to allowable LOD ranges.\n    return clamp(desiredMIPLevel, 0.0, maxMIPLevel);\n}\n\nvec4 getReflectionColor(const in float roughness, const in float depthModifier){\n    float mip = getSpecularMIPLevel(roughness + depthModifier, 5.0);\n    vec4 color = texture2D(tRefDiffuse, vRefUv.xy/vRefUv.w, mip);\n    float blurDist = saturate(2.0 / (1. + pow(abs(vViewPosition.z), 0.25))) * mip * 32. * color.a;\n\n    float rnd = PI2 * interleavedGradientNoise( vUv.xy, frameCount );\n    vec4 rotationMatrix = vec4(cos(rnd), -sin(rnd), 0.,0.);\n    rotationMatrix.z = -rotationMatrix.y;\n    rotationMatrix.w = rotationMatrix.x;\n\n    vec3 colorSum = color.rgb * color.a;\n    float weightSum = 0.001 + color.a;\n    vec2 ofs;\n\n    setPds(); // sets poisson_disk_samples\n\n    #pragma unroll_loop_start\n    for ( int i = 0; i < 16; i ++ ) {\n        ofs = poisson_disk_samples[UNROLLED_LOOP_INDEX];\n        ofs = vec2(dot(ofs, rotationMatrix.xy), dot(ofs, rotationMatrix.zw) );\n        ofs = vRefUv.xy + vRefUv.w * blurDist * ofs / tRefDiffuseSize.xy;\n        color = texture2D(tRefDiffuse, ofs / vRefUv.w, mip);\n        colorSum += color.rgb * color.a;\n        weightSum += color.a;\n    }\n    #pragma unroll_loop_end\n\n    return vec4(colorSum / weightSum, 1.0);\n}\n";

  // src/webgi-addon/shaders/randomHelpers.glsl
  var randomHelpers_default = "\n#ifndef BASIC_RANDOM_HELPERS\n#define BASIC_RANDOM_HELPERS\nuniform float frameCount;\n\nfloat random(float n){return fract(sin(n) * 43758.5453123);}\n\nfloat random2(vec2 n,float x){n+=x;return fract(sin(dot(n.xy,vec2(12.9898, 78.233)))*43758.5453);}\n\nfloat random3(vec3 v) {\n    v = fract(v * 443.8975);\n    v += dot(v, v.yzx + 19.19);\n    return fract((v.x + v.y) * v.z);\n}\n\n// https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Shaders/Private/Random.ush#L27\nfloat interleavedGradientNoise(const in vec2 fragCoord, const in float seed) {\n    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);\n    return fract(magic.z * fract(dot(fragCoord.xy + seed * vec2(2.083, 4.867), magic.xy)));\n}\n\nvec3 hash3( vec2 p )\n{\n    vec3 q = vec3( dot(p,vec2(127.1,311.7)),\n    dot(p,vec2(269.5,183.3)),\n    dot(p,vec2(419.2,371.9)) );\n    return fract(sin(q)*43758.5453);\n}\n\n#endif\n";

  // src/webgi-addon/shaders/poissonDiskSamples.glsl
  var poissonDiskSamples_default = "vec2 poisson_disk_samples[16];\nvoid setPds(){\n    poisson_disk_samples[0] = vec2(-0.399691779231, 0.728591545584);\n    poisson_disk_samples[1] = vec2(-0.48622557676, -0.84016533712);\n    poisson_disk_samples[2] = vec2(0.770309468987, -0.24906070432);\n    poisson_disk_samples[3] = vec2(0.556596796154, 0.820359876432);\n    poisson_disk_samples[4] = vec2(-0.933902004071, 0.0600539051593);\n    poisson_disk_samples[5] = vec2(0.330144964342, 0.207477293384);\n    poisson_disk_samples[6] = vec2(0.289013230975, -0.686749271417);\n    poisson_disk_samples[7] = vec2(-0.0832470893559, -0.187351643125);\n    poisson_disk_samples[8] = vec2(-0.296314525615, 0.254474834305);\n    poisson_disk_samples[9] = vec2(-0.850977666059, 0.484642744689);\n    poisson_disk_samples[10] = vec2(0.829287915319, 0.2345063545);\n    poisson_disk_samples[11] = vec2(-0.773042143899, -0.543741521254);\n    poisson_disk_samples[12] = vec2(0.0561133030864, 0.928419742597);\n    poisson_disk_samples[13] = vec2(-0.205799249508, -0.562072714492);\n    poisson_disk_samples[14] = vec2(-0.526991665882, -0.193690188118);\n    poisson_disk_samples[15] = vec2(-0.051789270667, -0.935374050821);\n}\n";

  // src/webgi-addon/DlssReflector2.ts
  var _DlssReflector2 = class _DlssReflector2 extends Mesh {
    constructor(geometry2, renderTarget, clipBias = 0) {
      super(geometry2);
      this.type = "Reflector";
      this.isReflector2 = true;
      this.enabled = true;
      this.reflectorModePhysical = true;
      // shows envmap in reflection
      this.reflectionTargetNeedsUpdate = true;
      this.transparentReflectionBackground = true;
      this.materialExtension = {
        extraUniforms: {
          // tRefDepth: {value: null},
          tRefDiffuse: { value: null },
          tRefDiffuseSize: { value: new Vector2() },
          refTextureMatrix: { value: null },
          frameCount: { value: 0 },
          sceneBoundingRadius: { value: 0 }
        },
        extraDefines: {
          // eslint-disable-next-line @typescript-eslint/naming-convention
          USE_UV: ""
        },
        updaters: [],
        shaderExtender: (shader, material, renderer) => {
          if (this.enabled) {
            shader.vertexShader = shaderReplaceString(shader.vertexShader, "void main() {", "void main() {\nvRefUv = refTextureMatrix * vec4( position, 1.0 );");
            const ls = "#glMarker beforeModulation";
            shader.fragmentShader = shaderReplaceString(shader.fragmentShader, ls, `
                    if(roughnessFactor < 0.95) {
                        float d = 0.;//textureProj(tRefDepth, vRefUv).r;
                        // d = min(2., max(0., (d-0.06) * ((7./3.-ior)) * sceneBoundingRadius));
                        vec4 refBaseColor = getReflectionColor(material.roughness, material.roughness * d);
                        // refBaseColor.rgb = vec3(refBaseColor.a);
                        // refBaseColor.a *= 1.0 - clamp(material.roughness * .3, 0., 1.);
                        ` + (this.reflectorModePhysical ? `
                        #if !defined(SSR_ENABLED) || SSR_ENABLED < 1 
                        vec3 specularColor = EnvironmentBRDF(geometryNormal, geometryViewDir, material.specularColor.rgb, material.specularF90, material.roughness);
                        #endif
                        reflectedLight.indirectSpecular = mix(vec3(reflectedLight.indirectSpecular), saturate(specularColor.rgb * refBaseColor.rgb), refBaseColor.a);
                        ` : `
                        reflectedLight.indirectSpecular = saturate(diffuseColor.rgb * refBaseColor.rgb);
                        diffuseColor.a *= refBaseColor.a;
                        `) + "}\n" + ls);
          }
        },
        parsVertexSnippet: () => !this.enabled ? "" : `
		uniform mat4 refTextureMatrix;
		varying vec4 vRefUv;
`,
        parsFragmentSnippet: () => this.enabled ? poissonDiskSamples_default + "\n" + randomHelpers_default + "\n" + reflectorSample_default : "",
        computeCacheKey: (material) => {
          return this.enabled + " " + material.materialObject.transparent + " " + this.reflectorModePhysical + " ";
        },
        onObjectRender: (object, { materialObject }) => {
          if (materialObject.userData.__lastTransparent !== materialObject.transparent) {
            materialObject.needsUpdate = true;
            materialObject.userData.__lastTransparent = materialObject.transparent;
          }
        },
        isCompatible: (material) => {
          return material.isMeshStandardMaterial2;
        }
      };
      this.material = void 0;
      this._renderTarget = renderTarget;
      const reflectorPlane = new Plane();
      const normal = new Vector3();
      const reflectorWorldPosition = new Vector3();
      const cameraWorldPosition = new Vector3();
      const rotationMatrix = new Matrix4();
      const lookAtPosition = new Vector3(0, 0, -1);
      const clipPlane = new Vector4();
      const view = new Vector3();
      const target = new Vector3();
      const q = new Vector4();
      const textureMatrix = new Matrix4();
      const virtualCamera = new PerspectiveCamera();
      if (!MathUtils.isPowerOfTwo(renderTarget.texture.image.width) || !MathUtils.isPowerOfTwo(renderTarget.texture.image.height))
        this._renderTarget.texture.generateMipmaps = false;
      this.onBeforeRender = (renderer, scene, camera2) => {
        if (!this.enabled || !renderer.userData.mainRenderPass) return;
        if (!this.reflectionTargetNeedsUpdate) {
          return;
        }
        const viewOffset = camera2.view ? Object.assign({}, camera2.view) : null;
        viewOffset && camera2.clearViewOffset && camera2.clearViewOffset();
        reflectorWorldPosition.setFromMatrixPosition(this.matrixWorld);
        cameraWorldPosition.setFromMatrixPosition(camera2.matrixWorld);
        rotationMatrix.extractRotation(this.matrixWorld);
        normal.set(0, 0, 1);
        normal.applyMatrix4(rotationMatrix);
        view.subVectors(reflectorWorldPosition, cameraWorldPosition);
        if (view.dot(normal) > 0) return;
        view.reflect(normal).negate();
        view.add(reflectorWorldPosition);
        rotationMatrix.extractRotation(camera2.matrixWorld);
        lookAtPosition.set(0, 0, -1);
        lookAtPosition.applyMatrix4(rotationMatrix);
        lookAtPosition.add(cameraWorldPosition);
        target.subVectors(reflectorWorldPosition, lookAtPosition);
        target.reflect(normal).negate();
        target.add(reflectorWorldPosition);
        virtualCamera.position.copy(view);
        virtualCamera.up.set(0, 1, 0);
        virtualCamera.up.applyMatrix4(rotationMatrix);
        virtualCamera.up.reflect(normal);
        virtualCamera.lookAt(target);
        virtualCamera.far = 2;
        virtualCamera.near = 0;
        virtualCamera.updateMatrixWorld();
        virtualCamera.projectionMatrix.copy(camera2.projectionMatrix);
        textureMatrix.set(
          0.5,
          0,
          0,
          0.5,
          0,
          0.5,
          0,
          0.5,
          0,
          0,
          0.5,
          0.5,
          0,
          0,
          0,
          1
        );
        textureMatrix.multiply(virtualCamera.projectionMatrix);
        textureMatrix.multiply(virtualCamera.matrixWorldInverse);
        textureMatrix.multiply(this.matrixWorld);
        reflectorPlane.setFromNormalAndCoplanarPoint(normal, reflectorWorldPosition);
        reflectorPlane.applyMatrix4(virtualCamera.matrixWorldInverse);
        clipPlane.set(reflectorPlane.normal.x, reflectorPlane.normal.y, reflectorPlane.normal.z, reflectorPlane.constant);
        const projectionMatrix = virtualCamera.projectionMatrix;
        q.x = (Math.sign(clipPlane.x) + projectionMatrix.elements[8]) / projectionMatrix.elements[0];
        q.y = (Math.sign(clipPlane.y) + projectionMatrix.elements[9]) / projectionMatrix.elements[5];
        q.z = -1;
        q.w = (1 + projectionMatrix.elements[10]) / projectionMatrix.elements[14];
        clipPlane.multiplyScalar(2 / clipPlane.dot(q));
        projectionMatrix.elements[2] = clipPlane.x;
        projectionMatrix.elements[6] = clipPlane.y;
        projectionMatrix.elements[10] = clipPlane.z + 1 - clipBias;
        projectionMatrix.elements[14] = clipPlane.w;
        this.visible = false;
        const currentRenderTarget = renderer.getRenderTarget();
        const currentXrEnabled = renderer.xr.enabled;
        const currentShadowAutoUpdate = renderer.shadowMap.autoUpdate;
        renderer.xr.enabled = false;
        renderer.shadowMap.autoUpdate = false;
        renderer.setRenderTarget(this._renderTarget);
        renderer.state.buffers.depth.setMask(true);
        if (renderer.autoClear === false) renderer.clear();
        const sceneBackground = scene.background;
        const renderBackground = !this.transparentReflectionBackground;
        if (sceneBackground?.isTexture && renderBackground) {
          if (!sceneBackground.userData) sceneBackground.userData = {};
          sceneBackground.userData.flipX = !sceneBackground.userData.flipX;
        }
        setThreeRendererMode(renderer, {
          shadowMapRender: false,
          backgroundRender: renderBackground,
          opaqueRender: true,
          transparentRender: true,
          transmissionRender: false,
          // todo: render transmissive objects somehow
          screenSpaceRendering: false
        }, () => renderer.render(scene, virtualCamera));
        if (sceneBackground?.isTexture && renderBackground) {
          sceneBackground.userData.flipX = !sceneBackground.userData.flipX || void 0;
        }
        renderer.xr.enabled = currentXrEnabled;
        renderer.shadowMap.autoUpdate = currentShadowAutoUpdate;
        renderer.setRenderTarget(currentRenderTarget);
        if (viewOffset?.enabled && camera2.setViewOffset)
          camera2.setViewOffset(viewOffset.fullWidth, viewOffset.fullHeight, viewOffset.offsetX, viewOffset.offsetY, viewOffset.width, viewOffset.height);
        const viewport = camera2.viewport;
        if (viewport !== void 0) {
          renderer.state.viewport(viewport);
        }
        this.visible = true;
        this.reflectionTargetNeedsUpdate = false;
      };
      this.textureMatrix = textureMatrix;
      this.materialExtension.extraUniforms.tRefDiffuse.value = this._renderTarget.texture;
      this.materialExtension.extraUniforms.tRefDiffuseSize.value = new Vector2(this._renderTarget.width, this._renderTarget.height);
      this.materialExtension.extraUniforms.refTextureMatrix.value = textureMatrix;
    }
    // todo change to reflectBackground or something
    _updateExtension() {
      this.transparentReflectionBackground = this.reflectorModePhysical;
      this.materialExtension?.setDirty?.();
    }
    getRenderTarget() {
      return this._renderTarget;
    }
  };
  __decorateClass([
    onChange(_DlssReflector2.prototype._updateExtension)
  ], _DlssReflector2.prototype, "enabled", 2);
  __decorateClass([
    onChange(_DlssReflector2.prototype._updateExtension)
  ], _DlssReflector2.prototype, "reflectorModePhysical", 2);
  var DlssReflector2 = _DlssReflector2;
  DlssReflector2.prototype.isReflector = true;

  // src/webgi-addon/DlssGroundPlugin.ts
  var _DlssGroundPlugin = class _DlssGroundPlugin extends BaseGroundPlugin {
    constructor(options = {}, showDebug = false) {
      super(options);
      this.bakedShadows = true;
      this.groundReflection = false;
      this.physicalReflections = false;
      this.autoFrustumSize = true;
      /**
       * autoBakeShadows - when true, shadows are baked automatically on scene update(whenever any object in the scene changes), set it to `false` to trigger baking manually with {@see bakeShadows()}
       */
      this.autoBakeShadows = true;
      this._showDebug = showDebug;
      if (showDebug) this.dependencies.push(DebugPlugin);
      this._onSceneUpdate = this._onSceneUpdate.bind(this);
    }
    get shadowBaker() {
      return this._shadowBaker;
    }
    static {
      // set mesh(value: IModel<TMesh>|undefined) {
      //     this._iMesh = value
      // }
      this.PluginType = "Ground";
    }
    /**
     * bake shadows manually, to be used with {@see autoBakeShadows} set to false
     */
    bakeShadows() {
      this._shadowBaker?.reset();
    }
    _createMesh() {
      const reflector = new DlssReflector2(this._geometry, this._viewer.renderer.createTarget({
        // type: HalfFloatType,
        type: UnsignedByteType,
        format: RGBAFormat,
        colorSpace: NoColorSpace,
        // todo: we can do rgbm if only opaque objects will be reflected
        size: { width: 1024, height: 1024 },
        generateMipmaps: true,
        depthBuffer: true,
        minFilter: LinearMipmapLinearFilter,
        magFilter: LinearFilter
        // isAntialiased: this._viewer.isAntialiased,
      }));
      const superOnBeforeRender = reflector.onBeforeRender;
      reflector.onBeforeRender = (...params) => {
        let ssr = this._viewer?.getPluginByType("SSReflection")?.passes.ssr.passObject;
        if (ssr && !ssr.enabled) ssr = void 0;
        if (ssr) ssr.enabled = false;
        let ssbevel = this._viewer?.getPluginByType("SSBevelPlugin")?.pass?.passObject;
        if (ssbevel && !ssbevel.enabled) ssbevel = void 0;
        if (ssbevel) ssbevel.enabled = false;
        superOnBeforeRender(...params);
        if (ssr) ssr.enabled = true;
        if (ssbevel) ssbevel.enabled = true;
      };
      return reflector;
    }
    async onAdded(viewer) {
      await super.onAdded(viewer);
      if (this._showDebug) {
        viewer.getPlugin(DebugPlugin)?.addTexture("bake_ground_1", () => {
          return this._shadowBaker?.light.shadow.map?.texture;
        }, [100, 100, 200, 200]);
        viewer.getPlugin(DebugPlugin)?.addTexture("bake_ground_2", () => {
          return this._shadowBaker?.target?.texture;
        }, [100, 400, 400, 400], "texel = vec4(vec3(unpackRGBAToDepth(texel)), 1.0);");
      }
    }
    _postFrame() {
      super._postFrame();
      if (!this._viewer) return;
      if (!this.enabled) return;
      if (this._shadowBaker && this.bakedShadows) {
        this._shadowBaker.autoUpdateShadow();
      }
    }
    _preRender() {
      super._preRender();
      if (!this._viewer) return;
      this._mesh.reflectionTargetNeedsUpdate = this._viewer.renderer.frameCount < 1;
    }
    async onDispose(viewer) {
      return super.onDispose(viewer);
    }
    async onRemove(viewer) {
      return super.onRemove(viewer);
    }
    _removeMaterial() {
      if (!this._material) return;
      if (this._shadowBaker && this._material.groundMatExtension) {
        this._material.unregisterMaterialExtensions?.([this._shadowBaker.materialExtension]);
        delete this._material.groundMatExtension;
      }
      if (this._material.reflectorMatExtension) {
        const ext = this._mesh.materialExtension;
        if (!ext) console.warn("WebGi GroundPlugin: unable to find the extension to unregister");
        this._material.unregisterMaterialExtensions?.([ext]);
        delete this._material.reflectorMatExtension;
      }
      super._removeMaterial();
    }
    _onSceneUpdate(event) {
      super._onSceneUpdate(event);
      if (event.geometryChanged === false) return;
      if (this.autoBakeShadows) this._shadowBaker?.reset();
    }
    refreshOptions() {
      if (!this._viewer) return;
      if (this.bakedShadows && !this._shadowBaker) {
        this._shadowBaker = new ShadowMapBaker(this._viewer);
        this._shadowBaker.attachedMesh = this._mesh;
      } else if (!this.bakedShadows && this._shadowBaker) {
        this._shadowBaker.reset();
        this._shadowBaker.cleanupMaterial();
      }
      const ref = this._mesh;
      if (ref.isReflector2) {
        ref.enabled = this.groundReflection;
        ref.reflectorModePhysical = this.physicalReflections;
      }
      super.refreshOptions();
      this._viewer.setDirty(this);
    }
    _refreshTransform() {
      if (this.autoFrustumSize) {
        const baker = this.shadowBaker;
        if (baker) {
          const fs = this.size / 2;
          if (fs !== baker.light.shadowParams.frustumSize) {
            baker.light.shadowParams.frustumSize = fs;
            baker.light.updateShadowParams();
            baker.reset();
          }
        }
      }
      super._refreshTransform();
    }
    // see BaseGroundPlugin
    fromJSON(data, meta) {
      if (!super.fromJSON(data, meta)) return null;
      if (data.autoFrustumSize === void 0) this.autoFrustumSize = false;
      return this;
    }
    _refreshMaterial() {
      if (!this._viewer) return false;
      const isNewMaterial = super._refreshMaterial();
      if (!this._material) return isNewMaterial;
      if (this.groundReflection && this._mesh.isReflector2 && !this._material.reflectorMatExtension) {
        const ext = this._mesh.materialExtension;
        ext.updaters = [this._viewer.scene, this._viewer.renderer];
        this._material.registerMaterialExtensions?.([ext]);
        this._material.reflectorMatExtension = true;
      }
      if (this.bakedShadows && this._shadowBaker && !this._material.groundMatExtension) {
        this._material.registerMaterialExtensions?.([this._shadowBaker.materialExtension]);
        this._material.groundMatExtension = true;
      }
      this._material.materialObject.userData.ssreflDisabled = this.groundReflection;
      this._material.materialObject.userData.ssreflNonPhysical = !this.physicalReflections;
      this._viewer.setDirty(this);
      return isNewMaterial;
    }
    _extraUiConfig() {
      return [
        {
          label: "Baked Shadows",
          type: "checkbox",
          property: [this, "bakedShadows"]
        },
        {
          label: "Shadow Frames",
          type: "input",
          hidden: () => !this._shadowBaker,
          stepSize: 1,
          bounds: [1, 1e3],
          property: [this._shadowBaker, "maxFrameNumber"]
        },
        {
          label: "Alpha Vignette",
          type: "checkbox",
          hidden: () => !this._material || this._material.transmission < 1e-4 && !this._material.transparent,
          property: [this._shadowBaker, "alphaVignette"],
          limitedUi: true,
          onChange: () => this._uiConfig?.uiRefresh?.("postFrame", true)
        },
        {
          label: "Alpha Vignette Axis",
          type: "dropdown",
          hidden: () => !this._shadowBaker?.alphaVignette || !this._material || this._material.transmission < 1e-4 && !this._material.transparent,
          property: [this._shadowBaker, "alphaVignetteAxis"],
          children: ["x", "y", "xy"].map((v) => ({ label: v, value: v })),
          limitedUi: true
        },
        {
          label: "Planar Reflections",
          type: "checkbox",
          property: [this, "groundReflection"]
        },
        {
          label: "Auto Frustum Size",
          type: "checkbox",
          property: [this, "autoFrustumSize"]
        },
        {
          label: "Physical Reflections",
          type: "checkbox",
          // hidden: ()=> !this._options.groundReflection || !(this._mesh as Reflector2).isReflector2,
          // property: [this._mesh as Reflector2, 'reflectorModePhysical'],
          property: [this, "physicalReflections"],
          limitedUi: true
        },
        {
          label: "Shadow type",
          type: "dropdown",
          hidden: () => !this._shadowBaker,
          property: [this._shadowBaker, "groundMapMode"],
          children: [
            { label: "aoMap" },
            { label: "map" },
            { label: "alphaMap" }
          ],
          limitedUi: true
        },
        {
          label: "Smooth Shadow",
          type: "checkbox",
          property: [this._shadowBaker, "smoothShadow"]
        },
        {
          label: "Baked shadow type",
          type: "dropdown",
          children: [["Basic", BasicShadowMap], ["PCF", PCFShadowMap], ["PCFSoft", PCFSoftShadowMap], ["VSM", VSMShadowMap]].map((v) => ({ label: v[0].toString(), value: v[1] })),
          property: [this._shadowBaker, "shadowMapType"]
        },
        {
          type: "folder",
          label: "Randomized Light",
          hidden: () => !this._shadowBaker,
          limitedUi: true,
          children: [
            {
              type: "color",
              label: "Color",
              property: [this._shadowBaker?.light, "color"]
            },
            {
              type: "slider",
              label: "Intensity",
              bounds: [0, 100],
              property: [this._shadowBaker?.light, "intensity"]
            },
            {
              type: "checkbox",
              label: "Shadow Enabled",
              property: [this._shadowBaker?.light?.shadowParams, "enabled"],
              onChange: [this._shadowBaker?.light?.updateShadowParams, this._onSceneUpdate]
            },
            {
              type: "slider",
              bounds: [0, 1],
              property: [this._shadowBaker?.light?.randomParams, "focus"],
              onChange: [this._onSceneUpdate]
            },
            {
              type: "slider",
              bounds: [0, 1],
              property: [this._shadowBaker?.light?.randomParams, "spread"],
              onChange: [this._onSceneUpdate],
              limitedUi: true
            },
            {
              type: "slider",
              bounds: [0.01, 60],
              property: [this._shadowBaker?.light?.randomParams, "distanceScale"],
              onChange: [this._shadowBaker?.light?.updateShadowParams, this._onSceneUpdate]
            },
            {
              type: "vec3",
              bounds: [-1, 1],
              property: [this._shadowBaker?.light?.randomParams, "direction"],
              onChange: [this._onSceneUpdate],
              limitedUi: true
            },
            {
              type: "vec3",
              bounds: [-1, 1],
              property: [this._shadowBaker?.light?.randomParams, "normalDirection"],
              onChange: [this._onSceneUpdate],
              limitedUi: true
            },
            {
              type: "slider",
              bounds: [0.01, 10],
              property: [this._shadowBaker?.light?.shadowParams, "radius"],
              onChange: [this._shadowBaker?.light?.updateShadowParams, this._onSceneUpdate]
            },
            {
              type: "input",
              property: [this._shadowBaker?.light?.shadowParams, "frustumSize"],
              hidden: () => this.autoFrustumSize,
              onChange: [this._shadowBaker?.light?.updateShadowParams, this._onSceneUpdate]
            },
            {
              type: "slider",
              bounds: [-0.1, 0.1],
              property: [this._shadowBaker?.light?.shadowParams, "bias"],
              onChange: [this._shadowBaker?.light?.updateShadowParams, this._onSceneUpdate]
            }
          ]
        },
        ...super._extraUiConfig()
      ];
    }
  };
  __decorateClass([
    onChange(_DlssGroundPlugin.prototype.refreshOptions),
    serialize()
  ], _DlssGroundPlugin.prototype, "bakedShadows", 2);
  __decorateClass([
    onChange(_DlssGroundPlugin.prototype.refreshOptions),
    serialize()
  ], _DlssGroundPlugin.prototype, "groundReflection", 2);
  __decorateClass([
    onChange(_DlssGroundPlugin.prototype.refreshOptions),
    serialize()
  ], _DlssGroundPlugin.prototype, "physicalReflections", 2);
  __decorateClass([
    onChange(_DlssGroundPlugin.prototype.refreshOptions),
    serialize()
  ], _DlssGroundPlugin.prototype, "autoFrustumSize", 2);
  __decorateClass([
    serialize("shadowBaker")
  ], _DlssGroundPlugin.prototype, "_shadowBaker", 2);
  var DlssGroundPlugin = _DlssGroundPlugin;

  // src/scenes/nr-groom-rings.ts
  function decodeGroomRings(buffer, part) {
    const { rings, strands, vertices, indices, centerStep, radiusStep, frameScale } = part;
    if (![rings, strands, vertices, indices].every((n) => Number.isSafeInteger(n) && n > 0) || vertices !== rings * 3 || indices !== (rings - strands) * 18 || ![centerStep, radiusStep, frameScale].every((n) => Number.isFinite(n) && n > 0) || buffer.byteLength !== strands * 2 + rings * 22) throw new Error("Invalid compact groom metadata");
    const bytes = new Uint8Array(buffer);
    const centerOffset = strands * 2;
    const frameOffset = centerOffset + rings * 12;
    const omittedOffset = frameOffset + rings * 6;
    const radiusOffset = omittedOffset + rings;
    const opacityOffset = radiusOffset + rings * 2;
    const position = new Float32Array(vertices * 3);
    const normal = new Float32Array(vertices * 3);
    const color = new Uint8Array(vertices * 4);
    const index = new Uint32Array(indices);
    const read16 = (base, i, count) => bytes[base + i] | bytes[base + count + i] << 8;
    const read32 = (base, i) => bytes[base + i] | bytes[base + rings + i] << 8 | bytes[base + rings * 2 + i] << 16 | bytes[base + rings * 3 + i] << 24;
    let cx = 0, cy = 0, cz = 0;
    const q = new Float64Array(4);
    for (let i = 0; i < rings; i++) {
      cx += read32(centerOffset, i);
      cy += read32(centerOffset + rings * 4, i);
      cz += read32(centerOffset + rings * 8, i);
      const omitted = bytes[omittedOffset + i];
      if (omitted > 3) throw new Error("Invalid groom rotation");
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
        position[j] = cx * centerStep + nx * radius;
        position[j + 1] = cy * centerStep + ny * radius;
        position[j + 2] = cz * centerStep + nz * radius;
        normal[j] = nx;
        normal[j + 1] = ny;
        normal[j + 2] = nz;
        color[vertex * 4] = color[vertex * 4 + 1] = color[vertex * 4 + 2] = 255;
        color[vertex * 4 + 3] = opacity;
      }
    }
    let ring = 0, cursor = 0;
    for (let strand = 0; strand < strands; strand++) {
      const length = read16(0, strand, strands);
      if (length < 2 || ring + length > rings) throw new Error("Invalid groom strand length");
      for (let j = 0; j < length - 1; j++) {
        const base = (ring + j) * 3;
        for (let side = 0; side < 3; side++) {
          const a = base + side, b = base + (side + 1) % 3;
          index[cursor++] = a;
          index[cursor++] = b;
          index[cursor++] = a + 3;
          index[cursor++] = b;
          index[cursor++] = b + 3;
          index[cursor++] = a + 3;
        }
      }
      ring += length;
    }
    if (ring !== rings || cursor !== indices) throw new Error("Incomplete compact groom");
    return { position, normal, color, index };
  }

  // src/scenes/nr-culled-instances.ts
  var scatterBounds = /* @__PURE__ */ new WeakMap();
  function createCulledInstances(geometry2, material, original, groups, renderer) {
    const total = original.length / 16;
    const mesh = new InstancedMesh(geometry2, material, total);
    const matrices = mesh.instanceMatrix.array;
    mesh.instanceMatrix.setUsage(DynamicDrawUsage);
    geometry2.computeBoundingSphere();
    const source = geometry2.boundingSphere;
    const centers = new Float64Array(total * 4);
    const clusters = [];
    const box = new Box3(), fullBox = new Box3(), point = new Vector3();
    let cursor = 0;
    for (const group of groups) {
      const start = cursor;
      box.makeEmpty();
      for (const index of group) {
        const j = index * 16, d = cursor * 16, k = cursor * 4;
        for (let c = 0; c < 16; c++) matrices[d + c] = original[j + c];
        const x = original[j] * source.center.x + original[j + 4] * source.center.y + original[j + 8] * source.center.z + original[j + 12];
        const y = original[j + 1] * source.center.x + original[j + 5] * source.center.y + original[j + 9] * source.center.z + original[j + 13];
        const z = original[j + 2] * source.center.x + original[j + 6] * source.center.y + original[j + 10] * source.center.z + original[j + 14];
        const dot = (a, b) => original[j + a] * original[j + b] + original[j + a + 1] * original[j + b + 1] + original[j + a + 2] * original[j + b + 2];
        const xy = Math.abs(dot(0, 4)), xz = Math.abs(dot(0, 8)), yz = Math.abs(dot(4, 8));
        const scale = Math.sqrt(Math.max(dot(0, 0) + xy + xz, dot(4, 4) + xy + yz, dot(8, 8) + xz + yz));
        const r = source.radius * scale + 1e-5;
        centers[k] = x;
        centers[k + 1] = y;
        centers[k + 2] = z;
        centers[k + 3] = r;
        box.expandByPoint(point.set(x - r, y - r, z - r));
        box.expandByPoint(point.set(x + r, y + r, z + r));
        cursor++;
      }
      fullBox.union(box);
      clusters.push({ start, end: cursor, box: box.clone(), sphere: box.getBoundingSphere(new Sphere()) });
    }
    if (cursor !== total) throw new Error("Incomplete scatter partition");
    mesh.boundingBox = fullBox;
    const masks = /* @__PURE__ */ new WeakMap();
    scatterBounds.set(mesh, { centers, masks });
    mesh.boundingSphere = fullBox.getBoundingSphere(new Sphere());
    const frustum = new Frustum(), projection = new Matrix4();
    const cameras = /* @__PURE__ */ new WeakMap();
    const gl = renderer.getContext();
    let buffer = null;
    let uploaded = null, uploadedRevision = -1;
    mesh.instanceMatrix.onUpload(() => {
      buffer = gl.getParameter(gl.ARRAY_BUFFER_BINDING);
      uploaded = null;
    });
    const prepare = (camera2) => {
      if (!buffer) return;
      if (!mesh.frustumCulled) {
        if (uploaded) {
          const previous = gl.getParameter(gl.ARRAY_BUFFER_BINDING);
          gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
          gl.bufferData(gl.ARRAY_BUFFER, matrices, gl.DYNAMIC_DRAW);
          gl.bindBuffer(gl.ARRAY_BUFFER, previous);
          uploaded = null;
        }
        mesh.count = total;
        return;
      }
      projection.multiplyMatrices(camera2.projectionMatrix, camera2.matrixWorldInverse);
      const candidateMask = masks.get(camera2);
      const target = renderer.getRenderTarget();
      const mask = candidateMask?.matrix.equals(projection) && target && !target.samples && candidateMask.width === target.width && candidateMask.height === target.height ? candidateMask : void 0;
      projection.multiply(mesh.matrixWorld);
      let visible = cameras.get(camera2);
      if (!visible) {
        visible = {
          matrix: new Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
          matrices: new Float32Array(matrices.length),
          count: 0,
          revision: 0,
          maskRevision: -1
        };
        cameras.set(camera2, visible);
      }
      const maskRevision = mask?.revision ?? -1;
      if (!visible.matrix.equals(projection) || visible.maskRevision !== maskRevision) {
        visible.matrix.copy(projection);
        visible.maskRevision = maskRevision;
        frustum.setFromProjectionMatrix(projection);
        const planes = frustum.planes;
        let count = 0;
        for (const cluster of clusters) {
          if (!frustum.intersectsSphere(cluster.sphere)) continue;
          const c = cluster.sphere.center, r = cluster.sphere.radius;
          const inside = planes.every((p) => p.distanceToPoint(c) >= r);
          if (inside && !mask) {
            visible.matrices.set(matrices.subarray(cluster.start * 16, cluster.end * 16), count * 16);
            count += cluster.end - cluster.start;
            continue;
          }
          for (let i = cluster.start; i < cluster.end; i++) {
            if (mask && !mask.bytes[mask.offset + i]) continue;
            const k = i * 4, x = centers[k], y = centers[k + 1], z = centers[k + 2], radius = centers[k + 3];
            let keep = true;
            for (const p of planes) if (p.normal.x * x + p.normal.y * y + p.normal.z * z + p.constant < -radius) {
              keep = false;
              break;
            }
            if (!keep) continue;
            const a = i * 16, b = count++ * 16;
            for (let j = 0; j < 16; j++) visible.matrices[b + j] = matrices[a + j];
          }
        }
        visible.count = count;
        visible.revision++;
      }
      mesh.count = visible.count;
      if (uploaded !== visible || uploadedRevision !== visible.revision) {
        const previous = gl.getParameter(gl.ARRAY_BUFFER_BINDING);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, Math.max(64, visible.count * 64), gl.DYNAMIC_DRAW);
        if (visible.count) gl.bufferSubData(gl.ARRAY_BUFFER, 0, visible.matrices, 0, visible.count * 16);
        gl.bindBuffer(gl.ARRAY_BUFFER, previous);
        uploaded = visible;
        uploadedRevision = visible.revision;
      }
    };
    mesh.onBeforeRender = (_renderer, _scene, camera2) => prepare(camera2);
    mesh.onBeforeShadow = (_renderer, _object, _camera, shadowCamera) => prepare(shadowCamera);
    mesh.onAfterRender = mesh.onAfterShadow = () => {
      mesh.count = total;
    };
    return mesh;
  }

  // src/scenes/nr-vegetation-occlusion.ts
  var vegetationQuadVertex = `void main() { gl_Position = vec4(position.xy, 0., 1.); }`;
  var vegetationReduceDepth = `
uniform sampler2D sourceDepth;
uniform vec4 sourceRect;
uniform vec2 destinationOrigin;
out vec4 result;
float fetchDepth(ivec2 p) {
    if (any(greaterThanEqual(p, ivec2(sourceRect.zw)))) return 1.;
    return texelFetch(sourceDepth, p + ivec2(sourceRect.xy), 0).r;
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy - destinationOrigin) * 2;
    float d = max(max(fetchDepth(p), fetchDepth(p+ivec2(1,0))),
                  max(fetchDepth(p+ivec2(0,1)), fetchDepth(p+ivec2(1,1))));
    result = vec4(d, 0., 0., 1.);
}`;
  var vegetationTestBounds = `
uniform sampler2D boundsTexture, depth0, depthA, depthB;
uniform mat4 viewProjectionBounds;
uniform vec4 levels[16];
uniform vec2 screenSize;
uniform float rasterPadding;
uniform int boundsWidth, outputWidth, instanceCount, lastLevel;
out vec4 result;
float depthAt(ivec2 p, int level) {
    vec4 rect = levels[level];
    if (any(lessThan(p,ivec2(0))) || any(greaterThanEqual(p,ivec2(rect.zw)))) return 1.;
    p += ivec2(rect.xy);
    if (level == 0) return texelFetch(depth0,p,0).r;
    if ((level & 1) == 1) return texelFetch(depthA,p,0).r;
    return texelFetch(depthB,p,0).r;
}
float visible(int index) {
    if (index >= instanceCount) return 1.;
    ivec2 address = ivec2((index*4) % boundsWidth,(index*4) / boundsWidth);
    vec4 center = texelFetch(boundsTexture,address,0);
    vec4 clip = viewProjectionBounds * vec4(center.xyz,1.);
    vec4 axisX = viewProjectionBounds * vec4(texelFetch(boundsTexture,address+ivec2(1,0),0).xyz,0.);
    vec4 axisY = viewProjectionBounds * vec4(texelFetch(boundsTexture,address+ivec2(2,0),0).xyz,0.);
    vec4 axisZ = viewProjectionBounds * vec4(texelFetch(boundsTexture,address+ivec2(3,0),0).xyz,0.);
    vec4 pad = (abs(viewProjectionBounds[0])+abs(viewProjectionBounds[1])+abs(viewProjectionBounds[2])) * center.w;
    vec4 extent = abs(axisX)+abs(axisY)+abs(axisZ)+pad;
    vec4 lo = clip-extent, hi = clip+extent;
    if (hi.w <= 0. || hi.z < -hi.w || lo.z > hi.w ||
        hi.x < -hi.w || lo.x > hi.w || hi.y < -hi.w || lo.y > hi.w) return 0.;
    if (lo.w <= 0. || lo.z <= -hi.w) return 1.;
    vec3 ndcLo = vec3(1e30), ndcHi = vec3(-1e30);
    // Project the transformed geometry box, with explicit float-rounding padding.
    for (int corner=0; corner<8; corner++) {
        vec3 signCorner = vec3((corner&1)==0 ? -1. : 1., (corner&2)==0 ? -1. : 1., (corner&4)==0 ? -1. : 1.);
        vec4 p = clip + signCorner.x*axisX + signCorner.y*axisY + signCorner.z*axisZ;
        vec4 a = p-pad, b = p+pad;
        ndcLo = min(ndcLo,min(min(a.xyz/a.w,a.xyz/b.w),min(b.xyz/a.w,b.xyz/b.w)));
        ndcHi = max(ndcHi,max(max(a.xyz/a.w,a.xyz/b.w),max(b.xyz/a.w,b.xyz/b.w)));
    }
    // Single-sample rasterization tests pixel centers. Include two units of
    // the implementation's reported subpixel precision around those centers.
    vec2 a = ceil((ndcLo.xy*.5+.5)*screenSize-.5-rasterPadding);
    vec2 b = floor((ndcHi.xy*.5+.5)*screenSize-.5+rasterPadding);
    if (any(greaterThan(a,b))) return 0.;
    if (any(lessThan(b,vec2(0.))) || any(greaterThanEqual(a,screenSize))) return 0.;
    a = max(a,vec2(0.)); b = min(b,screenSize-1.);
    int level = min(lastLevel,int(ceil(log2(max(1.,max(b.x-a.x,b.y-a.y)+1.)))));
    float scale = exp2(float(level));
    ivec2 p = ivec2(floor(a/scale)), q = ivec2(floor(b/scale));
    float farthest = max(max(depthAt(p,level),depthAt(ivec2(q.x,p.y),level)),
                         max(depthAt(ivec2(p.x,q.y),level),depthAt(q,level)));
    float nearest = ndcLo.z*.5+.5 - .000002;
    if (nearest > farthest) return 0.;
    // Refine the rectangle, not the geometry. Four coarse texels can include
    // a large amount of sky outside a thin plant's actual screen bounds.
    level = max(0,level-2); scale = exp2(float(level));
    p = ivec2(floor(a/scale)); q = ivec2(floor(b/scale));
    for (int y=0; y<5; y++) for (int x=0; x<5; x++) {
        ivec2 cell = p+ivec2(x,y);
        if (any(greaterThan(cell,q))) continue;
        if (nearest <= depthAt(cell,level)) return 1.;
    }
    return 0.;
}
void main() {
    int index = (int(gl_FragCoord.y)*outputWidth+int(gl_FragCoord.x))*4;
    result = vec4(visible(index),visible(index+1),visible(index+2),visible(index+3));
}`;
  function createVegetationOcclusion(viewer) {
    const renderer = viewer.renderer.rendererObject, gl = renderer.getContext();
    if (!renderer.capabilities.isWebGL2 || viewer.isAntialiased || renderer.capabilities.logarithmicDepthBuffer || !gl.getExtension("EXT_color_buffer_float")) return void 0;
    const mainOccluders = new Scene(), shadowOccluders = new Scene();
    const materials = /* @__PURE__ */ new Map();
    const entries = [];
    let count = 0, occluderTriangles = 0;
    const materialFor = (source, side) => {
      const key = source.id * 3 + side;
      let material = materials.get(key);
      if (!material) {
        material = new MeshDepthMaterial({
          side,
          colorWrite: false,
          map: source.map,
          alphaTest: Math.max(1e-3, source.alphaTest)
        });
        materials.set(key, material);
      }
      return material;
    };
    viewer.scene.modelRoot.updateWorldMatrix(true, true);
    viewer.scene.modelRoot.traverseVisible((object) => {
      const mesh = object;
      if (!mesh.isMesh) return;
      const scatter = scatterBounds.get(mesh);
      if (scatter) {
        const length = scatter.centers.length / 4;
        entries.push({ mesh, offset: count, count: length });
        count += length;
        return;
      }
      if (mesh.isInstancedMesh || Array.isArray(mesh.material)) return;
      const m = mesh.material;
      if (!m.visible || m.polygonOffset || m.userData.renderToDepth === false || m.userData.pluginsDisabled || m.transparent || m.transmission || m.alphaHash || m.alphaTest || m.alphaMap || m.displacementMap || m.depthWrite === false || m.clippingPlanes?.length || mesh.isSkinnedMesh || mesh.morphTargetInfluences?.length || mesh.customDepthMaterial) return;
      const add = (scene, side) => {
        const proxy = new Mesh(mesh.geometry, materialFor(m, side));
        proxy.matrix.copy(mesh.matrixWorld);
        proxy.matrixAutoUpdate = false;
        scene.add(proxy);
      };
      add(mainOccluders, m.side);
      if (mesh.castShadow) add(
        shadowOccluders,
        m.shadowSide ?? (m.side === FrontSide ? BackSide : m.side === BackSide ? FrontSide : m.side)
      );
      occluderTriangles += (mesh.geometry.index?.count ?? 0) / 3;
    });
    if (!count) return void 0;
    const width = Math.min(renderer.capabilities.maxTextureSize, Math.max(2048, 2 ** Math.ceil(Math.log2(Math.sqrt(count * 4)))));
    const height = Math.ceil(count * 4 / width);
    if (height > renderer.capabilities.maxTextureSize) {
      materials.forEach((m) => m.dispose());
      return void 0;
    }
    const bounds = new Float32Array(width * height * 4), point = new Vector3(), half = new Vector3(), center = new Vector3();
    const transform = new Matrix4();
    for (const entry of entries) {
      const mesh = entry.mesh, matrices = mesh.instanceMatrix.array;
      mesh.geometry.computeBoundingBox();
      mesh.geometry.boundingBox.getCenter(center);
      mesh.geometry.boundingBox.getSize(half).multiplyScalar(0.5);
      for (let i = 0; i < entry.count; i++) {
        transform.fromArray(matrices, i * 16).premultiply(mesh.matrixWorld);
        point.copy(center).applyMatrix4(transform);
        const k = (entry.offset + i) * 16, e = transform.elements;
        bounds[k] = point.x;
        bounds[k + 1] = point.y;
        bounds[k + 2] = point.z;
        bounds[k + 3] = 1e-4 + Math.max(Math.abs(point.x), Math.abs(point.y), Math.abs(point.z)) * 1e-6;
        for (let j = 0; j < 3; j++) {
          bounds[k + 4 + j] = e[j] * half.x;
          bounds[k + 8 + j] = e[4 + j] * half.y;
          bounds[k + 12 + j] = e[8 + j] * half.z;
        }
      }
    }
    const boundsTexture = new DataTexture(bounds, width, height, RGBAFormat, FloatType);
    boundsTexture.needsUpdate = true;
    const maskWidth = width, maskHeight = Math.ceil(count / 4 / maskWidth);
    const makeTarget = (w, h, depth = false, float = false) => {
      const target = new WebGLRenderTarget(w, h, {
        depthBuffer: depth,
        format: float || depth ? RedFormat : RGBAFormat,
        type: float ? FloatType : void 0,
        minFilter: NearestFilter,
        magFilter: NearestFilter
      });
      return target;
    };
    const maskTarget = makeTarget(maskWidth, maskHeight);
    const reduce = new ShaderMaterial({
      glslVersion: GLSL3,
      vertexShader: vegetationQuadVertex,
      fragmentShader: vegetationReduceDepth,
      depthTest: false,
      depthWrite: false,
      blending: NoBlending,
      uniforms: {
        sourceDepth: { value: null },
        sourceRect: { value: new Vector4() },
        destinationOrigin: { value: new Vector2() }
      }
    });
    const levels = Array.from({ length: 16 }, () => new Vector4());
    const test = new ShaderMaterial({
      glslVersion: GLSL3,
      vertexShader: vegetationQuadVertex,
      fragmentShader: vegetationTestBounds,
      depthTest: false,
      depthWrite: false,
      blending: NoBlending,
      uniforms: {
        boundsTexture: { value: boundsTexture },
        depth0: { value: null },
        depthA: { value: null },
        depthB: { value: null },
        viewProjectionBounds: { value: new Matrix4() },
        levels: { value: levels },
        screenSize: { value: new Vector2() },
        boundsWidth: { value: width },
        outputWidth: { value: maskWidth },
        rasterPadding: { value: 2 / 2 ** gl.getParameter(gl.SUBPIXEL_BITS) },
        instanceCount: { value: count },
        lastLevel: { value: 0 }
      }
    });
    const quad = new FullScreenQuad(reduce);
    const resources = /* @__PURE__ */ new Map();
    let revision = 0;
    const cameras = /* @__PURE__ */ new Map();
    const projection = new Matrix4(), previousColor = new Color(), scissor = new Vector4();
    const stats = { occluderTriangles, totalInstances: count, visibleInstances: 0, milliseconds: 0, updates: 0 };
    const clearMasks = () => {
      for (const camera2 of cameras.keys()) for (const entry of entries) scatterBounds.get(entry.mesh).masks.delete(camera2);
      cameras.clear();
    };
    const update = (camera2, w, h, shadow = false) => {
      camera2.updateMatrixWorld(true);
      projection.multiplyMatrices(camera2.projectionMatrix, camera2.matrixWorldInverse);
      let cache = cameras.get(camera2);
      if (cache && cache.width === w && cache.height === h && cache.matrix.equals(projection)) return;
      if (!cache) {
        cache = { matrix: new Matrix4(), width: w, height: h, bytes: new Uint8Array(maskWidth * maskHeight * 4) };
        cameras.set(camera2, cache);
      }
      cache.matrix.copy(projection);
      cache.width = w;
      cache.height = h;
      const key = shadow ? "shadow" : "camera";
      let targets = resources.get(key);
      if (targets && (targets.depthTarget.width !== w || targets.depthTarget.height !== h)) {
        targets.depthTarget.dispose();
        targets.atlas.forEach((t) => t.dispose());
        targets = void 0;
      }
      if (!targets) {
        const depthTarget2 = viewer.renderer.createTarget({
          size: { width: w, height: h },
          depthBuffer: true,
          depthTexture: true,
          format: RedFormat,
          minFilter: NearestFilter,
          magFilter: NearestFilter
        }, false);
        const atlas2 = [makeTarget(Math.ceil(w / 2), h, false, true), makeTarget(Math.ceil(w / 2), h, false, true)];
        targets = { depthTarget: depthTarget2, atlas: atlas2 };
        resources.set(key, targets);
      }
      const { depthTarget, atlas } = targets;
      const start = performance.now(), oldTarget = renderer.getRenderTarget(), autoClear = renderer.autoClear;
      const alpha = renderer.getClearAlpha(), scissorTest = renderer.getScissorTest();
      const userData = renderer.userData;
      const modes = {
        shadowMapRender: userData.shadowMapRender,
        backgroundRender: userData.backgroundRender,
        opaqueRender: userData.opaqueRender,
        transparentRender: userData.transparentRender,
        transmissionRender: userData.transmissionRender,
        mainRenderPass: userData.mainRenderPass
      };
      renderer.getClearColor(previousColor);
      renderer.getScissor(scissor);
      try {
        renderer.autoClear = false;
        renderer.setScissorTest(false);
        setThreeRendererMode(renderer, {
          shadowMapRender: false,
          backgroundRender: false,
          opaqueRender: true,
          transparentRender: false,
          transmissionRender: false,
          mainRenderPass: false
        }, () => {
          renderer.setRenderTarget(depthTarget);
          renderer.clear(false, true, false);
          renderer.render(shadow ? shadowOccluders : mainOccluders, camera2);
          levels[0].set(0, 0, w, h);
          let lw = w, lh = h, level = 0;
          const offsets = [0, 0];
          quad.material = reduce;
          while ((lw > 1 || lh > 1) && level < 15) {
            level++;
            const index = (level - 1) % 2, nw = Math.ceil(lw / 2), nh = Math.ceil(lh / 2), y = offsets[index];
            reduce.uniforms.sourceDepth.value = level === 1 ? depthTarget.depthTexture : atlas[1 - index].texture;
            reduce.uniforms.sourceRect.value.copy(levels[level - 1]);
            reduce.uniforms.destinationOrigin.value.set(0, y);
            levels[level].set(0, y, nw, nh);
            atlas[index].viewport.set(0, y, nw, nh);
            renderer.setRenderTarget(atlas[index]);
            quad.render(renderer);
            offsets[index] += nh;
            lw = nw;
            lh = nh;
          }
          test.uniforms.lastLevel.value = level;
          test.uniforms.depth0.value = depthTarget.depthTexture;
          test.uniforms.depthA.value = atlas[0].texture;
          test.uniforms.depthB.value = atlas[1].texture;
          test.uniforms.screenSize.value.set(w, h);
          test.uniforms.viewProjectionBounds.value.copy(cache.matrix);
          quad.material = test;
          renderer.setRenderTarget(maskTarget);
          quad.render(renderer);
          renderer.readRenderTargetPixels(maskTarget, 0, 0, maskWidth, maskHeight, cache.bytes);
          revision++;
          let visible = 0;
          for (let i = 0; i < count; i++) if (cache.bytes[i]) visible++;
          for (const entry of entries) scatterBounds.get(entry.mesh).masks.set(
            camera2,
            { bytes: cache.bytes, offset: entry.offset, revision, matrix: cache.matrix, width: w, height: h }
          );
          stats.visibleInstances = visible;
        });
        stats.milliseconds = performance.now() - start;
        stats.updates++;
      } finally {
        Object.assign(userData, modes);
        renderer.setRenderTarget(oldTarget);
        renderer.setScissor(scissor);
        renderer.setScissorTest(scissorTest);
        renderer.autoClear = autoClear;
        renderer.setClearColor(previousColor, alpha);
      }
    };
    const controller = {
      enabled: true,
      shadows: true,
      stats,
      update,
      clearMasks,
      probeChunks: async (levels2 = 3) => {
        const enabled = viewer.renderEnabled;
        viewer.renderEnabled = false;
        try {
          const camera2 = viewer.scene.activeCamera.cameraObject;
          renderer.getDrawingBufferSize(size);
          update(camera2, size.x, size.y);
          const { probeVegetationChunks } = await import("./nr-vegetation-chunk-probe");
          return await probeVegetationChunks(
            viewer,
            entries,
            camera2,
            cameras.get(camera2),
            resources.get("camera"),
            mainOccluders,
            levels2
          );
        } finally {
          viewer.renderEnabled = enabled;
        }
      },
      dispose: () => {
        viewer.removeEventListener("preRender", preRender);
        clearMasks();
        resources.forEach(({ depthTarget, atlas }) => {
          depthTarget.dispose();
          atlas.forEach((t) => t.dispose());
        });
        maskTarget.dispose();
        boundsTexture.dispose();
        reduce.dispose();
        test.dispose();
        materials.forEach((m) => m.dispose());
      }
    };
    const size = new Vector2();
    const preRender = () => {
      if (!controller.enabled) {
        if (cameras.size) clearMasks();
        return;
      }
      try {
        renderer.getDrawingBufferSize(size);
        const source = viewer.getPlugin(GBufferPlugin)?.getTarget();
        if (source) size.set(source.width, source.height);
        update(viewer.scene.activeCamera.cameraObject, size.x, size.y);
        if (controller.shadows && (renderer.shadowMap.autoUpdate || renderer.shadowMap.needsUpdate)) {
          viewer.scene.traverseVisible((object) => {
            if (!object.isDirectionalLight || !object.castShadow) return;
            object.shadow.updateMatrices(object);
            update(object.shadow.camera, object.shadow.mapSize.x, object.shadow.mapSize.y, true);
          });
        }
      } catch (error) {
        controller.enabled = false;
        clearMasks();
        console.warn("Vegetation occlusion unavailable; retaining all instances", error);
      }
    };
    viewer.addEventListener("preRender", preRender);
    return controller;
  }

  // src/scenes/nr-opaque-pass-fusion.ts
  function createOpaquePassFusion(viewer) {
    const renderer = viewer.renderer.rendererObject, gl = renderer.getContext();
    const gb = viewer.getPlugin(GBufferPlugin), velocity = viewer.getPlugin(DlssVelocityBufferPlugin);
    const gp = gb?.pass?.passObject, vp = velocity?.pass?.passObject;
    const gt = gb?.getTarget(), vt = velocity?.getVelocityBuffer();
    if (!renderer.capabilities.isWebGL2 || !gp || !vp || !gt || !vt || gt.textures.length !== 2 || gt.samples || vt.samples || gt.depthTexture && gt.depthTexture.type !== UnsignedIntType || renderer.capabilities.logarithmicDepthBuffer) return void 0;
    const certified = /* @__PURE__ */ new Set([
      "Foliage001_2K-JPG_Color",
      "Grass procedural .003",
      "Brown grass.003",
      "Bark.001",
      "Stem",
      "Flower.001",
      "Flower.002"
    ]);
    const eligible = /* @__PURE__ */ new Set();
    viewer.scene.modelRoot.traverseVisible((object) => {
      const mesh = object, m = mesh.material;
      if (!mesh.isInstancedMesh || Array.isArray(m) || !certified.has(m.name) || m.transparent || m.alphaHash || m.alphaMap || m.alphaTest || m.transmission || m.displacementMap || m.polygonOffset || m.clippingPlanes?.length || m.depthWrite === false || m.userData.pluginsDisabled || m.userData.renderToDepth === false || mesh.morphTargetInfluences?.length) return;
      eligible.add(mesh);
    });
    if (!eligible.size) return void 0;
    const gm = gp.overrideMaterial;
    const vm = vp.overrideMaterial;
    const original = {
      gRender: gp.render,
      vRender: vp.render,
      compile: gm.onBeforeCompile,
      key: gm.customProgramCacheKey,
      before: gm.onBeforeRender
    };
    const previousUniform = gm.extraUniformsToUpload.modelMatrixPrevious;
    const fused = gt.clone();
    fused.textures.push(vt.texture.clone());
    fused.textures[2].name = "opaqueScatterMotion";
    let enabled = false, fusedFrame = false, drawingOpaque = false;
    const motionBody = ssVelocityFrag_default.slice(0, ssVelocityFrag_default.indexOf("void main()"));
    const motionWrite = velocity.rawOutput ? "gScatterMotion = vec4(clamp(computeScreenSpaceVelocity2(),-1.0,1.0),0.,1.);" : `
        vec2 scatterVelocity=clamp(computeScreenSpaceVelocity2(),-1.0,1.0);
        scatterVelocity=sign(scatterVelocity)*pow(abs(scatterVelocity),vec2(1./4.));
        gScatterMotion=vec4(scatterVelocity*.5+.5,1.,1.);`;
    gm.onBeforeCompile = function(shader, r) {
      original.compile.call(this, shader, r);
      if (!enabled) return;
      shader.uniforms.currentProjectionViewMatrix = vm.uniforms.currentProjectionViewMatrix;
      shader.uniforms.lastProjectionViewMatrix = vm.uniforms.lastProjectionViewMatrix;
      shader.vertexShader = `varying vec3 vWorldPosition; varying vec3 vWorldPositionPrevious;
            uniform mat4 modelMatrixPrevious;
` + shader.vertexShader.replace("#include <project_vertex>", `
            vec4 scatterPosition = vec4(transformed,1.);
            #ifdef USE_INSTANCING
            scatterPosition = instanceMatrix * scatterPosition;
            #endif
            vWorldPosition = (modelMatrix * scatterPosition).xyz;
            vWorldPositionPrevious = (modelMatrixPrevious * scatterPosition).xyz;
            #include <project_vertex>`);
      shader.fragmentShader = "layout(location=2) out vec4 gScatterMotion;\n" + motionBody + shader.fragmentShader.replace(/}\s*$/, motionWrite + "\n}");
    };
    gm.customProgramCacheKey = function() {
      return original.key.call(this) + "|scatter-motion:" + enabled;
    };
    gm.onBeforeRender = function(r, s, c, geometry2, object, group) {
      if (drawingOpaque) {
        vm.onBeforeRender(r, s, c, geometry2, object, group);
        gm.extraUniformsToUpload.modelMatrixPrevious = vm.extraUniformsToUpload.modelMatrixPrevious;
      }
      original.before.call(this, r, s, c, geometry2, object, group);
    };
    const select = (opaque, callback) => {
      const hidden = [];
      viewer.scene.modelRoot.traverseVisible((object) => {
        const mesh = object;
        if (mesh.isMesh && eligible.has(mesh) !== opaque) {
          hidden.push(mesh);
          mesh.visible = false;
        }
      });
      try {
        callback();
      } finally {
        for (const mesh of hidden) mesh.visible = true;
      }
    };
    const copy = (source, destination, attachment, output, depth) => {
      renderer.setRenderTarget(destination);
      const oldRead = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING), oldDraw = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING);
      const buffers = destination.textures.map((_, i) => gl.COLOR_ATTACHMENT0 + i);
      try {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, source);
        gl.readBuffer(gl.COLOR_ATTACHMENT0 + attachment);
        gl.drawBuffers(buffers.map((b, i) => i === output ? b : gl.NONE));
        gl.blitFramebuffer(
          0,
          0,
          fused.width,
          fused.height,
          0,
          0,
          destination.width,
          destination.height,
          gl.COLOR_BUFFER_BIT | (depth ? gl.DEPTH_BUFFER_BIT : 0),
          gl.NEAREST
        );
        gl.readBuffer(gl.COLOR_ATTACHMENT0);
        gl.drawBuffers(buffers);
      } finally {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, oldRead);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, oldDraw);
      }
    };
    vp.render = function(...args) {
      fusedFrame = false;
      if (!enabled || !gb.enabled || gt.width !== vt.width || gt.height !== vt.height || gt.samples || vt.samples) {
        return original.vRender.apply(this, args);
      }
      fused.setSize(gt.width, gt.height);
      gb.pass?.update?.();
      const render = renderer.render;
      renderer.render = function(scene, camera2) {
        if (drawingOpaque) {
          const zero = velocity.rawOutput ? 0 : 0.5;
          gl.clearBufferfv(gl.COLOR, 2, new Float32Array([zero, zero, 1, 1]));
          return render.call(this, scene, camera2);
        }
        if (scene.overrideMaterial !== vm) return render.call(this, scene, camera2);
        const old = { target: gp.target, scene: gp.scene, camera: gp.camera, clear: gp.clear };
        try {
          drawingOpaque = true;
          gp.target = fused;
          gp.scene = scene;
          gp.camera = camera2;
          gp.clear = true;
          select(true, () => original.gRender.call(gp, renderer, null));
          renderer.setRenderTarget(fused);
          const framebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
          copy(framebuffer, vt, 2, 0, true);
          copy(framebuffer, gt, 0, 0, true);
          copy(framebuffer, gt, 1, 1, false);
          fusedFrame = true;
        } finally {
          drawingOpaque = false;
          gp.target = old.target;
          gp.scene = old.scene;
          gp.camera = old.camera;
          gp.clear = old.clear;
          renderer.setRenderTarget(vt);
        }
        select(false, () => render.call(this, scene, camera2));
      };
      try {
        return original.vRender.apply(this, args);
      } finally {
        renderer.render = render;
      }
    };
    gp.render = function(...args) {
      if (!fusedFrame) return original.gRender.apply(this, args);
      const clear = this.clear;
      try {
        this.clear = false;
        select(false, () => original.gRender.apply(this, args));
      } finally {
        this.clear = clear;
        fusedFrame = false;
      }
    };
    const setEnabled = (value) => {
      enabled = value;
      gm.needsUpdate = true;
      viewer.setDirty();
    };
    setEnabled(true);
    return { setEnabled, get enabled() {
      return enabled;
    }, eligibleMeshes: eligible.size, dispose: () => {
      gp.render = original.gRender;
      vp.render = original.vRender;
      gm.onBeforeCompile = original.compile;
      gm.customProgramCacheKey = original.key;
      gm.onBeforeRender = original.before;
      if (previousUniform) gm.extraUniformsToUpload.modelMatrixPrevious = previousUniform;
      else delete gm.extraUniformsToUpload.modelMatrixPrevious;
      gm.needsUpdate = true;
      fused.dispose();
    } };
  }

  // src/scenes/nr-static-draw-groups.ts
  function createStaticDrawGroups(viewer) {
    const groups = /* @__PURE__ */ new Map();
    const lifecycle = /* @__PURE__ */ new Set([
      "name",
      "__iModelSetup",
      "__meshSetup",
      "setDirty",
      "dispose",
      "__objectUpdater",
      "setMaterial",
      "setGeometry",
      "__autoParentDispatchEvents",
      "parentRoot",
      "__materialUpdater",
      "__textureUpdater"
    ]);
    const drawData = (mesh) => Object.entries(mesh.userData).filter(([key]) => !lifecycle.has(key));
    const references = /* @__PURE__ */ new WeakMap();
    let nextReference = 0;
    const identity = (value) => {
      if ((typeof value !== "object" || !value) && typeof value !== "function") return value;
      if (!references.has(value)) references.set(value, nextReference++);
      return { reference: references.get(value) };
    };
    viewer.scene.modelRoot.traverseVisible((object) => {
      const mesh = object, m = mesh.material, g = mesh.geometry;
      if (!mesh.isMesh || mesh.isInstancedMesh || mesh.isSkinnedMesh || mesh.children.length || Array.isArray(m) || m.transparent || m.transmission || mesh.morphTargetInfluences?.length || mesh.customDepthMaterial || mesh.customDistanceMaterial || !mesh.parent || g.drawRange.start !== 0 || g.drawRange.count < (g.index?.count ?? g.attributes.position.count)) return;
      const attributes = Object.entries(g.attributes).map(([name, a]) => [name, a.itemSize, a.normalized, a.array.constructor.name, a.gpuType]).sort();
      const key = JSON.stringify([
        mesh.parent.uuid,
        mesh.matrix.elements,
        m.uuid,
        mesh.castShadow,
        mesh.receiveShadow,
        mesh.layers.mask,
        mesh.renderOrder,
        mesh.frustumCulled,
        drawData(mesh).map(([key2, value]) => [key2, identity(value)]).sort(),
        !!g.index,
        attributes
      ]);
      const row = groups.get(key);
      if (row) row.push(mesh);
      else groups.set(key, [mesh]);
    });
    const batches = [];
    for (const sources of groups.values()) {
      if (sources.length < 2) continue;
      const geometry2 = mergeGeometries(sources.map((m) => m.geometry));
      if (!geometry2) continue;
      const first = sources[0], mesh = new Mesh(geometry2, first.material);
      mesh.name = first.material.name + " static draw group";
      mesh.matrix.copy(first.matrix);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = first.castShadow;
      mesh.receiveShadow = first.receiveShadow;
      mesh.layers.mask = first.layers.mask;
      mesh.renderOrder = first.renderOrder;
      mesh.frustumCulled = first.frustumCulled;
      mesh.userData = Object.fromEntries(drawData(first));
      geometry2.computeBoundingBox();
      geometry2.computeBoundingSphere();
      batches.push({ sources, mesh, parent: first.parent });
    }
    let enabled = false;
    const setEnabled = (value) => {
      if (value === enabled) return;
      enabled = value;
      for (const batch of batches) {
        if (value) {
          for (const source of batch.sources) source.removeFromParent();
          batch.parent.add(batch.mesh);
        } else {
          batch.mesh.removeFromParent();
          for (const source of batch.sources) batch.parent.add(source);
        }
      }
      viewer.scene.modelRoot.updateWorldMatrix(true, true);
      viewer.setDirty();
    };
    setEnabled(true);
    return {
      setEnabled,
      get enabled() {
        return enabled;
      },
      sourceDraws: batches.reduce((n, b) => n + b.sources.length, 0),
      mergedDraws: batches.length,
      dispose: () => {
        setEnabled(false);
        for (const b of batches) b.mesh.geometry.dispose();
      }
    };
  }

  // src/scenes/nr-blendkit-scenes.ts
  var blendkitScenes = {
    "simple-lighting": "Mustang \xB7 Suspended studio",
    "vege-packshot": "Nature \xB7 Botanical packshot",
    arunthayan: "Arunthayan",
    "selection-five": "Cowboy Gramps",
    "selection-six": "Highlands \xB7 Golden hour"
  };
  async function loadBlendkitScene(viewer, id, dispose) {
    const base = assetUrl(`/scenes/blendkit/${id}/`);
    const response = await fetch(`${base}view.json`);
    if (!response.ok) throw new Error(`${blendkitScenes[id]} assets are unavailable`);
    const view = await response.json();
    await viewer.load(`${base}${view.model || "scene.glb"}`, { importConfig: false, autoScale: false, autoCenter: false });
    if (view.instances) {
      ;
      window.dlssLoading?.stage(`Preparing ${blendkitScenes[id]}`, "Restoring vegetation instances");
      const [metadata, binary] = await Promise.all([
        fetch(`${base}instances.json`).then((r) => {
          if (!r.ok) throw new Error("Missing instance manifest");
          return r.json();
        }),
        fetch(`${base}instances.bin`).then((r) => {
          if (!r.ok) throw new Error("Missing instance transforms");
          return r.arrayBuffer();
        })
      ]);
      viewer.scene.modelRoot.updateWorldMatrix(true, true);
      for (const entry of metadata) {
        const source = viewer.scene.modelRoot.getObjectByName(PropertyBinding.sanitizeNodeName(entry.name));
        if (!source) throw new Error(`Missing instance geometry: ${entry.name}`);
        const matrices = new Float32Array(binary, entry.offset, entry.count * 16);
        const cells = /* @__PURE__ */ new Map();
        const size = view.instanceCellSize ?? 1;
        for (let i = 0; i < entry.count; i++) {
          const key = `${Math.floor(matrices[i * 16 + 12] / size)},${Math.floor(matrices[i * 16 + 14] / size)}`;
          const indices = cells.get(key);
          if (indices) indices.push(i);
          else cells.set(key, [i]);
        }
        source.traverse((child) => {
          const mesh = child;
          if (!mesh.isMesh) return;
          const geometry2 = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);
          const batch = createCulledInstances(
            geometry2,
            mesh.material,
            matrices,
            [...cells.values()],
            viewer.renderer.rendererObject
          );
          batch.name = entry.name + " instances";
          viewer.scene.modelRoot.add(batch);
          dispose(() => {
            batch.removeFromParent();
            batch.dispose();
            geometry2.dispose();
          });
        });
        source.visible = false;
      }
    }
    if (view.groom) {
      ;
      window.dlssLoading?.stage(`Preparing ${blendkitScenes[id]}`, "Loading the original hair and beard groom");
      const response2 = await fetch(base + (typeof view.groom === "string" ? view.groom : "groom.json"));
      if (!response2.ok) throw new Error("The portrait groom manifest is missing");
      const groom = await response2.json();
      for (const [index, part] of groom.entries()) {
        const response3 = await fetch(base + part.file);
        if (!response3.ok || !response3.body) throw new Error(`Missing groom: ${part.name}`);
        let loaded = 0, updated = 0;
        const total = Number(response3.headers.get("content-length")) || 0;
        const progress = new TransformStream({ transform(chunk, controller) {
          loaded += chunk.byteLength;
          const now = performance.now();
          if (now - updated > 100 || loaded === total) {
            ;
            window.dlssLoading?.stage(
              `Loading ${blendkitScenes[id]}`,
              `Hair detail ${index + 1} of ${groom.length} \xB7 ${part.name}`,
              { loaded, total }
            );
            updated = now;
          }
          controller.enqueue(chunk);
        } });
        const bytes = await new Response(response3.body.pipeThrough(progress).pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
        const geometry2 = new BufferGeometry();
        if (part.encoding === "rings-v2") {
          const decoded = decodeGroomRings(bytes, part);
          geometry2.setAttribute("position", new BufferAttribute(decoded.position, 3));
          geometry2.setAttribute("normal", new BufferAttribute(decoded.normal, 3));
          geometry2.setIndex(new BufferAttribute(decoded.index, 1));
          geometry2.setAttribute("color", new BufferAttribute(decoded.color, 4, true));
        } else {
          geometry2.setAttribute("position", new BufferAttribute(new Float32Array(bytes, 0, part.vertices * 3), 3));
          geometry2.setAttribute("normal", new BufferAttribute(new Float32Array(bytes, part.vertices * 12, part.vertices * 3), 3));
          geometry2.setIndex(new BufferAttribute(new Uint32Array(bytes, part.vertices * 24, part.indices), 1));
          if (part.vertexOpacity) geometry2.setAttribute("color", new BufferAttribute(
            new Uint8Array(bytes, part.vertices * 24 + part.indices * 4, part.vertices * 4),
            4,
            true
          ));
        }
        const material = new MeshPhysicalMaterial({
          color: part.colorLinear ? new Color().fromArray(part.colorLinear) : part.color,
          roughness: 0.48,
          metalness: 0,
          opacity: part.rasterOpacity ?? 1,
          vertexColors: !!part.vertexOpacity,
          alphaHash: !!part.vertexOpacity
        });
        const hair = new Mesh(geometry2, material);
        hair.name = part.name;
        await viewer.getManager().addImportedSingle(hair, { autoScale: false, autoCenter: false, addToRoot: true });
      }
    }
    await viewer.setEnvironmentMap(`${base}lighting.hdr`);
    const scene = viewer.scene;
    scene.background = new Color(view.background);
    if (view.backgroundEnvironment) scene.background = scene.environment;
    scene.backgroundIntensity = view.backgroundIntensity ?? 1;
    scene.environmentIntensity = view.environmentIntensity;
    if (view.fogDensity) scene.fog = new FogExp2(view.background, view.fogDensity);
    const tonemap = viewer.getPlugin(TonemapPlugin);
    if (tonemap) {
      tonemap.toneMapping = view.toneMapping === "Neutral" ? NeutralToneMapping : view.toneMapping === "AgX" ? AgXToneMapping : ACESFilmicToneMapping;
      tonemap.exposure = view.exposure;
      if (tonemap.config) {
        const previous = tonemap.config.tonemapBackground;
        tonemap.config.tonemapBackground = !!view.backgroundEnvironment;
        tonemap.config.setDirty();
        dispose(() => {
          if (tonemap.config) {
            tonemap.config.tonemapBackground = previous;
            tonemap.config.setDirty();
          }
        });
      }
    }
    if (view.sun) {
      const light = new DirectionalLight2(new Color(view.sun.color), view.sun.intensity);
      const target = new Vector3().fromArray(view.camera.target);
      light.position.copy(target).addScaledVector(new Vector3().fromArray(view.sun.direction), 100);
      light.target.position.copy(target).sub(light.position);
      light.castShadow = true;
      light.shadow.mapSize.set(4096, 4096);
      const extent = view.sun.shadowExtent ?? 35;
      Object.assign(light.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent, near: 0.1, far: 400 });
      light.shadow.camera.updateProjectionMatrix();
      light.shadow.normalBias = view.sun.normalBias ?? 0.025;
      scene.add(light);
    }
    scene.modelRoot.traverse((object) => {
      const mesh = object;
      if (!mesh.isMesh) return;
      mesh.castShadow = mesh.receiveShadow = true;
      for (const mat of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
        const m = mat;
        if (id === "arunthayan" && m.name === "Body") {
          m.roughness = Math.max(m.roughness, 0.9);
          m.clearcoat = Math.min(m.clearcoat, 0.08);
          m.clearcoatRoughness = Math.max(m.clearcoatRoughness, 0.85);
          m.specularIntensity = Math.min(m.specularIntensity, 0.5);
          m.needsUpdate = true;
        }
        if (view.instances && m.transparent && !m.transmission) {
          m.transparent = false;
          m.alphaHash = true;
          m.alphaTest = Math.max(m.alphaTest, 1e-6);
          m.depthWrite = true;
          m.needsUpdate = true;
        }
        for (const key of ["map", "normalMap", "roughnessMap", "metalnessMap", "alphaMap", "emissiveMap"]) {
          if (m[key]) m[key].anisotropy = 16;
        }
      }
    });
    if (view.floor) {
      const original = scene.modelRoot.getObjectByName(view.floor);
      if (!original?.isMesh) throw new Error("The studio floor is missing");
      original.updateWorldMatrix(true, false);
      const ground = await viewer.addPlugin(new DlssGroundPlugin({ autoAdjustTransform: false }));
      ground.bakedShadows = false;
      ground.autoBakeShadows = false;
      ground.groundReflection = true;
      ground.physicalReflections = false;
      ground.size = 1;
      const geometry2 = original.geometry.clone().applyMatrix4(original.matrixWorld);
      geometry2.computeBoundingBox();
      const floorCenter = geometry2.boundingBox.getCenter(new Vector3());
      geometry2.translate(-floorCenter.x, -floorCenter.y, -floorCenter.z);
      geometry2.rotateX(Math.PI / 2);
      ground.setGeometry(geometry2);
      ground.material.color.set(11053224);
      ground.material.roughness = 0;
      ground.material.metalness = 1;
      original.visible = false;
      ground.refreshOptions();
      ground.mesh.modelObject.position.copy(floorCenter);
      ground.mesh.modelObject.rotation.set(-Math.PI / 2, 0, 0);
      ground.mesh.modelObject.updateMatrixWorld(true);
    }
    const ao = viewer.getPlugin(SSAOPlugin).passes.ssao.passObject.parameters;
    ao.occlusionWorldRadius = view.giRadius ?? (id.startsWith("selection-") || id === "arunthayan" ? 0.12 : 1);
    const camera2 = scene.activeCamera;
    Object.assign(camera2.cameraObject.userData, { autoNearFar: false, minNearPlane: view.camera.near, maxFarPlane: view.camera.far });
    camera2.setCameraOptions(view.camera);
    const orbit = camera2.getControls();
    if (orbit) {
      const distance = new Vector3().fromArray(view.camera.position).distanceTo(new Vector3().fromArray(view.camera.target));
      orbit.minDistance = distance * 0.12;
      orbit.maxDistance = distance * 2.5;
      orbit.maxPolarAngle = Math.PI * 0.75;
      orbit.update();
    }
    if (view.instances) {
      const groups = createStaticDrawGroups(viewer);
      window.dlssStaticDrawGroups = groups;
      dispose(() => {
        groups.dispose();
        delete window.dlssStaticDrawGroups;
      });
      const occlusion = createVegetationOcclusion(viewer);
      window.dlssVegetationOcclusion = occlusion;
      dispose(() => {
        occlusion?.dispose();
        delete window.dlssVegetationOcclusion;
      });
      const fusion = createOpaquePassFusion(viewer);
      window.dlssOpaquePassFusion = fusion;
      dispose(() => {
        fusion?.dispose();
        delete window.dlssOpaquePassFusion;
      });
    }
    viewer.setDirty();
  }

  // src/scenes/nr-demo-scenes.ts
  var sceneNames = { "selection-five": blendkitScenes["selection-five"] };
  var modelExtensions = /* @__PURE__ */ new Set(["glb", "gltf", "drc", "obj", "fbx", "stl", "3dm", "zip"]);
  var defaultLocalEnvironment = "studio-small-08";
  var localEnvironments = {
    embedded: { label: "Embedded in model" },
    "studio-small-08": { label: "Neutral studio \xB7 soft", path: assetUrl("/environments/studio_small_08_1k.hdr") },
    "studio-small-09": { label: "Neutral studio \xB7 directional", path: assetUrl("/environments/studio_small_09_1k.hdr") },
    "white-studio-04": { label: "Bright product studio", path: assetUrl("/environments/white_studio_04_1k.hdr") },
    "cloud-layers": { label: "Outdoor \xB7 bright midday", path: assetUrl("/environments/cloud_layers_1k.hdr") },
    "venice-sunset": { label: "Outdoor \xB7 warm sunset", path: assetUrl("/environments/venice_sunset_1k.hdr") },
    "autumn-forest-02": { label: "Forest \xB7 soft overcast", path: assetUrl("/environments/autumn_forest_02_1k.hdr") }
  };
  async function mountDemoScenes(viewer, bridge) {
    const globals = window;
    const { scene } = viewer;
    const roots = new Set(scene.children);
    const ambientOcclusion = await viewer.getOrAddPlugin(SSAOPlugin);
    ambientOcclusion.enabled = true;
    const defaultAoRadius = ambientOcclusion.passes.ssao.passObject.parameters.occlusionWorldRadius;
    const sharedPlugins = new Set(Object.values(viewer.plugins));
    const camera2 = scene.activeCamera;
    const cameraDefaults = camera2.getCameraOptions();
    const cameraData = { ...camera2.cameraObject.userData };
    const orbit = camera2.getControls();
    const orbitDefaults = orbit ? {
      minDistance: orbit.minDistance,
      maxDistance: orbit.maxDistance,
      minPolarAngle: orbit.minPolarAngle,
      maxPolarAngle: orbit.maxPolarAngle
    } : null;
    const tonemap = viewer.getPlugin("Tonemap");
    const defaultExposure = tonemap?.exposure;
    const defaultToneMapping = tonemap?.toneMapping;
    const progressive = viewer.getPlugin("Progressive");
    const defaultFrames = progressive?.maxFrameCount;
    const shadowType = viewer.renderer.rendererObject.shadowMap.type;
    let disposers = [];
    let changing = false;
    let loadingName = "scene";
    let localEnvironmentId = defaultLocalEnvironment;
    let embeddedEnvironment = null;
    const localEnvironmentTextures = /* @__PURE__ */ new Map();
    viewer.getManager()?.importer?.addEventListener("importFile", (event) => {
      if (!changing || event.state !== "downloading") return;
      const asset = /\.(hdr|exr)(\?|$)/i.test(event.path) ? "Environment" : "3D model";
      globals.dlssLoading?.stage(`Loading ${loadingName}`, `Downloading ${asset.toLowerCase()}`, {
        loaded: event.loadedBytes || 0,
        total: event.totalBytes || 0
      });
      if (event.totalBytes > 0 && event.loadedBytes >= event.totalBytes) {
        globals.dlssLoading?.stage(`Preparing ${loadingName}`, `${asset} downloaded \xB7 preparing materials and textures`);
      }
    });
    viewer.getManager()?.importer?.addEventListener("processFileStart", () => {
      if (changing) globals.dlssLoading?.stage(`Preparing ${loadingName}`, "Preparing scene materials and textures");
    });
    async function clearScene() {
      disposers.forEach((dispose) => dispose());
      disposers = [];
      for (const plugin of Object.values(viewer.plugins).reverse()) {
        if (!sharedPlugins.has(plugin)) await viewer.removePlugin(plugin);
      }
      ambientOcclusion.enabled = true;
      ambientOcclusion.passes.ssao.passObject.parameters.occlusionWorldRadius = defaultAoRadius;
      scene.disposeSceneModels();
      for (const object of [...scene.children]) {
        if (roots.has(object)) continue;
        object.traverse((child) => {
          const light = child;
          light.shadow?.dispose();
          const mesh = child;
          mesh.geometry?.dispose();
          if (mesh.material) for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) material.dispose();
        });
        object.removeFromParent();
      }
      const environmentResources = /* @__PURE__ */ new Set([scene.environment, ...localEnvironmentTextures.values()]);
      if (scene.background?.isTexture) environmentResources.add(scene.background);
      if (embeddedEnvironment?.environment) environmentResources.add(embeddedEnvironment.environment);
      if (embeddedEnvironment?.background?.isTexture) environmentResources.add(embeddedEnvironment.background);
      scene.environment = null;
      scene.background = null;
      for (const resource of environmentResources) resource?.dispose?.();
      localEnvironmentTextures.clear();
      embeddedEnvironment = null;
      localEnvironmentId = defaultLocalEnvironment;
      globals.dlssEmbeddedEnvironment = false;
      scene.fog = null;
      scene.environmentIntensity = 1;
      scene.backgroundIntensity = 1;
      scene.environmentRotation.set(0, 0, 0);
      scene.backgroundRotation.set(0, 0, 0);
      if (tonemap) {
        tonemap.exposure = defaultExposure;
        tonemap.toneMapping = defaultToneMapping;
      }
      if (progressive) progressive.maxFrameCount = defaultFrames;
      viewer.renderer.rendererObject.shadowMap.type = shadowType;
      for (const key of ["autoNearFar", "minNearPlane", "maxFarPlane"]) {
        if (key in cameraData) camera2.cameraObject.userData[key] = cameraData[key];
        else delete camera2.cameraObject.userData[key];
      }
      camera2.setCameraOptions(cameraDefaults);
      if (orbit && orbitDefaults) Object.assign(orbit, orbitDefaults);
    }
    async function applyLocalEnvironment(id) {
      const selection = localEnvironments[id];
      if (!selection) throw new Error("Unknown HDR environment");
      if (id === "embedded") {
        if (!embeddedEnvironment) throw new Error("This model does not contain an embedded WebGI environment");
        scene.environment = embeddedEnvironment.environment;
        scene.background = embeddedEnvironment.background;
        scene.environmentIntensity = embeddedEnvironment.environmentIntensity;
        scene.backgroundIntensity = embeddedEnvironment.backgroundIntensity;
        scene.environmentRotation.copy(embeddedEnvironment.environmentRotation);
        scene.backgroundRotation.copy(embeddedEnvironment.backgroundRotation);
      } else {
        let texture = localEnvironmentTextures.get(id);
        if (!texture) {
          texture = await viewer.setEnvironmentMap(selection.path, { setBackground: true });
          if (!texture) throw new Error(`Unable to load ${selection.label}`);
          localEnvironmentTextures.set(id, texture);
        } else {
          scene.environment = texture;
          scene.background = texture;
        }
        scene.environmentIntensity = 1;
        scene.backgroundIntensity = 1;
        scene.environmentRotation.set(0, 0, 0);
        scene.backgroundRotation.set(0, 0, 0);
      }
      localEnvironmentId = id;
      globals.dlssLocalEnvironment = id;
      window.dispatchEvent(new CustomEvent("dlss-local-environment-changed", { detail: { id } }));
    }
    async function changeScene(id) {
      if (!sceneNames[id]) throw new Error("Unknown demo scene");
      if (changing) throw new Error("A scene is already loading");
      changing = true;
      globals.dlssSceneLoading = true;
      document.body.dataset.sceneLoading = "true";
      globals.dlssSceneReady = false;
      loadingName = sceneNames[id];
      globals.dlssLoading?.begin(`Loading ${loadingName}`, "Preparing the scene");
      try {
        await globals.dlssDemoUi?.beforeSceneChange();
        viewer.renderEnabled = false;
        document.querySelector("#dlssWebGpuOutput")?.classList.remove("visible");
        const status = document.querySelector("#dlssWebGpuStatus");
        status.textContent = `Loading ${sceneNames[id]}\u2026`;
        status.dataset.state = "active";
        await clearScene();
        if (blendkitScenes[id]) {
          await loadBlendkitScene(viewer, id, (dispose) => disposers.push(dispose));
        } else if (id === "bistro") {
          const response = await fetch(assetUrl("/scenes/bistro/view.json"));
          if (!response.ok) throw new Error("Bistro camera data is unavailable");
          const view = await response.json();
          await viewer.setEnvironmentMap(assetUrl("/scenes/bistro/san_giuseppe_bridge_4k.hdr"), { setBackground: true });
          await viewer.load(assetUrl("/scenes/bistro/BistroExterior.glb"), { importConfig: false, autoScale: false });
          await configureBistroLighting(viewer);
          camera2.setCameraOptions({ position: view.position, target: view.target, fov: view.fov, near: 0.05, far: 1e3 });
        } else if (id === "lone-monk") {
          await viewer.load(assetUrl("/scenes/lone-monk/lone-monk.glb"), { importConfig: false, autoScale: false });
          await configureLoneMonkScene(viewer, (dispose) => disposers.push(dispose));
        }
        globals.dlssSceneId = id;
        globals.dlssSceneLabel = sceneNames[id];
        globals.dlssSceneReady = true;
        const query2 = new URLSearchParams(location.search);
        const mode = query2.get("sr") === "1" ? query2.get("srChain") === "1" ? "SR + Neural rendering" : "Super resolution" : "Neural rendering";
        document.title = `${sceneNames[id]} \xB7 DLSS 5 WebGPU`;
        const heading = document.querySelector("h1");
        if (heading) heading.textContent = `${sceneNames[id]} / ${mode}`;
        document.querySelector("#mcanvas")?.setAttribute("aria-label", `Interactive ${sceneNames[id]} viewer`);
        bridge.resetHistory();
        viewer.renderer.refreshPipeline();
        viewer.renderer.resetShadows();
        viewer.getPlugin("FrameFade")?.stopTransition();
        scene.setDirty({ sceneUpdate: true, frameFade: false });
        viewer.renderEnabled = true;
        viewer.setDirty();
        globals.dlssLoading?.stage("Preparing first frame", `${loadingName} loaded \xB7 preparing the rendered view`);
        globals.dlssDemoUi?.afterSceneChange();
        window.dispatchEvent(new CustomEvent("dlss-scene-ready", { detail: { id, label: sceneNames[id] } }));
      } catch (error) {
        globals.dlssLoading?.fail(error);
        await clearScene();
        throw error;
      } finally {
        changing = false;
        globals.dlssSceneLoading = false;
        delete document.body.dataset.sceneLoading;
        globals.dlssLoading?.releaseScene?.();
      }
    }
    async function importLocalFiles(files) {
      if (changing) throw new Error("A scene is already loading");
      const rootFiles = [...files.keys()].filter((path) => modelExtensions.has(path.split(".").pop()?.toLowerCase() ?? ""));
      if (!rootFiles.length) {
        const supported = [...modelExtensions].map((extension) => `.${extension}`).join(", ");
        throw new Error(`No supported 3D model found. Supported formats: ${supported}`);
      }
      changing = true;
      globals.dlssSceneLoading = true;
      document.body.dataset.sceneLoading = "true";
      globals.dlssSceneReady = false;
      const filename = rootFiles[0].split(/[\\/]/).pop() || "Local model";
      loadingName = filename.replace(/\.[^.]+$/, "") || filename;
      globals.dlssLoading?.begin(`Loading ${loadingName}`, "Preparing local files");
      try {
        await globals.dlssDemoUi?.beforeSceneChange();
        viewer.renderEnabled = false;
        document.querySelector("#dlssWebGpuOutput")?.classList.remove("visible");
        const status = document.querySelector("#dlssWebGpuStatus");
        status.textContent = `Loading ${loadingName}\u2026`;
        status.dataset.state = "active";
        await clearScene();
        const manager = viewer.getManager();
        const options = { autoScale: true, autoCenter: true, autoScaleRadius: 2 };
        const imported = await manager?.importer?.importFiles(files, options);
        const assets = [...imported?.values() ?? []].flat(2).filter(Boolean);
        if (!assets.length) throw new Error(`WebGI could not import ${filename}`);
        const added = manager?.addProcessedAssets(assets, options).filter(Boolean) ?? [];
        if (!added.length) throw new Error(`WebGI did not add ${filename} to the scene`);
        if (scene.environment) {
          embeddedEnvironment = {
            environment: scene.environment,
            background: scene.background,
            environmentIntensity: scene.environmentIntensity,
            backgroundIntensity: scene.backgroundIntensity,
            environmentRotation: scene.environmentRotation.clone(),
            backgroundRotation: scene.backgroundRotation.clone()
          };
          globals.dlssEmbeddedEnvironment = true;
          await applyLocalEnvironment("embedded");
        } else {
          globals.dlssEmbeddedEnvironment = false;
          await applyLocalEnvironment(defaultLocalEnvironment);
        }
        globals.dlssSceneId = "local-file";
        globals.dlssSceneLabel = loadingName;
        globals.dlssSceneReady = true;
        const query2 = new URLSearchParams(location.search);
        const mode = query2.get("sr") === "1" ? query2.get("srChain") === "1" ? "SR + Neural rendering" : "Super resolution" : "Neural rendering";
        document.title = `${loadingName} \xB7 DLSS 5 WebGPU`;
        const heading = document.querySelector("h1");
        if (heading) heading.textContent = `${loadingName} / ${mode}`;
        document.querySelector("#mcanvas")?.setAttribute("aria-label", `Interactive ${loadingName} viewer`);
        bridge.resetHistory();
        viewer.renderer.refreshPipeline();
        viewer.renderer.resetShadows();
        viewer.getPlugin("FrameFade")?.stopTransition();
        scene.setDirty({ sceneUpdate: true, frameFade: false });
        viewer.renderEnabled = true;
        viewer.setDirty();
        globals.dlssLoading?.stage("Preparing first frame", `${loadingName} loaded \xB7 preparing the rendered view`);
        globals.dlssDemoUi?.afterSceneChange();
        window.dispatchEvent(new CustomEvent("dlss-scene-ready", { detail: { id: "local-file", label: loadingName } }));
      } catch (error) {
        globals.dlssLoading?.fail(error);
        await clearScene();
        throw error;
      } finally {
        changing = false;
        globals.dlssSceneLoading = false;
        delete document.body.dataset.sceneLoading;
        globals.dlssLoading?.releaseScene?.();
      }
    }
    async function setLocalEnvironment(id) {
      if (globals.dlssSceneId !== "local-file" || !globals.dlssSceneReady) return;
      if (id === localEnvironmentId || changing) return;
      changing = true;
      globals.dlssSceneLoading = true;
      globals.dlssSceneReady = false;
      document.body.dataset.sceneLoading = "true";
      const label = localEnvironments[id]?.label || "HDR environment";
      globals.dlssLoading?.begin(`Loading ${label}`, "Preparing environment lighting");
      try {
        await globals.dlssDemoUi?.beforeSceneChange();
        viewer.renderEnabled = false;
        await applyLocalEnvironment(id);
        bridge.resetHistory();
        viewer.renderer.refreshPipeline();
        viewer.renderer.resetShadows();
        scene.setDirty({ sceneUpdate: true, frameFade: false });
        globals.dlssSceneReady = true;
        viewer.renderEnabled = true;
        viewer.setDirty();
        globals.dlssLoading?.stage("Preparing first frame", `${label} loaded \xB7 preparing the rendered view`);
        globals.dlssDemoUi?.afterSceneChange();
      } catch (error) {
        globals.dlssLoading?.fail(error);
        globals.dlssSceneReady = true;
        throw error;
      } finally {
        changing = false;
        globals.dlssSceneLoading = false;
        delete document.body.dataset.sceneLoading;
        globals.dlssLoading?.releaseScene?.();
      }
    }
    let initial = "selection-five";
    try {
      const saved = sessionStorage.getItem("dlss-demo-scene");
      sessionStorage.removeItem("dlss-demo-scene");
      if (saved && sceneNames[saved]) initial = saved;
    } catch {
    }
    globals.dlssChangeScene = changeScene;
    globals.dlssImportFiles = importLocalFiles;
    globals.dlssSetLocalEnvironment = setLocalEnvironment;
    globals.dlssLocalEnvironment = localEnvironmentId;
    globals.dlssEmbeddedEnvironment = false;
    globals.dlssLocalEnvironments = Object.fromEntries(Object.entries(localEnvironments).map(([id, environment]) => [environment.label, id]));
    await changeScene(initial);
  }

  // src/main.ts
  var query = new URLSearchParams(location.search);
  async function main() {
    ;
    window.dlssLoading?.stage("Preparing viewer", "Connecting to your GPU and loading the viewer");
    const canvas = document.querySelector("#mcanvas");
    if (!canvas) throw new Error("Viewer canvas is missing");
    const viewer = new ViewerApp({ canvas, useRgbm: false, useGBufferDepth: false });
    viewer.renderEnabled = false;
    const diagnostic = [
      "srReferenceHead",
      "srReferenceEnc0Down",
      "srReferenceDec5",
      "srReferenceDec4",
      "srReferenceDec0",
      "nrReference",
      "srDec0Sweep"
    ].some((key) => query.get(key) === "1");
    await viewer.addPlugin(new DlssVelocityBufferPlugin(true, true, true, true, false));
    KTX2LoadPlugin.TRANSCODER_LIBRARY_PATH = "/three/examples/jsm/libs/basis/";
    GLTFMeshOptPlugin.DECODER_URL = "/three/examples/jsm/libs/meshopt_decoder.module.js";
    await viewer.addPlugin(GLTFMeshOptPlugin);
    await addBasePlugins(viewer, {
      ground: false,
      bloom: false,
      depthTonemap: false,
      enableDrop: false,
      interactionPrompt: false
    });
    for (const name of ["SSGI", "SSContactShadows", "RandomizedDirectionalLight", "HDRiGroundPlugin", "WatchHandsPlugin"]) {
      const plugin = viewer.getPlugin(name);
      if (plugin) await viewer.removePlugin(plugin);
    }
    await viewer.getOrAddPlugin(DiamondPlugin);
    await viewer.getOrAddPlugin(GemRefractionPlugin);
    const bridge = await viewer.addPlugin(new DlssBridgePlugin(1, false, true, 1, false));
    bridge.srOnly = false;
    bridge.nrTemporal = !diagnostic;
    viewer.renderer.renderScale = 1;
    await mountDemoScenes(viewer, bridge);
    viewer.scene.activeCamera.interactionsEnabled = true;
    viewer.renderer.refreshPipeline();
    viewer.renderEnabled = true;
    window.dlssViewer = viewer;
    window.dispatchEvent(new CustomEvent("dlss-viewer-ready", { detail: viewer }));
    viewer.setDirty();
  }
  main().catch((error) => {
    ;
    window.dlssLoading?.fail(error);
    console.error(error);
    const status = document.querySelector("#dlssWebGpuStatus");
    if (status) {
      status.textContent = `Scene loading failed: ${error.message || error}`;
      status.setAttribute("data-state", "error");
    }
  });
})();
//# sourceMappingURL=viewer.js.map
