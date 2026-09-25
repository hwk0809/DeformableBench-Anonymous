import reflex as rx
from reflex_base.components.component import Component
from reflex_base.utils.imports import ImportVar


class ThreeViewport(Component):
    """Full-page Three.js viewport rendered by the frontend bundle."""

    tag = "div"
    _is_tag_in_global_scope = True

    def add_imports(self):
        return {
            "react": ["useEffect", "useRef"],
            "$/env.json": ImportVar(tag="env", is_default=True, install=False),
            "$/utils/state": ImportVar(tag="getBackendURL", install=False),
            "three": [
                "BoxGeometry",
                "BufferAttribute",
                "BufferGeometry",
                "CapsuleGeometry",
                "CanvasTexture",
                "Clock",
                "Color",
                "ConeGeometry",
                "CylinderGeometry",
                "DirectionalLight",
                "DoubleSide",
                "Fog",
                "GridHelper",
                "HemisphereLight",
                "LineBasicMaterial",
                "LineSegments",
                "Mesh",
                "MeshStandardMaterial",
                "MOUSE",
                "Object3D",
                "Points",
                "PointsMaterial",
                "PCFSoftShadowMap",
                "PerspectiveCamera",
                "PlaneGeometry",
                "Quaternion",
                "RepeatWrapping",
                "Scene",
                "SphereGeometry",
                "WebGLRenderer",
                "WireframeGeometry",
            ],
            "three/examples/jsm/controls/OrbitControls.js": "OrbitControls",
        }

    def add_custom_code(self) -> list[str]:
        return [
            """
const MATERIAL_PRESETS = {
  floor: {
    roughness: 0.3,
    metalness: 0.0,
  },
  rigidbody: {
    roughness: 0.3,
    metalness: 0.8,
  },
  cloth: {
    roughness: 0.4,
    metalness: 0.0,
  },
};

const TEXTURE_REPEAT = {
  floor: 500,
  heightfield: 5,
};

const STATS_SPINNER_STYLE_ID = "deformablebench-stats-spinner-style";

class DeformablebenchRenderer {
  constructor(mount) {
    this.mount = mount;
    this.scene = new Scene();
    this.camera = new PerspectiveCamera(55, 1, 0.1, 1000);
    this.renderer = new WebGLRenderer({ antialias: true });
    this.clock = new Clock();
    this.objects = new Map();
    this.meshGeometryCache = new Map();
    this.meshIndexCache = new Map();
    this.knownMeshKeys = new Set();
    this.floorTexture = this.createFloorTexture(TEXTURE_REPEAT.floor);
    this.heightfieldTexture = this.createFloorTexture(TEXTURE_REPEAT.heightfield);
    this.animationFrame = null;
    this.toolbar = null;
    this.select = null;
    this.wireframeCheckbox = null;
    this.status = null;
    this.statsPanel = null;
    this.frameValue = null;
    this.renderFpsValue = null;
    this.simFpsValue = null;
    this.dataFpsValue = null;
    this.wireframeEnabled = false;
    this.lastPollTime = 0;
    this.pollInterval = 1000 / 45;
    this.pollInFlight = false;
    this.renderFrameCount = 0;
    this.renderFps = 0;
    this.lastRenderFpsTime = 0;
    this.lastBackendFrame = null;
    this.lastBackendWallTime = null;
    this.simFpsBaseFrame = null;
    this.simFpsBaseWallTime = null;
    this.simFps = 0;
    this.dataFrameCount = 0;
    this.dataFps = 0;
    this.lastDataFpsTime = 0;
    this.lastDataFrame = null;
    this.currentExampleId = null;
    this.running = true;
    this.backendOrigin = null;
    this.cameraLight = null;
    this.cameraLightTarget = null;

    this.resize = this.resize.bind(this);
    this.render = this.render.bind(this);
    this.preventContextMenu = this.preventContextMenu.bind(this);
    this.handleKeyDown = this.handleKeyDown.bind(this);
    this.handleExampleChange = this.handleExampleChange.bind(this);
    this.handleWireframeChange = this.handleWireframeChange.bind(this);
  }

  start() {
    this.configureRenderer();
    this.configureScene();
    this.configureCamera();
    this.configureControls();
    this.addLights();
    this.addGridFloor();

    this.mount.appendChild(this.renderer.domElement);
    this.ensureStatsSpinnerStyle();
    this.createToolbar();
    this.createStatsPanel();
    this.mount.addEventListener("contextmenu", this.preventContextMenu);
    window.addEventListener("keydown", this.handleKeyDown);
    window.addEventListener("resize", this.resize);
    this.resize();
    this.loadExamples();
    this.render();
  }

  stop() {
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
    }
    window.removeEventListener("resize", this.resize);
    window.removeEventListener("keydown", this.handleKeyDown);
    this.mount.removeEventListener("contextmenu", this.preventContextMenu);
    this.controls.dispose();
    this.renderer.dispose();
    this.clearObjects();
    this.mount.replaceChildren();
  }

  preventContextMenu(event) {
    event.preventDefault();
  }

  createToolbar() {
    this.toolbar = document.createElement("div");
    this.toolbar.style.position = "absolute";
    this.toolbar.style.left = "12px";
    this.toolbar.style.top = "12px";
    this.toolbar.style.zIndex = "10";
    this.toolbar.style.display = "flex";
    this.toolbar.style.alignItems = "center";
    this.toolbar.style.flexWrap = "wrap";
    this.toolbar.style.gap = "8px";
    this.toolbar.style.padding = "8px";
    this.toolbar.style.background = "rgba(10, 14, 20, 0.72)";
    this.toolbar.style.border = "1px solid rgba(180, 205, 255, 0.24)";
    this.toolbar.style.borderRadius = "6px";
    this.toolbar.style.backdropFilter = "blur(8px)";

    this.select = document.createElement("select");
    this.select.style.maxWidth = "340px";
    this.select.style.height = "30px";
    this.select.style.color = "#e8f1ff";
    this.select.style.background = "#121923";
    this.select.style.border = "1px solid rgba(180, 205, 255, 0.28)";
    this.select.style.borderRadius = "4px";
    this.select.style.outline = "none";
    this.select.style.cursor = "default";
    this.select.addEventListener("change", this.handleExampleChange);

    const wireframeLabel = document.createElement("label");
    wireframeLabel.style.display = "inline-flex";
    wireframeLabel.style.alignItems = "center";
    wireframeLabel.style.gap = "6px";
    wireframeLabel.style.flexBasis = "100%";
    wireframeLabel.style.color = "#d7e4f5";
    wireframeLabel.style.font = "12px system-ui, sans-serif";
    wireframeLabel.style.cursor = "default";

    this.wireframeCheckbox = document.createElement("input");
    this.wireframeCheckbox.type = "checkbox";
    this.wireframeCheckbox.checked = this.wireframeEnabled;
    this.wireframeCheckbox.style.cursor = "default";
    this.wireframeCheckbox.addEventListener("change", this.handleWireframeChange);

    const wireframeText = document.createElement("span");
    wireframeText.textContent = "Wireframe";
    wireframeLabel.appendChild(this.wireframeCheckbox);
    wireframeLabel.appendChild(wireframeText);

    this.toolbar.appendChild(this.select);
    this.toolbar.appendChild(wireframeLabel);
    this.mount.appendChild(this.toolbar);
  }

  createStatsPanel() {
    this.statsPanel = document.createElement("div");
    this.statsPanel.style.position = "absolute";
    this.statsPanel.style.right = "12px";
    this.statsPanel.style.top = "12px";
    this.statsPanel.style.zIndex = "10";
    this.statsPanel.style.display = "grid";
    this.statsPanel.style.gridTemplateColumns = "auto auto";
    this.statsPanel.style.columnGap = "12px";
    this.statsPanel.style.rowGap = "4px";
    this.statsPanel.style.padding = "8px 10px";
    this.statsPanel.style.background = "rgba(10, 14, 20, 0.72)";
    this.statsPanel.style.border = "1px solid rgba(180, 205, 255, 0.24)";
    this.statsPanel.style.borderRadius = "6px";
    this.statsPanel.style.backdropFilter = "blur(8px)";
    this.statsPanel.style.color = "#e8f1ff";
    this.statsPanel.style.font = "12px ui-monospace, SFMono-Regular, Consolas, monospace";
    this.statsPanel.className = "deformablebench-stats-panel";

    this.frameValue = this.addStatsRow("Frame", "0");
    this.simFpsValue = this.addStatsRow("Sim FPS", "0.0");
    this.dataFpsValue = this.addStatsRow("Data FPS", "0.0");
    this.renderFpsValue = this.addStatsRow("Render FPS", "0.0");
    this.mount.appendChild(this.statsPanel);
    this.updateStatsRunningState();
  }

  ensureStatsSpinnerStyle() {
    if (document.getElementById(STATS_SPINNER_STYLE_ID)) {
      return;
    }
    const style = document.createElement("style");
    style.id = STATS_SPINNER_STYLE_ID;
    style.textContent = `
@property --stats-angle {
  syntax: "<angle>";
  inherits: false;
  initial-value: 0turn;
}
@keyframes deformablebenchStatsSpin {
  to { --stats-angle: 1turn; }
}
.deformablebench-stats-panel::before {
  content: "";
  position: absolute;
  inset: -2px;
  border-radius: 8px;
  padding: 2px;
  background: conic-gradient(
    from var(--stats-angle),
    rgba(52, 211, 153, 0.95) 0turn,
    rgba(52, 211, 153, 0.95) 0.5turn,
    rgba(52, 211, 153, 0) 0.56turn,
    rgba(52, 211, 153, 0) 1turn
  );
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  opacity: 0;
  pointer-events: none;
}
.deformablebench-stats-panel.running::before {
  opacity: 1;
  animation: deformablebenchStatsSpin 1.1s linear infinite;
}
.deformablebench-stats-panel.paused::before {
  opacity: 1;
  background: #f97316;
}
`;
    document.head.appendChild(style);
  }

  addStatsRow(label, value) {
    const labelNode = document.createElement("span");
    labelNode.style.color = "#9fb4cc";
    labelNode.textContent = label;

    const valueNode = document.createElement("span");
    valueNode.style.minWidth = "64px";
    valueNode.style.textAlign = "right";
    valueNode.textContent = value;

    this.statsPanel.appendChild(labelNode);
    this.statsPanel.appendChild(valueNode);
    return valueNode;
  }

  apiUrl(path) {
    const url = new URL(this.backendOrigin ?? getBackendURL(env.PING));
    const next = new URL(path, url.origin);
    url.pathname = next.pathname;
    url.search = next.search;
    url.hash = "";
    return url.href;
  }

  fallbackApiUrl(path) {
    const url = new URL(window.location.href);
    const next = new URL(path, url.origin);
    url.port = "8000";
    url.pathname = next.pathname;
    url.search = next.search;
    url.hash = "";
    return url.href;
  }

  async handleKeyDown(event) {
    if (event.code !== "Space") {
      return;
    }
    event.preventDefault();
    const response = await this.fetchNewton("/newton/toggle", { method: "POST" });
    const payload = await response.json();
    this.running = payload.running;
    this.updateStatsRunningState();
    this.setStatus(this.running ? "Running" : "Paused");
  }

  async handleExampleChange() {
    if (!this.select) {
      return;
    }
    this.setStatus("Loading");
    this.clearObjects();
    try {
      const response = await this.fetchNewton("/newton/example", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: this.select.value }),
      });
      const frame = await response.json();
      this.applyFrame(frame);
    } catch (error) {
      this.setStatus("Error");
      console.error(error);
    }
  }

  handleWireframeChange() {
    this.wireframeEnabled = this.wireframeCheckbox?.checked ?? false;
    for (const object of this.objects.values()) {
      this.updateWireframeOverlay(object);
    }
  }

  async loadExamples() {
    try {
      const response = await this.fetchNewton("/newton/examples");
      const payload = await response.json();
      this.select.replaceChildren();
      for (const example of payload.examples) {
        const option = document.createElement("option");
        option.value = example.id;
        option.textContent = example.label;
        this.select.appendChild(option);
      }
      this.select.value = payload.current;
      this.currentExampleId = payload.current;
      this.setStatus("Running");
    } catch (error) {
      this.setStatus("Error");
      console.error(error);
    }
  }

  setStatus(text) {
    this.status = text;
  }

  updateStatsRunningState() {
    if (this.statsPanel) {
      this.statsPanel.classList.toggle("running", this.running);
      this.statsPanel.classList.toggle("paused", !this.running);
    }
  }

  updateRenderFps(timestamp) {
    this.renderFrameCount += 1;
    if (this.lastRenderFpsTime === 0) {
      this.lastRenderFpsTime = timestamp;
      return;
    }
    const elapsed = timestamp - this.lastRenderFpsTime;
    if (elapsed >= 500) {
      this.renderFps = (this.renderFrameCount * 1000) / elapsed;
      this.renderFrameCount = 0;
      this.lastRenderFpsTime = timestamp;
    }
  }

  updateSimFps(frame) {
    if (this.simFpsBaseFrame === null || this.simFpsBaseWallTime === null) {
      this.simFpsBaseFrame = frame.frame;
      this.simFpsBaseWallTime = frame.wall_time;
      this.lastBackendFrame = frame.frame;
      this.lastBackendWallTime = frame.wall_time;
      return;
    }
    const elapsed = frame.wall_time - this.simFpsBaseWallTime;
    if (elapsed >= 0.5) {
      this.simFps = (frame.frame - this.simFpsBaseFrame) / elapsed;
      this.simFpsBaseFrame = frame.frame;
      this.simFpsBaseWallTime = frame.wall_time;
    }
    this.lastBackendFrame = frame.frame;
    this.lastBackendWallTime = frame.wall_time;
  }

  updateDataFps(frame) {
    if (frame.frame !== this.lastDataFrame) {
      this.dataFrameCount += 1;
      this.lastDataFrame = frame.frame;
    }
    if (this.lastDataFpsTime === 0) {
      this.lastDataFpsTime = performance.now();
      return;
    }
    const now = performance.now();
    const elapsed = now - this.lastDataFpsTime;
    if (elapsed >= 500) {
      this.dataFps = (this.dataFrameCount * 1000) / elapsed;
      this.dataFrameCount = 0;
      this.lastDataFpsTime = now;
    }
  }

  updateStatsPanel(frame = null) {
    if (frame) {
      if (this.frameValue) {
        this.frameValue.textContent = String(frame.frame);
      }
      if (this.simFpsValue) {
        this.simFpsValue.textContent = this.simFps.toFixed(1);
      }
      if (this.dataFpsValue) {
        this.dataFpsValue.textContent = this.dataFps.toFixed(1);
      }
    }
    if (this.renderFpsValue) {
      this.renderFpsValue.textContent = this.renderFps.toFixed(1);
    }
  }

  resetFrameStats() {
    this.lastBackendFrame = null;
    this.lastBackendWallTime = null;
    this.simFpsBaseFrame = null;
    this.simFpsBaseWallTime = null;
    this.simFps = 0;
    this.dataFrameCount = 0;
    this.dataFps = 0;
    this.lastDataFpsTime = 0;
    this.lastDataFrame = null;
    if (this.frameValue) {
      this.frameValue.textContent = "0";
    }
    if (this.simFpsValue) {
      this.simFpsValue.textContent = "0.0";
    }
    if (this.dataFpsValue) {
      this.dataFpsValue.textContent = "0.0";
    }
  }

  configureRenderer() {
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0d1117, 1);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFSoftShadowMap;
  }

  configureScene() {
    this.scene.background = new Color(0x0d1117);
    this.scene.fog = new Fog(0x0d1117, 20, 70);
  }

  configureCamera() {
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(10, -1.3, 4);
    this.camera.lookAt(0, 0, 0);
  }

  configureControls() {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 35;
    this.controls.target.set(0, 0, 1);
    this.controls.mouseButtons = {
      LEFT: null,
      MIDDLE: MOUSE.DOLLY,
      RIGHT: MOUSE.ROTATE,
    };
  }

  addLights() {
    const ambient = new HemisphereLight(0xb8d8ff, 0x18222e, 1.6);
    this.scene.add(ambient);

    const keyIntensity = 2.4;
    const key = new DirectionalLight(0xffffff, keyIntensity);
    key.position.set(6, -8, 12);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 40;
    key.shadow.camera.left = -12;
    key.shadow.camera.right = 12;
    key.shadow.camera.top = 12;
    key.shadow.camera.bottom = -12;
    this.scene.add(key);

    this.cameraLightTarget = new Object3D();
    this.scene.add(this.cameraLightTarget);

    this.cameraLight = new DirectionalLight(0xffffff, keyIntensity * 0.5);
    this.cameraLight.target = this.cameraLightTarget;
    this.scene.add(this.cameraLight);
  }

  updateCameraLight() {
    if (!this.cameraLight || !this.cameraLightTarget) {
      return;
    }
    this.cameraLight.position.copy(this.camera.position);
    this.cameraLightTarget.position.copy(this.controls.target);
  }

  addGridFloor() {
    const grid = new GridHelper(80, 80, 0x263849, 0x263849);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.005;
    this.scene.add(grid);

    const xAxisGeometry = new BufferGeometry();
    xAxisGeometry.setAttribute("position", new BufferAttribute(new Float32Array([-40, 0, 0.015, 40, 0, 0.015]), 3));
    this.scene.add(new LineSegments(xAxisGeometry, new LineBasicMaterial({ color: 0xff4a4a })));

    const yAxisGeometry = new BufferGeometry();
    yAxisGeometry.setAttribute("position", new BufferAttribute(new Float32Array([0, -40, 0.015, 0, 40, 0.015]), 3));
    this.scene.add(new LineSegments(yAxisGeometry, new LineBasicMaterial({ color: 0x43d17a })));
  }

  geometryForPrimitive(primitive) {
    const geoScale = primitive.geo_scale ?? [1, 1, 1];

    switch (primitive.type) {
      case "plane":
        return new PlaneGeometry(
          geoScale[0] > 0 ? geoScale[0] : 1000,
          geoScale[1] > 0 ? geoScale[1] : 1000
        );
      case "sphere":
        return new SphereGeometry(geoScale[0], 32, 16);
      case "ellipsoid":
        return new SphereGeometry(1, 32, 16);
      case "capsule":
        return new CapsuleGeometry(geoScale[0], geoScale[1] * 2, 12, 24).rotateX(Math.PI / 2);
      case "cylinder":
        return new CylinderGeometry(geoScale[0], geoScale[0], geoScale[1] * 2, 32).rotateX(Math.PI / 2);
      case "box":
        return new BoxGeometry(geoScale[0] * 2, geoScale[1] * 2, geoScale[2] * 2);
      case "cone":
        return new ConeGeometry(geoScale[0], geoScale[1] * 2, 32).rotateX(Math.PI / 2);
      case "mesh":
      case "convex_mesh":
        return this.meshGeometryForPrimitive(primitive);
      case "points":
        return this.pointsGeometryForPrimitive(primitive);
      case "lines":
        return this.pointsGeometryForPrimitive(primitive);
      default:
        return new BoxGeometry(1, 1, 1);
    }
  }

  meshGeometryForPrimitive(primitive) {
    const cacheKey = primitive.mesh_key ?? `${primitive.id}:${primitive.mesh_version ?? 0}`;
    if (this.meshGeometryCache.has(cacheKey)) {
      return this.meshGeometryCache.get(cacheKey);
    }
    const pos = primitive.mesh ? this.meshPositions(primitive.mesh) : null;
    if (!pos) {
      return new BoxGeometry(1, 1, 1);
    }

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(pos, 3));
    if (primitive.mesh.uvs) {
      geometry.setAttribute("uv", new BufferAttribute(new Float32Array(primitive.mesh.uvs.flat()), 2));
    }
    // Indices (static topology) are sent only on the first frame, then stripped for
    // dynamic meshes. Cache them client-side so a later geometry rebuild can still
    // index correctly — never setIndex(undefined) (that crashes the shadow pass).
    let indices = primitive.mesh.indices;
    if (indices) {
      this.meshIndexCache.set(cacheKey, indices);
    } else if (this.meshIndexCache.has(cacheKey)) {
      indices = this.meshIndexCache.get(cacheKey);
    }
    if (indices) {
      geometry.setIndex(indices);
    }
    geometry.computeVertexNormals();
    this.meshGeometryCache.set(cacheKey, geometry);
    // Tell the backend we have this topology so it stops re-sending it: whole mesh
    // for static keys (model:*), just the indices for dynamic (cloth) keys.
    if (primitive.mesh_key) {
      this.knownMeshKeys.add(primitive.mesh_key);
    }
    return geometry;
  }

  meshPositions(mesh) {
    // Vertex positions as a Float32Array, from base64 raw float32 (dynamic cloth)
    // or a nested JSON list (static model meshes).
    if (!mesh) {
      return null;
    }
    if (mesh.vertices_b64) {
      const bin = atob(mesh.vertices_b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i += 1) {
        bytes[i] = bin.charCodeAt(i);
      }
      return new Float32Array(bytes.buffer);
    }
    if (mesh.vertices) {
      return new Float32Array(mesh.vertices.flat());
    }
    return null;
  }

  updateDynamicMesh(object, primitive) {
    // Deforming mesh (cloth): reuse the geometry, just stream new vertex positions
    // into the existing position buffer — no rebuild, no setIndex, no re-parse of
    // static topology. Indices were sent once when the geometry was first built.
    const arr = this.meshPositions(primitive.mesh);
    if (!arr) {
      return;
    }
    const geometry = object.geometry;
    const position = geometry.getAttribute("position");
    if (position && position.array.length === arr.length) {
      position.array.set(arr);
      position.needsUpdate = true;
    } else {
      geometry.setAttribute("position", new BufferAttribute(arr, 3));
    }
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
  }

  pointsGeometryForPrimitive(primitive) {
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(new Float32Array(primitive.positions.flat()), 3));
    return geometry;
  }

  createFloorTexture(repeat = 1) {
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 128;
    const context = canvas.getContext("2d");
    const colors = ["#9aa0a8", "#6f767f"];
    const tileSize = 64;

    for (let y = 0; y < 2; y += 1) {
      for (let x = 0; x < 2; x += 1) {
        context.fillStyle = colors[(x + y) % 2];
        context.fillRect(x * tileSize, y * tileSize, tileSize, tileSize);
      }
    }

    const texture = new CanvasTexture(canvas);
    texture.wrapS = RepeatWrapping;
    texture.wrapT = RepeatWrapping;
    texture.repeat.set(repeat, repeat);
    return texture;
  }

  materialPresetForPrimitive(primitive) {
    if (primitive.type === "plane" || primitive.material_kind === "floor") {
      return MATERIAL_PRESETS.floor;
    }
    if (primitive.type === "mesh" || primitive.type === "convex_mesh") {
      return MATERIAL_PRESETS[primitive.material_kind] ?? MATERIAL_PRESETS.rigidbody;
    }
    return MATERIAL_PRESETS.rigidbody;
  }

  meshMaterialParams(primitive) {
    const params = {
      ...this.materialPresetForPrimitive(primitive),
      transparent: false,
    };
    if (primitive.type === "plane") {
      params.map = this.floorTexture;
    } else if (primitive.material_kind === "floor") {
      params.map = this.heightfieldTexture;
    }
    if (primitive.material_kind === "cloth") {
      params.side = DoubleSide;
    }
    return params;
  }

  materialForPrimitive(primitive) {
    const color = primitive.type === "plane" ? new Color(1, 1, 1) : new Color(...primitive.color);
    if (primitive.type === "points") {
      return new PointsMaterial({
        color,
        size: primitive.radius ?? 0.04,
        sizeAttenuation: true,
      });
    }
    if (primitive.type === "lines") {
      return new LineBasicMaterial({ color });
    }
    return new MeshStandardMaterial({
      color,
      ...this.meshMaterialParams(primitive),
    });
  }

  updateMaterialForPrimitive(material, primitive) {
    if (material.color) {
      if (primitive.type === "plane" || primitive.material_kind === "floor") {
        material.color.setRGB(1, 1, 1);
      } else {
        material.color.setRGB(...primitive.color);
      }
    }

    if (material.isMeshStandardMaterial) {
      Object.assign(material, this.meshMaterialParams(primitive));
      material.needsUpdate = true;
    }

    if (material.isPointsMaterial) {
      material.size = primitive.radius ?? 0.04;
      material.needsUpdate = true;
    }
  }

  createWireframeOverlay(geometry) {
    const wireframe = new LineSegments(
      new WireframeGeometry(geometry),
      new LineBasicMaterial({ color: 0x000000 })
    );
    wireframe.visible = this.wireframeEnabled;
    wireframe.userData.isWireframeOverlay = true;
    return wireframe;
  }

  updateWireframeOverlay(object) {
    if (!object.isMesh || object.userData.materialKind === "floor") {
      const wireframe = object.userData.wireframeOverlay;
      if (wireframe) {
        object.remove(wireframe);
        wireframe.geometry.dispose();
        wireframe.material.dispose();
        delete object.userData.wireframeOverlay;
      }
      return;
    }
    let wireframe = object.userData.wireframeOverlay;
    if (!wireframe) {
      wireframe = this.createWireframeOverlay(object.geometry);
      object.userData.wireframeOverlay = wireframe;
      object.add(wireframe);
    }
    wireframe.visible = this.wireframeEnabled;
  }

  replaceObjectGeometry(object, geometry) {
    object.geometry.dispose();
    object.geometry = geometry;
    const wireframe = object.userData.wireframeOverlay;
    if (wireframe) {
      wireframe.geometry.dispose();
      wireframe.geometry = new WireframeGeometry(geometry);
    }
  }

  disposeObject(object) {
    const wireframe = object.userData.wireframeOverlay;
    if (wireframe) {
      wireframe.geometry.dispose();
      wireframe.material.dispose();
    }
    object.geometry.dispose();
    object.material.dispose();
  }

  applyPrimitiveTransform(object, primitive) {
    if (primitive.type === "points" || primitive.type === "lines") {
      return;
    }

    object.position.set(...primitive.position);
    object.quaternion.copy(new Quaternion(...primitive.rotation));

    if (primitive.type === "plane") {
      object.scale.set(1, 1, 1);
      return;
    }

    if (primitive.type === "ellipsoid") {
      object.scale.set(...primitive.geo_scale);
      return;
    }

    if (primitive.type === "mesh") {
        object.scale.set(...primitive.scale);
        return;
    }

    object.scale.set(1, 1, 1);
  }

  upsertPrimitive(primitive) {
    let object = this.objects.get(primitive.id);
    if (!object) {
      const geometry = this.geometryForPrimitive(primitive);
      const material = this.materialForPrimitive(primitive);
      if (primitive.type === "points") {
        object = new Points(geometry, material);
      } else if (primitive.type === "lines") {
        object = new LineSegments(geometry, material);
      } else {
        object = new Mesh(geometry, material);
      }
      object.castShadow = primitive.type !== "plane";
      object.receiveShadow = true;
      this.scene.add(object);
      this.objects.set(primitive.id, object);
    } else {
      const nextKey = this.geometryKey(primitive);
      if (object.userData.geometryKey !== nextKey) {
        this.replaceObjectGeometry(object, this.geometryForPrimitive(primitive));
      }
    }
    this.updateMaterialForPrimitive(object.material, primitive);
    object.userData.geometryKey = this.geometryKey(primitive);
    object.userData.materialKind = primitive.type === "plane" ? "floor" : primitive.material_kind;
    this.updateWireframeOverlay(object);
    this.applyPrimitiveTransform(object, primitive);
    if (primitive.dynamic) {
      this.updateDynamicMesh(object, primitive);  // stream new cloth vertices in place
    }
  }

  geometryKey(primitive) {
    return `${primitive.id}:${primitive.type}:${primitive.mesh_key ?? primitive.mesh_version ?? 0}`;
  }

  removeMissingPrimitives(activeIds) {
    for (const [id, object] of this.objects.entries()) {
      if (!activeIds.has(id)) {
        this.scene.remove(object);
        this.disposeObject(object);
        this.objects.delete(id);
      }
    }
  }

  async pollNewtonFrame(timestamp) {
    if (this.pollInFlight || timestamp - this.lastPollTime < this.pollInterval) {
      return;
    }

    this.pollInFlight = true;
    this.lastPollTime = timestamp;
    try {
      const query = this.knownMeshKeys.size > 0
        ? `?known_meshes=${encodeURIComponent([...this.knownMeshKeys].join(","))}`
        : "";
      const response = await this.fetchNewton(`/newton/frame${query}`);
      const frame = await response.json();
      this.applyFrame(frame);
    } catch (error) {
      // Backend not ready yet (cold start / compiling warp kernels) or busy
      // building a scene. Poll again next frame instead of throwing — otherwise
      // the 45Hz loop floods the Reflex error overlay with "Failed to fetch".
      this.setStatus("Backend starting…");
    } finally {
      this.pollInFlight = false;
    }
  }

  applyFrame(frame) {
    this.running = frame.running;
    this.updateStatsRunningState();
    if (frame.example_id && frame.example_id !== this.currentExampleId) {
      this.currentExampleId = frame.example_id;
      this.clearObjects();
      if (this.select) {
        this.select.value = frame.example_id;
      }
    }
    if (frame.error) {
      this.setStatus("Error");
      console.error(frame.error);
      return;
    }
    this.updateSimFps(frame);
    this.updateDataFps(frame);

    const activeIds = new Set();
    for (const primitive of frame.primitives) {
      activeIds.add(primitive.id);
      this.upsertPrimitive(primitive);
    }
    this.removeMissingPrimitives(activeIds);
    this.setStatus(this.running ? "Running" : "Paused");
    this.updateStatsPanel(frame);
  }

  clearObjects() {
    for (const object of this.objects.values()) {
      this.scene.remove(object);
      this.disposeObject(object);
    }
    this.objects.clear();
    for (const geometry of this.meshGeometryCache.values()) {
      geometry.dispose();
    }
    this.meshGeometryCache.clear();
    this.meshIndexCache.clear();
    this.knownMeshKeys.clear();
    this.resetFrameStats();
  }

  async fetchNewton(path, options) {
    try {
      const response = await fetch(this.apiUrl(path), options);
      if (response.status !== 0) {
        this.backendOrigin = new URL(response.url).origin;
        return response;
      }
    } catch (_error) {
      // Reflex dev servers can leave env.json pointing at an old backend port.
    }

    const response = await fetch(this.fallbackApiUrl(path), options);
    if (response.ok) {
      this.backendOrigin = new URL(response.url).origin;
    }
    return response;
  }

  resize() {
    const width = this.mount.clientWidth;
    const height = this.mount.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  render(timestamp = 0) {
    this.clock.getDelta();
    this.updateRenderFps(timestamp);
    this.pollNewtonFrame(timestamp);
    this.controls.update();
    this.updateCameraLight();
    this.renderer.render(this.scene, this.camera);
    this.updateStatsPanel();
    this.animationFrame = requestAnimationFrame(this.render);
  }
}
"""
        ]

    def add_hooks(self) -> list[str]:
        return [
            "const viewportRef = useRef(null);",
            """
useEffect(() => {
  if (!viewportRef.current) {
    return;
  }

  const renderer = new DeformablebenchRenderer(viewportRef.current);
  renderer.start();
  return () => renderer.stop();
}, []);
""",
        ]

    @classmethod
    def create(cls, **props) -> Component:
        props.setdefault("width", "100vw")
        props.setdefault("height", "100vh")
        props.setdefault("overflow", "hidden")
        props.setdefault("background", "#0d1117")
        props.setdefault("ref", rx.Var(_js_expr="viewportRef"))
        return super().create(**props)


three_viewport = ThreeViewport.create
