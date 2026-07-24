/**
 * spatial.worker.ts - 3D OffscreenCanvas Web Worker Renderer with Three.js
 * Handles 100,000 node InstancedMesh, glowing semantic edge lines, and smooth camera lerping.
 */

import * as THREE from 'three';

// Message Protocol Interfaces
interface InitCanvasPayload {
  canvas: OffscreenCanvas;
  width: number;
  height: number;
  vectorCount: number;
}

interface TopMatchResult {
  id: number;
  score: number;
}

interface HighlightClusterPayload {
  topMatches: TopMatchResult[];
}

interface IncomingWorkerMessage {
  type: 'INIT_CANVAS' | 'HIGHLIGHT_CLUSTER' | 'RESET_HIGHLIGHTS' | 'RESIZE';
  payload: any;
}

// Global Three.js State in Worker Scope
let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let instancedMesh: THREE.InstancedMesh | null = null;
let lineSegments: THREE.LineSegments | null = null;
let lineGeometry: THREE.BufferGeometry | null = null;

// Node Positions (3 floats per node: x, y, z)
let nodePositions: Float32Array | null = null;
let totalVectorCount = 100000;

// Camera Animation Lerp Targets
const targetCameraPos = new THREE.Vector3(0, 0, 80);
const targetLookAt = new THREE.Vector3(0, 0, 0);
const currentLookAt = new THREE.Vector3(0, 0, 0);

// Default Colors
const DEFAULT_NODE_COLOR = new THREE.Color(0x111827); // Dark Slate Grey
const HIGHLIGHT_BASE_COLOR = new THREE.Color(0x00F0FF); // Cyan Glow
const HIGHLIGHT_TOP_COLOR = new THREE.Color(0xFFD700);  // Gold Glow

/**
 * Initializes Three.js WebGLRenderer on OffscreenCanvas and builds 100,000 node InstancedMesh.
 */
function init3DScene(payload: InitCanvasPayload): void {
  const { canvas, width, height, vectorCount } = payload;
  totalVectorCount = vectorCount;

  // 1. WebGLRenderer with OffscreenCanvas
  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: 'high-performance',
    alpha: true,
  });
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(Math.min(2, self.devicePixelRatio || 1));

  // 2. Scene & Camera Setup
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060810); // Deep Dark Space

  camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
  camera.position.copy(targetCameraPos);
  camera.lookAt(targetLookAt);

  // 3. Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0x00F0FF, 1.2);
  dirLight1.position.set(50, 50, 50);
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0xFF007F, 0.8);
  dirLight2.position.set(-50, -50, -50);
  scene.add(dirLight2);

  // 4. Generate 3D Positions for 100,000 Nodes (Distributed in a Fibonacci Sphere / UMAP Space)
  nodePositions = new Float32Array(vectorCount * 3);
  const dummyMatrix = new THREE.Matrix4();
  const radius = 45;

  const geometry = new THREE.IcosahedronGeometry(0.18, 1);
  const material = new THREE.MeshStandardMaterial({
    roughness: 0.2,
    metalness: 0.8,
  });

  instancedMesh = new THREE.InstancedMesh(geometry, material, vectorCount);

  for (let i = 0; i < vectorCount; i++) {
    // Golden Ratio Sphere distribution
    const phi = Math.acos(1 - 2 * (i + 0.5) / vectorCount);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;

    // Add subtle cluster perturbations
    const clusterOffset = (Math.sin(i * 0.05) + Math.cos(i * 0.03)) * 4.0;
    const r = radius + clusterOffset;

    const x = r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.sin(phi) * Math.sin(theta);
    const z = r * Math.cos(phi);

    nodePositions[i * 3] = x;
    nodePositions[i * 3 + 1] = y;
    nodePositions[i * 3 + 2] = z;

    dummyMatrix.setPosition(x, y, z);
    instancedMesh.setMatrixAt(i, dummyMatrix);
    instancedMesh.setColorAt(i, DEFAULT_NODE_COLOR);
  }

  instancedMesh.instanceMatrix.needsUpdate = true;
  if (instancedMesh.instanceColor) {
    instancedMesh.instanceColor.needsUpdate = true;
  }

  scene.add(instancedMesh);

  // 5. LineSegments Setup for Glowing Semantic Edges
  // Pre-allocate Buffer attribute capable of holding connections for up to 100 matches
  const maxLines = 500;
  const linePositions = new Float32Array(maxLines * 6); // 2 vertices per line (6 floats)

  lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
  lineGeometry.setDrawRange(0, 0); // Initially hidden

  const lineMaterial = new THREE.LineBasicMaterial({
    color: 0x00F0FF,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending,
    linewidth: 2,
  });

  lineSegments = new THREE.LineSegments(lineGeometry, lineMaterial);
  scene.add(lineSegments);

  console.log(`[3D Worker] Three.js Scene initialized with ${vectorCount} instanced nodes.`);

  // 6. Start Worker Animation Frame Loop
  requestAnimationFrame(animate);
}

/**
 * Updates instance colors and line segment geometries for Top-K query matches.
 */
function updateHighlightCluster(payload: HighlightClusterPayload): void {
  if (!instancedMesh || !nodePositions || !lineGeometry) return;

  const { topMatches } = payload;
  const k = topMatches.length;

  if (k === 0) return;

  // 1. Reset all instance colors back to default dim background color
  for (let i = 0; i < totalVectorCount; i++) {
    instancedMesh.setColorAt(i, DEFAULT_NODE_COLOR);
  }

  // 2. Highlight Top-K nodes with glowing color based on similarity score
  let centroidX = 0;
  let centroidY = 0;
  let centroidZ = 0;

  const matchPositions: THREE.Vector3[] = [];

  for (let rank = 0; rank < k; rank++) {
    const match = topMatches[rank];
    const nodeIdx = match.id;

    const x = nodePositions[nodeIdx * 3];
    const y = nodePositions[nodeIdx * 3 + 1];
    const z = nodePositions[nodeIdx * 3 + 2];

    matchPositions.push(new THREE.Vector3(x, y, z));

    centroidX += x;
    centroidY += y;
    centroidZ += z;

    // Interpolate glowing color from Cyan (#00F0FF) to Gold (#FFD700)
    const t = rank === 0 ? 1.0 : Math.max(0, 1.0 - rank / k);
    const glowColor = DEFAULT_NODE_COLOR.clone().lerp(HIGHLIGHT_TOP_COLOR, t);
    glowColor.multiplyScalar(3.0); // Boost intensity for glow effect

    instancedMesh.setColorAt(nodeIdx, glowColor);
  }

  if (instancedMesh.instanceColor) {
    instancedMesh.instanceColor.needsUpdate = true;
  }

  // 3. Compute Centroid of Highlighted Cluster
  centroidX /= k;
  centroidY /= k;
  centroidZ /= k;
  const centroid = new THREE.Vector3(centroidX, centroidY, centroidZ);

  // 4. Update Glowing Semantic Edge Lines Connecting Top Matches
  const linePosAttr = lineGeometry.attributes.position as THREE.BufferAttribute;
  const lineArray = linePosAttr.array as Float32Array;
  let lineVertexCount = 0;

  // Connect top rank node to all other top matches
  const topPos = matchPositions[0];
  for (let i = 1; i < k; i++) {
    const targetPos = matchPositions[i];

    lineArray[lineVertexCount * 3] = topPos.x;
    lineArray[lineVertexCount * 3 + 1] = topPos.y;
    lineArray[lineVertexCount * 3 + 2] = topPos.z;

    lineArray[(lineVertexCount + 1) * 3] = targetPos.x;
    lineArray[(lineVertexCount + 1) * 3 + 1] = targetPos.y;
    lineArray[(lineVertexCount + 1) * 3 + 2] = targetPos.z;

    lineVertexCount += 2;
  }

  lineGeometry.setDrawRange(0, lineVertexCount);
  linePosAttr.needsUpdate = true;

  // 5. Update Camera Target Position to smooth lerp toward cluster centroid
  targetLookAt.copy(centroid);
  
  const cameraOffset = centroid.clone().normalize().multiplyScalar(18.0);
  if (cameraOffset.length() === 0) cameraOffset.set(0, 0, 18);
  targetCameraPos.copy(centroid).add(cameraOffset);
}

/**
 * Continuous animation loop running on Worker Thread requestAnimationFrame.
 */
function animate(): void {
  requestAnimationFrame(animate);

  if (!renderer || !scene || !camera) return;

  // Smooth Camera Lerp Animation
  camera.position.lerp(targetCameraPos, 0.05);
  currentLookAt.lerp(targetLookAt, 0.05);
  camera.lookAt(currentLookAt);

  // Slowly rotate scene for dynamic spatial feel
  if (scene) {
    scene.rotation.y += 0.0005;
  }

  renderer.render(scene, camera);
}

// Handle Incoming Main Thread Messages
self.onmessage = (event: MessageEvent<IncomingWorkerMessage>) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'INIT_CANVAS':
      init3DScene(payload);
      break;

    case 'HIGHLIGHT_CLUSTER':
      updateHighlightCluster(payload);
      break;

    case 'RESIZE':
      if (camera && renderer) {
        camera.aspect = payload.width / payload.height;
        camera.updateProjectionMatrix();
        renderer.setSize(payload.width, payload.height, false);
      }
      break;
  }
};
