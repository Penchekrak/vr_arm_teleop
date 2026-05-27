import * as THREE from 'three';
import { robotToThreeVector } from './dashboard_scene.js';

function fmt(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

export class CubeLayer {
  constructor(world, listEl) {
    this._group = new THREE.Group();
    this._group.name = 'detected-cubes';
    world.add(this._group);
    this._listEl = listEl;
    this._cubes = [];
    this._meshes = new Map();
    this._selectedId = null;
    this._visible = true;
    this._raycaster = new THREE.Raycaster();
    this._pointer = new THREE.Vector2();
    this.onSelectionChange = null;
    this._renderList();
  }

  update(snapshotCubes) {
    const cubes = Array.isArray(snapshotCubes && snapshotCubes.items)
      ? snapshotCubes.items
      : [];
    this._cubes = cubes;
    const liveIds = new Set(cubes.map(cube => cube.id));

    for (const [id, mesh] of this._meshes.entries()) {
      if (!liveIds.has(id)) {
        this._group.remove(mesh);
        mesh.geometry.dispose();
        mesh.material.dispose();
        this._meshes.delete(id);
      }
    }

    for (const cube of cubes) {
      const mesh = this._meshFor(cube);
      const edge = Number(cube.edge_m || 0.038);
      mesh.scale.set(edge, edge, edge);
      const pos = robotToThreeVector(cube.center_m || [0, 0, edge / 2]);
      mesh.position.copy(pos);
      mesh.rotation.set(0, Number(cube.yaw_rad || 0), 0);
      mesh.userData.cubeId = cube.id;
    }

    if (this._selectedId && !liveIds.has(this._selectedId)) {
      this._selectedId = null;
    }
    this._syncSelection();
    this._renderList();
  }

  pick(clientX, clientY, camera, domElement) {
    const rect = domElement.getBoundingClientRect();
    this._pointer.x = ((clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    this._pointer.y = -(((clientY - rect.top) / Math.max(1, rect.height)) * 2 - 1);
    this._raycaster.setFromCamera(this._pointer, camera);
    const hits = this._raycaster.intersectObjects([...this._meshes.values()], false);
    if (!hits.length) return false;
    this.select(hits[0].object.userData.cubeId || null);
    return true;
  }

  select(id) {
    this._selectedId = id;
    this._syncSelection();
    this._renderList();
    if (this.onSelectionChange) this.onSelectionChange(this._selectedId);
  }

  getSelectedCubeId() {
    return this._selectedId;
  }

  setVisible(visible) {
    this._visible = !!visible;
    this._group.visible = this._visible;
  }

  _meshFor(cube) {
    let mesh = this._meshes.get(cube.id);
    if (mesh) return mesh;
    const geom = new THREE.BoxGeometry(1, 1, 1);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x4fb3ff,
      transparent: true,
      opacity: 0.22,
      roughness: 0.55,
      metalness: 0.0,
      depthWrite: false,
    });
    mesh = new THREE.Mesh(geom, mat);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geom),
      new THREE.LineBasicMaterial({ color: 0xd8f0ff }),
    );
    mesh.add(edges);
    this._group.add(mesh);
    this._meshes.set(cube.id, mesh);
    return mesh;
  }

  _syncSelection() {
    for (const [id, mesh] of this._meshes.entries()) {
      const selected = id === this._selectedId;
      mesh.material.color.setHex(selected ? 0xffc857 : 0x4fb3ff);
      mesh.material.opacity = selected ? 0.36 : 0.22;
      const edgeLines = mesh.children[0];
      if (edgeLines && edgeLines.material) {
        edgeLines.material.color.setHex(selected ? 0xfff0b0 : 0xd8f0ff);
      }
    }
  }

  _renderList() {
    if (!this._listEl) return;
    if (!this._cubes.length) {
      this._listEl.innerHTML = '<div class="cube-empty">No cubes detected</div>';
      return;
    }
    this._listEl.innerHTML = this._cubes.map(cube => {
      const selected = cube.id === this._selectedId;
      const c = cube.center_m || [0, 0, 0];
      const yawDeg = Number(cube.yaw_rad || 0) * 180 / Math.PI;
      return `
        <button class="cube-row${selected ? ' selected' : ''}" type="button" data-cube-id="${escapeHtml(cube.id)}">
          <span>${escapeHtml(cube.id)}</span>
          <span>${fmt(c[0])}, ${fmt(c[1])}, ${fmt(c[2])} m</span>
          <span>${fmt(yawDeg, 1)} deg · ${fmt(cube.confidence || 0, 2)}</span>
        </button>
      `;
    }).join('');
    for (const button of this._listEl.querySelectorAll('[data-cube-id]')) {
      button.addEventListener('click', () => this.select(button.dataset.cubeId));
    }
  }
}
