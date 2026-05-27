import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import {
  robotToThreeQuaternion,
  robotToThreeVector,
  threeToRobotQuaternion,
  threeToRobotVector,
} from './dashboard_scene.js';

const EDGE_PAIRS = [
  [0, 1], [1, 3], [3, 2], [2, 0],
  [4, 5], [5, 7], [7, 6], [6, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

export class WorkspaceLayer {
  constructor(world, camera, domElement, orbitControls) {
    this._geom = new THREE.BufferGeometry();
    this._positions = new Float32Array(EDGE_PAIRS.length * 2 * 3);
    this._geom.setAttribute('position', new THREE.BufferAttribute(this._positions, 3));
    this._mat = new THREE.LineBasicMaterial({ color: 0x49c4b5 });
    this._lines = new THREE.LineSegments(this._geom, this._mat);
    this._lines.visible = false;
    this._group = new THREE.Group();
    this._group.add(this._lines);
    world.add(this._group);
    this._baseHalfExtents = [0.0, 0.0, 0.0];
    this._visible = true;
    this._editing = false;
    this._lastWorkspace = null;
    this.onDraftChange = null;

    this._transform = new TransformControls(camera, domElement);
    this._transform.visible = false;
    this._transform.enabled = false;
    this._transform.setMode('translate');
    this._transform.addEventListener('dragging-changed', event => {
      if (orbitControls) orbitControls.enabled = !event.value;
    });
    this._transform.addEventListener('objectChange', () => {
      if (this.onDraftChange) this.onDraftChange(this.getDraftWorkspace());
    });
    world.add(this._transform);
  }

  setBounds(workspaceOrMin, maybeMax) {
    const workspace = Array.isArray(workspaceOrMin)
      ? {
          min: workspaceOrMin,
          max: maybeMax,
          center: [
            (workspaceOrMin[0] + maybeMax[0]) / 2,
            (workspaceOrMin[1] + maybeMax[1]) / 2,
            (workspaceOrMin[2] + maybeMax[2]) / 2,
          ],
          half_extents: [
            (maybeMax[0] - workspaceOrMin[0]) / 2,
            (maybeMax[1] - workspaceOrMin[1]) / 2,
            (maybeMax[2] - workspaceOrMin[2]) / 2,
          ],
          orientation: [0, 0, 0, 1],
          frame: 'world',
        }
      : workspaceOrMin;
    this._lastWorkspace = JSON.parse(JSON.stringify(workspace || {}));
    const center = workspace.center || [
      (workspace.min[0] + workspace.max[0]) / 2,
      (workspace.min[1] + workspace.max[1]) / 2,
      (workspace.min[2] + workspace.max[2]) / 2,
    ];
    const half = workspace.half_extents || [
      (workspace.max[0] - workspace.min[0]) / 2,
      (workspace.max[1] - workspace.min[1]) / 2,
      (workspace.max[2] - workspace.min[2]) / 2,
    ];
    const orientation = workspace.orientation || [0, 0, 0, 1];
    this._baseHalfExtents = half.map(v => Math.max(0.001, Math.abs(Number(v))));
    this._group.position.copy(robotToThreeVector(center));
    this._group.quaternion.copy(robotToThreeQuaternion(orientation));
    this._group.scale.set(1, 1, 1);
    this._writeGeometry(this._baseHalfExtents);
    this._lines.visible = this._visible;
  }

  _writeGeometry(half) {
    const corners = [
      [-half[0], -half[1], -half[2]],
      [ half[0], -half[1], -half[2]],
      [-half[0],  half[1], -half[2]],
      [ half[0],  half[1], -half[2]],
      [-half[0], -half[1],  half[2]],
      [ half[0], -half[1],  half[2]],
      [-half[0],  half[1],  half[2]],
      [ half[0],  half[1],  half[2]],
    ].map(robotToThreeVector);

    let offset = 0;
    for (const [a, b] of EDGE_PAIRS) {
      for (const v of [corners[a], corners[b]]) {
        this._positions[offset++] = v.x;
        this._positions[offset++] = v.y;
        this._positions[offset++] = v.z;
      }
    }
    this._geom.attributes.position.needsUpdate = true;
    this._geom.computeBoundingSphere();
  }

  setVisible(visible) {
    this._visible = !!visible;
    this._group.visible = this._visible;
    this._transform.visible = this._editing && this._visible;
  }

  enableEditing(enabled) {
    this._editing = !!enabled;
    if (this._editing) {
      this._transform.attach(this._group);
    } else {
      this._transform.detach();
    }
    this._transform.enabled = this._editing;
    this._transform.visible = this._editing && this._visible;
    this._mat.color.setHex(this._editing ? 0xffc857 : 0x49c4b5);
  }

  isEditing() {
    return this._editing;
  }

  setEditMode(mode) {
    if (!['translate', 'rotate', 'scale'].includes(mode)) return;
    this._transform.setMode(mode);
  }

  cancelEdit() {
    this.enableEditing(false);
    if (this._lastWorkspace) this.setBounds(this._lastWorkspace);
  }

  getDraftWorkspace() {
    const sx = Math.max(0.001, Math.abs(this._group.scale.x));
    const sy = Math.max(0.001, Math.abs(this._group.scale.y));
    const sz = Math.max(0.001, Math.abs(this._group.scale.z));
    return {
      center: threeToRobotVector(this._group.position),
      half_extents: [
        this._baseHalfExtents[0] * sx,
        this._baseHalfExtents[1] * sz,
        this._baseHalfExtents[2] * sy,
      ],
      orientation: threeToRobotQuaternion(this._group.quaternion),
      frame: 'world',
    };
  }
}
