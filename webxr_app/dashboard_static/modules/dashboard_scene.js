import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function robotToThreeVector(v) {
  return new THREE.Vector3(v[0], v[2], -v[1]);
}

export function threeToRobotVector(v) {
  return [v.x, -v.z, v.y];
}

const ROBOT_TO_THREE_BASIS = new THREE.Matrix4().set(
  1, 0, 0, 0,
  0, 0, 1, 0,
  0, -1, 0, 0,
  0, 0, 0, 1,
);
const THREE_TO_ROBOT_BASIS = ROBOT_TO_THREE_BASIS.clone().invert();

export function robotToThreeQuaternion(q) {
  const rot = new THREE.Matrix4().makeRotationFromQuaternion(
    new THREE.Quaternion(q[0], q[1], q[2], q[3]).normalize(),
  );
  const converted = ROBOT_TO_THREE_BASIS.clone().multiply(rot).multiply(THREE_TO_ROBOT_BASIS);
  return new THREE.Quaternion().setFromRotationMatrix(converted).normalize();
}

export function threeToRobotQuaternion(q) {
  const rot = new THREE.Matrix4().makeRotationFromQuaternion(q.clone().normalize());
  const converted = THREE_TO_ROBOT_BASIS.clone().multiply(rot).multiply(ROBOT_TO_THREE_BASIS);
  const out = new THREE.Quaternion().setFromRotationMatrix(converted).normalize();
  return [out.x, out.y, out.z, out.w];
}

export class DashboardScene {
  constructor(container) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0d10);
    this.world = new THREE.Group();
    this.scene.add(this.world);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.01, 50);
    this.camera.position.set(1.2, 1.0, 1.6);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setClearColor(0x0a0d10, 1);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(0.3, 0.3, 0.0);
    this.controls.update();

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x202428, 2.0));
    const dir = new THREE.DirectionalLight(0xffffff, 1.4);
    dir.position.set(2, 3, 2);
    this.scene.add(dir);

    const grid = new THREE.GridHelper(2.0, 20, 0x557788, 0x334455);
    this.world.add(grid);

    window.addEventListener('resize', () => this.resize());
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resize() {
    const rect = this.renderer.domElement.parentElement.getBoundingClientRect();
    this.camera.aspect = rect.width / Math.max(1, rect.height);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(rect.width, rect.height, false);
  }
}
