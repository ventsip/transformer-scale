import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { boxBounds, type Box } from './layout';

/** Vertical field of view, degrees. */
const FOV = 50;
/** Clipping planes: the scene spans from single elements to the 128K-wide LM head. */
const NEAR = 1;
const FAR = 1e8;
const ZOOM_SPEED = 4;
const MAX_DISTANCE = 5e6;
/** Framing leaves this much room around what it frames. */
const FRAME_MARGIN = 1.05;
/** Framing a single block keeps the view direction at least this steep (its Y in a unit vector). */
const MIN_ELEVATION = 0.35;
/** The overview looks from front right and above. */
const OVERVIEW_DIR = new THREE.Vector3(0.45, 0.35, 1);
/** Camera flights: eased interpolation of position and orbit target over this many seconds. */
const FLIGHT_SECONDS = 0.8;
/** The settings panel covers the screen's right side unless the window is narrower than this. */
const PANEL_MIN_WINDOW = 700;
const PANEL_MARGIN = 24;

/** A THREE.Box3 around boxes. */
export function box3Of(boxes: readonly Box[]): THREE.Box3 {
  const b = boxBounds(boxes);
  return new THREE.Box3(new THREE.Vector3(b.minX, b.minY, b.minZ), new THREE.Vector3(b.maxX, b.maxY, b.maxZ));
}

/** Width of the screen covered by the settings panel, kept out of the camera's center. */
function panelInset(): number {
  const panel = document.getElementById('panel')!;
  if (panel.hidden || window.innerWidth < PANEL_MIN_WINDOW) return 0;
  return panel.getBoundingClientRect().width + PANEL_MARGIN;
}

/** The camera and its orbit controls: framing, flights between views, and the zoom and pan pivot. */
export class CameraRig {
  readonly camera = new THREE.PerspectiveCamera(FOV, window.innerWidth / window.innerHeight, NEAR, FAR);
  readonly controls: OrbitControls;
  private flight: { from: THREE.Vector3; fromTarget: THREE.Vector3; to: THREE.Vector3; toTarget: THREE.Vector3; start: number } | null = null;

  constructor(dom: HTMLElement) {
    this.controls = new OrbitControls(this.camera, dom);
    this.controls.enableDamping = true;
    this.controls.zoomToCursor = true;
    this.controls.zoomSpeed = ZOOM_SPEED;
    this.controls.maxDistance = MAX_DISTANCE;
    // Any user input cancels a flight.
    this.controls.addEventListener('start', () => (this.flight = null));
  }

  updateProjection(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.camera.aspect = w / h;
    // Shift the center of projection to the middle of the area left of the panel.
    this.camera.setViewOffset(w, h, panelInset() / 2, 0, w, h);
    this.camera.updateProjectionMatrix();
  }

  /** Places the camera along `dir` from the box center, at the distance where every corner fits the visible area. */
  frame(box: THREE.Box3, dir: THREE.Vector3, instant = false): void {
    const center = box.getCenter(new THREE.Vector3());
    const back = dir.clone().normalize();
    const right = new THREE.Vector3(0, 1, 0).cross(back).normalize();
    const up = back.clone().cross(right);
    const tanV = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const tanH = (tanV * (window.innerWidth - panelInset())) / window.innerHeight;
    const corner = new THREE.Vector3();
    let dist = 0;
    for (let i = 0; i < 8; i++) {
      corner.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).sub(center);
      const z = corner.dot(back);
      dist = Math.max(dist, Math.abs(corner.dot(right)) / tanH + z, Math.abs(corner.dot(up)) / tanV + z);
    }
    this.flyTo(center.clone().addScaledVector(back, dist * FRAME_MARGIN), center, instant);
  }

  /** Frames boxes from the overview direction. */
  overview(boxes: readonly Box[], instant = false): void {
    this.frame(box3Of(boxes), OVERVIEW_DIR, instant);
  }

  /** Frames a single box from the current viewing direction, kept at least slightly above the floor. */
  frameBox(b: Box): void {
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    if (dir.y < MIN_ELEVATION) dir.setY(MIN_ELEVATION).normalize();
    this.frame(box3Of([b]), dir);
  }

  flyTo(position: THREE.Vector3, target: THREE.Vector3, instant = false): void {
    if (instant) {
      this.flight = null;
      this.camera.position.copy(position);
      this.controls.target.copy(target);
      return;
    }
    this.flight = {
      from: this.camera.position.clone(), fromTarget: this.controls.target.clone(),
      to: position, toTarget: target, start: performance.now(),
    };
  }

  /** Advances a flight, if any, and the controls' damping: once per frame. */
  update(): void {
    const flight = this.flight;
    if (flight) {
      const t = Math.min(1, (performance.now() - flight.start) / (FLIGHT_SECONDS * 1000));
      const e = t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2;
      this.camera.position.lerpVectors(flight.from, flight.to, e);
      this.controls.target.lerpVectors(flight.fromTarget, flight.toTarget, e);
      if (t === 1) this.flight = null;
    }
    this.controls.update();
  }

  /**
   * Before a zoom or pan, moves the orbit pivot along the view direction to the depth of what is under the cursor:
   * `ray` hits a block at `distance`, or else the plane at height `planeY` (the focused floor). OrbitControls scales
   * both by the camera–pivot distance, so a pivot left close to the camera (after zooming into empty space, or flying
   * to a small block) made them crawl. The camera keeps its position and orientation, so the view does not move.
   */
  pivotAt(ray: THREE.Ray, distance: number | null, planeY: number): void {
    if (this.flight) return;
    const point = distance !== null
      ? ray.at(distance, new THREE.Vector3())
      : ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -planeY), new THREE.Vector3());
    if (!point) return;
    const dir = this.camera.getWorldDirection(new THREE.Vector3());
    const depth = point.sub(this.camera.position).dot(dir);
    if (depth > 0) this.controls.target.copy(this.camera.position).addScaledVector(dir, depth);
  }
}
