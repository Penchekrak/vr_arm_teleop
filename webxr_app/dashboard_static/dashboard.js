import { DashboardComms } from './modules/dashboard_comms.js';
import { DashboardScene } from './modules/dashboard_scene.js';
import { RobotView } from './modules/robot_view.js';
import { DashboardPointCloudView } from './modules/dashboard_pointcloud_view.js';
import { WorkspaceLayer } from './modules/workspace_layer.js';
import { XRMarkers } from './modules/xr_markers.js';
import { StatusPanel } from './modules/status_panel.js';
import { CalibrationView } from './modules/calibration_view.js';
import { CalibrationDiagnosticsPanel } from './modules/calibration_diagnostics_panel.js';

const scene = new DashboardScene(document.getElementById('viewport'));
const robot = new RobotView(scene.world);
const cloud = new DashboardPointCloudView(scene.world);
const workspace = new WorkspaceLayer(scene.world);
const xr = new XRMarkers(scene.world);
const calibration = new CalibrationView(scene.world);
const calibrationDiagnostics = new CalibrationDiagnosticsPanel(
  document.getElementById('calibration-feeds'),
);
const status = new StatusPanel(
  document.getElementById('status'),
  document.getElementById('connection'),
);
const comms = new DashboardComms('/ws');
const finishCalibrationButton = document.getElementById('finish-calibration');

let modelLoaded = false;
let finishingCalibration = false;

comms.onConnectionState = state => status.setConnectionState(state);
comms.onJson = msg => {
  if (msg.type !== 'snapshot') return;
  if (!modelLoaded) {
    robot.load(msg.model, () => {
      if (msg.model && msg.model.pointcloud_frame === 'world') {
        cloud.setLinkParent(null);
        return;
      }
      // Parent the cloud to the first camera with a URDF link (the wrist
      // D405). When the arm moves, the cloud moves with it.
      const feeds = (msg.model && msg.model.camera_feeds) || [];
      const wristFeed = feeds.find(f => f.urdf_link);
      if (wristFeed) {
        const link = robot.findLink(wristFeed.urdf_link);
        if (link) cloud.setLinkParent(link);
        else console.warn('[dashboard] URDF link not found for cloud parent', wristFeed.urdf_link);
      }
    });
    workspace.setBounds(msg.workspace.min, msg.workspace.max);
    modelLoaded = true;
  }
  robot.applyJoints(msg.robot.joints);
  xr.update(msg.xr);
  calibration.update(msg.calibration);
  calibrationDiagnostics.update(msg);
  status.update(msg);
  updateFinishCalibrationButton(msg.calibration || null);
};
comms.onBinary = buf => cloud.ingest(buf);

function bindToggle(id, layer) {
  const el = document.getElementById(id);
  if (!el) return;
  layer.setVisible(el.checked);
  el.addEventListener('change', () => layer.setVisible(el.checked));
}

bindToggle('toggle-robot', robot);
bindToggle('toggle-cloud', cloud);
bindToggle('toggle-workspace', workspace);
bindToggle('toggle-xr', xr);
bindToggle('toggle-calibration', calibration);

function updateFinishCalibrationButton(calibrationState) {
  if (!finishCalibrationButton) return;
  if (!calibrationState) {
    finishCalibrationButton.disabled = true;
    finishCalibrationButton.textContent = 'Finish Calibration';
    return;
  }
  const finished = !!calibrationState.finished;
  finishCalibrationButton.disabled = finished || finishingCalibration;
  finishCalibrationButton.textContent = finished
    ? 'Calibration Finished'
    : (finishingCalibration ? 'Finishing...' : 'Finish Calibration');
}

if (finishCalibrationButton) {
  finishCalibrationButton.addEventListener('click', async () => {
    if (finishCalibrationButton.disabled) return;
    finishingCalibration = true;
    updateFinishCalibrationButton({ finished: false });
    try {
      await comms.postJson('/api/calibration/finish');
    } catch (err) {
      console.error('[dashboard] finish calibration failed', err);
      finishCalibrationButton.textContent = 'Finish Failed';
      setTimeout(() => {
        finishingCalibration = false;
        updateFinishCalibrationButton({ finished: false });
      }, 1200);
      return;
    }
    finishingCalibration = false;
  });
}
