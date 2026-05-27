import { DashboardComms } from './modules/dashboard_comms.js';
import { DashboardScene } from './modules/dashboard_scene.js';
import { RobotView } from './modules/robot_view.js';
import { DashboardPointCloudView } from './modules/dashboard_pointcloud_view.js';
import { WorkspaceLayer } from './modules/workspace_layer.js';
import { XRMarkers } from './modules/xr_markers.js';
import { StatusPanel } from './modules/status_panel.js';
import { CalibrationView } from './modules/calibration_view.js';
import { CalibrationDiagnosticsPanel } from './modules/calibration_diagnostics_panel.js';
import { CubeLayer } from './modules/cube_layer.js';

const scene = new DashboardScene(document.getElementById('viewport'));
const robot = new RobotView(scene.world);
const cloud = new DashboardPointCloudView(scene.world);
const workspace = new WorkspaceLayer(
  scene.world,
  scene.camera,
  scene.renderer.domElement,
  scene.controls,
);
const xr = new XRMarkers(scene.world);
const calibration = new CalibrationView(scene.world);
const cubes = new CubeLayer(scene.world, document.getElementById('cube-list'));
const calibrationDiagnostics = new CalibrationDiagnosticsPanel(
  document.getElementById('calibration-feeds'),
);
const status = new StatusPanel(
  document.getElementById('status'),
  document.getElementById('connection'),
);
const comms = new DashboardComms('/ws');
const finishCalibrationButton = document.getElementById('finish-calibration');
const enableControlButton = document.getElementById('enable-control');
const disableControlButton = document.getElementById('disable-control');
const adjustWorkspaceButton = document.getElementById('adjust-workspace');
const saveWorkspaceButton = document.getElementById('save-workspace');
const cancelWorkspaceButton = document.getElementById('cancel-workspace');
const planGraspButton = document.getElementById('plan-grasp');
const approveGraspButton = document.getElementById('approve-grasp');
const controlStatus = document.getElementById('control-status');

let modelLoaded = false;
let finishingCalibration = false;
let latestSnapshot = null;
let busyControlAction = false;

comms.onConnectionState = state => status.setConnectionState(state);
comms.onJson = msg => {
  if (msg.type !== 'snapshot') return;
  latestSnapshot = msg;
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
    modelLoaded = true;
  }
  if (!workspace.isEditing()) {
    workspace.setBounds(msg.workspace);
  }
  robot.applyJoints(msg.robot.joints);
  xr.update(msg.xr);
  calibration.update(msg.calibration);
  cubes.update(msg.cubes);
  calibrationDiagnostics.update(msg);
  status.update(msg);
  updateFinishCalibrationButton(msg.calibration || null);
  updateControlPanel();
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
bindToggle('toggle-cubes', cubes);

scene.renderer.domElement.addEventListener('pointerdown', ev => {
  if (workspace.isEditing()) return;
  cubes.pick(ev.clientX, ev.clientY, scene.camera, scene.renderer.domElement);
});

cubes.onSelectionChange = () => updateControlPanel();

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

function selectedCubeId() {
  return cubes.getSelectedCubeId();
}

function currentControl() {
  return (latestSnapshot && latestSnapshot.control) || {
    enabled: false,
    executing: false,
    pending_plan: null,
    last_error: null,
  };
}

function simulationSummary(plan) {
  if (!plan || !plan.simulation) return 'No simulated plan';
  const steps = Array.isArray(plan.simulation.steps) ? plan.simulation.steps : [];
  const failed = steps.find(step => !step.reached);
  if (failed) {
    return `${plan.simulation.message}: ${failed.name} error ${Number(failed.position_error_m || 0).toFixed(3)} m`;
  }
  return `${plan.simulation.message}; ${steps.length} step${steps.length === 1 ? '' : 's'} ready`;
}

function updateControlPanel() {
  const control = currentControl();
  const pending = control.pending_plan || null;
  const enabled = !!control.enabled;
  const editing = workspace.isEditing();
  const busy = busyControlAction || !!control.executing;
  if (enableControlButton) enableControlButton.disabled = enabled || busy;
  if (disableControlButton) disableControlButton.disabled = !enabled || busy;
  if (adjustWorkspaceButton) adjustWorkspaceButton.disabled = editing || busy;
  if (saveWorkspaceButton) saveWorkspaceButton.disabled = !editing || busy;
  if (cancelWorkspaceButton) cancelWorkspaceButton.disabled = !editing || busy;
  if (planGraspButton) {
    planGraspButton.disabled = !enabled || !selectedCubeId() || editing || busy;
    planGraspButton.textContent = busyControlAction ? 'Working...' : 'Simulate Grasp';
  }
  if (approveGraspButton) {
    approveGraspButton.disabled = !enabled || !pending || editing || busyControlAction;
    approveGraspButton.textContent = control.executing ? 'Executing...' : 'Approve Move';
  }
  if (controlStatus) {
    const selected = selectedCubeId() || 'none';
    const state = enabled ? 'Control enabled' : 'Control disabled';
    const planText = pending ? simulationSummary(pending) : 'No pending plan';
    const err = control.last_error ? `; ${control.last_error}` : '';
    controlStatus.textContent = `${state}; selected ${selected}; ${planText}${err}`;
  }
}

async function runControlAction(action) {
  if (busyControlAction) return null;
  busyControlAction = true;
  updateControlPanel();
  try {
    return await action();
  } catch (err) {
    console.error('[dashboard] control action failed', err);
    if (controlStatus) controlStatus.textContent = err.message || String(err);
    return null;
  } finally {
    busyControlAction = false;
    updateControlPanel();
  }
}

if (enableControlButton) {
  enableControlButton.addEventListener('click', () => {
    runControlAction(() => comms.postJson('/api/control/enable'));
  });
}

if (disableControlButton) {
  disableControlButton.addEventListener('click', () => {
    runControlAction(() => comms.postJson('/api/control/disable'));
  });
}

if (adjustWorkspaceButton) {
  adjustWorkspaceButton.addEventListener('click', () => {
    workspace.enableEditing(true);
    updateControlPanel();
  });
}

if (saveWorkspaceButton) {
  saveWorkspaceButton.addEventListener('click', () => {
    runControlAction(async () => {
      const response = await comms.postJson('/api/workspace', workspace.getDraftWorkspace());
      workspace.enableEditing(false);
      if (response && response.workspace) workspace.setBounds(response.workspace);
      return response;
    });
  });
}

if (cancelWorkspaceButton) {
  cancelWorkspaceButton.addEventListener('click', () => {
    workspace.cancelEdit();
    updateControlPanel();
  });
}

for (const button of document.querySelectorAll('[data-workspace-mode]')) {
  button.addEventListener('click', () => {
    workspace.setEditMode(button.dataset.workspaceMode);
    for (const item of document.querySelectorAll('[data-workspace-mode]')) {
      item.classList.toggle('selected', item === button);
    }
  });
}

if (planGraspButton) {
  planGraspButton.addEventListener('click', () => {
    const cubeId = selectedCubeId();
    if (!cubeId) return;
    runControlAction(() => comms.postJson('/api/grasp/plan', { cube_id: cubeId }));
  });
}

if (approveGraspButton) {
  approveGraspButton.addEventListener('click', () => {
    const pending = currentControl().pending_plan;
    if (!pending) return;
    runControlAction(() => comms.postJson('/api/grasp/approve', {
      plan_id: pending.plan_id,
    }));
  });
}

updateControlPanel();
