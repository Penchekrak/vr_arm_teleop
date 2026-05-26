function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function fmtNumber(value, digits = 3) {
  if (value === null || value === undefined) return 'none';
  return Number(value).toFixed(digits);
}

function row(key, value, className = '') {
  const cls = className ? ` ${className}` : '';
  return `
    <div class="status-row">
      <div class="status-key">${escapeHtml(key)}</div>
      <div class="status-value${cls}">${escapeHtml(value)}</div>
    </div>
  `;
}

export class StatusPanel {
  constructor(statusEl, connectionEl) {
    this._status = statusEl;
    this._connection = connectionEl;
    this.setConnectionState('connecting');
  }

  setConnectionState(state) {
    this._connection.textContent = state.charAt(0).toUpperCase() + state.slice(1);
    this._connection.className = state;
  }

  update(snapshot) {
    const model = snapshot.model || {};
    const robot = snapshot.robot || {};
    const cloud = snapshot.pointcloud || {};
    const xr = snapshot.xr || {};
    const status = snapshot.status || {};
    const calibration = snapshot.calibration || null;
    const jointCount = robot.joints ? Object.keys(robot.joints).length : 0;
    const cameraFeedCount = Array.isArray(model.camera_feeds)
      ? model.camera_feeds.length
      : 0;
    const xrState = xr.aligned ? 'Aligned' : 'XR unaligned';

    const robotError = robot.error || 'none';
    const cloudError = cloud.error || 'none';
    const calibrationHtml = calibration ? this._calibrationHtml(calibration) : '';
    this._status.innerHTML = `
      <div class="status-group">
        ${row('URDF URL', model.urdf_url || 'none')}
        ${row('URDF source', model.urdf_path || 'none')}
        ${row('Cloud frame', model.pointcloud_frame || 'camera-link')}
        ${row('Camera feeds', cameraFeedCount)}
        ${row('Robot joints', jointCount)}
        ${row('Robot Hz', fmtNumber(status.robot_hz, 1))}
      </div>
      <div class="status-group">
        ${row('Cloud points', cloud.n_points || 0)}
        ${row('Cloud seq', cloud.sequence || 0)}
        ${row('Cloud Hz', fmtNumber(status.pointcloud_hz, 1))}
      </div>
      <div class="status-group">
        ${row('XR', xrState, xr.aligned ? '' : 'warn')}
        ${row('Head pose', xr.head ? 'present' : 'none')}
        ${row('Right wrist', xr.right_wrist ? 'present' : 'none')}
      </div>
      <div class="status-group">
        ${row('Robot error', robotError, robot.error ? 'error' : '')}
        ${row('Cloud error', cloudError, cloud.error ? 'error' : '')}
      </div>
      ${calibrationHtml}
    `;
  }

  _calibrationHtml(calibration) {
    const autosave = calibration.autosave || {};
    const liveCorrection = calibration.live_correction || {};
    const targets = calibration.targets || {};
    const targetRows = Object.entries(targets).map(([name, target]) => {
      const stable = target.stable ? 'stable' : 'solving';
      const cls = target.stable ? '' : 'warn';
      const reproj = fmtNumber(target.reprojection_error_px, 2);
      return row(name, `${stable}, ${target.accepted_samples || 0} inliers, ${reproj}px`, cls);
    }).join('');
    const diagnostics = calibration.diagnostics || {};
    const diagnosticRows = Object.entries(diagnostics).map(([name, diagnostic]) => {
      const cls = diagnostic.detected ? '' : 'warn';
      const state = diagnostic.detected ? 'accepted' : (diagnostic.reason || 'rejected');
      const corners = diagnostic.charuco_corner_count ?? diagnostic.corner_count ?? 0;
      const depth = diagnostic.depth_valid_corners ?? 0;
      const rms = diagnostic.kabsch_rms_m === null || diagnostic.kabsch_rms_m === undefined
        ? 'none'
        : `${fmtNumber(diagnostic.kabsch_rms_m, 4)}m`;
      const fit = diagnostic.depth_fit_rms_m === null || diagnostic.depth_fit_rms_m === undefined
        ? 'none'
        : `${fmtNumber(diagnostic.depth_fit_rms_m, 4)}m`;
      const scale = diagnostic.depth_scale === null || diagnostic.depth_scale === undefined
        ? 'none'
        : fmtNumber(diagnostic.depth_scale, 3);
      return row(`${name} det`, `${state}, ${corners}/${depth}, ${rms}, fit ${fit}, s ${scale}`, cls);
    }).join('');
    const error = calibration.error || 'none';
    const correctionRows = Object.entries(liveCorrection.targets || {}).map(([name, target]) => {
      const active = target.active ? 'active' : 'inactive';
      const accepted = target.accepted ? 'accepted' : (target.reason || 'rejected');
      const rmse = target.rmse_m === null || target.rmse_m === undefined
        ? 'none'
        : `${fmtNumber(target.rmse_m, 4)}m`;
      const trans = target.translation_m === null || target.translation_m === undefined
        ? 'none'
        : `${fmtNumber(target.translation_m, 4)}m`;
      const cls = target.active ? '' : 'warn';
      return row(`${name} live`, `${active}, ${accepted}, ${trans}, ${rmse}`, cls);
    }).join('');
    const sampleRejected = calibration.sample_rejected_reason || 'none';
    const armMotion = calibration.arm_motion_m === null || calibration.arm_motion_m === undefined
      ? 'none'
      : `${fmtNumber(calibration.arm_motion_m, 4)}m`;
    return `
      <div class="status-group">
        ${row('Cal mode', calibration.mode || 'none')}
        ${row('Finished', calibration.finished ? 'yes' : 'no', calibration.finished ? '' : 'warn')}
        ${row('Live ICP', liveCorrection.enabled ? 'on' : 'off', liveCorrection.enabled ? '' : 'warn')}
        ${row('Anchor', calibration.anchor_camera || 'none')}
        ${row('Autosave', autosave.state || 'none', autosave.state === 'saved' ? '' : 'warn')}
        ${row('All stable', calibration.all_stable ? 'yes' : 'no', calibration.all_stable ? '' : 'warn')}
        ${row('Arm motion', armMotion, calibration.sample_rejected_reason ? 'warn' : '')}
        ${row('Sample gate', sampleRejected, calibration.sample_rejected_reason ? 'warn' : '')}
        ${targetRows}
        ${correctionRows}
        ${diagnosticRows}
        ${row('Cal error', error, calibration.error ? 'error' : '')}
      </div>
    `;
  }
}
