function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined) return 'none';
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : 'none';
}

function diagnosticLine(diagnostic) {
  if (!diagnostic) return 'not sampled';
  const status = diagnostic.detected ? 'accepted' : (diagnostic.reason || 'rejected');
  const corners = diagnostic.charuco_corner_count ?? diagnostic.corner_count ?? 0;
  const depth = diagnostic.depth_valid_corners ?? 0;
  const rms = diagnostic.kabsch_rms_m === null || diagnostic.kabsch_rms_m === undefined
    ? 'none'
    : `${fmt(diagnostic.kabsch_rms_m, 4)}m`;
  const fit = diagnostic.depth_fit_rms_m === null || diagnostic.depth_fit_rms_m === undefined
    ? 'none'
    : `${fmt(diagnostic.depth_fit_rms_m, 4)}m`;
  const scale = diagnostic.depth_scale === null || diagnostic.depth_scale === undefined
    ? 'none'
    : fmt(diagnostic.depth_scale, 3);
  const reproj = fmt(diagnostic.reprojection_error_px, 2);
  return `${status} | corners ${corners} | depth ${depth} | rigid ${rms} | fit ${fit} | scale ${scale} | reproj ${reproj}px`;
}

export class CalibrationDiagnosticsPanel {
  constructor(root) {
    this._root = root;
    this._lastImageRefresh = 0;
    this._imageUrls = new Map();
  }

  update(snapshot) {
    if (!this._root) return;
    const calibration = snapshot.calibration || null;
    const feeds = (snapshot.model && snapshot.model.camera_feeds) || [];
    if (!calibration || !Array.isArray(calibration.cameras)) {
      this._root.innerHTML = this._cameraFeedsHtml(feeds);
      return;
    }

    const diagnostics = calibration.diagnostics || {};
    const now = Date.now();
    const refreshImages = now - this._lastImageRefresh > 300;
    if (refreshImages) this._lastImageRefresh = now;

    const cards = calibration.cameras.map(camera => {
      const feed = feeds.find(candidate => candidate.name === camera.name) || {};
      const diagnostic = diagnostics[camera.name] || camera.detection || null;
      const url = feed.calibration_url || feed.url || '';
      if (url && refreshImages) {
        this._imageUrls.set(camera.name, `${url}${url.includes('?') ? '&' : '?'}t=${now}`);
      }
      const imageUrl = this._imageUrls.get(camera.name) || '';
      return `
        <div class="calibration-card" data-camera="${escapeHtml(camera.name)}">
          <div class="calibration-card-head">
            <span>${escapeHtml(camera.name)}</span>
            <span class="${diagnostic?.detected ? 'ok' : 'warn'}">
              ${escapeHtml(diagnostic?.detected ? 'accepted' : (diagnostic?.reason || 'waiting'))}
            </span>
          </div>
          ${imageUrl ? `<img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(camera.name)} calibration overlay">` : ''}
          <div class="calibration-card-metrics">${escapeHtml(diagnosticLine(diagnostic))}</div>
        </div>
      `;
    }).join('');

    this._root.innerHTML = `
      <div class="panel-section">
        <h2>Calibration Detection</h2>
        <div class="calibration-grid">${cards}</div>
      </div>
    `;
  }

  _cameraFeedsHtml(feeds) {
    if (!Array.isArray(feeds) || feeds.length === 0) return '';
    const now = Date.now();
    const refreshImages = now - this._lastImageRefresh > 300;
    if (refreshImages) this._lastImageRefresh = now;
    const cards = feeds.map(feed => {
      const name = feed.name || 'camera';
      const url = feed.url || feed.calibration_url || '';
      if (url && refreshImages) {
        this._imageUrls.set(name, `${url}${url.includes('?') ? '&' : '?'}t=${now}`);
      }
      const imageUrl = this._imageUrls.get(name) || '';
      const size = feed.width && feed.height ? `${feed.width}x${feed.height}` : 'live';
      return `
        <div class="calibration-card" data-camera="${escapeHtml(name)}">
          <div class="calibration-card-head">
            <span>${escapeHtml(name)}</span>
            <span>${escapeHtml(size)}</span>
          </div>
          ${imageUrl ? `<img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(name)} camera feed">` : ''}
        </div>
      `;
    }).join('');
    return `
      <div class="panel-section">
        <h2>Camera Feeds</h2>
        <div class="calibration-grid">${cards}</div>
      </div>
    `;
  }
}
