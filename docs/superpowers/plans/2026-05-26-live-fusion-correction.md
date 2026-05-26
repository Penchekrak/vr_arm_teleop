# Live Fusion Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live, ChArUco-gated point-cloud correction during calibration, then let the dashboard freeze the corrected transforms for normal-operation display.

**Architecture:** Keep ChArUco as the authoritative coarse transform. Add a small bounded ICP estimator in `teleop_backends/pointcloud/live_alignment.py`, call it only from the calibration source when the anchor and target detections are accepted, and apply the smoothed correction to non-anchor transforms used for fusion. Add a dashboard POST action that freezes the latest corrected transforms and moves the calibration source into a finished mode without autosaving camera JSON.

**Tech Stack:** Python, NumPy, SciPy `cKDTree`, aiohttp dashboard routes, existing dashboard JavaScript.

---

### Task 1: Point-Cloud Correction Estimator

**Files:**
- Create: `teleop_backends/pointcloud/live_alignment.py`
- Test: `tests/test_live_pointcloud_alignment.py`

- [x] **Step 1: Write failing tests**

```python
def test_estimate_icp_correction_recovers_small_world_offset():
    anchor = make_asymmetric_cloud()
    target = anchor + np.array([0.012, -0.006, 0.004], dtype=np.float32)
    result = estimate_icp_correction(anchor, target, LiveCorrectionConfig())
    assert result.accepted is True
    assert np.allclose(result.transform[:3, 3], [-0.012, 0.006, -0.004], atol=0.003)
```

```python
def test_estimate_icp_correction_rejects_large_transform():
    anchor = make_asymmetric_cloud()
    target = anchor + np.array([0.20, 0.0, 0.0], dtype=np.float32)
    result = estimate_icp_correction(anchor, target, LiveCorrectionConfig(max_translation_m=0.03))
    assert result.accepted is False
    assert result.reason == "translation_too_large"
```

- [x] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=. pytest tests/test_live_pointcloud_alignment.py -q`

Expected: FAIL because `teleop_backends.pointcloud.live_alignment` does not exist.

- [x] **Step 3: Implement the estimator**

Create a bounded point-to-point ICP estimator that:
- samples input clouds deterministically,
- uses `scipy.spatial.cKDTree` for correspondences,
- solves each incremental rigid correction with Kabsch,
- rejects corrections that exceed configured translation, rotation, RMSE, or inlier thresholds,
- returns serializable metrics.

- [x] **Step 4: Run tests to verify GREEN**

Run: `PYTHONPATH=. pytest tests/test_live_pointcloud_alignment.py -q`

Expected: PASS.

### Task 2: Calibration Source Integration

**Files:**
- Modify: `scripts/calibrate_two_cameras_charuco.py`
- Test: `tests/test_camera_calibration.py`

- [x] **Step 1: Write failing tests**

Add a source-level unit test with fake raw anchor/target frames and accepted detections. Verify the target transform used for fusion includes the live correction while calibration is active, and that `finish_calibration()` freezes the corrected transform and stops updating it.

- [x] **Step 2: Run test to verify RED**

Run: `PYTHONPATH=. pytest tests/test_camera_calibration.py -q`

Expected: FAIL because the source has no live correction or finish state.

- [x] **Step 3: Implement integration**

Add `LiveCorrectionConfig` wiring, per-target correction state, correction metrics in `calibration_snapshot()`, and a `finish_calibration()` method. During active calibration, compute corrections only when ChArUco detections and arm-motion gates are valid. During finished mode, fuse with frozen transforms and skip optimizer/correction updates.

- [x] **Step 4: Run tests to verify GREEN**

Run: `PYTHONPATH=. pytest tests/test_camera_calibration.py -q`

Expected: PASS.

### Task 3: Dashboard Finish Control

**Files:**
- Modify: `scripts/calibrate_two_cameras_charuco.py`
- Modify: `webxr_app/dashboard_static/index.html`
- Modify: `webxr_app/dashboard_static/dashboard.js`
- Modify: `webxr_app/dashboard_static/modules/dashboard_comms.js`
- Modify: `webxr_app/dashboard_static/modules/status_panel.js`
- Modify: `webxr_app/dashboard_static/style.css`
- Test: `tests/test_camera_calibration_cli.py`

- [x] **Step 1: Write failing tests**

Add parser tests for live-correction tuning flags and route tests or direct source tests for finished state fields.

- [x] **Step 2: Run test to verify RED**

Run: `PYTHONPATH=. pytest tests/test_camera_calibration_cli.py tests/test_camera_calibration.py -q`

Expected: FAIL because flags and finished state fields do not exist.

- [x] **Step 3: Implement dashboard action**

Add a POST `/api/calibration/finish` route, `DashboardComms.postJson()`, a Finish Calibration button, and status rows for calibration mode plus live correction metrics. Disable the button after finish.

- [x] **Step 4: Run focused tests**

Run: `PYTHONPATH=. pytest tests/test_camera_calibration.py tests/test_camera_calibration_cli.py tests/test_live_pointcloud_alignment.py -q`

Expected: PASS.

### Task 4: Verification and Commit

**Files:**
- All modified implementation, test, and dashboard files.

- [x] **Step 1: Run full test suite**

Run: `PYTHONPATH=. pytest -q`

Expected: PASS, or document any unrelated environment-only failures.

- [ ] **Step 2: Commit live correction**

Run:

```bash
git add docs/superpowers/plans/2026-05-26-live-fusion-correction.md \
  teleop_backends/pointcloud/live_alignment.py \
  scripts/calibrate_two_cameras_charuco.py \
  tests/test_live_pointcloud_alignment.py \
  tests/test_camera_calibration.py \
  tests/test_camera_calibration_cli.py \
  webxr_app/dashboard_static/index.html \
  webxr_app/dashboard_static/dashboard.js \
  webxr_app/dashboard_static/modules/dashboard_comms.js \
  webxr_app/dashboard_static/modules/status_panel.js \
  webxr_app/dashboard_static/style.css
git commit -m "feat: add live point-cloud fusion correction"
```
