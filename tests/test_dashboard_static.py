import json
import subprocess
import textwrap


def _run_node_module(script: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_camera_feed_panel_renders_model_feeds_without_calibration_snapshot():
    script = textwrap.dedent(
        """
        import { CalibrationDiagnosticsPanel } from './webxr_app/dashboard_static/modules/calibration_diagnostics_panel.js';

        const root = { innerHTML: '' };
        const panel = new CalibrationDiagnosticsPanel(root);
        panel.update({
          model: {
            camera_feeds: [
              {
                name: 'd435i',
                url: '/api/cameras/d435i/color.jpg',
                width: 640,
                height: 480,
              },
            ],
          },
        });

        console.log(JSON.stringify({ html: root.innerHTML }));
        """
    )

    result = _run_node_module(script)

    assert "Camera Feeds" in result["html"]
    assert "d435i" in result["html"]
    assert "/api/cameras/d435i/color.jpg" in result["html"]

