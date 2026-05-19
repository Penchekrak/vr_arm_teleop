import subprocess
import sys


def test_cli_help_does_not_require_pybullet():
    result = subprocess.run(
        [sys.executable, "-m", "webxr_app", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--robot-backend" in result.stdout
