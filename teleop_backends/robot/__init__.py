"""Robot driver backends."""

from .noop import NoopRobotDriver

__all__ = [
    "NoopRobotDriver",
    "PybulletRobotDriver",
    "PybulletPlanSimulator",
    "FloatingWristDriver",
    "AeroArmDriver",
]


def __getattr__(name: str):
    if name == "PybulletRobotDriver":
        from .pybullet_driver import PybulletRobotDriver
        return PybulletRobotDriver
    if name == "PybulletPlanSimulator":
        from .pybullet_preview import PybulletPlanSimulator
        return PybulletPlanSimulator
    if name == "FloatingWristDriver":
        from .floating_wrist_driver import FloatingWristDriver
        return FloatingWristDriver
    if name == "AeroArmDriver":
        from .aero_arm import AeroArmDriver
        return AeroArmDriver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
