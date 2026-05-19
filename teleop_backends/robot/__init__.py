"""Robot driver backends."""

__all__ = [
    "NoopRobotDriver",
    "PybulletRobotDriver",
    "FloatingWristDriver",
    "AeroArmDriver",
]

_EXPORTS = {
    "NoopRobotDriver": ("noop", "NoopRobotDriver"),
    "PybulletRobotDriver": ("pybullet_driver", "PybulletRobotDriver"),
    "FloatingWristDriver": ("floating_wrist_driver", "FloatingWristDriver"),
    "AeroArmDriver": ("aero_arm", "AeroArmDriver"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
