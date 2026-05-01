"""Smart ``--task`` resolver for rollout / eval scripts.

Accepts three input forms and returns a registered ``gymnasium`` task id:

1. A gym id already present in ``gym.registry`` (returned unchanged).
2. A path to a ``.py`` file that performs ``gym.register`` at import time.
3. A ``module:Class`` reference whose import side-effect registers the task,
   or whose ``module:Class`` string equals an existing task's
   ``env_cfg_entry_point``.

See ``AUT-80`` for the design rationale.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import uuid
from pathlib import Path

import gymnasium as gym

__all__ = ["resolve_task"]

# Maps absolute file path -> gym id, so repeated loads of the same .py
# file in one process do not re-execute the module or re-register.
_FILE_LOAD_CACHE: dict[str, str] = {}

_SKELETON_EXAMPLE = """\
Example skeleton for an external task file:

    import gymnasium as gym
    from isaaclab.utils import configclass
    from simulator.tasks.template.single_arm_franka_cfg import (
        SingleArmFrankaTaskEnvCfg,
    )

    @configclass
    class MyEvalEnvCfg(SingleArmFrankaTaskEnvCfg):
        ...

    TASK_ID = "Private-MyEval-SingleArm-v0"

    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}:MyEvalEnvCfg"},
    )
"""


def resolve_task(spec: str) -> str:
    """Return a registered gym task id for ``spec``.

    Args:
        spec: gym id, ``.py`` path, or ``module:Class`` reference.

    Raises:
        ValueError: ``spec`` does not match any supported form.
        RuntimeError: the referenced file/module did not register a task.
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError(f"--task spec must be a non-empty string, got {spec!r}")

    if spec in gym.registry:
        return spec

    p = Path(spec).expanduser()
    if p.suffix == ".py":
        if not p.is_file():
            raise ValueError(f"--task '{spec}' looks like a .py path but file does not exist")
        return _load_from_file(p.resolve())

    if ":" in spec:
        return _load_from_module_ref(spec)

    raise ValueError(
        f"--task '{spec}' is not a registered gym id, .py path, or module:Class ref"
    )


def _load_from_file(path: Path) -> str:
    abs_path = str(path)
    cached = _FILE_LOAD_CACHE.get(abs_path)
    if cached is not None and cached in gym.registry:
        return cached

    pre = set(gym.registry.keys())
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    mod_name = f"_aicapstone_external_task_{uuid.uuid4().hex}"
    spec_obj = importlib.util.spec_from_file_location(mod_name, path)
    if spec_obj is None or spec_obj.loader is None:
        raise RuntimeError(f"Could not build import spec for '{path}'")

    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[mod_name] = module
    try:
        spec_obj.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise

    declared = getattr(module, "TASK_ID", None)
    new_ids = set(gym.registry.keys()) - pre

    if declared is not None:
        if declared not in gym.registry:
            raise RuntimeError(
                f"External file '{path}' declares TASK_ID={declared!r} but did not "
                f"register it via gym.register"
            )
        _FILE_LOAD_CACHE[abs_path] = declared
        return declared

    if len(new_ids) == 1:
        task_id = next(iter(new_ids))
        _FILE_LOAD_CACHE[abs_path] = task_id
        return task_id

    if len(new_ids) > 1:
        raise RuntimeError(
            f"External file '{path}' registered multiple gym ids {sorted(new_ids)}; "
            f"declare TASK_ID in the file to disambiguate"
        )

    raise RuntimeError(
        f"External file '{path}' did not call gym.register (0 new ids registered) "
        f"and has no TASK_ID attribute.\n\n{_SKELETON_EXAMPLE}"
    )


def _load_from_module_ref(spec: str) -> str:
    if spec.count(":") != 1:
        raise ValueError(
            f"--task '{spec}' must contain exactly one ':' separating module and class"
        )

    mod_path, cls_name = spec.split(":", 1)
    if not mod_path or not cls_name:
        raise ValueError(
            f"--task '{spec}' must be of the form 'module.path:ClassName'"
        )

    try:
        importlib.import_module(mod_path)
    except ImportError as exc:
        raise ValueError(
            f"--task '{spec}' module '{mod_path}' could not be imported: {exc}"
        ) from exc

    for tid, entry in gym.registry.items():
        kwargs = getattr(entry, "kwargs", None) or {}
        if kwargs.get("env_cfg_entry_point") == spec:
            return tid

    raise RuntimeError(
        f"module:Class '{spec}' did not register a gym id whose "
        f"env_cfg_entry_point matches it"
    )
