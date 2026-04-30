"""Smart ``--task`` resolver supporting gym id, ``.py`` path, or ``module:Class`` refs.

The resolver lets callers point ``--task`` at:

* an existing gym id (the in-tree contract);
* a path to a ``.py`` file that calls ``gym.register(...)`` at import time
  (private/external eval scenarios that must not be checked into the public
  repo);
* a ``module:Class`` reference for tasks shipped as installed packages.

The loader is intentionally side-effect light: it never imports IsaacLab and
expects ``AppLauncher`` to have been booted before it is called only because
the *user file* may import ``isaaclab.*`` symbols at the top level.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import uuid
from pathlib import Path

import gymnasium as gym


_LOADED_FILES: dict[str, str] = {}


_REGISTER_SKELETON = (
    "Expected the file to call `gym.register(...)`. Minimal example:\n"
    "    import gymnasium as gym\n"
    "    TASK_ID = 'Private-MyEval-v0'\n"
    "    gym.register(\n"
    "        id=TASK_ID,\n"
    "        entry_point='isaaclab.envs:ManagerBasedRLEnv',\n"
    "        kwargs={'env_cfg_entry_point': f'{__name__}:MyEvalEnvCfg'},\n"
    "    )\n"
)


def resolve_task(spec: str) -> str:
    """Return a registered gym task id for ``spec``.

    Accepted forms (auto-detected, in order):

    1. Already-registered gym id — returned as-is.
    2. Path to a ``.py`` file (``Path.is_file()`` and ``.py`` suffix) — imported
       under a synthetic module name so ``gym.register`` runs, then the task id
       is taken from ``module.TASK_ID`` or inferred from the single new
       registry entry.
    3. ``module:Class`` reference — module is imported (which is expected to
       call ``gym.register``), then the registry is scanned for an entry whose
       ``env_cfg_entry_point`` equals ``spec``.

    Raises:
        ValueError: ``spec`` matches none of the three forms (e.g. unknown id,
            non-existent ``.py`` path, or string with no colon).
        RuntimeError: a ``.py`` file imported successfully but registered no
            gym id (or registered multiple ids without declaring ``TASK_ID``),
            or a ``module:Class`` ref imported but no matching id was found.
    """
    if spec in gym.registry:
        return spec

    candidate = Path(spec).expanduser()
    if candidate.suffix == ".py" and candidate.is_file():
        return _load_from_file(candidate.resolve())

    if ":" in spec:
        return _load_from_module_ref(spec)

    raise ValueError(
        f"--task '{spec}' is not a registered gym id, an existing .py path, "
        f"or a module:Class reference"
    )


def _load_from_file(path: Path) -> str:
    abs_path = str(path)
    cached = _LOADED_FILES.get(abs_path)
    if cached is not None:
        return cached

    pre_ids = set(gym.registry.keys())
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    mod_name = f"_aicapstone_external_task_{uuid.uuid4().hex}"
    file_spec = importlib.util.spec_from_file_location(mod_name, path)
    if file_spec is None or file_spec.loader is None:
        raise RuntimeError(f"Could not build import spec for '{path}'")

    module = importlib.util.module_from_spec(file_spec)
    sys.modules[mod_name] = module
    try:
        file_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise

    declared = getattr(module, "TASK_ID", None)
    if isinstance(declared, str) and declared in gym.registry:
        _LOADED_FILES[abs_path] = declared
        return declared

    new_ids = set(gym.registry.keys()) - pre_ids
    if len(new_ids) == 1:
        tid = next(iter(new_ids))
        _LOADED_FILES[abs_path] = tid
        return tid

    if len(new_ids) > 1:
        raise RuntimeError(
            f"External task file '{path}' registered multiple gym ids "
            f"{sorted(new_ids)}. Declare `TASK_ID = '<id>'` to disambiguate."
        )

    raise RuntimeError(
        f"External task file '{path}' did not call `gym.register(...)`.\n"
        + _REGISTER_SKELETON
    )


def _load_from_module_ref(spec: str) -> str:
    mod_path, _, cls_name = spec.partition(":")
    if not mod_path or not cls_name:
        raise ValueError(
            f"--task '{spec}' is not a valid module:Class reference"
        )

    importlib.import_module(mod_path)

    for tid, entry in gym.registry.items():
        kwargs = getattr(entry, "kwargs", None) or {}
        if kwargs.get("env_cfg_entry_point") == spec:
            return tid

    raise RuntimeError(
        f"module:Class '{spec}' did not register a gym id with a matching "
        f"`env_cfg_entry_point`. Make sure the module's import-time "
        f"`gym.register(...)` sets `kwargs={{'env_cfg_entry_point': '{spec}'}}`."
    )
