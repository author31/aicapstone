"""Unit tests for ``simulator.tasks.external.resolve_task``.

The resolver is exercised end-to-end without booting IsaacLab / AppLauncher:
fixtures use a stub dataclass cfg and only string ``env_cfg_entry_point``
references, so neither the loader nor these tests need to actually instantiate
an Isaac environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_SRC = REPO_ROOT / "packages" / "simulator" / "src"
FIXTURES = Path(__file__).parent / "fixtures" / "external_tasks"

if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))


@pytest.fixture(autouse=True)
def _isolate_gym_registry():
    """Snapshot ``gym.registry`` and remove anything tests added."""
    snapshot = set(gym.registry.keys())
    yield
    for tid in list(gym.registry.keys()):
        if tid not in snapshot:
            del gym.registry[tid]


@pytest.fixture(autouse=True)
def _reset_loader_cache_and_modules():
    """Clear loader file cache and drop fixture modules from ``sys.modules``."""
    yield
    from simulator.tasks import external as ext

    ext._LOADED_FILES.clear()
    for name in list(sys.modules):
        if name.startswith("_aicapstone_external_task_"):
            sys.modules.pop(name, None)
    for name in (
        "valid_task",
        "no_register",
        "multi_register",
        "single_register_no_task_id",
    ):
        sys.modules.pop(name, None)


def _resolver():
    from simulator.tasks.external import resolve_task

    return resolve_task


def test_passthrough_for_already_registered_id():
    resolve_task = _resolver()
    gym.register(
        id="Test-Passthrough-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": "fake:Cfg"},
    )

    assert resolve_task("Test-Passthrough-v0") == "Test-Passthrough-v0"


def test_py_file_with_task_id_returns_declared_id():
    resolve_task = _resolver()

    task_id = resolve_task(str(FIXTURES / "valid_task.py"))

    assert task_id == "Private-Test-v0"
    assert "Private-Test-v0" in gym.registry


def test_py_file_infers_single_new_id_without_task_id():
    resolve_task = _resolver()

    task_id = resolve_task(str(FIXTURES / "single_register_no_task_id.py"))

    assert task_id == "Private-Single-v0"


def test_module_class_import_then_scan(monkeypatch):
    monkeypatch.syspath_prepend(str(FIXTURES))
    resolve_task = _resolver()

    task_id = resolve_task("valid_task:DummyCfg")

    assert task_id == "Private-Test-v0"


def test_relative_py_path_is_resolved(tmp_path, monkeypatch):
    """Relative paths resolve via the current working directory."""
    src = (FIXTURES / "single_register_no_task_id.py").read_text()
    (tmp_path / "rel_task.py").write_text(src)
    monkeypatch.chdir(tmp_path)
    resolve_task = _resolver()

    assert resolve_task("./rel_task.py") == "Private-Single-v0"


def test_idempotent_reload_does_not_re_register():
    resolve_task = _resolver()
    path = str(FIXTURES / "valid_task.py")

    first = resolve_task(path)
    # Second call must hit the de-dupe cache. Without it ``exec_module`` would
    # run again and re-call ``gym.register`` for the same id.
    second = resolve_task(path)

    assert first == second == "Private-Test-v0"


def test_unregistered_id_raises_value_error():
    resolve_task = _resolver()

    with pytest.raises(ValueError, match="not a registered gym id"):
        resolve_task("Totally-Not-Registered-v999")


def test_missing_py_path_raises_value_error(tmp_path):
    resolve_task = _resolver()
    missing = tmp_path / "does_not_exist.py"

    with pytest.raises(ValueError, match="not a registered gym id"):
        resolve_task(str(missing))


def test_no_register_call_raises_runtime_error():
    resolve_task = _resolver()

    with pytest.raises(RuntimeError, match=r"did not call `gym\.register"):
        resolve_task(str(FIXTURES / "no_register.py"))


def test_multi_register_without_task_id_raises_runtime_error():
    resolve_task = _resolver()

    with pytest.raises(RuntimeError, match="registered multiple gym ids"):
        resolve_task(str(FIXTURES / "multi_register.py"))


def test_module_class_with_no_matching_entry_point_raises(monkeypatch):
    monkeypatch.syspath_prepend(str(FIXTURES))
    resolve_task = _resolver()

    with pytest.raises(RuntimeError, match="did not register a gym id"):
        # valid_task module registers Private-Test-v0 with env_cfg_entry_point
        # 'valid_task:DummyCfg'. Asking for a different class raises.
        resolve_task("valid_task:NotAClassThatExists")


def test_invalid_module_class_ref_raises_value_error():
    resolve_task = _resolver()

    with pytest.raises(ValueError, match="valid module:Class"):
        resolve_task("missing_class_part:")
