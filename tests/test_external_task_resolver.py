"""Unit tests for ``simulator.tasks.external.resolve_task``.

These tests deliberately avoid pulling in ``isaaclab`` — they exercise the
loader's branching, registry interaction, dedupe, and error messages with
synthetic ``.py`` fixtures that only call ``gym.register``.
"""

from __future__ import annotations

import sys
import textwrap
import uuid
from pathlib import Path

import gymnasium as gym
import pytest

ROOT = Path(__file__).resolve().parents[1]
SIM_SRC = ROOT / "packages" / "simulator" / "src"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))

from simulator.tasks import external as external_mod  # noqa: E402
from simulator.tasks.external import resolve_task  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_registry_and_cache():
    """Snapshot gym registry + loader cache so tests do not bleed state."""
    saved_registry = dict(gym.registry)
    saved_cache = dict(external_mod._FILE_LOAD_CACHE)
    saved_modules = set(sys.modules.keys())
    yield
    for tid in list(gym.registry.keys()):
        if tid not in saved_registry:
            del gym.registry[tid]
    external_mod._FILE_LOAD_CACHE.clear()
    external_mod._FILE_LOAD_CACHE.update(saved_cache)
    for mod_name in list(sys.modules.keys()):
        if mod_name not in saved_modules and mod_name.startswith("_aicapstone_external_task_"):
            del sys.modules[mod_name]


def _unique_id(prefix: str = "Test-Resolver") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}-v0"


def _write_task_file(tmp_path: Path, body: str, name: str = "task.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


# ---------- registered gym id ----------


def test_resolves_registered_gym_id():
    task_id = _unique_id()
    gym.register(id=task_id, entry_point="cartpole:CartPole")
    assert resolve_task(task_id) == task_id


# ---------- .py file form ----------


def test_loads_py_file_with_task_id_attribute(tmp_path: Path):
    task_id = _unique_id()
    f = _write_task_file(
        tmp_path,
        f"""
        import gymnasium as gym
        TASK_ID = "{task_id}"
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={{"env_cfg_entry_point": f"{{__name__}}:Cfg"}},
        )
        """,
    )
    assert resolve_task(str(f)) == task_id
    assert task_id in gym.registry


def test_loads_py_file_without_task_id_single_register(tmp_path: Path):
    task_id = _unique_id()
    f = _write_task_file(
        tmp_path,
        f"""
        import gymnasium as gym
        gym.register(id="{task_id}", entry_point="isaaclab.envs:ManagerBasedRLEnv")
        """,
    )
    assert resolve_task(str(f)) == task_id


def test_py_file_with_no_register_raises_runtime_error(tmp_path: Path):
    f = _write_task_file(tmp_path, "x = 1\n")
    with pytest.raises(RuntimeError, match="did not call gym.register"):
        resolve_task(str(f))


def test_py_file_with_multiple_registers_requires_task_id(tmp_path: Path):
    a = _unique_id("Multi-A")
    b = _unique_id("Multi-B")
    f = _write_task_file(
        tmp_path,
        f"""
        import gymnasium as gym
        gym.register(id="{a}", entry_point="isaaclab.envs:ManagerBasedRLEnv")
        gym.register(id="{b}", entry_point="isaaclab.envs:ManagerBasedRLEnv")
        """,
    )
    with pytest.raises(RuntimeError, match="multiple gym ids"):
        resolve_task(str(f))


def test_py_file_task_id_must_be_registered(tmp_path: Path):
    f = _write_task_file(
        tmp_path,
        """
        TASK_ID = "Declared-But-Not-Registered-v0"
        """,
    )
    with pytest.raises(RuntimeError, match="declares TASK_ID"):
        resolve_task(str(f))


def test_repeated_load_of_same_file_is_deduped(tmp_path: Path):
    task_id = _unique_id()
    f = _write_task_file(
        tmp_path,
        f"""
        import gymnasium as gym
        TASK_ID = "{task_id}"
        gym.register(id=TASK_ID, entry_point="isaaclab.envs:ManagerBasedRLEnv")
        """,
    )
    first = resolve_task(str(f))
    # If the loader re-executed, gym.register would raise on duplicate id.
    second = resolve_task(str(f))
    assert first == second == task_id


def test_nonexistent_py_path_raises_value_error(tmp_path: Path):
    missing = tmp_path / "does_not_exist.py"
    with pytest.raises(ValueError, match="file does not exist"):
        resolve_task(str(missing))


# ---------- module:Class form ----------


def test_loads_module_ref_matching_env_cfg_entry_point(tmp_path: Path, monkeypatch):
    pkg_dir = tmp_path / "pkg_for_module_ref"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    task_id = _unique_id("ModuleRef")
    entry = f"pkg_for_module_ref.eval:MyEvalCfg"
    (pkg_dir / "eval.py").write_text(
        textwrap.dedent(
            f"""
            import gymnasium as gym
            class MyEvalCfg:  # placeholder
                pass
            gym.register(
                id="{task_id}",
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={{"env_cfg_entry_point": "{entry}"}},
            )
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert resolve_task(entry) == task_id


def test_module_ref_unimportable_raises_value_error():
    with pytest.raises(ValueError, match="could not be imported"):
        resolve_task("definitely_not_a_real_module_xyz:Cls")


def test_module_ref_imported_but_no_matching_entry_point(tmp_path: Path, monkeypatch):
    pkg_dir = tmp_path / "pkg_no_match"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "mod.py").write_text("class Cfg: pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RuntimeError, match="did not register a gym id"):
        resolve_task("pkg_no_match.mod:Cfg")


# ---------- bad inputs ----------


def test_unregistered_plain_id_raises_value_error():
    with pytest.raises(ValueError, match="not a registered gym id"):
        resolve_task("Totally-Made-Up-Task-v0")


def test_empty_spec_raises_value_error():
    with pytest.raises(ValueError):
        resolve_task("")


def test_missing_class_after_colon_raises_value_error():
    with pytest.raises(ValueError, match="module.path:ClassName"):
        resolve_task("some.module:")


def test_too_many_colons_raises_value_error():
    with pytest.raises(ValueError, match="exactly one ':'"):
        resolve_task("a:b:c")
