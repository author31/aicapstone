"""Static checks for the resolve_task wiring in scripts/rollout.py (AUT-84).

Importing rollout.py directly pulls in IsaacLab and AppLauncher, which is too
heavy for a unit test. Parse the source via ast instead and assert the wiring
contract: import order, the resolve_task call, and the --task help string.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT = ROOT / "scripts" / "rollout.py"


@pytest.fixture(scope="module")
def source() -> str:
    return ROLLOUT.read_text()


@pytest.fixture(scope="module")
def tree(source: str) -> ast.Module:
    return ast.parse(source)


def _import_names(tree: ast.Module) -> list[str]:
    """Return module-level imported names in source order ('mod' or 'mod:name')."""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                names.append(f"{node.module}:{alias.name}")
    return names


def test_rollout_script_exists() -> None:
    assert ROLLOUT.is_file(), "scripts/rollout.py must exist for resolve_task wiring"


def test_resolve_task_imported_after_simulator_tasks(tree: ast.Module) -> None:
    names = _import_names(tree)
    assert "simulator.tasks" in names, "simulator.tasks must be imported to register in-tree gym ids"
    assert "simulator.tasks.external:resolve_task" in names, "resolve_task must be imported from simulator.tasks.external"
    assert names.index("simulator.tasks") < names.index(
        "simulator.tasks.external:resolve_task"
    ), "simulator.tasks must be imported before resolve_task so registrations happen first"


def test_resolve_task_imported_after_app_launcher_block(source: str) -> None:
    """AppLauncher must boot before any isaaclab.envs imports; resolve_task
    pulls in modules that may import isaaclab, so it must come after the
    `simulation_app = ...app` line."""
    app_idx = source.index("simulation_app = app_launcher.app")
    resolve_idx = source.index("from simulator.tasks.external import resolve_task")
    assert app_idx < resolve_idx


def test_main_resolves_task_id(source: str) -> None:
    assert "task_id = resolve_task(args_cli.task)" in source
    assert "parse_env_cfg(task_id" in source, "parse_env_cfg must be called with the resolved task_id"
    assert "gym.make(task_id" in source, "gym.make must be called with the resolved task_id"
    assert "parse_env_cfg(args_cli.task" not in source, "parse_env_cfg must not consume args_cli.task directly"
    assert "gym.make(args_cli.task" not in source, "gym.make must not consume args_cli.task directly"


def test_task_help_describes_three_forms(tree: ast.Module) -> None:
    """The --task argument's help string must describe gym id, .py path, and module:Class forms."""
    help_text: str | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            args = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if "--task" not in args:
                continue
            for kw in node.keywords:
                if kw.arg == "help":
                    if isinstance(kw.value, ast.Constant):
                        help_text = kw.value.value
                    elif isinstance(kw.value, ast.Call) and getattr(kw.value.func, "id", None) is None:
                        # parenthesised string concat — render the constants
                        help_text = "".join(
                            a.value for a in ast.walk(kw.value) if isinstance(a, ast.Constant)
                        )
                    else:
                        help_text = ast.unparse(kw.value)
            break
    assert help_text is not None, "--task argument must declare a help string"
    lowered = help_text.lower()
    assert "gym" in lowered, "help text must mention the gym id form"
    assert ".py" in lowered, "help text must mention the .py file form"
    assert "module:class" in lowered or "module:" in lowered, "help text must mention the module:Class form"


def test_no_env_cfg_file_flag(source: str) -> None:
    """AC: no --env_cfg_file flag introduced (the alternative AUT-32 design)."""
    assert "--env_cfg_file" not in source
