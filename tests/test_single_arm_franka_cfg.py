import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "packages" / "leisaac_overlay" / "src" / "leisaac" / "tasks" / "template" / "single_arm_franka_cfg.py"


def test_single_arm_franka_base_env_cfg_declares_object_poses_path() -> None:
    source = CFG_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)

    env_cfg_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "SingleArmFrankaTaskEnvCfg"
    )
    object_poses_field = next(
        node
        for node in env_cfg_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "object_poses_path"
    )

    assert ast.get_source_segment(source, object_poses_field.annotation) == "str | None"
    assert isinstance(object_poses_field.value, ast.Constant)
    assert object_poses_field.value.value is None
