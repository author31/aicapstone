from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_top_level_layout_exists() -> None:
    for name in ["umi", "source", "scripts", "data", "checkpoints"]:
        assert (ROOT / name).is_dir(), name


def test_single_top_level_pyproject_is_present() -> None:
    assert (ROOT / "pyproject.toml").is_file()
    assert not (ROOT / "umi" / "pyproject.toml").exists()
    assert not (ROOT / "source" / "leisaac" / "pyproject.toml").exists()


def test_leisaac_is_dependency_with_local_extensions() -> None:
    assert not (ROOT / "umi" / ".git").exists()
    assert not (ROOT / "leisaac").exists()
    assert (ROOT / "source" / "leisaac" / "leisaac" / "tasks" / "cup_stacking").is_dir()
    assert (ROOT / "source" / "leisaac" / "leisaac" / "tasks" / "template" / "single_arm_franka_cfg.py").is_file()


def test_top_level_pyproject_declares_leisaac_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "leisaac @ git+https://github.com/author31/leisaac.git@feat/add-joint-position-action-space#subdirectory=source/leisaac" in pyproject


def test_runtime_directories_are_tracked() -> None:
    assert (ROOT / "data" / ".gitkeep").is_file()
    assert (ROOT / "checkpoints" / ".gitkeep").is_file()
