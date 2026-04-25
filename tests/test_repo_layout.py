from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_root_scaffolding_exists() -> None:
    for path in [
        "packages/umi",
        "packages/leisaac",
        "scripts",
        "data",
        "checkpoints",
        "Dockerfile",
        "Makefile",
        "README.md",
        "pyproject.toml",
    ]:
        assert (ROOT / path).exists(), path


def test_workspace_members_have_pyproject_files() -> None:
    assert (ROOT / "packages" / "umi" / "pyproject.toml").is_file()
    assert (ROOT / "packages" / "leisaac" / "pyproject.toml").is_file()


def test_root_pyproject_declares_uv_workspace() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'members = ["packages/*"]' in pyproject
    assert 'umi = { workspace = true }' in pyproject
    assert 'leisaac = { workspace = true }' in pyproject


def test_workspace_members_are_flat_vendored_copies() -> None:
    assert not (ROOT / "packages" / "umi" / ".git").exists()
    assert not (ROOT / "packages" / "leisaac" / ".git").exists()
    assert not (ROOT / "packages" / "leisaac" / ".gitmodules").exists()


def test_leisaac_member_is_extension_only() -> None:
    assert (ROOT / "packages" / "leisaac" / "source" / "leisaac" / "leisaac" / "tasks" / "cup_stacking").is_dir()
    assert (ROOT / "packages" / "leisaac" / "source" / "leisaac" / "leisaac" / "tasks" / "template" / "single_arm_franka_cfg.py").is_file()
    assert not (ROOT / "packages" / "leisaac" / "docs").exists()
    assert not (ROOT / "packages" / "leisaac" / "scripts").exists()
    assert not (ROOT / "packages" / "leisaac" / "assets").exists()
    assert not (ROOT / "packages" / "leisaac" / "Dockerfile").exists()
    assert not (ROOT / "packages" / "leisaac" / "Makefile").exists()


def test_runtime_directories_are_tracked() -> None:
    assert (ROOT / "data" / ".gitkeep").is_file()
    assert (ROOT / "checkpoints" / ".gitkeep").is_file()
