from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_root_scaffolding_exists() -> None:
    for path in [
        "packages/umi",
        "packages/simulator",
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
    assert (ROOT / "packages" / "simulator" / "pyproject.toml").is_file()


def test_root_pyproject_declares_uv_workspace() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'members = ["packages/*"]' in pyproject
    assert 'umi = { workspace = true }' in pyproject
    assert 'simulator = { workspace = true }' in pyproject


def test_workspace_members_are_flat_vendored_copies() -> None:
    assert not (ROOT / "packages" / "umi" / ".git").exists()
    assert not (ROOT / "packages" / "simulator" / ".git").exists()
    assert not (ROOT / "packages" / "simulator" / ".gitmodules").exists()


def test_simulator_member_is_extension_only() -> None:
    assert (ROOT / "packages" / "simulator" / "src" / "simulator" / "tasks" / "cup_stacking").is_dir()
    assert (ROOT / "packages" / "simulator" / "src" / "simulator" / "tasks" / "template" / "single_arm_franka_cfg.py").is_file()
    assert not (ROOT / "packages" / "simulator" / "source").exists()
    assert not (ROOT / "packages" / "simulator" / "docs").exists()
    assert not (ROOT / "packages" / "simulator" / "scripts").exists()
    assert not (ROOT / "packages" / "simulator" / "assets").exists()
    assert not (ROOT / "packages" / "simulator" / "Dockerfile").exists()
    assert not (ROOT / "packages" / "simulator" / "Makefile").exists()


def test_umi_member_omits_requested_subtrees() -> None:
    assert not (ROOT / "packages" / "umi" / "src" / "umi" / "pipeline" / "aruco_detection.py").exists()
    assert not (ROOT / "packages" / "umi" / "src" / "umi" / "real_world").exists()
    assert not (ROOT / "packages" / "umi" / "src" / "umi" / "shared_memory").exists()
    assert not (ROOT / "packages" / "umi" / "src" / "umi" / "traj_eval").exists()


def test_runtime_directories_are_tracked() -> None:
    assert (ROOT / "data" / ".gitkeep").is_file()
    assert (ROOT / "checkpoints" / ".gitkeep").is_file()
