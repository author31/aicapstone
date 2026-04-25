from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEISAAC_OVERLAY = ROOT / "packages" / "leisaac_overlay"


def test_expected_root_scaffolding_exists() -> None:
    for path in [
        "packages/umi",
        "packages/leisaac_overlay",
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
    assert (LEISAAC_OVERLAY / "pyproject.toml").is_file()


def test_root_pyproject_declares_uv_workspace_with_upstream_leisaac_git_source() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'members = ["packages/*"]' in pyproject
    assert 'umi = { workspace = true }' in pyproject
    assert 'leisaac-overlay = { workspace = true }' in pyproject
    # Upstream leisaac is consumed as a pinned uv git source rather than a
    # workspace member to avoid the dist-name collision with the overlay.
    assert 'leisaac = { git = "https://github.com/author31/leisaac.git"' in pyproject
    assert 'subdirectory = "source/leisaac"' in pyproject
    assert 'rev = "' in pyproject


def test_workspace_members_are_flat_vendored_copies() -> None:
    assert not (ROOT / "packages" / "umi" / ".git").exists()
    assert not (LEISAAC_OVERLAY / ".git").exists()
    assert not (LEISAAC_OVERLAY / ".gitmodules").exists()


def test_leisaac_overlay_member_is_extension_only() -> None:
    assert (LEISAAC_OVERLAY / "src" / "leisaac" / "tasks" / "cup_stacking").is_dir()
    assert (LEISAAC_OVERLAY / "src" / "leisaac" / "tasks" / "template" / "single_arm_franka_cfg.py").is_file()
    assert not (LEISAAC_OVERLAY / "source").exists()
    assert not (LEISAAC_OVERLAY / "docs").exists()
    assert not (LEISAAC_OVERLAY / "scripts").exists()
    assert not (LEISAAC_OVERLAY / "assets").exists()
    assert not (LEISAAC_OVERLAY / "Dockerfile").exists()
    assert not (LEISAAC_OVERLAY / "Makefile").exists()


def test_leisaac_overlay_keeps_extend_path_shim() -> None:
    init_text = (LEISAAC_OVERLAY / "src" / "leisaac" / "__init__.py").read_text()
    assert "extend_path" in init_text
    tasks_init_text = (LEISAAC_OVERLAY / "src" / "leisaac" / "tasks" / "__init__.py").read_text()
    assert "extend_path" in tasks_init_text


def test_legacy_leisaac_overlay_path_is_gone() -> None:
    # Old `packages/leisaac/` shadowed the upstream dist name; the rename must
    # leave nothing at the legacy path.
    assert not (ROOT / "packages" / "leisaac").exists()


def test_umi_member_omits_requested_subtrees() -> None:
    assert not (ROOT / "packages" / "umi" / "src" / "umi" / "pipeline" / "aruco_detection.py").exists()
    assert not (ROOT / "packages" / "umi" / "src" / "umi" / "real_world").exists()
    assert not (ROOT / "packages" / "umi" / "src" / "umi" / "shared_memory").exists()
    assert not (ROOT / "packages" / "umi" / "src" / "umi" / "traj_eval").exists()


def test_runtime_directories_are_tracked() -> None:
    assert (ROOT / "data" / ".gitkeep").is_file()
    assert (ROOT / "checkpoints" / ".gitkeep").is_file()


def test_dockerfile_no_longer_overlays_leisaac_via_pythonpath() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "packages/leisaac/source/leisaac" not in dockerfile
    assert "packages/leisaac_overlay" not in dockerfile


def test_scripts_tree_mirrors_upstream_leisaac() -> None:
    # The bulk-copy ticket mirrors the upstream leisaac/scripts/ tree verbatim;
    # downstream tickets (AUT-6 and follow-ups) own any per-file edits.
    for subdir in ("datagen", "evaluation", "mimic", "environments", "tutorials", "convert"):
        assert (ROOT / "scripts" / subdir).is_dir(), subdir
    assert (ROOT / "scripts" / "datagen" / "state_machine" / "generate.py").is_file()
