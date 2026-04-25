from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_top_level_layout_exists() -> None:
    for name in ["umi", "leisaac", "scripts", "data", "checkpoints"]:
        assert (ROOT / name).is_dir(), name


def test_single_top_level_pyproject_is_present() -> None:
    assert (ROOT / "pyproject.toml").is_file()
    assert not (ROOT / "umi" / "pyproject.toml").exists()
    assert not (ROOT / "leisaac" / "source" / "leisaac" / "pyproject.toml").exists()


def test_vendored_repos_are_flat_copies() -> None:
    assert not (ROOT / "umi" / ".git").exists()
    assert not (ROOT / "leisaac" / ".git").exists()
    assert not (ROOT / "leisaac" / ".gitmodules").exists()


def test_runtime_directories_are_tracked() -> None:
    assert (ROOT / "data" / ".gitkeep").is_file()
    assert (ROOT / "checkpoints" / ".gitkeep").is_file()
