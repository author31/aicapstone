import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from umi.cli import cli

POSE_SUBPATH = Path("demos") / "mapping" / "object_poses.json"


def _make_session(tmpdir: Path, name: str, episodes: list) -> Path:
    session = tmpdir / name
    pose_file = session / POSE_SUBPATH
    pose_file.parent.mkdir(parents=True)
    pose_file.write_text(json.dumps(episodes, indent=4))
    return session


def _sample_episode(start: int, end: int) -> dict:
    return {
        "video_name": "converted_60fps_raw_video.mp4",
        "episode_range": [start, end],
        "objects": [
            {"object_name": "pink_cup", "rvec": [1.0, 2.0, 3.0], "tvec": [0.1, 0.2, 0.3]}
        ],
        "status": "full",
    }


def test_merge_creates_output_dir():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dir1 = _make_session(tmpdir, "session_a", [_sample_episode(0, 100)])
        dir2 = _make_session(tmpdir, "session_b", [_sample_episode(100, 200), _sample_episode(200, 300)])

        result = runner.invoke(cli, ["merge-object-poses", str(dir1), str(dir2)])
        assert result.exit_code == 0, result.output

        out_file = tmpdir / "merged_session_a_session_b" / POSE_SUBPATH
        assert out_file.exists()
        merged = json.loads(out_file.read_text())
        assert len(merged) == 3


def test_merge_with_output_option():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dir1 = _make_session(tmpdir, "s1", [_sample_episode(0, 50)])
        dir2 = _make_session(tmpdir, "s2", [_sample_episode(50, 100)])
        custom_out = tmpdir / "custom_output"

        result = runner.invoke(cli, ["merge-object-poses", str(dir1), str(dir2), "-o", str(custom_out)])
        assert result.exit_code == 0, result.output

        out_file = custom_out / POSE_SUBPATH
        assert out_file.exists()
        merged = json.loads(out_file.read_text())
        assert len(merged) == 2


def test_missing_object_poses_json():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dir1 = _make_session(tmpdir, "has_poses", [_sample_episode(0, 100)])
        dir2 = tmpdir / "no_poses"
        dir2.mkdir()

        result = runner.invoke(cli, ["merge-object-poses", str(dir1), str(dir2)])
        assert result.exit_code != 0
        assert "not found" in result.output


def test_empty_arrays():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dir1 = _make_session(tmpdir, "empty1", [])
        dir2 = _make_session(tmpdir, "empty2", [])

        result = runner.invoke(cli, ["merge-object-poses", str(dir1), str(dir2)])
        assert result.exit_code == 0, result.output

        out_file = tmpdir / "merged_empty1_empty2" / POSE_SUBPATH
        merged = json.loads(out_file.read_text())
        assert merged == []


def test_malformed_json():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dir1 = _make_session(tmpdir, "good", [_sample_episode(0, 100)])
        dir2 = tmpdir / "bad"
        bad_file = dir2 / POSE_SUBPATH
        bad_file.parent.mkdir(parents=True)
        bad_file.write_text("{not valid json")

        result = runner.invoke(cli, ["merge-object-poses", str(dir1), str(dir2)])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output
