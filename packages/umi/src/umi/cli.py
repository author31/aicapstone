import json
from pathlib import Path

import click
from umi.pipeline_executor import PipelineExecutor
from umi.profiler import PipelineProfiler
from umi.services.visualize_slam_gui import VisualizeSLAMGUI


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_path")
@click.option("--session-dir", type=click.Path(exists=True), help="Override session directory from config file")
@click.option("--task", type=click.Choice(["kitchen", "living_room", "dining_room"]), help="Specify task type")
@click.option("--profile/--no-profile", "-p", default=False,
              help="Profile per-stage runtime and video counts; write CSV report.")
@click.option("--profile-output", type=click.Path(),
              help="Path for profile CSV. Default: {session_dir}/pipeline_profile.csv or ./pipeline_profile.csv")
def run_slam_pipeline(config_path: str, session_dir: str, task: str, profile: bool, profile_output: str):
    profiler = None
    if profile:
        if profile_output:
            out_path = Path(profile_output)
        elif session_dir:
            out_path = Path(session_dir) / "pipeline_profile.csv"
        else:
            out_path = Path("pipeline_profile.csv")
        profiler = PipelineProfiler(out_path)

    executor = PipelineExecutor(
        config_path,
        session_dir_override=session_dir,
        task_override=task,
        profiler=profiler,
    )
    executor.execute_all()


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--session-dir", type=click.Path(exists=True), required=True,
              help="Session directory for SLAM output")
@click.option("--docker-image", default="chicheng/orb_slam3:latest",
              help="ORB-SLAM3 Docker image")
@click.option("--settings-file",
              help="ORB-SLAM3 settings file path")
@click.option("--force", is_flag=True,
              help="Force re-run even if GUI already running")
def visualize_slam_gui(video_path: str, session_dir: str, docker_image: str,
                       settings_file: str, force: bool):
    """Launch ORB-SLAM3 GUI for debugging specified video."""
    config = {
        "session_dir": session_dir,
        "video_path": video_path,
        "docker_image": docker_image,
        "slam_settings_file": settings_file,
        "force": force
    }

    try:
        service = VisualizeSLAMGUI(config)
        result = service.execute()

        if result["status"] == "completed":
            click.echo(f"SLAM GUI execution completed successfully")
        elif result["status"] == "interrupted":
            click.echo(f"SLAM GUI execution interrupted by user")
        else:
            click.echo(f"SLAM GUI execution failed with return code {result.get('return_code', 'unknown')}")

        click.echo(f"Session directory: {result['session_dir']}")
        click.echo(f"Video file: {result['video_path']}")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.ClickException(str(e))


@cli.command()
@click.argument("session_dir_1", type=click.Path(exists=True))
@click.argument("session_dir_2", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(),
              help="Override output session directory path.")
def merge_object_poses(session_dir_1: str, session_dir_2: str, output: str):
    """Merge object_poses.json files from two session directories into a new session directory."""
    pose_subpath = Path("demos") / "mapping" / "object_poses.json"
    dir1, dir2 = Path(session_dir_1), Path(session_dir_2)
    path1, path2 = dir1 / pose_subpath, dir2 / pose_subpath

    datasets = []
    for p, d in [(path1, session_dir_1), (path2, session_dir_2)]:
        if not p.exists():
            raise click.ClickException(f"object_poses.json not found in {d}/demos/mapping/")
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in {p}: {e}")
        if not isinstance(data, list):
            raise click.ClickException(f"Expected JSON array in {p}, got {type(data).__name__}")
        datasets.append(data)

    merged = datasets[0] + datasets[1]

    if output:
        out_dir = Path(output)
    else:
        out_dir = dir1.parent / f"merged_{dir1.name}_{dir2.name}"

    out_path = out_dir / pose_subpath
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=4) + "\n")

    click.echo(f"Merged {len(datasets[0])} + {len(datasets[1])} = {len(merged)} episodes")
    click.echo(f"Output: {out_path}")


if __name__ == "__main__":
    cli()
