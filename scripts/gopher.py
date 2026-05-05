import sys
import shutil
from pathlib import Path


def copy_if_exists(src: Path, dst: Path, label: str):
    if src.exists():
        shutil.copy(src, dst)
        print(f"[OK] {label}: {src} -> {dst}")
        return True
    else:
        print(f"[WARN] missing {label}: {src}")
        return False


def safe_remove_dir(path: Path):
    if not path.exists():
        return

    name = path.name

    if name not in ["demos", "raw_videos"]:
        print(f"[SKIP] unsafe delete target: {path}")
        return

    print(f"[DELETE] removing {path}")
    shutil.rmtree(path)


def main(root_path: str):
    root = Path(root_path)
    demos = root / "demos"
    raw_videos = root/ "raw_videos"

    output_dir = Path(".")
    output_dir = root / output_dir
    output_dir.mkdir(exist_ok=True)

    print(f"Processing root: {root.resolve()}")

    # -----------------------
    # 1. mapping -> 1.mp4
    # -----------------------
    copy_if_exists(
        demos / "mapping" / "raw_video.mp4",
        output_dir / "1.mp4",
        "mapping"
    )

    # -----------------------
    # 2. gripper* -> 2.mp4
    # -----------------------
    gripper_files = list(demos.glob("gripper*/raw_video.mp4"))
    if gripper_files:
        copy_if_exists(gripper_files[0], output_dir / "2.mp4", "gripper")
    else:
        print("[WARN] no gripper video found")

    # -----------------------
    # 3. demo_* -> 3.mp4...
    # -----------------------
    demo_files = sorted(demos.glob("demo_*/raw_video.mp4"))

    idx = 3
    for f in demo_files:
        copy_if_exists(f, output_dir / f"{idx}.mp4", f"demo {idx}")
        idx += 1

    print(f"Done. total videos: {idx - 1}")

    # -----------------------
    # 4. CLEANUP (rm -rf equivalent)
    # -----------------------

    # delete demos directory
    safe_remove_dir(demos)
    safe_remove_dir(raw_videos)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/gopher.py <root_path>")
        sys.exit(1)

    main(sys.argv[1])
