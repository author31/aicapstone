# LeRobot Dataset Visualizer

The Hugging Face Space at <https://huggingface.co/spaces/lerobot/visualize_dataset> is a hosted GUI for inspecting any LeRobot-format dataset published on the Hub. It pairs the recorded multi-camera video with per-frame `observation.state` / `action` plots and an episode picker — useful for spot-checking demonstrations after teleop or datagen, before sinking GPU time into training.

## When to use it

- After running `scripts/teleop.py` or `scripts/datagen/generate.py` with `--use_lerobot_recorder` and pushing the resulting dataset to the Hub.
- To audit episode boundaries, action-trace continuity, and camera framing without standing up a local viewer.
- To share concrete examples with collaborators by URL — the Space accepts a `dataset` query parameter pointing at any public Hub repo.

It is **not** a recorder, decoder, or editor. Read-only inspection of already-pushed Hub datasets only.

## Prerequisite: dataset on the Hub

The Space loads via `repo_id`, so the dataset must already be published. Our recorder writes it locally and uploads when given a `repo_id`:

```bash
uv run python scripts/teleop.py \
    --task=HCIS-CupStacking-SingleArm-v0 \
    --teleop_device=keyboard \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id=<hf_user>/<dataset_name> \
    --lerobot_dataset_fps=30
```

The same flags exist on `scripts/datagen/generate.py`. `LeRobotRecorderManager` (in the leisaac package) handles the Hub push.

If you only have a local dataset, push it manually:

```bash
uv run huggingface-cli login
uv run python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('<hf_user>/<dataset_name>', root='<local_dataset_path>')
ds.push_to_hub()
"
```

Public datasets work without auth in the Space; private datasets require the viewer to be signed into the Hub with read access.

## Opening a dataset in the Space

Two equivalent forms:

1. Visit <https://huggingface.co/spaces/lerobot/visualize_dataset>, paste `<hf_user>/<dataset_name>` into the input, hit load.
2. Direct deep link — append the repo as a query parameter:

   ```
   https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=<hf_user>/<dataset_name>
   ```

   Optional: append `&episode=<N>` to jump to a specific episode.

## What the UI shows

| Panel | Source field | What to look for |
|-------|-------------|------------------|
| Video player | `observation.images.<camera>` features (we use `wrist`, `front`) | Smooth playback, correct framing, no missing frames between episodes |
| State plot | `observation.state` (Franka joint positions) | Continuous trajectories within joint limits; no jumps at episode boundaries |
| Action plot | `action` (joint targets + gripper command) | Aligned with state, gripper open/close transitions match the task |
| Episode picker | `episode_index` | Episode count matches expectation; per-episode length looks sane |
| Frame scrubber | `frame_index` / `timestamp` | Total frames ≈ `fps × episode_seconds` (we record at 30 fps by default) |

The Space reads features from the dataset's `meta/info.json`, so anything the recorder did not declare there will not appear. If a camera is missing from the video panel, check `LeRobotDatasetCfg.features` wiring in `leisaac.enhance.datasets.lerobot_dataset_handler`.

## Common issues

- **"Dataset not found"** — repo id typo, or the dataset is private and the Space session is not authenticated. Open the repo page directly first to confirm visibility.
- **Video panel is blank but state plots load** — videos failed to encode at recording time, or the `meta/videos/` shards never uploaded. Check the local dataset directory for `videos/chunk-*/episode_*.mp4`.
- **Action and state plots have different lengths** — recorder dropped frames; inspect `episode_data_index` in `meta/episodes.jsonl`.
- **Episode boundaries look wrong** — typically a `done` / termination misfire. Cross-reference against the env's `terminations.success` predicate (e.g. `cutlery_arranged`, `toys_in_box`).

## Local alternative

If the dataset is too large to push, or you need offline inspection, use the LeRobot CLI shipped with the `lerobot` package:

```bash
uv run python -m lerobot.scripts.visualize_dataset \
    --repo-id <hf_user>/<dataset_name> \
    --root <local_dataset_path>
```

Same plot/video layout, served from a local rerun.io viewer. Good for iterating on recording configs before committing to a Hub push.

## References

- Hosted visualizer: <https://huggingface.co/spaces/lerobot/visualize_dataset>
- LeRobot dataset format spec: <https://huggingface.co/docs/lerobot>
- Recorder wiring in this repo: `scripts/teleop.py:215-230`, `scripts/datagen/generate.py:162-179`
