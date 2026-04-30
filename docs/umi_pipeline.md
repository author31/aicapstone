# UMI Pipeline — Data Collection & Processing Guide

This guide walks through the end-to-end UMI data workflow: how to record a session in a way that maximizes processing success, how to verify recordings before running the heavy pipeline, and how to run the full reconstruction.

The pipeline is fragile at two points:

- `02_create_map` — depends entirely on the quality of the **mapping video**.
- `05_run_calibrations` — depends on the SLAM map quality, the **gripper calibration video**, and on tag #13 being clearly visible during mapping.

Failures at either stage cascade silently into stages 06/07/08, where errors look unrelated to the real cause. The verification pipeline introduced below catches both classes of failure before you spend hours on the full run.

---

## 1. Record a session

Each session needs three kinds of footage, recorded with the same GoPro camera in one continuous folder:

| Footage | Purpose | Used by |
|---|---|---|
| **mapping video** | Build the ORB-SLAM3 map of the workspace and locate ArUco tag #13 in the SLAM frame | `02_create_map`, `05_run_calibrations` |
| **gripper calibration video(s)** | Measure each gripper's min/max finger separation (one video per physical gripper) | `05_run_calibrations` |
| **demo videos** | The actual demonstrations to learn from | `03_batch_slam`, `07_frame_to_pose`, `08_generate_replay_buffer` |

### Recording tips for higher success rate

**Mapping video** (drives `02_create_map`)
- Move slowly and smoothly. Fast motion → motion blur → lost SLAM tracking.
- Sweep the workspace from multiple angles. ORB-SLAM3 builds the map from parallax — pure rotation without translation produces a degenerate map.
- Keep ArUco tag #13 visible for a substantial portion of the video, ideally near the image center and roughly 0.3–4.0 m from the camera. The verification stage filters tag detections outside that range and fails if too few survive.
- Avoid blank walls, glossy reflections, and uniform textures. ORB-SLAM3 needs textured regions to extract features.
- Lock GoPro stabilization off (or to "standard"), avoid HyperSmooth — heavy stabilization warps the optical flow that SLAM relies on.

**Gripper calibration video** (drives `05_run_calibrations` gripper range step)
- Hold the gripper still while opening and closing it through its full travel range.
- Keep both finger tags fully visible. The pipeline picks the gripper with the highest detection rate and refuses gripper IDs whose tags appear in fewer than 10% of frames.
- One video per physical gripper, in its own `gripper_calibration_*` directory.

**Demo videos**
- Keep the recording continuous; do not trim or cut.
- Tag #13 does not need to be visible during demos (it's only needed during mapping), but the gripper finger tags must be in view.

---

## 2. Place recorded videos

Place the videos under any directory you like, as long as the layout matches the structure below. We recommend `videos/raw_videos/` at the repo root or under your dataset directory:

```
<NAME_OF_YOUR_DIR>/
├── raw_videos/
│   ├── .gitkeep
│   ├── video1.mp4
│   └── ...
└── ...
```

`00_process_video` (the first stage in every pipeline) walks `raw_videos/` (pattern: `*.MP4` / `*.mp4`) and reorganizes the videos into the expected `demos/mapping/`, `demos/demo_*/`, and `demos/gripper_calibration_*/` layout based on filenames and metadata. You don't need to create those subdirectories yourself.

Pass the chosen directory as `--session-dir` to the pipeline runner, e.g. `--session-dir datasets/1120_team_LMVC/`.

---

## 3. Run the verification pipeline

Before kicking off the full reconstruction, run the verification pipeline. It executes stages 00 → 05 plus a new `05b_verify_calibration` quality gate, and stops there:

```
uv run umi run-slam-pipeline \
    umi_pipeline_configs/gopro13_fisheye_2-7k_verify_pipeline_config.yaml \
    --session-dir <NAME_OF_YOUR_DIR>/
```

The `env -u VIRTUAL_ENV -u PYTHONPATH` prefix shields the `uv` environment from polluted shell variables (lerobot venv pointer, ROS Humble Python paths). Drop it once your shell init no longer sets those.

### 3.1 Stage `02_create_map` — interactive SLAM viewer

The verify pipeline runs `02_create_map` with `enable_gui: true`, which forwards `--enable_gui` to the `chicheng/orb_slam3:latest` Docker container. A Pangolin viewer window pops up while ORB-SLAM3 processes the mapping video. You can watch:

- Green dots = currently tracked map points.
- Red dots = newly added points.
- Blue boxes = keyframes.
- Camera frustum = current pose estimate.

What to look for:

| What you see | Meaning | Action |
|---|---|---|
| Steady stream of new keyframes, dense point cloud | Healthy mapping | Let it finish |
| Frequent "TRACK LOST" messages, viewer freezes | Camera moved too fast or hit a textureless region | Stop the run, re-record more slowly |
| Sparse points, viewer mostly empty | Workspace lacks texture, or lighting is too dim | Re-record with more textured surroundings |
| Tag #13 never appears in the camera view | Calibration step will fail | Re-record ensuring tag is visible |

If the recording is bad, hit `Ctrl+C` and re-record before wasting time on the heavy stages. If it looks healthy, the pipeline continues automatically.

### 3.2 Stage `05b_verify_calibration` — quality gate

After stage `05_run_calibrations` produces `tx_slam_tag.json` and one `gripper_range.json` per gripper, the new `CalibrationVerificationService` runs a battery of checks against those outputs and against the upstream tag-detection pickles. It fails fast with a structured error listing every check that didn't pass.

What it checks:

- **`tx_slam_tag` matrix sanity** — file exists, parses to 4×4, all entries finite, bottom row is `[0,0,0,1]`, rotation block is orthogonal with `det(R) ≈ 1`, translation magnitude is reasonable, condition number is bounded (so `np.linalg.inv` in stage 06 won't blow up).
- **`tx_slam_tag` statistical quality** — re-runs the distance and image-center filtering from `calibration.py` against the same inputs. Verifies that enough detections survived (`min_valid_detections`) and that tag #13 was actually visible (`min_tag_visibility_ratio`).
- **Per-gripper `gripper_range.json` sanity** — file present, schema complete, widths finite and within `[min_gripper_width_m, max_gripper_width_m]`, `min_width < max_width` with a meaningful spread, and tag-id layout matches the calibration assumption (`right == left + 1`, `left == gripper_id * 6`).
- **Downstream dry-run** — actually inverts `tx_slam_tag` (the operation that `dataset_planning.py:56` will perform) and constructs the gripper calibration interpolator (the call from `dataset_planning.py:60-74`), so any failure mode that would otherwise surface in stage 06 is caught here.

When the verifier fails, a typical log line looks like:

```
[FAIL] stats.valid_detections: value=17 threshold=30 :: 17 detections survived filtering (skipped distance=1, center=8)
```

If you trust your recording but the count threshold is too strict, lower `min_valid_detections` in the yaml:

```yaml
05b_verify_calibration:
  config:
    min_valid_detections: 10        # relax count gate
```

---

## 4. Run the full reconstruction pipeline

Once the verification pipeline exits 0, run the full reconstruct pipeline:

```
uv run umi run-slam-pipeline \
    umi_pipeline_configs/gopro13_fisheye_2-7k_reconstruct_pipeline_config.yaml \
    --session-dir <NAME_OF_YOUR_DIR>/
```

This executes the same stages 00–05 plus `06_generate_dataset_plan`, `07_frame_to_pose`, `08_generate_replay_buffer`, ending with a `dataset.zarr.zip` ready for training. Because stages 00–05 already ran during verification, the existing artifacts on disk will be reused (the SLAM mapping stage skips when `map_atlas.osa` is already present unless `force: true` is set).

If anything fails at this point, it is downstream of calibration and unrelated to recording quality — check the stage logs directly.
