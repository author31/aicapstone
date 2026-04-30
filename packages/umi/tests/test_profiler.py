import csv
import tempfile
import time
from pathlib import Path

from umi.profiler import STAGE_COUNT_EXTRACTORS, PipelineProfiler


def test_extractors_handle_each_stage_shape():
    cases = {
        "00_process_video":         ({"organized_demos": 7}, 7),
        "01_extract_gopro_imu":     ({"extracted": [{"a": 1}, {"a": 2}, {"a": 3}], "failed": []}, 3),
        "02_create_map":            ({"map_path": "/tmp/x.osa", "trajectory_csv": "/tmp/t.csv"}, 1),
        "03_batch_slam":            ({"processed_videos": [], "total_processed": 5}, 5),
        "04_detect_aruco":          ({"total_videos_found": 8, "videos_processed": 6, "videos_skipped": 2}, 6),
        "05_run_calibrations":      ({"slam_tag_calibration": {}, "gripper_range_calibration": {}}, None),
        "05b_verify_calibration":   ({"checks": [], "passed": 4, "failed": 1}, 5),
        "06_generate_dataset_plan": ({"total_episodes": 12, "total_frames": 100}, 12),
        "07_generate_replay_buffer":({"num_videos": 9, "num_episodes": 12}, 9),
    }
    for stage_name, (result, expected) in cases.items():
        label, fn = STAGE_COUNT_EXTRACTORS[stage_name]
        assert isinstance(label, str)
        assert fn(result) == expected, f"{stage_name}: expected {expected}"


def test_extractors_robust_to_garbage():
    for stage_name, (_, fn) in STAGE_COUNT_EXTRACTORS.items():
        assert fn(None) is None
        assert fn("not a dict") is None
        assert fn({}) is None or stage_name == "05_run_calibrations"


def test_profiler_writes_csv_on_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.csv"
        profiler = PipelineProfiler(out_path)

        now = time.time()
        profiler.record_stage(
            stage_index=1,
            stage_name="00_process_video",
            service_class="VideoOrganizationService",
            result={"organized_demos": 4},
            start_time=now,
            duration_sec=1.234,
            status="success",
        )
        profiler.record_stage(
            stage_index=2,
            stage_name="01_extract_gopro_imu",
            service_class="IMUExtractionService",
            result=None,
            start_time=now + 1.5,
            duration_sec=0.5,
            status="failed",
            error="RuntimeError: boom",
        )
        profiler.finalize()

        with open(out_path, newline="") as f:
            rows = list(csv.reader(f))

        assert rows[0] == [
            "stage_index", "stage_name", "service_class", "count_label",
            "videos_processed", "start_time_iso", "duration_sec", "status", "error",
        ]
        assert len(rows) == 3

        ok_row = rows[1]
        assert ok_row[1] == "00_process_video"
        assert ok_row[3] == "organized_demos"
        assert ok_row[4] == "4"
        assert ok_row[7] == "success"
        assert ok_row[8] == ""

        fail_row = rows[2]
        assert fail_row[1] == "01_extract_gopro_imu"
        assert fail_row[7] == "failed"
        assert "RuntimeError: boom" in fail_row[8]


def test_profiler_finalize_idempotent():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.csv"
        profiler = PipelineProfiler(out_path)
        profiler.record_stage(
            stage_index=1,
            stage_name="unknown_stage",
            service_class="Foo",
            result={},
            start_time=time.time(),
            duration_sec=0.1,
            status="success",
        )
        profiler.finalize()
        profiler.finalize()
        with open(out_path, newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2
        assert rows[1][3] == "n/a"
        assert rows[1][4] == ""
