"""Pipeline profiler: per-stage runtime + video count, written to CSV."""
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger


def _get(r: Any, k: str) -> int | None:
    if isinstance(r, dict):
        v = r.get(k)
        if isinstance(v, int):
            return v
    return None


def _verify_count(r: Any) -> int | None:
    if not isinstance(r, dict):
        return None
    passed = r.get("passed")
    failed = r.get("failed")
    if isinstance(passed, int) and isinstance(failed, int):
        return passed + failed
    return None


def _imu_count(r: Any) -> int | None:
    if isinstance(r, dict) and isinstance(r.get("extracted"), list):
        return len(r["extracted"])
    return None


def _create_map_count(r: Any) -> int | None:
    if isinstance(r, dict) and r.get("map_path"):
        return 1
    return None


STAGE_COUNT_EXTRACTORS: dict[str, tuple[str, Callable[[Any], int | None]]] = {
    "00_process_video":         ("organized_demos",  lambda r: _get(r, "organized_demos")),
    "01_extract_gopro_imu":     ("extracted",        _imu_count),
    "02_create_map":            ("mapping_videos",   _create_map_count),
    "03_batch_slam":            ("total_processed",  lambda r: _get(r, "total_processed")),
    "04_detect_aruco":          ("videos_processed", lambda r: _get(r, "videos_processed")),
    "05_run_calibrations":      ("n/a",              lambda r: None),
    "05b_verify_calibration":   ("checks",           _verify_count),
    "06_generate_dataset_plan": ("total_episodes",   lambda r: _get(r, "total_episodes")),
    "07_generate_replay_buffer":("num_videos",       lambda r: _get(r, "num_videos")),
}


CSV_HEADER = [
    "stage_index",
    "stage_name",
    "service_class",
    "count_label",
    "videos_processed",
    "start_time_iso",
    "duration_sec",
    "status",
    "error",
]


class PipelineProfiler:
    """Collects per-stage timing + video counts, streams CSV rows to disk."""

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
        self._file = None
        self._writer: csv.writer | None = None
        self._closed = False

    def _ensure_open(self) -> None:
        if self._writer is not None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_HEADER)
        self._file.flush()

    def record_stage(
        self,
        stage_index: int,
        stage_name: str,
        service_class: str,
        result: Any | None,
        start_time: float,
        duration_sec: float,
        status: str,
        error: str = "",
    ) -> None:
        if self._closed:
            return
        extractor = STAGE_COUNT_EXTRACTORS.get(stage_name)
        if extractor is None:
            count_label, count = "n/a", None
        else:
            count_label, fn = extractor
            try:
                count = fn(result)
            except Exception:
                count = None
        count_cell = "" if count is None else int(count)
        start_iso = datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat()
        err_str = error[:500] if error else ""

        self._ensure_open()
        self._writer.writerow([
            stage_index,
            stage_name,
            service_class,
            count_label,
            count_cell,
            start_iso,
            f"{duration_sec:.4f}",
            status,
            err_str,
        ])
        self._file.flush()

    def finalize(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._file is not None:
            self._file.close()
            logger.info(f"Wrote pipeline profile to {self.output_path}")
