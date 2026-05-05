from __future__ import annotations

"""Artifact logging and snapshot utilities for training runs.

Purpose:
    Provide small, side-effect-focused helpers to resolve artifact paths,
    write JSON/JSONL metrics logs, and snapshot config/vocab files.

Input / Output format:
    Input:
        - Run directory path.
        - Metrics/event dictionaries and source snapshot paths.
    Output:
        - Files under run directory:
          `metrics.json`, `train_log.jsonl`, `config_snapshot.yaml`, `vocab.json`.

Pipeline fit:
    Upstream input:
        - Called by `train.py` during startup and epoch/finish logging.
    Downstream output:
        - Artifacts consumed by operators, analysis scripts, and resume diagnostics.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def now_utc_iso() -> str:
    """Return current UTC timestamp in compact ISO-8601 format.

    Args:
        None.

    Returns:
        UTC timestamp string like `2026-05-05T08:30:00Z`.

    Important notes:
        - Microseconds are stripped for stable, readable logs.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TrainArtifacts:
    """Resolved canonical artifact paths for a run.

    Fields:
        run_dir: Root run directory.
        metrics_path: Path to summary metrics JSON.
        log_jsonl_path: Path to streaming JSONL events log.
        config_snapshot_path: Path to copied train config.
        vocab_snapshot_path: Path to copied vocab JSON.
    """

    run_dir: Path
    metrics_path: Path
    log_jsonl_path: Path
    config_snapshot_path: Path
    vocab_snapshot_path: Path


def resolve_train_artifacts(run_dir: Path) -> TrainArtifacts:
    """Resolve standard artifact file paths under a run directory.

    Args:
        run_dir: Root directory for the current training run.

    Returns:
        `TrainArtifacts` with canonical file locations.
    """
    return TrainArtifacts(
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.json",
        log_jsonl_path=run_dir / "train_log.jsonl",
        config_snapshot_path=run_dir / "config_snapshot.yaml",
        vocab_snapshot_path=run_dir / "vocab.json",
    )


def write_config_snapshot(config_path: Path, snapshot_path: Path) -> None:
    """Copy config file into run artifact directory.

    Args:
        config_path: Source config path.
        snapshot_path: Destination snapshot path.

    Returns:
        None.

    Important notes:
        - Parent directories are created automatically.
    """
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, snapshot_path)


def write_vocab_snapshot(vocab_source_path: Path, vocab_snapshot_path: Path) -> None:
    """Copy source vocab JSON into run artifact directory.

    Args:
        vocab_source_path: Source vocab path.
        vocab_snapshot_path: Destination snapshot path.

    Returns:
        None.

    Important notes:
        - Snapshot is a byte-for-byte file copy.
    """
    vocab_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vocab_source_path, vocab_snapshot_path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append one JSON object line to JSONL log file.

    Args:
        path: Target JSONL path.
        payload: Serializable event/metrics object.

    Returns:
        None.

    Important notes:
        - Writes UTF-8 with one object per line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False, sort_keys=False) + "\n")


def write_metrics(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write run summary metrics JSON.

    Args:
        path: Destination metrics path.
        payload: Serializable summary object.

    Returns:
        None.

    Important notes:
        - Uses temp-file + replace to avoid partial writes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)
