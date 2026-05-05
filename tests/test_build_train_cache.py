from __future__ import annotations

import csv
import shutil
import sys
import types
import unittest
import wave
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from scripts.build_train_cache import (
    CACHE_SCHEMA_VERSION,
    CacheSettings,
    ProcessedSample,
    build_train_cache,
)


def _make_wav(path: Path, *, sample_rate: int = 22050, seconds: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(sample_rate * seconds))
    silence = (b"\x00\x00") * frame_count
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(silence)


def _make_osu(path: Path, *, audio_filename: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "osu file format v14",
            "",
            "[General]",
            f"AudioFilename: {audio_filename}",
            "Mode: 3",
            "",
            "[Difficulty]",
            "CircleSize: 7",
            "",
            "[TimingPoints]",
            "0,500,4,1,0,100,1,0",
            "",
            "[HitObjects]",
            "36,192,0,1,0,0:0:0:0:",
            "109,192,250,1,0,0:0:0:0:",
            "182,192,500,1,0,0:0:0:0:",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "sample_id",
                "osu_path",
                "audio_path",
                "set_id",
                "difficulty_name",
                "key_count",
                "valid",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _make_settings(
    root: Path,
    *,
    manifest_name: str = "manifest.csv",
    shard_size: int = 2,
    include_invalid: bool = False,
    overwrite: bool = True,
    skip_failed_samples: bool = True,
) -> CacheSettings:
    cache_dir = root / "cache"
    return CacheSettings(
        project_root=root,
        manifest_path=root / manifest_name,
        cache_dir=cache_dir,
        cache_index_path=cache_dir / "cache_index.csv",
        shard_size=shard_size,
        cache_format="npz",
        overwrite=overwrite,
        num_workers=1,
        include_invalid=include_invalid,
        skip_failed_samples=skip_failed_samples,
        allowed_key_counts={7},
        sample_rate=22050,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=80,
        fmin=30.0,
        fmax=None,
        clip_seconds=None,
        ticks_per_beat=48,
        context_events=1,
        audio_window_size=2,
        expected_n_mels=80,
        num_lanes=7,
    )


def _read_cache_index(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


class BuildTrainCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._workspace_tmp_root = Path(__file__).resolve().parents[1] / ".tmp_test_build_train_cache"
        cls._workspace_tmp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self) -> None:
        self.root = self._workspace_tmp_root / f"case_{uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_normal_sample_generates_shard_and_cache_index(self) -> None:
        maps_dir = self.root / "maps"
        audio_path = maps_dir / "audio.wav"
        osu_path = maps_dir / "map.osu"
        _make_wav(audio_path)
        _make_osu(osu_path, audio_filename="audio.wav")
        _write_manifest(
            self.root / "manifest.csv",
            [
                {
                    "sample_id": "sample-1",
                    "osu_path": "./maps/map.osu",
                    "audio_path": "./maps/audio.wav",
                    "set_id": "1",
                    "difficulty_name": "test",
                    "key_count": "7",
                    "valid": "true",
                }
            ],
        )
        settings = _make_settings(self.root, shard_size=10, include_invalid=False, overwrite=True)

        fake_mel_module = types.ModuleType("audio.mel_extractor")
        fake_mel_module.extract_mel = lambda *args, **kwargs: np.zeros((80, 256), dtype=np.float32)

        with patch.dict(sys.modules, {"audio.mel_extractor": fake_mel_module}):
            summary = build_train_cache(settings)

        self.assertEqual(summary.manifest_rows, 1)
        self.assertEqual(summary.cached_rows, 1)
        self.assertEqual(summary.shards_written, 1)

        index_rows = _read_cache_index(settings.cache_index_path)
        self.assertEqual(len(index_rows), 1)
        self.assertEqual(index_rows[0]["sample_id"], "sample-1")
        self.assertEqual(index_rows[0]["valid"], "true")
        self.assertEqual(index_rows[0]["error"], "")
        self.assertTrue(index_rows[0]["shard_path"])

        shard = settings.cache_dir / "train_cache_shard_00000.npz"
        self.assertTrue(shard.exists())

    def test_shard_size_splits_cache_files(self) -> None:
        rows: List[Dict[str, Any]] = []
        for i in range(5):
            rows.append(
                {
                    "sample_id": f"s{i}",
                    "osu_path": f"./maps/s{i}.osu",
                    "audio_path": f"./maps/s{i}.wav",
                    "set_id": "1",
                    "difficulty_name": "x",
                    "key_count": "7",
                    "valid": "true",
                }
            )
        _write_manifest(self.root / "manifest.csv", rows)
        settings = _make_settings(self.root, shard_size=2, overwrite=True)

        def fake_process(row: Dict[str, Any], _: CacheSettings) -> ProcessedSample:
            sid = str(row["sample_id"])
            return ProcessedSample(
                sample_id=sid,
                key_count=7,
                osu_path=str(self.root / "maps" / f"{sid}.osu"),
                audio_path=str(self.root / "maps" / f"{sid}.wav"),
                valid=True,
                error="",
                record={
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "sample_id": sid,
                    "valid": True,
                    "error": "",
                    "payload": {"x": sid},
                },
            )

        summary = build_train_cache(settings, process_row_fn=fake_process)
        self.assertEqual(summary.cached_rows, 5)
        self.assertEqual(summary.shards_written, 3)

        self.assertTrue((settings.cache_dir / "train_cache_shard_00000.npz").exists())
        self.assertTrue((settings.cache_dir / "train_cache_shard_00001.npz").exists())
        self.assertTrue((settings.cache_dir / "train_cache_shard_00002.npz").exists())

        index_rows = _read_cache_index(settings.cache_index_path)
        self.assertEqual(len(index_rows), 5)
        self.assertEqual([r["shard_index"] for r in index_rows], ["0", "1", "0", "1", "0"])

    def test_invalid_sample_skipped_when_include_invalid_false(self) -> None:
        _write_manifest(
            self.root / "manifest.csv",
            [
                {
                    "sample_id": "bad-1",
                    "osu_path": "./maps/missing.osu",
                    "audio_path": "./maps/missing.wav",
                    "set_id": "1",
                    "difficulty_name": "x",
                    "key_count": "7",
                    "valid": "false",
                }
            ],
        )
        settings = _make_settings(self.root, include_invalid=False, overwrite=True)
        summary = build_train_cache(settings)

        self.assertEqual(summary.cached_rows, 0)
        self.assertEqual(summary.skipped_rows, 1)
        self.assertEqual(summary.failed_rows, 0)
        self.assertEqual(summary.shards_written, 0)

        index_rows = _read_cache_index(settings.cache_index_path)
        self.assertEqual(len(index_rows), 1)
        self.assertEqual(index_rows[0]["valid"], "false")
        self.assertIn("manifest valid=false", index_rows[0]["error"])
        self.assertEqual(index_rows[0]["shard_path"], "")

    def test_single_sample_failure_records_error_and_does_not_abort(self) -> None:
        _write_manifest(
            self.root / "manifest.csv",
            [
                {
                    "sample_id": "bad-1",
                    "osu_path": "./maps/bad.osu",
                    "audio_path": "./maps/bad.wav",
                    "set_id": "1",
                    "difficulty_name": "x",
                    "key_count": "7",
                    "valid": "true",
                },
                {
                    "sample_id": "ok-1",
                    "osu_path": "./maps/ok.osu",
                    "audio_path": "./maps/ok.wav",
                    "set_id": "1",
                    "difficulty_name": "x",
                    "key_count": "7",
                    "valid": "true",
                },
            ],
        )
        settings = _make_settings(self.root, overwrite=True, skip_failed_samples=True)

        def fake_process(row: Dict[str, Any], _: CacheSettings) -> ProcessedSample:
            sid = str(row["sample_id"])
            if sid == "bad-1":
                return ProcessedSample(
                    sample_id=sid,
                    key_count=7,
                    osu_path="bad.osu",
                    audio_path="bad.wav",
                    valid=False,
                    error="RuntimeError: synthetic failure",
                    record=None,
                    skipped=False,
                )
            return ProcessedSample(
                sample_id=sid,
                key_count=7,
                osu_path="ok.osu",
                audio_path="ok.wav",
                valid=True,
                error="",
                record={
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "sample_id": sid,
                    "valid": True,
                    "error": "",
                    "payload": {"ok": True},
                },
                skipped=False,
            )

        summary = build_train_cache(settings, process_row_fn=fake_process)
        self.assertEqual(summary.failed_rows, 1)
        self.assertEqual(summary.cached_rows, 1)
        self.assertEqual(summary.shards_written, 1)

        index_rows = _read_cache_index(settings.cache_index_path)
        self.assertEqual(len(index_rows), 2)
        self.assertEqual(index_rows[0]["sample_id"], "bad-1")
        self.assertIn("synthetic failure", index_rows[0]["error"])
        self.assertEqual(index_rows[0]["shard_path"], "")
        self.assertEqual(index_rows[1]["sample_id"], "ok-1")
        self.assertEqual(index_rows[1]["valid"], "true")

    def test_overwrite_false_raises_when_cache_exists(self) -> None:
        _write_manifest(
            self.root / "manifest.csv",
            [
                {
                    "sample_id": "s1",
                    "osu_path": "./maps/s1.osu",
                    "audio_path": "./maps/s1.wav",
                    "set_id": "1",
                    "difficulty_name": "x",
                    "key_count": "7",
                    "valid": "true",
                }
            ],
        )
        settings_first = _make_settings(self.root, overwrite=True)
        settings_second = _make_settings(self.root, overwrite=False)

        def fake_process(row: Dict[str, Any], _: CacheSettings) -> ProcessedSample:
            sid = str(row["sample_id"])
            return ProcessedSample(
                sample_id=sid,
                key_count=7,
                osu_path="s1.osu",
                audio_path="s1.wav",
                valid=True,
                error="",
                record={
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "sample_id": sid,
                    "valid": True,
                    "error": "",
                    "payload": {"ok": True},
                },
            )

        first_summary = build_train_cache(settings_first, process_row_fn=fake_process)
        self.assertEqual(first_summary.cached_rows, 1)

        with self.assertRaises(FileExistsError):
            build_train_cache(settings_second, process_row_fn=fake_process)

    def test_overwrite_true_cleans_and_rebuilds(self) -> None:
        _write_manifest(
            self.root / "manifest.csv",
            [
                {
                    "sample_id": "s1",
                    "osu_path": "./maps/s1.osu",
                    "audio_path": "./maps/s1.wav",
                    "set_id": "1",
                    "difficulty_name": "x",
                    "key_count": "7",
                    "valid": "true",
                }
            ],
        )
        settings = _make_settings(self.root, overwrite=True)

        def fake_process(_: Dict[str, Any], __: CacheSettings) -> ProcessedSample:
            return ProcessedSample(
                sample_id="s1",
                key_count=7,
                osu_path="s1.osu",
                audio_path="s1.wav",
                valid=True,
                error="",
                record={
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "sample_id": "s1",
                    "valid": True,
                    "error": "",
                    "payload": {"run": 1},
                },
            )

        first = build_train_cache(settings, process_row_fn=fake_process)
        self.assertEqual(first.cached_rows, 1)

        stale_file = settings.cache_dir / "train_cache_shard_99999.npz"
        stale_file.write_bytes(b"stale")
        self.assertTrue(stale_file.exists())

        second = build_train_cache(settings, process_row_fn=fake_process)
        self.assertEqual(second.cached_rows, 1)
        self.assertFalse(stale_file.exists())


if __name__ == "__main__":
    unittest.main()
