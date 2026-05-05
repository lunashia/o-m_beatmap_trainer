from __future__ import annotations

import copy
import csv
import shutil
import sys
import types
import unittest
import wave
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    DataLoader = None

if torch is not None:
    from model.next_event_model import NextEventPredictor, compute_next_event_loss
    from scripts.build_train_cache import CacheSettings, build_train_cache
    from scripts.inference_common import (
        collect_cache_samples_for_ids,
        filter_valid_cache_sample_records,
        read_split_sample_ids,
    )
    from scripts.train_minimal import NextEventDataset, _load_shard_records
    from vocab.delta_tick_vocab import VocabSettings, build_delta_tick_vocab_from_train_cache
else:  # pragma: no cover
    CacheSettings = Any  # type: ignore[misc,assignment]
    VocabSettings = Any  # type: ignore[misc,assignment]


def _write_wav(path: Path, *, sample_rate: int = 22050, seconds: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(sample_rate * seconds))
    silence = (b"\x00\x00") * frame_count
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(silence)


def _write_osu(path: Path, *, audio_filename: str) -> None:
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
            "256,192,1000,1,0,0:0:0:0:",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
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


def _write_split(path: Path, sample_ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["sample_id", "valid"])
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow({"sample_id": sample_id, "valid": "true"})


def _make_cache_settings(root: Path) -> CacheSettings:
    cache_dir = root / "cache"
    return CacheSettings(
        project_root=root,
        manifest_path=root / "manifest.csv",
        cache_dir=cache_dir,
        cache_index_path=cache_dir / "cache_index.csv",
        shard_size=8,
        cache_format="npz",
        overwrite=True,
        num_workers=1,
        include_invalid=False,
        skip_failed_samples=True,
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
        context_events=2,
        audio_window_size=4,
        expected_n_mels=80,
        num_lanes=7,
    )


def _make_vocab_settings(root: Path, train_split_path: Path) -> VocabSettings:
    cache_dir = root / "cache"
    return VocabSettings(
        project_root=root,
        cache_dir=cache_dir,
        cache_index_path=cache_dir / "cache_index.csv",
        split_train_index_path=train_split_path,
        vocab_path=root / "artifacts" / "vocab.json",
        fit_from="train",
        field="delta_tick",
        sources=("target", "past"),
        overwrite=True,
        frozen=True,
        fail_on_unknown=False,
        special_tokens={"pad": "<PAD>", "unk": "<UNK>"},
    )


def _validate_cached_record_contract(
    record: Dict[str, Any],
    *,
    context_events: int,
    audio_window_size: int,
    n_mels: int,
) -> None:
    required_record_fields = {
        "schema_version",
        "sample_id",
        "key_count",
        "osu_path",
        "audio_path",
        "valid",
        "error",
        "payload",
    }
    missing_record = sorted(required_record_fields.difference(record.keys()))
    if missing_record:
        raise AssertionError(f"record missing fields: {missing_record}")

    payload = record["payload"]
    if not isinstance(payload, dict):
        raise AssertionError("payload must be a dict.")
    required_payload_fields = {
        "parsed_chart",
        "quantized_notes",
        "events",
        "grid",
        "grid_stats",
        "mel",
        "samples",
    }
    missing_payload = sorted(required_payload_fields.difference(payload.keys()))
    if missing_payload:
        raise AssertionError(f"payload missing fields: {missing_payload}")

    parsed_chart = payload["parsed_chart"]
    quantized_notes = payload["quantized_notes"]
    events = payload["events"]
    grid = payload["grid"]
    grid_stats = payload["grid_stats"]
    mel = payload["mel"]
    samples = payload["samples"]

    if not isinstance(parsed_chart, dict):
        raise AssertionError("parsed_chart must be dict.")
    if not isinstance(quantized_notes, list):
        raise AssertionError("quantized_notes must be list.")
    if not isinstance(events, list):
        raise AssertionError("events must be list.")
    if not isinstance(samples, list):
        raise AssertionError("samples must be list.")

    grid_arr = np.asarray(grid)
    if grid_arr.ndim != 2 or grid_arr.shape[1] != 7:
        raise AssertionError(f"grid must be [T,7], got {grid_arr.shape!r}")

    stats_arr = np.asarray(grid_stats)
    if stats_arr.ndim != 2 or stats_arr.shape[1] != 10:
        raise AssertionError(f"grid_stats must be [N,10], got {stats_arr.shape!r}")
    if len(events) != int(stats_arr.shape[0]):
        raise AssertionError("events and grid_stats length mismatch.")

    mel_arr = np.asarray(mel)
    if mel_arr.ndim != 2 or mel_arr.shape[0] != n_mels:
        raise AssertionError(f"mel must be [n_mels,T], got {mel_arr.shape!r}")

    for event in events:
        for key in ("start_tick", "delta_tick", "lane_mask", "chord_size"):
            if key not in event:
                raise AssertionError(f"event missing field: {key}")
            if not isinstance(event[key], int):
                raise AssertionError(f"event field {key} must be int.")
        if event["delta_tick"] < 0:
            raise AssertionError("delta_tick must be >= 0.")
        if not (0 <= event["lane_mask"] <= 127):
            raise AssertionError("lane_mask out of range [0,127].")
        if not (1 <= event["chord_size"] <= 7):
            raise AssertionError("chord_size out of range [1,7].")

    hit_objects = parsed_chart.get("hit_objects", [])
    for note, hit in zip(quantized_notes, hit_objects):
        start_tick = note.get("start_tick")
        time_ms = hit.get("time_ms")
        if not isinstance(start_tick, int) or not isinstance(time_ms, int):
            raise AssertionError("quantized note start_tick and hit time_ms must be int.")
        if time_ms > 0 and start_tick == time_ms:
            raise AssertionError(
                "semantic mismatch: quantized start_tick equals ms timestamp."
            )

    if not samples:
        raise AssertionError("samples must be non-empty.")
    for sample in samples:
        if not (isinstance(sample, (list, tuple)) and len(sample) >= 2):
            raise AssertionError("sample must be (x, y).")
        x, y = sample[0], sample[1]
        if not isinstance(x, dict) or not isinstance(y, dict):
            raise AssertionError("sample x/y must be dict.")
        for field in ("past_events", "grid_stats", "audio_window"):
            if field not in x:
                raise AssertionError(f"x missing field: {field}")
        for field in ("delta_tick", "lane_mask"):
            if field not in y:
                raise AssertionError(f"y missing field: {field}")
        if not isinstance(y["delta_tick"], int):
            raise AssertionError("y.delta_tick must be int.")
        if not isinstance(y["lane_mask"], int):
            raise AssertionError("y.lane_mask must be int.")
        if y["delta_tick"] <= 0:
            raise AssertionError("y.delta_tick must be > 0.")
        if not (0 <= y["lane_mask"] <= 127):
            raise AssertionError("y.lane_mask out of range [0,127].")

        past_events = x["past_events"]
        if not isinstance(past_events, list) or len(past_events) != context_events:
            raise AssertionError("x.past_events length mismatch.")
        for pe in past_events:
            if not isinstance(pe, dict):
                raise AssertionError("past_events item must be dict.")
            if not isinstance(pe.get("delta_tick"), int):
                raise AssertionError("past_events.delta_tick must be int.")
            if not isinstance(pe.get("lane_mask"), int):
                raise AssertionError("past_events.lane_mask must be int.")

        x_stats = np.asarray(x["grid_stats"])
        if x_stats.ndim != 1 or x_stats.shape[0] != 10:
            raise AssertionError("x.grid_stats must be [10].")

        x_audio = np.asarray(x["audio_window"])
        if x_audio.ndim != 2 or x_audio.shape != (n_mels, audio_window_size):
            raise AssertionError(
                f"x.audio_window must be [{n_mels},{audio_window_size}], got {x_audio.shape!r}"
            )


@unittest.skipIf(torch is None, "torch is required for integration pipeline tests.")
class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_root = Path(__file__).resolve().parents[1] / ".tmp_test_pipeline_integration"
        cls._tmp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self) -> None:
        self.root = self._tmp_root / f"case_{uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _prepare_dummy_project(self) -> tuple[CacheSettings, Path]:
        maps = self.root / "maps"
        _write_wav(maps / "train.wav")
        _write_wav(maps / "val.wav")
        _write_osu(maps / "train.osu", audio_filename="train.wav")
        _write_osu(maps / "val.osu", audio_filename="val.wav")

        _write_manifest(
            self.root / "manifest.csv",
            [
                {
                    "sample_id": "sample-train",
                    "osu_path": "./maps/train.osu",
                    "audio_path": "./maps/train.wav",
                    "set_id": "set-train",
                    "difficulty_name": "dummy",
                    "key_count": "7",
                    "valid": "true",
                },
                {
                    "sample_id": "sample-val",
                    "osu_path": "./maps/val.osu",
                    "audio_path": "./maps/val.wav",
                    "set_id": "set-val",
                    "difficulty_name": "dummy",
                    "key_count": "7",
                    "valid": "true",
                },
            ],
        )
        split_train = self.root / "artifacts" / "splits" / "train.csv"
        _write_split(split_train, ["sample-train"])
        return _make_cache_settings(self.root), split_train

    def test_small_end_to_end_module_connections(self) -> None:
        cache_settings, train_split = self._prepare_dummy_project()

        fake_mel_module = types.ModuleType("audio.mel_extractor")
        fake_mel_module.extract_mel = lambda *args, **kwargs: np.zeros((80, 256), dtype=np.float32)

        with unittest.mock.patch.dict(sys.modules, {"audio.mel_extractor": fake_mel_module}):
            summary = build_train_cache(cache_settings)

        self.assertEqual(summary.cached_rows, 2)
        self.assertGreaterEqual(summary.shards_written, 1)
        self.assertTrue(cache_settings.cache_index_path.exists())

        first_shard = cache_settings.cache_dir / "train_cache_shard_00000.npz"
        self.assertTrue(first_shard.exists())
        records = _load_shard_records(first_shard)
        self.assertGreater(len(records), 0)

        _validate_cached_record_contract(
            records[0],
            context_events=cache_settings.context_events,
            audio_window_size=cache_settings.audio_window_size,
            n_mels=cache_settings.n_mels,
        )

        vocab = build_delta_tick_vocab_from_train_cache(
            _make_vocab_settings(self.root, train_split)
        )
        self.assertGreater(vocab.num_tokens, 2)

        train_ids = read_split_sample_ids(train_split)
        raw_records, missing_count = collect_cache_samples_for_ids(
            cache_settings.cache_index_path,
            self.root,
            train_ids,
        )
        self.assertEqual(missing_count, 0)
        filtered_records, failed_pairs = filter_valid_cache_sample_records(raw_records, vocab)
        self.assertEqual(failed_pairs, 0)
        self.assertGreater(len(filtered_records), 0)

        pairs = [(r.x, r.y) for r in filtered_records]
        dataset = NextEventDataset(pairs, vocab)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        x1, y1 = dataset[0]
        self.assertEqual(tuple(x1["past_delta_id"].shape), (cache_settings.context_events,))
        self.assertEqual(tuple(x1["past_lane_mask_id"].shape), (cache_settings.context_events,))
        self.assertEqual(tuple(x1["grid_stats"].shape), (10,))
        self.assertEqual(tuple(x1["audio_window"].shape), (80, cache_settings.audio_window_size))
        self.assertGreaterEqual(int(y1["delta_tick"].item()), 0)
        self.assertTrue(0 <= int(y1["lane_mask"].item()) <= 127)

        batch_x, batch_y = next(iter(loader))
        model = NextEventPredictor(
            delta_vocab_size=vocab.num_tokens,
            grid_stats_dim=int(batch_x["grid_stats"].shape[-1]),
        )
        logits = model(batch_x)
        self.assertIn("delta_tick_logits", logits)
        self.assertIn("lane_mask_logits", logits)
        self.assertEqual(logits["delta_tick_logits"].shape[0], batch_y["delta_tick"].shape[0])
        self.assertEqual(logits["delta_tick_logits"].shape[1], vocab.num_tokens)
        self.assertEqual(logits["lane_mask_logits"].shape[1], 128)

        losses = compute_next_event_loss(logits, batch_y["delta_tick"], batch_y["lane_mask"])
        self.assertTrue(torch.isfinite(losses["total_loss"]).item())
        self.assertGreater(float(losses["total_loss"].item()), 0.0)

    def test_contract_validator_catches_missing_field_wrong_type_and_tick_ms_mismatch(self) -> None:
        cache_settings, _ = self._prepare_dummy_project()

        fake_mel_module = types.ModuleType("audio.mel_extractor")
        fake_mel_module.extract_mel = lambda *args, **kwargs: np.zeros((80, 256), dtype=np.float32)
        with unittest.mock.patch.dict(sys.modules, {"audio.mel_extractor": fake_mel_module}):
            _ = build_train_cache(cache_settings)

        records = _load_shard_records(cache_settings.cache_dir / "train_cache_shard_00000.npz")
        base = records[0]

        broken_missing = copy.deepcopy(base)
        del broken_missing["payload"]["samples"][0][1]["lane_mask"]
        with self.assertRaises(AssertionError):
            _validate_cached_record_contract(
                broken_missing,
                context_events=cache_settings.context_events,
                audio_window_size=cache_settings.audio_window_size,
                n_mels=cache_settings.n_mels,
            )

        broken_type = copy.deepcopy(base)
        broken_type["payload"]["samples"][0][1]["delta_tick"] = "24"
        with self.assertRaises(AssertionError):
            _validate_cached_record_contract(
                broken_type,
                context_events=cache_settings.context_events,
                audio_window_size=cache_settings.audio_window_size,
                n_mels=cache_settings.n_mels,
            )

        broken_semantic = copy.deepcopy(base)
        for idx, hit in enumerate(broken_semantic["payload"]["parsed_chart"]["hit_objects"]):
            if int(hit.get("time_ms", 0)) > 0:
                broken_semantic["payload"]["quantized_notes"][idx]["start_tick"] = int(hit["time_ms"])
                break
        with self.assertRaises(AssertionError):
            _validate_cached_record_contract(
                broken_semantic,
                context_events=cache_settings.context_events,
                audio_window_size=cache_settings.audio_window_size,
                n_mels=cache_settings.n_mels,
            )


if __name__ == "__main__":
    unittest.main()
