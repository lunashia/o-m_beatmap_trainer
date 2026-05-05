"""Regression tests for delta_tick vocab fitting and reuse.

Purpose:
    Verify that the frozen vocab is fitted from train-only cached samples,
    saved and reloaded consistently, and consumed by dataset encoding without
    mutating the token space.

Input / Output:
    Input:
        - synthetic train-cache shards built through `build_train_cache`
        - temporary vocab JSON artifacts written under the test workspace
    Output:
        - assertions over fitted vocab contents and encoded dataset fields

Pipeline fit:
    Upstream input is the cache-building pipeline and the new vocab module.
    Downstream output is a set of regression checks guarding the shared vocab
    contract used by training, validation, and generation.
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from dataset_builder import build_dataset
from scripts.build_train_cache import CACHE_SCHEMA_VERSION, CacheSettings, ProcessedSample, build_train_cache
from vocab.delta_tick_vocab import (
    VocabSettings,
    build_delta_tick_vocab_from_train_cache,
    load_vocab,
    save_vocab,
)


def _make_cache_settings(root: Path) -> CacheSettings:
    """Create an isolated cache configuration for synthetic test data."""
    cache_dir = root / "cache"
    return CacheSettings(
        project_root=root,
        manifest_path=root / "manifest.csv",
        cache_dir=cache_dir,
        cache_index_path=cache_dir / "cache_index.csv",
        shard_size=10,
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
        context_events=1,
        audio_window_size=2,
        expected_n_mels=80,
        num_lanes=7,
    )


def _make_vocab_settings(root: Path) -> VocabSettings:
    """Create an isolated vocab configuration for the test workspace."""
    cache_dir = root / "cache"
    return VocabSettings(
        project_root=root,
        cache_dir=cache_dir,
        cache_index_path=cache_dir / "cache_index.csv",
        split_train_index_path=None,
        vocab_path=root / "artifacts" / "vocab.json",
        fit_from="train",
        field="delta_tick",
        sources=("target", "past"),
        overwrite=True,
        frozen=True,
        fail_on_unknown=False,
        special_tokens={"pad": "<PAD>", "unk": "<UNK>"},
    )


class VocabTests(unittest.TestCase):
    """End-to-end regression tests for the frozen delta_tick vocab."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create the shared temporary root used by all test cases."""
        cls._tmp_root = Path(__file__).resolve().parents[1] / ".tmp_test_vocab"
        cls._tmp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self) -> None:
        """Create a fresh per-test workspace."""
        self.root = self._tmp_root / "case"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Remove the per-test workspace after each run."""
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_cache(self) -> None:
        """Populate a small synthetic cache shard for vocab fitting."""
        settings = _make_cache_settings(self.root)
        manifest_rows = [
            {
                "sample_id": "sample-1",
                "osu_path": "./maps/map.osu",
                "audio_path": "./maps/audio.wav",
                "set_id": "1",
                "difficulty_name": "test",
                "key_count": "7",
                "valid": "true",
            }
        ]

        def fake_process(_: Dict[str, Any], __: CacheSettings) -> ProcessedSample:
            record = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "sample_id": "sample-1",
                "key_count": 7,
                "osu_path": "./maps/map.osu",
                "audio_path": "./maps/audio.wav",
                "valid": True,
                "error": "",
                "payload": {
                    "samples": [
                        (
                            {
                                "past_events": [
                                    {"delta_tick": 0, "lane_mask": 1},
                                    {"delta_tick": 12, "lane_mask": 2},
                                ],
                                "grid_stats": np.zeros(10, dtype=np.float32),
                                "audio_window": np.zeros((80, 2), dtype=np.float32),
                            },
                            {"delta_tick": 24, "lane_mask": 3},
                        ),
                        (
                            {
                                "past_events": [
                                    {"delta_tick": 24, "lane_mask": 4},
                                ],
                                "grid_stats": np.zeros(10, dtype=np.float32),
                                "audio_window": np.zeros((80, 2), dtype=np.float32),
                            },
                            {"delta_tick": 48, "lane_mask": 5},
                        ),
                    ]
                },
            }
            return ProcessedSample(
                sample_id="sample-1",
                key_count=7,
                osu_path="map.osu",
                audio_path="audio.wav",
                valid=True,
                error="",
                record=record,
            )

        build_train_cache(settings, manifest_rows=manifest_rows, process_row_fn=fake_process)

    def test_fit_save_load_and_encode(self) -> None:
        """The fitted vocab should round-trip cleanly and encode values."""
        self._write_cache()
        settings = _make_vocab_settings(self.root)
        vocab = build_delta_tick_vocab_from_train_cache(settings)

        self.assertEqual(vocab.field, "delta_tick")
        self.assertTrue(vocab.frozen)
        self.assertEqual(vocab.sources, ("target", "past"))
        self.assertEqual(vocab.delta_tick_values, (0, 12, 24, 48))
        self.assertEqual(vocab.pad_id, 0)
        self.assertEqual(vocab.unk_id, 1)
        self.assertEqual(vocab.encode_delta_tick(24), 4)
        self.assertEqual(vocab.encode_delta_tick(999), vocab.unk_id)
        self.assertEqual(vocab.decode_delta_tick(4), 24)
        with self.assertRaises(KeyError):
            vocab.encode_delta_tick(999, fail_on_unknown=True)

        save_vocab(vocab, settings.vocab_path, overwrite=True)
        loaded = load_vocab(settings.vocab_path)
        self.assertEqual(loaded.to_dict(), vocab.to_dict())

    def test_build_dataset_encodes_with_vocab(self) -> None:
        """Dataset assembly should emit encoded delta_tick fields when enabled."""
        self._write_cache()
        vocab = build_delta_tick_vocab_from_train_cache(_make_vocab_settings(self.root))
        events = [
            {"start_tick": 0, "delta_tick": 0, "lane_mask": 1, "chord_size": 1},
            {"start_tick": 48, "delta_tick": 12, "lane_mask": 2, "chord_size": 1},
        ]
        grid_stats = [np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32)]
        mel = np.zeros((80, 6), dtype=np.float32)
        tick_to_frame = lambda tick: 3

        samples = build_dataset(
            events,
            grid_stats,
            mel,
            tick_to_frame,
            context_events=1,
            audio_window_size=2,
            expected_n_mels=80,
            delta_tick_vocab=vocab,
        )

        self.assertEqual(len(samples), 1)
        x, y = samples[0]
        self.assertIn("past_delta_tick", x)
        self.assertIn("past_lane_mask", x)
        self.assertEqual(list(x["past_delta_tick"]), [vocab.encode_delta_tick(0)])
        self.assertEqual(list(x["past_lane_mask"]), [1])
        self.assertEqual(y["delta_tick_id"], vocab.encode_delta_tick(12))

    def test_save_rejects_overwrite_false(self) -> None:
        """Saving should fail when overwrite is disabled and the file exists."""
        self._write_cache()
        settings = _make_vocab_settings(self.root)
        vocab = build_delta_tick_vocab_from_train_cache(settings)
        save_vocab(vocab, settings.vocab_path, overwrite=True)
        with self.assertRaises(FileExistsError):
            save_vocab(vocab, settings.vocab_path, overwrite=False)

    def test_load_rejects_unfrozen_vocab(self) -> None:
        """Loading should reject payloads that violate the frozen contract."""
        self._write_cache()
        settings = _make_vocab_settings(self.root)
        vocab = build_delta_tick_vocab_from_train_cache(settings)
        payload = vocab.to_dict()
        payload["frozen"] = False
        settings.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        settings.vocab_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with self.assertRaises(ValueError):
            load_vocab(settings.vocab_path)

    def test_vocab_respects_split_train_sample_ids(self) -> None:
        """Train-only fitting should ignore samples excluded by the split file."""
        settings = _make_cache_settings(self.root)
        manifest_rows = [
            {
                "sample_id": "sample-train",
                "osu_path": "./maps/a.osu",
                "audio_path": "./maps/a.wav",
                "set_id": "1",
                "difficulty_name": "a",
                "key_count": "7",
                "valid": "true",
            },
            {
                "sample_id": "sample-val",
                "osu_path": "./maps/b.osu",
                "audio_path": "./maps/b.wav",
                "set_id": "2",
                "difficulty_name": "b",
                "key_count": "7",
                "valid": "true",
            },
        ]

        def fake_process(row: Dict[str, Any], _: CacheSettings) -> ProcessedSample:
            sid = str(row["sample_id"])
            if sid == "sample-train":
                target_delta = 24
                past_delta = 12
            else:
                target_delta = 777
                past_delta = 888
            record = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "sample_id": sid,
                "key_count": 7,
                "osu_path": row["osu_path"],
                "audio_path": row["audio_path"],
                "valid": True,
                "error": "",
                "payload": {
                    "samples": [
                        (
                            {
                                "past_events": [
                                    {"delta_tick": past_delta, "lane_mask": 1},
                                ],
                                "grid_stats": np.zeros(10, dtype=np.float32),
                                "audio_window": np.zeros((80, 2), dtype=np.float32),
                            },
                            {"delta_tick": target_delta, "lane_mask": 3},
                        ),
                    ]
                },
            }
            return ProcessedSample(
                sample_id=sid,
                key_count=7,
                osu_path="x.osu",
                audio_path="x.wav",
                valid=True,
                error="",
                record=record,
            )

        build_train_cache(settings, manifest_rows=manifest_rows, process_row_fn=fake_process)

        split_train_path = self.root / "artifacts" / "splits" / "train.csv"
        split_train_path.parent.mkdir(parents=True, exist_ok=True)
        split_train_path.write_text(
            "sample_id,valid\nsample-train,true\n",
            encoding="utf-8",
        )

        vocab_settings = _make_vocab_settings(self.root)
        vocab_settings = VocabSettings(
            project_root=vocab_settings.project_root,
            cache_dir=vocab_settings.cache_dir,
            cache_index_path=vocab_settings.cache_index_path,
            split_train_index_path=split_train_path,
            vocab_path=vocab_settings.vocab_path,
            fit_from=vocab_settings.fit_from,
            field=vocab_settings.field,
            sources=vocab_settings.sources,
            overwrite=vocab_settings.overwrite,
            frozen=vocab_settings.frozen,
            fail_on_unknown=vocab_settings.fail_on_unknown,
            special_tokens=vocab_settings.special_tokens,
        )

        vocab = build_delta_tick_vocab_from_train_cache(vocab_settings)
        self.assertIn(12, vocab.delta_tick_values)
        self.assertIn(24, vocab.delta_tick_values)
        self.assertNotIn(777, vocab.delta_tick_values)
        self.assertNotIn(888, vocab.delta_tick_values)


if __name__ == "__main__":
    unittest.main()
