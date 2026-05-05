from __future__ import annotations

import csv
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import torch

from model.next_event_model import NextEventPredictor
from scripts.build_train_cache import (
    CACHE_SCHEMA_VERSION,
    CacheSettings,
    ProcessedSample,
    build_train_cache,
)
from scripts.infer_minimal import run_inference, InferSettings
from vocab.delta_tick_vocab import (
    VocabSettings,
    build_delta_tick_vocab_from_train_cache,
    save_vocab,
)


def _make_cache_settings(root: Path) -> CacheSettings:
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
        context_events=2,
        audio_window_size=4,
        expected_n_mels=80,
        num_lanes=7,
    )


def _make_vocab_settings(root: Path) -> VocabSettings:
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


class InferMinimalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_root = Path(__file__).resolve().parents[1] / ".tmp_test_infer_minimal"
        cls._tmp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self) -> None:
        self.root = self._tmp_root / f"case_{uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.cache_dir = self.root / "cache"
        self.splits_dir = self.root / "artifacts" / "splits"
        self.infer_dir = self.root / "artifacts" / "infer"
        self.ckpt_dir = self.root / "artifacts" / "checkpoints"
        self.vocab_path = self.root / "artifacts" / "vocab.json"
        self.val_split_path = self.splits_dir / "val.csv"
        self.checkpoint_path = self.ckpt_dir / "latest.pt"
        self.cache_index_path = self.cache_dir / "cache_index.csv"

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_cache_and_split(self) -> None:
        manifest_rows = [
            {
                "sample_id": "sample-val",
                "osu_path": "./maps/a.osu",
                "audio_path": "./maps/a.wav",
                "set_id": "set-1",
                "difficulty_name": "d",
                "key_count": "7",
                "valid": "true",
            }
        ]
        settings = _make_cache_settings(self.root)

        def fake_process(row: dict, _: CacheSettings) -> ProcessedSample:
            sid = str(row["sample_id"])
            samples = [
                (
                    {
                        "past_events": [
                            {"delta_tick": 12, "lane_mask": 1},
                            {"delta_tick": 24, "lane_mask": 2},
                        ],
                        "grid_stats": np.zeros(10, dtype=np.float32),
                        "audio_window": np.zeros((80, 4), dtype=np.float32),
                    },
                    {"delta_tick": 24, "lane_mask": 3},
                ),
                (
                    {
                        "past_events": [
                            {"delta_tick": 24, "lane_mask": 4},
                            {"delta_tick": 48, "lane_mask": 8},
                        ],
                        "grid_stats": np.ones(10, dtype=np.float32),
                        "audio_window": np.ones((80, 4), dtype=np.float32),
                    },
                    {"delta_tick": 48, "lane_mask": 16},
                ),
            ]
            record = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "sample_id": sid,
                "key_count": 7,
                "osu_path": row["osu_path"],
                "audio_path": row["audio_path"],
                "valid": True,
                "error": "",
                "payload": {"samples": samples},
            }
            return ProcessedSample(
                sample_id=sid,
                key_count=7,
                osu_path=row["osu_path"],
                audio_path=row["audio_path"],
                valid=True,
                error="",
                record=record,
            )

        build_train_cache(settings, manifest_rows=manifest_rows, process_row_fn=fake_process)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.val_split_path.write_text(
            "sample_id,valid\nsample-val,true\n",
            encoding="utf-8",
        )

    def _build_vocab_and_checkpoint(self) -> None:
        vocab_settings = _make_vocab_settings(self.root)
        vocab = build_delta_tick_vocab_from_train_cache(vocab_settings)
        save_vocab(vocab, self.vocab_path, overwrite=True)

        model = NextEventPredictor(delta_vocab_size=vocab.num_tokens, grid_stats_dim=10)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, self.checkpoint_path)

    def _make_settings(self) -> InferSettings:
        return InferSettings(
            project_root=self.root,
            checkpoint_path=self.checkpoint_path,
            input_split_path=self.val_split_path,
            cache_index_path=self.cache_index_path,
            vocab_path=self.vocab_path,
            device="cpu",
            batch_size=2,
            num_workers=0,
            output_dir=self.infer_dir,
            output_filename="pred.csv",
            summary_filename="summary.json",
            top_k=1,
            temperature=1.0,
            sampling=False,
            max_steps=0,
            seed=7,
        )

    def test_inference_runs_and_outputs_are_stable(self) -> None:
        self._write_cache_and_split()
        self._build_vocab_and_checkpoint()
        settings = self._make_settings()

        with patch("scripts.infer_minimal.load_vocab_from_config") as mock_load_vocab:
            from vocab.delta_tick_vocab import load_vocab

            mock_load_vocab.side_effect = lambda force_reload=True: load_vocab(self.vocab_path)
            summary1 = run_inference(settings)
            summary2 = run_inference(settings)

        pred_path = self.infer_dir / "pred.csv"
        sum_path = self.infer_dir / "summary.json"
        self.assertTrue(pred_path.exists())
        self.assertTrue(sum_path.exists())

        rows = list(csv.DictReader(pred_path.open("r", encoding="utf-8")))
        self.assertGreater(len(rows), 0)
        self.assertEqual(summary1["output_row_count"], len(rows))
        self.assertEqual(summary1["output_row_count"], summary2["output_row_count"])
        self.assertEqual(summary1["step_count"], summary2["step_count"])

        text1 = pred_path.read_text(encoding="utf-8")
        text2 = pred_path.read_text(encoding="utf-8")
        self.assertEqual(text1, text2)

        summary_payload = json.loads(sum_path.read_text(encoding="utf-8"))
        self.assertIn("checkpoint_path", summary_payload)
        self.assertIn("vocab_path", summary_payload)
        self.assertEqual(summary_payload["failed_sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
