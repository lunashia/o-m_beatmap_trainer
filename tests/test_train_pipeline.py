from __future__ import annotations

import csv
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    torch = None

import config_loader
from scripts.build_train_cache import CACHE_SCHEMA_VERSION, CacheSettings, ProcessedSample, build_train_cache
from vocab.delta_tick_vocab import fit_delta_tick_vocab, load_vocab, save_vocab

if torch is not None:
    import train
else:  # pragma: no cover
    train = None


def _write_split(path: Path, sample_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["sample_id", "valid"])
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow({"sample_id": sample_id, "valid": "true"})


def _build_cache(root: Path) -> Path:
    cache_dir = root / "cache"
    settings = CacheSettings(
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
    manifest_rows = [
        {
            "sample_id": "sample-train",
            "osu_path": "./maps/train.osu",
            "audio_path": "./maps/train.wav",
            "set_id": "set-train",
            "difficulty_name": "d",
            "key_count": "7",
            "valid": "true",
        },
        {
            "sample_id": "sample-val",
            "osu_path": "./maps/val.osu",
            "audio_path": "./maps/val.wav",
            "set_id": "set-val",
            "difficulty_name": "d",
            "key_count": "7",
            "valid": "true",
        },
    ]

    def fake_process(row: dict, _: CacheSettings) -> ProcessedSample:
        sid = str(row["sample_id"])
        base = 1 if sid == "sample-train" else 2
        samples = [
            (
                {
                    "past_events": [
                        {"delta_tick": 12 * base, "lane_mask": 1},
                        {"delta_tick": 24 * base, "lane_mask": 2},
                    ],
                    "grid_stats": np.zeros(10, dtype=np.float32) + base,
                    "audio_window": np.zeros((80, 4), dtype=np.float32) + base,
                },
                {"delta_tick": 24 * base, "lane_mask": 3},
            ),
            (
                {
                    "past_events": [
                        {"delta_tick": 24 * base, "lane_mask": 4},
                        {"delta_tick": 48 * base, "lane_mask": 8},
                    ],
                    "grid_stats": np.ones(10, dtype=np.float32) * base,
                    "audio_window": np.ones((80, 4), dtype=np.float32) * base,
                },
                {"delta_tick": 48 * base, "lane_mask": 16},
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
    return settings.cache_index_path


def _write_vocab(path: Path) -> None:
    vocab = fit_delta_tick_vocab([12, 24, 48, 96], frozen=True, fail_on_unknown=False)
    save_vocab(vocab, path, overwrite=True)


def _write_config(
    *,
    path: Path,
    cache_index_path: Path,
    train_split_path: Path,
    val_split_path: Path,
    vocab_path: Path,
    run_dir: Path,
    max_epochs: int,
    resume_enabled: bool,
) -> None:
    text = f"""
data:
  output_dir: "{(run_dir / 'output').as_posix()}"

vocab:
  path: "{vocab_path.as_posix()}"
  frozen: true
  field: "delta_tick"
  fit_from: "train"
  sources: ["target","past"]
  fail_on_unknown: false
  special_tokens:
    pad: "<PAD>"
    unk: "<UNK>"

cache:
  cache_index_path: "{cache_index_path.as_posix()}"
  cache_dir: "{cache_index_path.parent.as_posix()}"

split:
  train_path: "{train_split_path.as_posix()}"
  val_path: "{val_split_path.as_posix()}"

model:
  event_encoder_type: "gru"
  delta_embed_dim: 8
  lane_embed_dim: 8
  event_hidden_dim: 16
  transformer_heads: 2
  transformer_layers: 1
  dropout: 0.1
  audio_hidden_dim: 16
  audio_out_dim: 16
  stats_hidden_dim: 8
  stats_out_dim: 8
  fusion_hidden_dim: 16
  in_mels: 80

train:
  seed: 7
  batch_size: 2
  num_workers: 0
  device: "cpu"
  learning_rate: 0.0005
  weight_decay: 0.0
  max_epochs: {max_epochs}
  max_steps: 0
  grad_accum_steps: 1
  grad_clip_norm: 1.0
  log_every_steps: 1
  amp:
    enabled: false
    dtype: "fp16"
  scheduler:
    name: "none"
  resume:
    enabled: {"true" if resume_enabled else "false"}
    path: ""
    strict: true
  output:
    run_dir: "{run_dir.as_posix()}"
  dataloader:
    pin_memory: false
    persistent_workers: false
""".strip()
    path.write_text(text + "\n", encoding="utf-8")


@unittest.skipIf(torch is None, "torch is required for training pipeline tests.")
class TrainPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_root = Path(__file__).resolve().parents[1] / ".tmp_test_train_pipeline"
        cls._tmp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self) -> None:
        self.root = self._tmp_root / f"case_{uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_train_pipeline_outputs_and_resume(self) -> None:
        cache_index_path = _build_cache(self.root)
        splits_dir = self.root / "splits"
        train_split_path = splits_dir / "train.csv"
        val_split_path = splits_dir / "val.csv"
        _write_split(train_split_path, ["sample-train"])
        _write_split(val_split_path, ["sample-val"])

        vocab_path = self.root / "artifacts" / "vocab.json"
        _write_vocab(vocab_path)

        run_dir = self.root / "artifacts" / "train_run"
        config_path = self.root / "train_test.yaml"
        _write_config(
            path=config_path,
            cache_index_path=cache_index_path,
            train_split_path=train_split_path,
            val_split_path=val_split_path,
            vocab_path=vocab_path,
            run_dir=run_dir,
            max_epochs=1,
            resume_enabled=False,
        )

        with patch.object(config_loader, "CONFIG_PATH", config_path), patch.object(train, "CONFIG_PATH", config_path):
            summary_first = train.run_training()

        best_path = run_dir / "checkpoints" / "best.pt"
        last_path = run_dir / "checkpoints" / "last.pt"
        metrics_path = run_dir / "metrics.json"
        log_path = run_dir / "train_log.jsonl"
        config_snapshot = run_dir / "config_snapshot.yaml"
        vocab_snapshot = run_dir / "vocab.json"

        self.assertTrue(best_path.exists())
        self.assertTrue(last_path.exists())
        self.assertTrue(metrics_path.exists())
        self.assertTrue(log_path.exists())
        self.assertTrue(config_snapshot.exists())
        self.assertTrue(vocab_snapshot.exists())
        self.assertTrue((run_dir / "checkpoints").exists())
        self.assertEqual(summary_first["epochs_ran"], 1)
        self.assertEqual(summary_first["resumed_from"], "")

        source_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        copied_vocab = json.loads(vocab_snapshot.read_text(encoding="utf-8"))
        self.assertEqual(source_vocab, copied_vocab)
        loaded_source = load_vocab(vocab_path)
        loaded_snapshot = load_vocab(vocab_snapshot)
        self.assertEqual(loaded_source.token_to_id, loaded_snapshot.token_to_id)

        logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        epoch_logs = [item for item in logs if item.get("event") == "epoch_end"]
        self.assertEqual(len(epoch_logs), 1)
        self.assertIn("val_loss", epoch_logs[0])

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertIn("best_checkpoint_path", metrics)
        self.assertIn("last_checkpoint_path", metrics)
        self.assertGreaterEqual(metrics["global_step"], 1)

        # resume for another epoch
        _write_config(
            path=config_path,
            cache_index_path=cache_index_path,
            train_split_path=train_split_path,
            val_split_path=val_split_path,
            vocab_path=vocab_path,
            run_dir=run_dir,
            max_epochs=2,
            resume_enabled=True,
        )
        with patch.object(config_loader, "CONFIG_PATH", config_path), patch.object(train, "CONFIG_PATH", config_path):
            summary_second = train.run_training()

        self.assertTrue(summary_second["resumed_from"])
        self.assertGreater(summary_second["global_step"], summary_first["global_step"])
        logs_after_resume = [
            json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        epoch_logs_after = [item for item in logs_after_resume if item.get("event") == "epoch_end"]
        self.assertGreaterEqual(len(epoch_logs_after), 2)


if __name__ == "__main__":
    unittest.main()
