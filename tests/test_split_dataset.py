from __future__ import annotations

import csv
import json
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from scripts.split_dataset import SplitSettings, run_split


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


class SplitDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_root = Path(__file__).resolve().parents[1] / ".tmp_test_split_dataset"
        cls._tmp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self) -> None:
        self.root = self._tmp_root / f"case_{uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _make_settings(
        self,
        *,
        input_name: str = "manifest.csv",
        overwrite: bool = True,
        valid_only: bool = True,
        group_by: str = "set_id",
        fallback_group_by: str = "audio_path",
    ) -> SplitSettings:
        return SplitSettings(
            project_root=self.root,
            input_index_path=self.root / input_name,
            output_dir=self.root / "artifacts" / "splits",
            train_path=self.root / "artifacts" / "splits" / "train.csv",
            val_path=self.root / "artifacts" / "splits" / "val.csv",
            summary_path=self.root / "artifacts" / "splits" / "summary.json",
            val_ratio=0.4,
            seed=42,
            group_by=group_by,
            fallback_group_by=fallback_group_by,
            valid_only=valid_only,
            shuffle=True,
            overwrite=overwrite,
        )

    def test_group_split_no_overlap_and_valid_only(self) -> None:
        rows = [
            {"sample_id": "a1", "set_id": "A", "audio_path": "./songs/a.mp3", "valid": "true"},
            {"sample_id": "a2", "set_id": "A", "audio_path": "./songs/a.mp3", "valid": "true"},
            {"sample_id": "b1", "set_id": "B", "audio_path": "./songs/b.mp3", "valid": "true"},
            {"sample_id": "c1", "set_id": "C", "audio_path": "./songs/c.mp3", "valid": "false"},
        ]
        _write_csv(self.root / "manifest.csv", rows, ["sample_id", "set_id", "audio_path", "valid"])
        settings = self._make_settings(valid_only=True)

        summary = run_split(settings)
        train_rows = _read_csv(settings.train_path)
        val_rows = _read_csv(settings.val_path)
        train_sets = {r["set_id"] for r in train_rows}
        val_sets = {r["set_id"] for r in val_rows}

        self.assertEqual(summary["total_rows_read"], 4)
        self.assertEqual(summary["total_rows_split"], 3)
        self.assertTrue(train_sets.isdisjoint(val_sets))
        self.assertNotIn("C", train_sets | val_sets)

    def test_fallback_group_used_when_primary_missing(self) -> None:
        rows = [
            {"sample_id": "x1", "set_id": "", "audio_path": "./songs/shared.mp3", "valid": "true"},
            {"sample_id": "x2", "set_id": "", "audio_path": "./songs/shared.mp3", "valid": "true"},
            {"sample_id": "y1", "set_id": "", "audio_path": "./songs/other.mp3", "valid": "true"},
        ]
        _write_csv(self.root / "manifest.csv", rows, ["sample_id", "set_id", "audio_path", "valid"])
        settings = self._make_settings(valid_only=True)

        run_split(settings)
        train_rows = _read_csv(settings.train_path)
        val_rows = _read_csv(settings.val_path)

        all_rows = train_rows + val_rows
        group_map: dict[str, set[str]] = {}
        for row in all_rows:
            group_map.setdefault(row["audio_path"], set()).add(
                "train" if row in train_rows else "val"
            )
        self.assertEqual(len(group_map["./songs/shared.mp3"]), 1)

    def test_deterministic_output_for_same_input_even_if_row_order_changes(self) -> None:
        rows_order1 = [
            {"sample_id": "a1", "set_id": "A", "audio_path": "./a.mp3", "valid": "true"},
            {"sample_id": "b1", "set_id": "B", "audio_path": "./b.mp3", "valid": "true"},
            {"sample_id": "c1", "set_id": "C", "audio_path": "./c.mp3", "valid": "true"},
            {"sample_id": "d1", "set_id": "D", "audio_path": "./d.mp3", "valid": "true"},
        ]
        rows_order2 = list(reversed(rows_order1))
        fieldnames = ["sample_id", "set_id", "audio_path", "valid"]

        _write_csv(self.root / "manifest1.csv", rows_order1, fieldnames)
        _write_csv(self.root / "manifest2.csv", rows_order2, fieldnames)

        settings1 = self._make_settings(input_name="manifest1.csv", overwrite=True)
        settings1 = replace(
            settings1,
            output_dir=self.root / "s1",
            train_path=self.root / "s1" / "train.csv",
            val_path=self.root / "s1" / "val.csv",
            summary_path=self.root / "s1" / "summary.json",
        )
        settings2 = self._make_settings(input_name="manifest2.csv", overwrite=True)
        settings2 = replace(
            settings2,
            output_dir=self.root / "s2",
            train_path=self.root / "s2" / "train.csv",
            val_path=self.root / "s2" / "val.csv",
            summary_path=self.root / "s2" / "summary.json",
        )

        run_split(settings1)
        run_split(settings2)
        train1 = sorted(r["sample_id"] for r in _read_csv(settings1.train_path))
        val1 = sorted(r["sample_id"] for r in _read_csv(settings1.val_path))
        train2 = sorted(r["sample_id"] for r in _read_csv(settings2.train_path))
        val2 = sorted(r["sample_id"] for r in _read_csv(settings2.val_path))

        self.assertEqual(train1, train2)
        self.assertEqual(val1, val2)

    def test_summary_json_written(self) -> None:
        rows = [
            {"sample_id": "a1", "set_id": "A", "audio_path": "./a.mp3", "valid": "true"},
            {"sample_id": "b1", "set_id": "B", "audio_path": "./b.mp3", "valid": "true"},
        ]
        _write_csv(self.root / "manifest.csv", rows, ["sample_id", "set_id", "audio_path", "valid"])
        settings = self._make_settings()

        summary = run_split(settings)
        payload = json.loads(settings.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["val_ratio"], 0.4)
        self.assertEqual(payload["train_rows"], summary["train_rows"])
        self.assertEqual(payload["val_rows"], summary["val_rows"])


if __name__ == "__main__":
    unittest.main()
