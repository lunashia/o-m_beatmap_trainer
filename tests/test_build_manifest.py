from __future__ import annotations

import unittest
from pathlib import Path
from typing import Dict, Iterable, Optional

from build_manifest import _build_manifest_rows


def _osu_text(
    *,
    audio_filename: Optional[str] = "audio.mp3",
    mode: Optional[str] = "3",
    circle_size: Optional[str] = "7",
    version: str = "Hard",
    beatmap_set_id: str = "123",
) -> str:
    general_lines = []
    if audio_filename is not None:
        general_lines.append(f"AudioFilename: {audio_filename}")
    if mode is not None:
        general_lines.append(f"Mode: {mode}")

    difficulty_lines = []
    if circle_size is not None:
        difficulty_lines.append(f"CircleSize: {circle_size}")

    metadata_lines = [
        f"Version: {version}",
        f"BeatmapSetID: {beatmap_set_id}",
    ]

    return "\n".join(
        [
            "[General]",
            *general_lines,
            "",
            "[Difficulty]",
            *difficulty_lines,
            "",
            "[Metadata]",
            *metadata_lines,
        ]
    )


class BuildManifestRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path("C:/repo")
        self.dataset_root = self.project_root / "dataset"
        self.osu_path = self.dataset_root / "123 Set" / "map.osu"

    def _run(
        self,
        text_by_path: Dict[Path, Optional[str]],
        *,
        include_invalid: bool,
        audio_exists: bool = True,
        allowed_modes: set[int] | None = None,
        allowed_key_counts: set[int] | None = None,
    ):
        def text_loader(path: Path, encodings: Iterable[str]) -> Optional[str]:
            _ = encodings
            return text_by_path.get(path)

        def path_exists(path: Path) -> bool:
            _ = path
            return audio_exists

        return _build_manifest_rows(
            [self.osu_path],
            self.dataset_root,
            self.project_root,
            encodings=["utf-8"],
            allowed_modes=allowed_modes if allowed_modes is not None else {3},
            allowed_key_counts=(
                allowed_key_counts if allowed_key_counts is not None else {7}
            ),
            require_audio=True,
            include_invalid=include_invalid,
            text_loader=text_loader,
            path_exists=path_exists,
        )

    def test_valid_osu_and_existing_audio(self) -> None:
        text = _osu_text()
        rows, valid_count, invalid_count = self._run(
            {self.osu_path: text},
            include_invalid=False,
            audio_exists=True,
        )

        self.assertEqual(valid_count, 1)
        self.assertEqual(invalid_count, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["valid"], "true")
        self.assertEqual(rows[0]["key_count"], 7)

    def test_missing_audio_include_invalid_true(self) -> None:
        text = _osu_text(audio_filename="missing.mp3")
        rows, valid_count, invalid_count = self._run(
            {self.osu_path: text},
            include_invalid=True,
            audio_exists=False,
        )

        self.assertEqual(valid_count, 0)
        self.assertEqual(invalid_count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["valid"], "false")

    def test_missing_audio_include_invalid_false_skips(self) -> None:
        text = _osu_text(audio_filename="missing.mp3")
        rows, valid_count, invalid_count = self._run(
            {self.osu_path: text},
            include_invalid=False,
            audio_exists=False,
        )

        self.assertEqual(valid_count, 0)
        self.assertEqual(invalid_count, 1)
        self.assertEqual(len(rows), 0)

    def test_unparseable_audiofilename(self) -> None:
        text = _osu_text(audio_filename=None)
        rows, valid_count, invalid_count = self._run(
            {self.osu_path: text},
            include_invalid=True,
            audio_exists=True,
        )

        self.assertEqual(valid_count, 0)
        self.assertEqual(invalid_count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["valid"], "false")

    def test_mode_unparseable(self) -> None:
        text = _osu_text(mode="abc")
        rows, valid_count, invalid_count = self._run(
            {self.osu_path: text},
            include_invalid=True,
            audio_exists=True,
        )

        self.assertEqual(valid_count, 0)
        self.assertEqual(invalid_count, 1)
        self.assertEqual(rows[0]["valid"], "false")

    def test_mode_not_mania(self) -> None:
        text = _osu_text(mode="0")
        rows, valid_count, invalid_count = self._run(
            {self.osu_path: text},
            include_invalid=True,
            audio_exists=True,
        )

        self.assertEqual(valid_count, 0)
        self.assertEqual(invalid_count, 1)
        self.assertEqual(rows[0]["valid"], "false")

    def test_key_count_unparseable(self) -> None:
        text = _osu_text(circle_size="seven")
        rows, valid_count, invalid_count = self._run(
            {self.osu_path: text},
            include_invalid=True,
            audio_exists=True,
        )

        self.assertEqual(valid_count, 0)
        self.assertEqual(invalid_count, 1)
        self.assertEqual(rows[0]["valid"], "false")

    def test_allowed_key_counts_filter(self) -> None:
        text = _osu_text(circle_size="4")
        rows, valid_count, invalid_count = self._run(
            {self.osu_path: text},
            include_invalid=True,
            audio_exists=True,
            allowed_key_counts={7},
        )

        self.assertEqual(valid_count, 0)
        self.assertEqual(invalid_count, 1)
        self.assertEqual(rows[0]["valid"], "false")


if __name__ == "__main__":
    unittest.main()
