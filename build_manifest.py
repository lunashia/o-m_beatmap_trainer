"""Build a unified osu!mania manifest index from `.osu` beatmap files.

Purpose:
- Recursively scan configured dataset folders for `.osu` files.
- Extract minimal metadata needed by training-time indexing.
- Validate each sample with configurable filters and write `manifest.csv`.

Input/Output format:
- Input:
  - Runtime config loaded from `configs/train.yaml` under `manifest`:
    `dataset_root`, `manifest_path`, `allowed_modes`, `allowed_key_counts`,
    `require_audio`, `include_invalid`, `osu_pattern`/`file_glob`, `encodings`.
  - `.osu` text content parsed for:
    `[General] AudioFilename/Mode`, `[Difficulty] CircleSize`,
    `[Metadata] Version/BeatmapSetID`.
- Output:
  - CSV file with fixed columns:
    `sample_id,osu_path,audio_path,set_id,difficulty_name,key_count,valid`.
  - Console summary: scanned/valid/invalid counts and output path.

Pipeline integration:
- Upstream input producer:
  - Dataset preparation/download step that places beatmap folders and audio
    files under `manifest.dataset_root`.
- Downstream output consumer:
  - Training data entry/index loader that reads `manifest.csv` as the unified
    sample source.
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from config_loader import get_config_value, load_train_config


MANIFEST_COLUMNS = [
    "sample_id",
    "osu_path",
    "audio_path",
    "set_id",
    "difficulty_name",
    "key_count",
    "valid",
]


def _parse_list_config(raw: Any) -> List[Any]:
    """Normalize a config value into a flat Python list.

    Args:
        raw (Any):
            Config value that may be:
            - list/tuple/set
            - scalar (`int`, `float`, `bool`)
            - string list formats such as `"[1, 2]"` or `"a,b,c"`
            - `None`

    Returns:
        List[Any]:
            Parsed list representation. Invalid/unrecognized inputs return `[]`.

    Important notes:
        - Uses `ast.literal_eval` only for bracketed list-like strings.
        - Falls back to comma-splitting for plain strings.
        - No type coercion is applied here beyond preserving scalar items.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return list(raw)
    if isinstance(raw, (int, float, bool)):
        return [raw]
    if not isinstance(raw, str):
        return []

    text = raw.strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            # Support list-like strings from YAML subsets, e.g. "[3, 7]".
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            value = None
        if isinstance(value, (list, tuple, set)):
            return list(value)

    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_int_list(raw: Any) -> List[int]:
    """Parse a config value into a list of integers.

    Args:
        raw (Any):
            Any list-compatible config value accepted by `_parse_list_config`.

    Returns:
        List[int]:
            Integer items successfully cast from the input list.

    Important notes:
        - Non-castable items are ignored instead of raising.
    """
    values: List[int] = []
    for item in _parse_list_config(raw):
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    return values


def _parse_str_list(raw: Any) -> List[str]:
    """Parse a config value into a list of non-empty strings.

    Args:
        raw (Any):
            Any list-compatible config value accepted by `_parse_list_config`.

    Returns:
        List[str]:
            String items with surrounding whitespace removed and empty items
            discarded.

    Important notes:
        - All items are converted via `str(...)`.
    """
    values: List[str] = []
    for item in _parse_list_config(raw):
        text = str(item).strip()
        if text:
            values.append(text)
    return values


def _read_text_with_fallback(path: Path, encodings: Iterable[str]) -> Optional[str]:
    """Read text file content using ordered encoding fallbacks.

    Args:
        path (Path):
            Target file path.
        encodings (Iterable[str]):
            Encoding candidates attempted in order.

    Returns:
        Optional[str]:
            File text if decoding succeeds with any encoding, otherwise `None`.

    Important notes:
        - `UnicodeDecodeError` triggers fallback to the next encoding.
        - `OSError` (missing/unreadable file) returns `None` immediately.
    """
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def _parse_osu_fields(text: str) -> Dict[str, Any]:
    """Extract required manifest fields from one `.osu` text blob.

    Args:
        text (str):
            Full `.osu` file content.

    Returns:
        Dict[str, Any]:
            Parsed fields:
            - `audio_filename` (str)
            - `mode` (Optional[int])
            - `key_count` (Optional[int])
            - `difficulty_name` (str)
            - `set_id` (str)

    Important notes:
        - Ignores comment lines (`//`) and malformed `key:value` lines.
        - `CircleSize` is parsed with `float -> int` to tolerate values like
          `"7.0"`.
        - Missing/invalid values remain empty or `None` for downstream validation.
    """
    section = ""
    audio_filename = ""
    mode: Optional[int] = None
    key_count: Optional[int] = None
    difficulty_name = ""
    set_id = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue

        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if section == "General":
            if key == "AudioFilename":
                audio_filename = value
            elif key == "Mode":
                try:
                    mode = int(value)
                except ValueError:
                    mode = None
        elif section == "Difficulty":
            if key == "CircleSize":
                try:
                    key_count = int(float(value))
                except ValueError:
                    key_count = None
        elif section == "Metadata":
            if key == "Version":
                difficulty_name = value
            elif key == "BeatmapSetID":
                set_id = value

    return {
        "audio_filename": audio_filename,
        "mode": mode,
        "key_count": key_count,
        "difficulty_name": difficulty_name,
        "set_id": set_id,
    }


def _infer_set_id_from_path(osu_path: Path) -> str:
    """Infer beatmap set id from the parent directory name.

    Args:
        osu_path (Path):
            Path to a `.osu` file.

    Returns:
        str:
            Leading digit sequence from the parent folder name, or `""` when
            no leading digits are present.

    Important notes:
        - This is a best-effort fallback when metadata `BeatmapSetID` is empty.
    """
    parent_name = osu_path.parent.name
    match = re.match(r"^\s*(\d+)", parent_name)
    if match:
        return match.group(1)
    return ""


def _canonical_path(path: Path, project_root: Path) -> str:
    """Format path output using project-relative form when possible.

    Args:
        path (Path):
            Source path to normalize.
        project_root (Path):
            Project root used for relative canonicalization.

    Returns:
        str:
            `./posix/style/path` if `path` is inside `project_root`, otherwise
            absolute resolved path.

    Important notes:
        - Uses `.resolve()` to collapse `..` and normalize separators.
    """
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(project_root.resolve())
        return f"./{relative.as_posix()}"
    except ValueError:
        return str(resolved)


def _discover_osu_files(dataset_root: Path, osu_patterns: List[str]) -> List[Path]:
    """Discover `.osu` files under dataset root with glob pattern support.

    Args:
        dataset_root (Path):
            Root directory containing beatmap files.
        osu_patterns (List[str]):
            File patterns (for example `"*.osu"` or `"**/*.osu"`).

    Returns:
        List[Path]:
            Deduplicated, case-insensitive sorted list of `.osu` file paths.

    Important notes:
        - Plain filename patterns (no path separators or `**`) use `rglob` for
          recursive matching by default.
        - Paths are deduplicated by lowercase resolved path to avoid duplicate
          matches across overlapping patterns.
    """
    found: Dict[str, Path] = {}
    patterns = osu_patterns or ["*.osu"]

    for pattern in patterns:
        normalized = pattern.strip() or "*.osu"
        use_rglob = (
            "**" not in normalized
            and "/" not in normalized
            and "\\" not in normalized
        )
        # Keep simple patterns recursive; preserve explicit caller intent otherwise.
        iterator = dataset_root.rglob(normalized) if use_rglob else dataset_root.glob(normalized)
        for path in iterator:
            if path.is_file() and path.suffix.lower() == ".osu":
                found[str(path.resolve()).lower()] = path

    return sorted(found.values(), key=lambda p: str(p).lower())


def _build_manifest_rows(
    osu_files: List[Path],
    dataset_root: Path,
    project_root: Path,
    *,
    encodings: Iterable[str],
    allowed_modes: set[int],
    allowed_key_counts: set[int],
    require_audio: bool,
    include_invalid: bool,
    text_loader: Callable[[Path, Iterable[str]], Optional[str]] = _read_text_with_fallback,
    path_exists: Callable[[Path], bool] = lambda p: p.exists(),
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Build manifest row dictionaries and validity counters from `.osu` paths.

    Args:
        osu_files (List[Path]):
            Candidate `.osu` file paths.
        dataset_root (Path):
            Root used to derive stable `sample_id` as relative path.
        project_root (Path):
            Root used to canonicalize output path strings.
        encodings (Iterable[str]):
            Encoding fallback order for reading `.osu` files.
        allowed_modes (set[int]):
            Allowed osu! game modes. Empty set disables mode filtering.
        allowed_key_counts (set[int]):
            Allowed key counts. Empty set disables key-count filtering.
        require_audio (bool):
            If `True`, sample is invalid when referenced audio file is missing.
        include_invalid (bool):
            If `True`, invalid samples are still emitted with `valid=false`.
        text_loader (Callable[[Path, Iterable[str]], Optional[str]]):
            Injectable reader used by tests to avoid filesystem I/O.
        path_exists (Callable[[Path], bool]):
            Injectable existence checker used by tests.

    Returns:
        Tuple[List[Dict[str, Any]], int, int]:
            - Manifest rows that pass inclusion policy.
            - Valid sample count.
            - Invalid sample count.

    Important notes:
        - `valid` is computed from readability, `AudioFilename`, audio existence
          (when required), parseable `CircleSize`, mode filter, and key filter.
        - `set_id` uses metadata first, then falls back to path inference.
        - `sample_id` is stable across runs as dataset-relative POSIX path.
    """
    rows: List[Dict[str, Any]] = []
    valid_count = 0
    invalid_count = 0

    for osu_file in osu_files:
        rel_path = osu_file.relative_to(dataset_root)
        sample_id = rel_path.as_posix()

        parsed = {
            "audio_filename": "",
            "mode": None,
            "key_count": None,
            "difficulty_name": "",
            "set_id": "",
        }
        valid = True

        text = text_loader(osu_file, encodings)
        if text is None:
            valid = False
        else:
            parsed = _parse_osu_fields(text)

        audio_filename = str(parsed.get("audio_filename", "") or "").strip()
        mode = parsed.get("mode")
        key_count = parsed.get("key_count")
        difficulty_name = str(parsed.get("difficulty_name", "") or "").strip()
        set_id = str(parsed.get("set_id", "") or "").strip()

        if not set_id:
            set_id = _infer_set_id_from_path(osu_file)

        if not audio_filename:
            valid = False

        audio_abs: Optional[Path] = None
        if audio_filename:
            audio_abs = (osu_file.parent / audio_filename).resolve()
            if require_audio and not path_exists(audio_abs):
                valid = False
        elif require_audio:
            valid = False

        if key_count is None:
            valid = False
        elif allowed_key_counts and key_count not in allowed_key_counts:
            valid = False

        if allowed_modes:
            if not isinstance(mode, int) or mode not in allowed_modes:
                valid = False

        if valid:
            valid_count += 1
        else:
            invalid_count += 1
            # Preserve invalid rows only when explicitly requested by config.
            if not include_invalid:
                continue

        row = {
            "sample_id": sample_id,
            "osu_path": _canonical_path(osu_file, project_root),
            "audio_path": _canonical_path(audio_abs, project_root) if audio_abs else "",
            "set_id": set_id,
            "difficulty_name": difficulty_name,
            "key_count": key_count if key_count is not None else "",
            "valid": str(valid).lower(),
        }
        rows.append(row)

    return rows, valid_count, invalid_count


def main() -> None:
    """Entry point for manifest generation from `configs/train.yaml`.

    Args:
        None.

    Returns:
        None.

    Important notes:
        - Reads runtime options from `manifest` config section with fallbacks.
        - Ensures output directory exists before writing CSV.
        - Raises `ValueError` when required paths are not configured.
        - Upstream input: dataset files under configured root.
        - Downstream output: `manifest.csv` consumed by training/index loaders.
    """
    project_root = Path(__file__).resolve().parent
    config = load_train_config()
    manifest_cfg = config.get("manifest", {}) if isinstance(config, dict) else {}

    dataset_root_raw = manifest_cfg.get("dataset_root", get_config_value("data.osu_root", ""))
    manifest_path_raw = manifest_cfg.get(
        "manifest_path", get_config_value("data.cache_dir", "./data/cache") + "/manifest.csv"
    )

    if not dataset_root_raw:
        raise ValueError("manifest.dataset_root (or data.osu_root) must be configured.")
    if not manifest_path_raw:
        raise ValueError("manifest.manifest_path must be configured.")

    dataset_root = Path(dataset_root_raw).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (project_root / dataset_root).resolve()

    manifest_path = Path(manifest_path_raw).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (project_root / manifest_path).resolve()

    allowed_modes = set(
        _parse_int_list(
            manifest_cfg.get(
                "allowed_modes",
                manifest_cfg.get("game_mode", get_config_value("chart.expected_mode", 3)),
            )
        )
    )
    allowed_key_counts = set(
        _parse_int_list(
            manifest_cfg.get(
                "allowed_key_counts",
                get_config_value("chart.key_count", 7),
            )
        )
    )
    require_audio = bool(manifest_cfg.get("require_audio", True))
    include_invalid = bool(manifest_cfg.get("include_invalid", False))
    osu_patterns = _parse_str_list(
        manifest_cfg.get("osu_pattern", manifest_cfg.get("file_glob", "*.osu"))
    )
    encodings = _parse_str_list(
        manifest_cfg.get("encodings", "utf-8-sig,utf-8,latin-1")
    )
    if not encodings:
        encodings = ["utf-8-sig", "utf-8", "latin-1"]

    osu_files = _discover_osu_files(dataset_root, osu_patterns)

    rows, valid_count, invalid_count = _build_manifest_rows(
        osu_files,
        dataset_root,
        project_root,
        encodings=encodings,
        allowed_modes=allowed_modes,
        allowed_key_counts=allowed_key_counts,
        require_audio=require_audio,
        include_invalid=include_invalid,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scanned: {len(osu_files)}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {invalid_count}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
