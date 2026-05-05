"""Fit, freeze, save, load, and apply the project delta_tick vocab.

Purpose:
    Build a deterministic integer vocabulary for `delta_tick` values from the
    training split only, then reuse that frozen mapping everywhere else.

Input / Output:
    Input:
        - train cache shards produced by `scripts.build_train_cache`
        - optional train split manifest for train-only vocab fitting
        - `configs/train.yaml` vocab settings
    Output:
        - `artifacts/vocab.json` with stable token metadata
        - `DeltaTickVocab` objects for encoding/decoding `delta_tick`

Pipeline fit:
    Upstream input is the cached training dataset built from beatmaps.
    Downstream output is consumed by dataset encoding, training, validation,
    and generation so every stage uses the same frozen token mapping.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np

from config_loader import load_train_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELD = "delta_tick"
SUPPORTED_FIT_FROM = {"train"}
SUPPORTED_SOURCES = {"target", "past"}


def _resolve_path(raw_path: Any, base_dir: Path) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _as_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_sources(raw: Any) -> tuple[str, ...]:
    if raw is None:
        values = ["target", "past"]
    elif isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item).strip() for item in raw if str(item).strip()]
    else:
        values = [str(raw).strip()] if str(raw).strip() else []

    normalized: List[str] = []
    for value in values:
        lowered = value.lower()
        if lowered not in SUPPORTED_SOURCES:
            raise ValueError(f"Unsupported vocab source: {value!r}")
        if lowered not in normalized:
            normalized.append(lowered)
    if not normalized:
        raise ValueError("vocab.sources must not be empty.")
    return tuple(normalized)


def _normalize_special_tokens(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ValueError("vocab.special_tokens must be a mapping.")
    pad = str(raw.get("pad", "")).strip()
    unk = str(raw.get("unk", "")).strip()
    if not pad or not unk:
        raise ValueError("vocab.special_tokens must define pad and unk.")
    if pad == unk:
        raise ValueError("vocab.special_tokens.pad and vocab.special_tokens.unk must differ.")
    return {"pad": pad, "unk": unk}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_delta_tick_values(values: Iterable[Any]) -> List[int]:
    observed: set[int] = set()
    for raw in values:
        if isinstance(raw, bool):
            raise ValueError("delta_tick values must be integers, not bools.")
        try:
            observed.add(int(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid delta_tick value: {raw!r}") from exc
    if not observed:
        raise ValueError("No delta_tick values were collected from the training set.")
    return sorted(observed)


def _build_mapping(
    delta_tick_values: Sequence[int],
    special_tokens: Mapping[str, str],
) -> tuple[Dict[str, int], tuple[Any, ...]]:
    # Reserve the first two ids for deterministic special tokens.
    token_to_id: Dict[str, int] = {
        special_tokens["pad"]: 0,
        special_tokens["unk"]: 1,
    }
    id_to_token: List[Any] = [special_tokens["pad"], special_tokens["unk"]]
    for idx, value in enumerate(delta_tick_values, start=2):
        token_to_id[str(int(value))] = idx
        id_to_token.append(int(value))
    return token_to_id, tuple(id_to_token)


@dataclass(frozen=True)
class VocabSettings:
    """Resolved settings used to fit or load the delta_tick vocab.

    Attributes:
        project_root: Repository root used to resolve relative paths.
        cache_dir: Directory that contains cached train shards.
        cache_index_path: CSV index produced by train-cache building.
        split_train_index_path: Optional train split manifest used to restrict
            fitting to train sample ids only.
        vocab_path: Destination JSON path for the frozen vocab.
        fit_from: Source split name used for fitting; this project supports
            `train` only.
        field: Token field name; this module supports `delta_tick` only.
        sources: Which sample fields to scan (`target`, `past`, or both).
        overwrite: Whether to replace an existing vocab file.
        frozen: Whether the resulting vocab must be immutable.
        fail_on_unknown: Whether unknown values should raise instead of
            mapping to UNK at encode time.
        special_tokens: Names for PAD and UNK.
    """
    project_root: Path
    cache_dir: Path
    cache_index_path: Path
    split_train_index_path: Path | None
    vocab_path: Path
    fit_from: str
    field: str
    sources: tuple[str, ...]
    overwrite: bool
    frozen: bool
    fail_on_unknown: bool
    special_tokens: Dict[str, str]


@dataclass(frozen=True)
class DeltaTickVocab:
    """Frozen integer vocabulary for `delta_tick`.

    The vocabulary stores a stable mapping from observed integer values to
    contiguous token ids. Special tokens are always reserved at ids 0 and 1.
    """
    field: str
    sources: tuple[str, ...]
    fitted_from: str
    frozen: bool
    fail_on_unknown: bool
    created_at: str
    special_tokens: Dict[str, str]
    delta_tick_values: tuple[int, ...]
    token_to_id: Dict[str, int]
    id_to_token: tuple[Any, ...]

    @property
    def num_tokens(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.special_tokens["pad"]]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.special_tokens["unk"]]

    def encode_delta_tick(self, value: Any, *, fail_on_unknown: bool | None = None) -> int:
        """Encode one raw `delta_tick` value into a token id.

        Args:
            value (Any):
                Raw `delta_tick` value, typically an integer from a sample's
                `target` or `past` fields.
            fail_on_unknown (bool | None):
                Overrides the vocab-level unknown-token policy when provided.
                `None` means use the vocab's configured default.

        Returns:
            int:
                Token id for the value, or `UNK` when unknown values are allowed.

        Important notes:
            - Non-integer inputs are rejected when unknowns are disallowed.
            - Values not seen during fitting map to `UNK` unless strict mode is on.
        """
        effective_fail = self.fail_on_unknown if fail_on_unknown is None else bool(fail_on_unknown)
        if isinstance(value, bool):
            if effective_fail:
                raise ValueError("delta_tick must be an integer.")
            return self.unk_id
        try:
            key = str(int(value))
        except (TypeError, ValueError):
            if effective_fail:
                raise ValueError(f"Invalid delta_tick value: {value!r}")
            return self.unk_id
        token_id = self.token_to_id.get(key)
        if token_id is not None:
            return token_id
        if effective_fail:
            raise KeyError(f"Unknown delta_tick value: {value!r}")
        return self.unk_id

    def decode_delta_tick(self, token_id: Any) -> Any:
        """Decode a token id back to the stored `delta_tick` value.

        Args:
            token_id (Any):
                Integer token id produced by `encode_delta_tick`.

        Returns:
            Any:
                The original integer delta_tick value, or the special token
                string for reserved ids.

        Important notes:
            - Out-of-range ids raise `IndexError`.
            - This method does not infer unseen values; it is a pure lookup.
        """
        if isinstance(token_id, bool):
            raise ValueError("token_id must be an integer.")
        try:
            idx = int(token_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid token id: {token_id!r}") from exc
        if idx < 0 or idx >= len(self.id_to_token):
            raise IndexError(f"token id out of range: {idx}")
        return self.id_to_token[idx]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the vocab into a JSON-friendly dictionary.

        Returns:
            Dict[str, Any]:
                Stable, diff-friendly payload that can be written to disk.

        Important notes:
            - Integer keys are converted to JSON-safe representations.
            - The output includes audit metadata such as source split and time.
        """
        return {
            "field": self.field,
            "sources": list(self.sources),
            "fitted_from": self.fitted_from,
            "frozen": self.frozen,
            "fail_on_unknown": self.fail_on_unknown,
            "created_at": self.created_at,
            "num_tokens": self.num_tokens,
            "special_tokens": dict(self.special_tokens),
            "delta_tick_values": list(self.delta_tick_values),
            "token_to_id": dict(self.token_to_id),
            "id_to_token": list(self.id_to_token),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeltaTickVocab":
        """Rebuild a vocab from a previously saved JSON object.

        Args:
            data (Mapping[str, Any]):
                Parsed JSON object from `vocab.json`.

        Returns:
            DeltaTickVocab:
                Frozen vocab instance reconstructed from disk.

        Important notes:
            - Validates required metadata and consistency of token tables.
            - Rejects non-frozen payloads and mismatched fields.
        """
        required = {
            "field",
            "sources",
            "fitted_from",
            "frozen",
            "fail_on_unknown",
            "created_at",
            "num_tokens",
            "special_tokens",
            "delta_tick_values",
            "token_to_id",
            "id_to_token",
        }
        missing = sorted(required.difference(data.keys()))
        if missing:
            raise ValueError(f"Missing required vocab fields: {missing!r}")

        field = str(data["field"]).strip()
        if field != DEFAULT_FIELD:
            raise ValueError(f"Unsupported vocab field: {field!r}")

        sources = _normalize_sources(data["sources"])
        fitted_from = str(data["fitted_from"]).strip()
        if fitted_from != "train":
            raise ValueError(f"Unsupported fitted_from: {fitted_from!r}")

        frozen = bool(data["frozen"])
        if not frozen:
            raise ValueError("Loaded vocab must be frozen.")

        special_tokens = _normalize_special_tokens(data["special_tokens"])
        delta_tick_values = tuple(_normalize_delta_tick_values(data["delta_tick_values"]))
        token_to_id, id_to_token = _build_mapping(delta_tick_values, special_tokens)

        loaded_token_to_id = {str(k): int(v) for k, v in dict(data["token_to_id"]).items()}
        loaded_id_to_token = list(data["id_to_token"])
        if loaded_token_to_id != token_to_id:
            raise ValueError("token_to_id is inconsistent with delta_tick_values.")
        if loaded_id_to_token != list(id_to_token):
            raise ValueError("id_to_token is inconsistent with delta_tick_values.")

        num_tokens = int(data["num_tokens"])
        if num_tokens != len(id_to_token):
            raise ValueError("num_tokens is inconsistent with id_to_token.")

        return cls(
            field=field,
            sources=sources,
            fitted_from=fitted_from,
            frozen=True,
            fail_on_unknown=bool(data["fail_on_unknown"]),
            created_at=str(data["created_at"]),
            special_tokens=special_tokens,
            delta_tick_values=delta_tick_values,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
        )


def fit_delta_tick_vocab(
    values: Iterable[Any],
    *,
    field: str = DEFAULT_FIELD,
    sources: Sequence[str] = ("target", "past"),
    fitted_from: str = "train",
    frozen: bool = True,
    fail_on_unknown: bool = False,
    special_tokens: Mapping[str, str] | None = None,
    created_at: str | None = None,
) -> DeltaTickVocab:
    """Fit a frozen vocab from raw delta_tick values.

    Args:
        values (Iterable[Any]):
            Raw `delta_tick` values collected from the training split.
        field (str):
            Token field name; kept for metadata and validation.
        sources (Sequence[str]):
            Which sample fields contributed values, typically `target`,
            `past`, or both.
        fitted_from (str):
            Split name used for fitting; this project expects `train`.
        frozen (bool):
            Must remain `True` so the mapping cannot expand after fitting.
        fail_on_unknown (bool):
            Whether encoding unknown values should raise instead of returning UNK.
        special_tokens (Mapping[str, str] | None):
            Names for the reserved PAD and UNK tokens.
        created_at (str | None):
            Optional explicit timestamp for deterministic test fixtures.

    Returns:
        DeltaTickVocab:
            Frozen vocab with sorted, reproducible ids.

    Important notes:
        - Values are deduplicated and sorted numerically before ids are assigned.
        - Special token ids are fixed and stable across runs.
    """
    if field != DEFAULT_FIELD:
        raise ValueError(f"Only field={DEFAULT_FIELD!r} is supported, got {field!r}.")
    if fitted_from != "train":
        raise ValueError(f"Only fitted_from='train' is supported, got {fitted_from!r}.")
    if not frozen:
        raise ValueError("delta_tick vocab must be frozen.")

    normalized_sources = _normalize_sources(sources)
    normalized_special_tokens = _normalize_special_tokens(
        special_tokens or {"pad": "<PAD>", "unk": "<UNK>"}
    )
    delta_tick_values = tuple(_normalize_delta_tick_values(values))
    token_to_id, id_to_token = _build_mapping(delta_tick_values, normalized_special_tokens)

    return DeltaTickVocab(
        field=field,
        sources=normalized_sources,
        fitted_from=fitted_from,
        frozen=True,
        fail_on_unknown=bool(fail_on_unknown),
        created_at=created_at or _now_utc_iso(),
        special_tokens=normalized_special_tokens,
        delta_tick_values=delta_tick_values,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
    )


def _read_cache_index(cache_index_path: Path) -> List[Dict[str, str]]:
    """Read the cache index CSV into a list of dictionaries."""
    if not cache_index_path.exists():
        return []
    with cache_index_path.open("r", encoding="utf-8-sig", newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def _read_allowed_sample_ids(path: Path | None) -> set[str] | None:
    """Load the train split sample-id whitelist when configured.

    Args:
        path (Path | None):
            Optional CSV file with a `sample_id` column.

    Returns:
        set[str] | None:
            Allowed sample ids, or `None` when no split restriction is active.

    Important notes:
        - The file is expected to be a train-only split manifest.
        - Missing `sample_id` values are ignored.
    """
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"split.train_path not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        sample_ids: set[str] = set()
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id:
                sample_ids.add(sample_id)
    return sample_ids


def _is_valid_row(row: Mapping[str, Any]) -> bool:
    """Interpret the `valid` CSV field used by the cache index."""
    value = row.get("valid", "")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _resolve_shard_paths(settings: VocabSettings, allowed_sample_ids: set[str] | None) -> List[Path]:
    """Resolve candidate train-cache shard paths.

    Args:
        settings (VocabSettings):
            Resolved vocab settings and project paths.
        allowed_sample_ids (set[str] | None):
            Optional train-split whitelist. When set, only shards referenced by
            matching rows in the cache index are considered.

    Returns:
        List[Path]:
            Deterministically ordered shard paths to scan.

    Important notes:
        - Prefers the cache index when available.
        - Falls back to shard globbing only when no indexed shards are found.
    """
    shard_paths: Dict[str, Path] = {}
    for row in _read_cache_index(settings.cache_index_path):
        if not _is_valid_row(row):
            continue
        if allowed_sample_ids is not None:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id or sample_id not in allowed_sample_ids:
                continue
        shard_path_raw = str(row.get("shard_path", "")).strip()
        if not shard_path_raw:
            continue
        shard_path = _resolve_path(shard_path_raw, settings.project_root)
        if shard_path is None:
            continue
        shard_paths[str(shard_path.resolve()).lower()] = shard_path

    if allowed_sample_ids is not None:
        return [shard_paths[key] for key in sorted(shard_paths)]

    if shard_paths:
        return [shard_paths[key] for key in sorted(shard_paths)]

    for suffix in ("pt", "npz"):
        for path in settings.cache_dir.glob(f"train_cache_shard_*.{suffix}"):
            shard_paths[str(path.resolve()).lower()] = path
    return [shard_paths[key] for key in sorted(shard_paths)]


def _load_shard_records(shard_path: Path) -> List[Dict[str, Any]]:
    """Load one train-cache shard payload into a list of records."""
    suffix = shard_path.suffix.lower()
    if suffix == ".pt":
        import torch

        payload = torch.load(str(shard_path), map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Unsupported PT shard payload: {shard_path}")
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError(f"Unsupported PT shard records layout: {shard_path}")
        return list(records)

    if suffix == ".npz":
        with np.load(str(shard_path), allow_pickle=True) as archive:
            if "records" not in archive:
                raise ValueError(f"NPZ shard is missing records: {shard_path}")
            records_arr = archive["records"]
            if records_arr.size == 0:
                return []
            records_obj = records_arr.tolist()
            if isinstance(records_obj, list):
                if records_obj and isinstance(records_obj[0], list):
                    return list(records_obj[0])
                return list(records_obj)
            if isinstance(records_obj, tuple):
                return list(records_obj)
            raise ValueError(f"Unsupported NPZ shard records layout: {shard_path}")

    raise ValueError(f"Unsupported shard format: {shard_path.suffix!r}")


def _iter_sample_pairs(record: Mapping[str, Any]) -> Iterator[tuple[Any, Any]]:
    """Yield `(X, y)` sample pairs from a cached record payload."""
    if not _is_valid_row(record):
        return
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return
    samples = payload.get("samples")
    if not isinstance(samples, (list, tuple)):
        raise ValueError("Cache record payload must contain a samples list.")
    for idx, sample in enumerate(samples):
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            raise ValueError(f"Invalid sample layout at index {idx}.")
        yield sample[0], sample[1]


def _iter_delta_tick_values_from_sample(sample: tuple[Any, Any], sources: Sequence[str]) -> Iterator[int]:
    """Extract raw delta_tick integers from one cached sample pair."""
    x, y = sample
    if "target" in sources:
        if not isinstance(y, Mapping):
            raise ValueError("Target sample must be a mapping.")
        delta_tick = y.get("delta_tick")
        if isinstance(delta_tick, bool):
            raise ValueError("Target delta_tick must be an integer.")
        yield int(delta_tick)

    if "past" in sources:
        if not isinstance(x, Mapping):
            raise ValueError("Past sample must be a mapping.")
        past_events = x.get("past_events")
        if not isinstance(past_events, list):
            raise ValueError("past_events must be a list.")
        for event in past_events:
            if not isinstance(event, Mapping):
                raise ValueError("Each past event must be a mapping.")
            delta_tick = event.get("delta_tick")
            if isinstance(delta_tick, bool):
                raise ValueError("Past delta_tick must be an integer.")
            yield int(delta_tick)


def build_delta_tick_vocab_from_train_cache(settings: VocabSettings) -> DeltaTickVocab:
    """Fit the vocab by scanning only the train cache.

    Args:
        settings (VocabSettings):
            Resolved config, cache locations, and policy flags.

    Returns:
        DeltaTickVocab:
            Frozen vocab fitted from train-only cached samples.

    Important notes:
        - Only the training split is eligible for fitting.
        - When `split_train_index_path` is provided, it acts as a whitelist of
          sample ids to prevent leakage from other splits.
        - Both `target` and `past` sources are scanned when configured.
    """
    if settings.fit_from != "train":
        raise NotImplementedError("Only fit_from='train' is supported.")
    if settings.field != DEFAULT_FIELD:
        raise ValueError(f"Only field={DEFAULT_FIELD!r} is supported.")

    allowed_sample_ids = _read_allowed_sample_ids(settings.split_train_index_path)
    shard_paths = _resolve_shard_paths(settings, allowed_sample_ids)
    if not shard_paths:
        raise FileNotFoundError(
            f"No train cache shards found in {settings.cache_dir} or {settings.cache_index_path}."
        )

    observed: List[int] = []
    for shard_path in shard_paths:
        for record in _load_shard_records(shard_path):
            if not _is_valid_row(record):
                continue
            if allowed_sample_ids is not None:
                sample_id = str(record.get("sample_id", "")).strip()
                if not sample_id or sample_id not in allowed_sample_ids:
                    continue
            for sample in _iter_sample_pairs(record):
                observed.extend(_iter_delta_tick_values_from_sample(sample, settings.sources))

    return fit_delta_tick_vocab(
        observed,
        field=settings.field,
        sources=settings.sources,
        fitted_from=settings.fit_from,
        frozen=settings.frozen,
        fail_on_unknown=settings.fail_on_unknown,
        special_tokens=settings.special_tokens,
    )


def save_vocab(vocab: DeltaTickVocab, path: Path | str, *, overwrite: bool = False) -> None:
    """Write a frozen vocab to disk as stable JSON.

    Args:
        vocab (DeltaTickVocab):
            Frozen vocab instance to serialize.
        path (Path | str):
            Destination JSON file, typically `artifacts/vocab.json`.
        overwrite (bool):
            When `False`, an existing file raises `FileExistsError`.

    Returns:
        None

    Important notes:
        - The parent directory is created automatically.
        - JSON is written with stable indentation to keep diffs reviewable.
    """
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Vocab already exists at {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(vocab.to_dict(), fp, indent=2, ensure_ascii=False, sort_keys=False)
        fp.write("\n")
    tmp_path.replace(output_path)


def load_vocab(path: Path | str) -> DeltaTickVocab:
    """Load and validate a vocab JSON file.

    Args:
        path (Path | str):
            Path to the saved vocab JSON file.

    Returns:
        DeltaTickVocab:
            Frozen vocab reconstructed from disk.

    Important notes:
        - Rejects missing files and malformed JSON.
        - Validates that the loaded payload is frozen and matches the
          `delta_tick` field contract.
    """
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(str(payload_path))
    data = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Vocab JSON must contain an object.")
    vocab = DeltaTickVocab.from_dict(data)
    if not vocab.frozen:
        raise ValueError("Loaded vocab must be frozen.")
    if vocab.field != DEFAULT_FIELD:
        raise ValueError(f"Loaded vocab field must be {DEFAULT_FIELD!r}.")
    return vocab


def load_vocab_settings(force_reload: bool = True) -> VocabSettings:
    """Resolve vocab-related settings from the shared train config.

    Args:
        force_reload (bool):
            Reload the config file before resolving settings.

    Returns:
        VocabSettings:
            Paths, flags, and policies normalized for runtime use.

    Important notes:
        - All relative paths are resolved from the repository root.
        - This enforces `field == "delta_tick"` and `frozen == true`.
    """
    config = load_train_config(force_reload=force_reload)
    if not isinstance(config, dict):
        raise ValueError("Invalid train config.")

    data_cfg = config.get("data", {})
    cache_cfg = config.get("cache", {})
    vocab_cfg = config.get("vocab", {})
    split_cfg = config.get("split", {})

    if not isinstance(vocab_cfg, dict):
        raise ValueError("vocab config must be a mapping.")

    vocab_path = _resolve_path(vocab_cfg.get("path"), PROJECT_ROOT)
    if vocab_path is None:
        raise ValueError("vocab.path must be configured.")

    cache_dir = _resolve_path(
        cache_cfg.get("cache_dir", data_cfg.get("cache_dir", "./data/cache/train_cache")),
        PROJECT_ROOT,
    )
    if cache_dir is None:
        raise ValueError("cache_dir must be configured.")

    cache_index_path = _resolve_path(
        cache_cfg.get("cache_index_path", cache_dir / "cache_index.csv"),
        PROJECT_ROOT,
    )
    if cache_index_path is None:
        raise ValueError("cache_index_path must be configured.")

    split_train_index_path: Path | None = None
    if isinstance(split_cfg, dict):
        split_train_index_path = _resolve_path(split_cfg.get("train_path"), PROJECT_ROOT)

    fit_from = str(vocab_cfg.get("fit_from", "train")).strip().lower()
    field = str(vocab_cfg.get("field", DEFAULT_FIELD)).strip()
    sources = _normalize_sources(vocab_cfg.get("sources", ["target", "past"]))
    overwrite = _as_bool(vocab_cfg.get("overwrite", False), False)
    frozen = _as_bool(vocab_cfg.get("frozen", True), True)
    fail_on_unknown = _as_bool(vocab_cfg.get("fail_on_unknown", False), False)
    special_tokens = _normalize_special_tokens(vocab_cfg.get("special_tokens", {}))

    if fit_from not in SUPPORTED_FIT_FROM:
        raise ValueError(f"Unsupported vocab.fit_from: {fit_from!r}")
    if field != DEFAULT_FIELD:
        raise ValueError(f"Unsupported vocab.field: {field!r}")
    if not frozen:
        raise ValueError("vocab.frozen must be true.")

    return VocabSettings(
        project_root=PROJECT_ROOT,
        cache_dir=cache_dir,
        cache_index_path=cache_index_path,
        split_train_index_path=split_train_index_path,
        vocab_path=vocab_path,
        fit_from=fit_from,
        field=field,
        sources=sources,
        overwrite=overwrite,
        frozen=frozen,
        fail_on_unknown=fail_on_unknown,
        special_tokens=special_tokens,
    )


def load_vocab_from_config(force_reload: bool = True) -> DeltaTickVocab:
    """Load the configured vocab file from `configs/train.yaml`.

    Args:
        force_reload (bool):
            Reload configuration before resolving the vocab path.

    Returns:
        DeltaTickVocab:
            Frozen vocab loaded from the configured path.

    Important notes:
        - This is the canonical runtime entry point for train/val/generation.
    """
    return load_vocab(load_vocab_settings(force_reload=force_reload).vocab_path)


def encode_delta_tick(vocab: DeltaTickVocab, value: Any, *, fail_on_unknown: bool | None = None) -> int:
    """Convenience wrapper around `DeltaTickVocab.encode_delta_tick`."""
    return vocab.encode_delta_tick(value, fail_on_unknown=fail_on_unknown)


def decode_delta_tick(vocab: DeltaTickVocab, token_id: Any) -> Any:
    """Convenience wrapper around `DeltaTickVocab.decode_delta_tick`."""
    return vocab.decode_delta_tick(token_id)
