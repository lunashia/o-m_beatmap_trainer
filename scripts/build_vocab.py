"""CLI for fitting and freezing the project delta_tick vocab.

Purpose:
    Read the shared train config, scan the train cache only, and emit a single
    frozen `artifacts/vocab.json` used by training, validation, and generation.

Input / Output:
    Input:
        - `configs/train.yaml`
        - cached train shards produced by `scripts/build_train_cache.py`
    Output:
        - `artifacts/vocab.json`
        - console summary with token count, delta_tick range, and output path

Pipeline fit:
    Upstream input is the train cache + vocab config.
    Downstream output is the reusable vocab artifact consumed by later stages.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vocab.delta_tick_vocab import (
    build_delta_tick_vocab_from_train_cache,
    load_vocab,
    load_vocab_settings,
    save_vocab,
)

LOGGER = logging.getLogger("build_vocab")


def _configure_logging() -> None:
    """Set the CLI logging format used by the vocab build command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    """Build or reuse the configured delta_tick vocab artifact.

    Args:
        None

    Returns:
        None

    Important notes:
        - Existing vocab files are reused when `overwrite=false`.
        - If the vocab file is absent, the command fits only from train cache.
    """
    _configure_logging()
    settings = load_vocab_settings(force_reload=True)

    if settings.vocab_path.exists() and not settings.overwrite:
        vocab = load_vocab(settings.vocab_path)
        values = list(vocab.delta_tick_values)
        LOGGER.info(
            "Reused vocab: tokens=%d delta_tick_range=[%s, %s] path=%s",
            vocab.num_tokens,
            values[0] if values else "n/a",
            values[-1] if values else "n/a",
            settings.vocab_path,
        )
        return

    vocab = build_delta_tick_vocab_from_train_cache(settings)
    save_vocab(vocab, settings.vocab_path, overwrite=settings.overwrite)
    values = list(vocab.delta_tick_values)
    LOGGER.info(
        "Built vocab: tokens=%d delta_tick_range=[%s, %s] path=%s",
        vocab.num_tokens,
        values[0] if values else "n/a",
        values[-1] if values else "n/a",
        settings.vocab_path,
    )


if __name__ == "__main__":
    main()
