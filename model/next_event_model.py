"""Neural model components for osu!mania next-event prediction.

Purpose:
    Define a minimal multimodal PyTorch model that predicts the next event from:
      1) past event tokens (`delta_tick`, `lane_mask`)
      2) per-step rhythm/grid statistics
      3) local mel spectrogram window

Input / Output format:
    Model input (`NextEventPredictor.forward`):
        x = {
            "past_delta_id":   LongTensor [B, L],
            "past_lane_mask_id": LongTensor [B, L],  # class ids in [0, 127]
            "grid_stats":      FloatTensor [B, G],
            "audio_window":    FloatTensor [B, 80, T],
        }
    Model output:
        {
            "delta_tick_logits": FloatTensor [B, D_delta],
            "lane_mask_logits":  FloatTensor [B, 128],
        }

Pipeline fit:
    Upstream input:
        - `src.dataset_builder.build_dataset` produces sample tuples (X, y).
        - Typical training code converts `X["past_events"]` into
          `past_delta_id`/`past_lane_mask_id` tensors.
    Downstream output:
        - Training scripts consume logits via `compute_next_event_loss`.
        - Inference code can decode logits with `argmax` or sampling to generate
          next `(delta_tick, lane_mask)` predictions.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_loader import get_config_value


class EventEncoder(nn.Module):
    """Encode past event sequences into a fixed-size context vector.

    Args:
        delta_vocab_size: Number of classes for `delta_tick` token ids.
        delta_embed_dim: Embedding dimension for `delta_tick`.
        lane_embed_dim: Embedding dimension for `lane_mask` ids (0..127).
        hidden_dim: Output hidden size for sequence encoder.
        encoder_type: Sequence backbone, either `"gru"` or `"transformer"`.
        transformer_heads: Attention heads when `encoder_type="transformer"`.
        transformer_layers: Encoder layer count for Transformer mode.
        dropout: Dropout used by Transformer encoder layers.

    Notes:
        - `past_delta_tick` values must already be mapped to class ids in
          `[0, delta_vocab_size - 1]`.
        - `past_lane_mask` values are treated as categorical ids, not bitsets.
    """

    def __init__(
        self,
        delta_vocab_size: int,
        delta_embed_dim: int,
        lane_embed_dim: int,
        hidden_dim: int,
        encoder_type: str = "gru",
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if encoder_type not in {"gru", "transformer"}:
            raise ValueError("encoder_type must be either 'gru' or 'transformer'.")

        self.encoder_type = encoder_type
        self.delta_embedding = nn.Embedding(delta_vocab_size, delta_embed_dim)
        self.lane_embedding = nn.Embedding(128, lane_embed_dim)
        in_dim = delta_embed_dim + lane_embed_dim

        if encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.output_dim = hidden_dim
        else:
            self.input_proj = nn.Linear(in_dim, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
            self.output_dim = hidden_dim

    def forward(
        self,
        past_delta_tick: torch.LongTensor,
        past_lane_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            past_delta_tick: Tensor[int64] of shape `[B, L]`.
            past_lane_mask: Tensor[int64] of shape `[B, L]`, class ids in
                `[0, 127]`.

        Returns:
            Tensor[float] of shape `[B, D]`, one context vector per batch item.

        Important notes:
            - GRU mode uses the final hidden state.
            - Transformer mode returns the last sequence position embedding.
        """
        delta_e = self.delta_embedding(past_delta_tick)
        lane_e = self.lane_embedding(past_lane_mask)
        x = torch.cat([delta_e, lane_e], dim=-1)  # [B, L, E]

        if self.encoder_type == "gru":
            _, h_n = self.encoder(x)
            return h_n[-1]

        x = self.input_proj(x)
        x = self.encoder(x)
        # Use the last context position as the event summary.
        return x[:, -1, :]


class AudioEncoder(nn.Module):
    """Encode mel windows with a small Conv1D stack.

    Args:
        in_mels: Mel-bin channel count (default 80).
        hidden_dim: Internal convolution channel width.
        out_dim: Final projected embedding dimension.
    """

    def __init__(self, in_mels: int = 80, hidden_dim: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_mels, out_channels=hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, audio_window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_window: Tensor[float] with shape `[B, 80, T]`.

        Returns:
            Tensor[float] with shape `[B, out_dim]`.

        Important notes:
            - Adaptive average pooling removes time-length dependence.
            - Any `T >= 1` is valid at runtime.
        """
        x = self.net(audio_window).squeeze(-1)
        return self.proj(x)


class GridStatsEncoder(nn.Module):
    """Encode per-event grid statistics with a small MLP.

    Args:
        in_dim: Feature dimension of `grid_stats`.
        hidden_dim: Hidden layer width.
        out_dim: Output embedding size.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, grid_stats: torch.Tensor) -> torch.Tensor:
        """Encode `grid_stats`.

        Args:
            grid_stats: Tensor[float] with shape `[B, G]`.

        Returns:
            Tensor[float] with shape `[B, out_dim]`.
        """
        return self.net(grid_stats)


class NextEventPredictor(nn.Module):
    """Multimodal classifier for next-event prediction.

    This module fuses:
      - event context embedding from `EventEncoder`
      - audio embedding from `AudioEncoder`
      - stats embedding from `GridStatsEncoder`
    and emits logits for two classification heads:
      - `delta_tick` class
      - `lane_mask` class (0..127)

    Args:
        delta_vocab_size: Number of `delta_tick` classes.
        grid_stats_dim: Input feature size `G` for `grid_stats`.
        event_encoder_type: `"gru"` or `"transformer"`.
        delta_embed_dim: Embedding size for `delta_tick`.
        lane_embed_dim: Embedding size for `lane_mask`.
        event_hidden_dim: Output width of event encoder.
        audio_hidden_dim: Internal channel width in audio encoder.
        audio_out_dim: Output width of audio encoder.
        stats_hidden_dim: Hidden width of grid stats MLP.
        stats_out_dim: Output width of grid stats MLP.
        fusion_hidden_dim: Hidden width after modality concatenation.
    """

    def __init__(
        self,
        delta_vocab_size: int | None = None,
        grid_stats_dim: int | None = None,
        event_encoder_type: str | None = None,
        delta_embed_dim: int | None = None,
        lane_embed_dim: int | None = None,
        event_hidden_dim: int | None = None,
        audio_hidden_dim: int | None = None,
        audio_out_dim: int | None = None,
        stats_hidden_dim: int | None = None,
        stats_out_dim: int | None = None,
        fusion_hidden_dim: int | None = None,
        in_mels: int | None = None,
    ) -> None:
        super().__init__()

        if delta_vocab_size is None:
            delta_vocab_size = int(get_config_value("model.delta_vocab_size", 256))
        if grid_stats_dim is None:
            grid_stats_dim = int(get_config_value("model.grid_stats_dim", 10))
        if event_encoder_type is None:
            event_encoder_type = str(get_config_value("model.event_encoder_type", "gru"))
        if delta_embed_dim is None:
            delta_embed_dim = int(get_config_value("model.delta_embed_dim", 64))
        if lane_embed_dim is None:
            lane_embed_dim = int(get_config_value("model.lane_embed_dim", 32))
        if event_hidden_dim is None:
            event_hidden_dim = int(get_config_value("model.event_hidden_dim", 128))
        if audio_hidden_dim is None:
            audio_hidden_dim = int(get_config_value("model.audio_hidden_dim", 128))
        if audio_out_dim is None:
            audio_out_dim = int(get_config_value("model.audio_out_dim", 128))
        if stats_hidden_dim is None:
            stats_hidden_dim = int(get_config_value("model.stats_hidden_dim", 64))
        if stats_out_dim is None:
            stats_out_dim = int(get_config_value("model.stats_out_dim", 64))
        if fusion_hidden_dim is None:
            fusion_hidden_dim = int(get_config_value("model.fusion_hidden_dim", 256))
        if in_mels is None:
            in_mels = int(get_config_value("model.in_mels", 80))

        transformer_heads = int(get_config_value("model.transformer_heads", 4))
        transformer_layers = int(get_config_value("model.transformer_layers", 2))
        dropout = float(get_config_value("model.dropout", 0.1))

        self.delta_vocab_size = delta_vocab_size
        self.grid_stats_dim = grid_stats_dim

        self.event_encoder = EventEncoder(
            delta_vocab_size=delta_vocab_size,
            delta_embed_dim=delta_embed_dim,
            lane_embed_dim=lane_embed_dim,
            hidden_dim=event_hidden_dim,
            encoder_type=event_encoder_type,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            dropout=dropout,
        )
        self.audio_encoder = AudioEncoder(
            in_mels=in_mels,
            hidden_dim=audio_hidden_dim,
            out_dim=audio_out_dim,
        )
        self.grid_encoder = GridStatsEncoder(
            in_dim=grid_stats_dim,
            hidden_dim=stats_hidden_dim,
            out_dim=stats_out_dim,
        )

        fusion_in_dim = self.event_encoder.output_dim + audio_out_dim + stats_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.ReLU(),
        )

        self.delta_head = nn.Linear(fusion_hidden_dim, delta_vocab_size)
        self.lane_head = nn.Linear(fusion_hidden_dim, 128)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
        x: Input dict with required keys:
                - `past_delta_id`: LongTensor `[B, L]`
                - `past_lane_mask_id`: LongTensor `[B, L]`
                - `grid_stats`: FloatTensor `[B, G]`
                - `audio_window`: FloatTensor `[B, 80, T]`

        Returns:
            Dict with logits:
                - `delta_tick_logits`: FloatTensor `[B, delta_vocab_size]`
                - `lane_mask_logits`: FloatTensor `[B, 128]`

        Important notes:
            - This method does not apply softmax.
            - Use `compute_next_event_loss` or `torch.argmax` downstream.
        """
        past_delta = x.get("past_delta_id")
        if past_delta is None:
            # Backward compatibility with older sample key name.
            past_delta = x["past_delta_tick"]
        past_lane = x.get("past_lane_mask_id")
        if past_lane is None:
            # Backward compatibility with older sample key name.
            past_lane = x["past_lane_mask"]

        event_vec = self.event_encoder(past_delta, past_lane)
        audio_vec = self.audio_encoder(x["audio_window"])
        stats_vec = self.grid_encoder(x["grid_stats"])

        fused = self.fusion(torch.cat([event_vec, audio_vec, stats_vec], dim=-1))
        return {
            "delta_tick_logits": self.delta_head(fused),
            "lane_mask_logits": self.lane_head(fused),
        }


def compute_next_event_loss(
    logits: Dict[str, torch.Tensor],
    target_delta_tick: torch.LongTensor,
    target_lane_mask: torch.LongTensor,
    class_weights_delta: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute dual-head cross-entropy loss.

    Args:
        logits: Dict returned by `NextEventPredictor.forward`:
            - `delta_tick_logits`: FloatTensor `[B, D_delta]`
            - `lane_mask_logits`: FloatTensor `[B, 128]`
        target_delta_tick: LongTensor `[B]`, class ids in `[0, D_delta - 1]`.
        target_lane_mask: LongTensor `[B]`, class ids in `[0, 127]`.
        class_weights_delta: Optional class weights for `delta_tick` head.

    Returns:
        Dict[str, Tensor] containing:
            - `total_loss`: scalar tensor
            - `delta_loss`: scalar tensor
            - `lane_loss`: scalar tensor

    Important notes:
        - `total_loss = delta_loss + lane_loss`.
        - No label smoothing or auxiliary terms are applied here.
    """
    delta_loss = F.cross_entropy(
        logits["delta_tick_logits"],
        target_delta_tick,
        weight=class_weights_delta,
    )
    lane_loss = F.cross_entropy(logits["lane_mask_logits"], target_lane_mask)
    total_loss = delta_loss + lane_loss
    return {
        "total_loss": total_loss,
        "delta_loss": delta_loss,
        "lane_loss": lane_loss,
    }


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    context_len = 32
    audio_t = 64
    grid_stats_dim = 10
    delta_vocab_size = 256

    model = NextEventPredictor(
        delta_vocab_size=delta_vocab_size,
        grid_stats_dim=grid_stats_dim,
        event_encoder_type="gru",
    )

    x_dummy = {
        "past_delta_id": torch.randint(0, delta_vocab_size, (batch_size, context_len), dtype=torch.long),
        "past_lane_mask_id": torch.randint(0, 128, (batch_size, context_len), dtype=torch.long),
        "grid_stats": torch.randn(batch_size, grid_stats_dim),
        "audio_window": torch.randn(batch_size, 80, audio_t),
    }
    y_delta = torch.randint(0, delta_vocab_size, (batch_size,), dtype=torch.long)
    y_lane = torch.randint(0, 128, (batch_size,), dtype=torch.long)

    logits_dummy = model(x_dummy)
    losses = compute_next_event_loss(logits_dummy, y_delta, y_lane)

    print("delta_tick_logits:", logits_dummy["delta_tick_logits"].shape)
    print("lane_mask_logits:", logits_dummy["lane_mask_logits"].shape)
    print("total_loss:", float(losses["total_loss"]))
