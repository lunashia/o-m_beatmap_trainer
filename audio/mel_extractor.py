"""
Mel spectrogram feature extraction for audio-model training pipelines.

Purpose:
    Convert a single audio file into a log-mel spectrogram tensor.

Input/Output:
    Input: audio file path (any format decodable by `librosa.load`).
    Output: 2D NumPy array `mel` with shape [n_mels, n_frames], dtype float32.

Pipeline placement:
    Upstream input:
        Produced by dataset/indexing components that provide per-sample audio paths
        (for example, manifest readers or training data loaders).
    Downstream output:
        Consumed by feature batching/model-input stages that expect fixed mel bins
        across time, and by optional augmentation/normalization steps.
"""

import numpy as np
import librosa

from config_loader import get_config_value


def extract_mel(
    audio_path: str,
    *,
    sample_rate: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    n_mels: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Extract an 80-bin log-mel spectrogram from an audio file.

    Args:
        audio_path (str):
            Path to the source audio file. The file is resampled to 22050 Hz and
            mixed to mono during loading.

    Returns:
        np.ndarray:
            Log-mel spectrogram with shape [80, n_frames] and dtype float32.
            Axis 0 is mel-frequency bins; axis 1 is time frames.

    Notes:
        - Assumes librosa-supported decoding for the input file format.
        - Empty audio is replaced with one zero sample to keep feature extraction
          defined for edge-case inputs.
        - Output is sanitized with `np.nan_to_num` to remove NaN/+inf/-inf before
          training-time consumption.
    """
    if sample_rate is None:
        sample_rate = int(get_config_value("audio.sample_rate", 22050))
    if n_fft is None:
        n_fft = int(get_config_value("audio.n_fft", 2048))
    if hop_length is None:
        hop_length = int(get_config_value("audio.hop_length", 512))
    if n_mels is None:
        n_mels = int(get_config_value("audio.n_mels", 80))
    if fmin is None:
        fmin = float(get_config_value("audio.fmin", 30.0))
    if fmax is None:
        cfg_fmax = get_config_value("audio.fmax", None)
        fmax = None if cfg_fmax is None else float(cfg_fmax)

    # Load audio as mono at target sample rate.
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Handle empty/very short audio safely.
    if y.size == 0:
        y = np.zeros(1, dtype=np.float32)
    else:
        y = y.astype(np.float32, copy=False)

    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel_power, ref=np.max)

    # Sanitize non-finite values before returning training features.
    mel_db = np.nan_to_num(mel_db, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )

    return mel_db
