import numpy as np
import librosa


def extract_mel(audio_path: str) -> np.ndarray:
    """
    Extract a log-mel spectrogram from an audio file.

    Returns:
        mel: np.ndarray with shape [80, n_frames], dtype float32
    """
    sample_rate = 22050
    n_fft = 2048
    hop_length = 512
    n_mels = 80

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
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel_power, ref=np.max)

    # Ensure finite values and stable dtype for training.
    mel_db = np.nan_to_num(mel_db, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )

    return mel_db
