import librosa
from functools import lru_cache
from pathlib import Path
import numpy as np
from src.new.config.model_config import AudioProperties
from tqdm import tqdm


def spectrogram(samples, sample_rate, fft: float, hop_length: float, window, max_freq=None):
    """Converts audio samples to spectrogram.

    Args:
        samples (ndarray): 1D array of samples.
        sample_rate (float): Sample rate.
        fft (int): FFT frame width in seconds.
        hop_length (int): Hop length in seconds.
        window (str): Window function to apply on frame.
        max_freq (int, optional): Maximal frequency to cut spectrogram values. If None do not cut. Defaults to None.

    Returns:
        ndarray: 2D array of spectrogram values.
    """
    D = librosa.amplitude_to_db(np.abs(librosa.stft(
        samples, n_fft=int(fft * sample_rate), hop_length=int(hop_length * sample_rate), window=window)))
    if max_freq is not None:
        bins_amount = int(len(D) * max_freq / (sample_rate / 2))
        D = D[:bins_amount, :]
    return D


def normalize_spectrogram(D: np.ndarray):
    """Normalize spectrogram.

    Args:
        D (ndarray): 2D array of spectrogram values.
        audio_mean (float): Mean of spectrogram values used to normalize.
        audio_std (float): Standard deviation of spectrogram values used to normalize.

    Returns:
        ndarray: 2D array of normalized spectrogram values.
    """
    D -= D.mean()
    D /= D.std()
    dmin = D.min()
    dmax = D.max()
    assert dmin < dmax
    D = (D - dmin) / (dmax - dmin)
    return D


@lru_cache
def get_wav_length(filename: Path):
    """Gets length of audio file.

    Args:
        filename (str): Path to audio file.

    Returns:
        float: Length of audio file in seconds.
    """
    return librosa.get_duration(path=filename)


def get_wav_samples(filename: Path, sample_rate: int):
    return librosa.load(
        str(filename),
        sr=sample_rate
    )


def get_normalized_spectrogram(filename: Path, audio_props: AudioProperties):
    samples, sr = get_wav_samples(filename, audio_props.sample_rate)

    # 2s chunks
    chunk_size = int(sr*2.0)

    all_spectrogram_chunks = []
    for slice in range(0, len(samples), chunk_size):
        spec = spectrogram(
            samples[slice:min(len(samples), slice+chunk_size)],
            audio_props.sample_rate,
            audio_props.spectrogram_fft,
            audio_props.spectrogram_hop_length,
            audio_props.spectrogram_window_type,
            audio_props.max_frequency,
        )
        normalized_spec = normalize_spectrogram(spec)
        all_spectrogram_chunks.append(normalized_spec)

    spec_full = np.concatenate(all_spectrogram_chunks, axis=1)
    return spec_full
