"""Simulate harder acoustic conditions.

A trauma bay differs from a quiet clinic room in two main ways:
  1. many people talk at once (overlapping, competing speech, i.e. "babble")
  2. background noise from monitors, alarms and equipment

We approximate both by adding noise to the clean consultation at a controlled
signal-to-noise ratio (SNR, in dB). Lower SNR means a noisier recording.
"""

from __future__ import annotations

import numpy as np


def _power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2)) + 1e-12


def add_noise_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale ``noise`` so that 10*log10(P_signal / P_noise) == snr_db, then add it."""
    if len(noise) < len(signal):  # loop the noise to cover the whole signal
        reps = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[: len(signal)]
    scale = np.sqrt(_power(signal) / (_power(noise) * 10 ** (snr_db / 10)))
    mixed = signal + scale * noise
    peak = np.abs(mixed).max()
    return (mixed / peak * 0.95).astype(np.float32) if peak > 0 else mixed.astype(np.float32)


def make_babble(other_audios: list[np.ndarray], length: int, n_talkers: int = 4,
                seed: int = 0) -> np.ndarray:
    """Build multi-talker babble by summing random segments of *other* consultations.

    Using different consultations as the interfering speech avoids leaking the
    target words into the noise.
    """
    rng = np.random.default_rng(seed)
    babble = np.zeros(length, dtype=np.float32)
    pool = [a for a in other_audios if len(a) > 0]
    if not pool:
        raise ValueError("need at least one other recording to build babble noise")
    for _ in range(n_talkers):
        src = pool[rng.integers(len(pool))]
        if len(src) < length:
            src = np.tile(src, int(np.ceil(length / len(src))))
        start = rng.integers(0, len(src) - length + 1)
        seg = src[start : start + length]
        babble += seg / (np.sqrt(_power(seg)) + 1e-8)  # equal loudness per talker
    return babble


def make_white_noise(length: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(length).astype(np.float32)
