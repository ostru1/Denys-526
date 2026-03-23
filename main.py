# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
import soundfile as sf
from skimage.restoration import (
    denoise_bilateral,
    denoise_invariant,
    denoise_tv_chambolle,
    denoise_wavelet,
)

SAMPLE_RATE = 44100
SOUNDS_DIR = Path("./Sounds")

NAME_ORIGINAL_WAV = SOUNDS_DIR / "Sound_44100[Hz]_2[byte].wav"
NAME_RESAMPLED_WAV = SOUNDS_DIR / "Sound_4000[Hz]_2[byte].wav"
NAME_FILTERED_WAV = SOUNDS_DIR / "Filtered_4000[Hz]_2[byte].wav"

NAME_INVARIANCE_WAV = SOUNDS_DIR / "Filtered_Invariance.wav"
NAME_TOTAL_VARIATION_WAV = SOUNDS_DIR / "Filtered_Total_Variation.wav"
NAME_BILATERAL_WAV = SOUNDS_DIR / "Filtered_Bilateral.wav"
NAME_WAVELET_WAV = SOUNDS_DIR / "Filtered_Wavelet.wav"

PLOT_PRACTICAL_2 = SOUNDS_DIR / "Plot_Practical_2_Comparison.png"
PLOT_INVARIANCE = SOUNDS_DIR / "Plot_J_Invariance.png"
PLOT_TOTAL_VARIATION = SOUNDS_DIR / "Plot_Total_Variation.png"
PLOT_BILATERAL = SOUNDS_DIR / "Plot_Bilateral.png"
PLOT_WAVELET = SOUNDS_DIR / "Plot_Wavelet.png"


def load_mono_signal(path: Path):
    data, sample_rate = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    return data.astype(np.float32), sample_rate


def save_plot(time_axis, original, processed, title, processed_label, output_path: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, original, "b-", label="Original Clean Signal")
    plt.plot(time_axis, processed, "g-", linewidth=2, label=processed_label)
    plt.title(title)
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_practical_2_plot():
    original, fs_original = load_mono_signal(NAME_ORIGINAL_WAV)
    resampled, fs_resampled = load_mono_signal(NAME_RESAMPLED_WAV)
    filtered, fs_filtered = load_mono_signal(NAME_FILTERED_WAV)

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(original)) / fs_original, original, label=f"Original (fs={fs_original} Hz)")
    plt.plot(np.arange(len(resampled)) / fs_resampled, resampled, label=f"Resampled (fs={fs_resampled} Hz)")
    plt.plot(np.arange(len(filtered)) / fs_filtered, filtered, label=f"Filtered (LPF 4000 Hz)")
    plt.title("Comparison of Signals in Time Domain")
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PRACTICAL_2, dpi=200)
    plt.close()


def wavelet_denoiser(signal, level, mode, wavelet):
    """
    Denoise a 1D signal using Discrete Wavelet Transform thresholding.

    Args:
        signal (np.ndarray): The input signal.
        wavelet (str): The name of the mother wavelet.
        level (int): The level of decomposition.
        mode (str): Thresholding mode, "soft" or "hard".

    Returns:
        np.ndarray: The denoised signal.
    """
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(signal), wavelet_obj.dec_len)
    decomposition_level = max(1, min(level, max_level))
    coeffs = pywt.wavedec(signal, wavelet_obj, level=decomposition_level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(coef, threshold, mode=mode) for coef in coeffs[1:]
    ]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet_obj)
    return denoised_signal[: len(signal)].astype(np.float32)


def invarince_denoiser(image, **kwargs):
    signal = np.asarray(image, dtype=np.float32).reshape(-1)
    denoised = denoise_wavelet(
        signal,
        sigma=0.5,
        wavelet="db4",
        mode="soft",
        channel_axis=None,
    )
    return denoised.reshape(np.asarray(image).shape)


def sound_filter():
    if not NAME_ORIGINAL_WAV.exists():
        raise FileNotFoundError(
            f"Missing source file: {NAME_ORIGINAL_WAV}. Run practical work 2 recording first."
        )

    SOUNDS_DIR.mkdir(exist_ok=True)

    data, fs_original = load_mono_signal(NAME_ORIGINAL_WAV)
    time = np.arange(len(data)) / fs_original
    data_2d = data.reshape(1, -1)

    invariance = denoise_invariant(
        data_2d, denoise_function=invarince_denoiser
    ).flatten().astype(np.float32)
    total_variation = denoise_tv_chambolle(
        data_2d, weight=0.1, channel_axis=None
    ).flatten().astype(np.float32)
    bilateral = denoise_bilateral(
        data_2d, sigma_color=0.05, sigma_spatial=15, channel_axis=None
    ).flatten().astype(np.float32)
    wavelet = wavelet_denoiser(data, level=5, mode="soft", wavelet="db4")

    sf.write(NAME_INVARIANCE_WAV, invariance, SAMPLE_RATE)
    sf.write(NAME_TOTAL_VARIATION_WAV, total_variation, SAMPLE_RATE)
    sf.write(NAME_BILATERAL_WAV, bilateral, SAMPLE_RATE)
    sf.write(NAME_WAVELET_WAV, wavelet, SAMPLE_RATE)

    build_practical_2_plot()
    save_plot(time, data, invariance, "J-Invariance", "J-Invariance", PLOT_INVARIANCE)
    save_plot(
        time,
        data,
        total_variation,
        "Total Variation",
        "Total Variation",
        PLOT_TOTAL_VARIATION,
    )
    save_plot(time, data, bilateral, "Bilateral", "Bilateral", PLOT_BILATERAL)
    save_plot(time, data, wavelet, "Wavelet", "Wavelet", PLOT_WAVELET)


if __name__ == "__main__":
    sound_filter()
