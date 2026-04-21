# -*- coding: utf-8 -*-
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
import soundfile as sf
from scipy.signal import convolve, resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.restoration import (
    cycle_spin,
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
NAME_GAUSSIAN_WAV = SOUNDS_DIR / "Filtered_Gaussian_Filter.wav"

PLOT_PRACTICAL_2 = SOUNDS_DIR / "Plot_Practical_2_Comparison.png"
PLOT_INVARIANCE = SOUNDS_DIR / "Plot_J_Invariance.png"
PLOT_TOTAL_VARIATION = SOUNDS_DIR / "Plot_Total_Variation.png"
PLOT_BILATERAL = SOUNDS_DIR / "Plot_Bilateral.png"
PLOT_WAVELET = SOUNDS_DIR / "Plot_Wavelet.png"
PLOT_SHIFTED_WAVELET = SOUNDS_DIR / "Plot_Shifted_Wavelet.png"
PLOT_GAUSSIAN = SOUNDS_DIR / "Plot_Gaussian_Filter.png"
PLOT_METRICS = SOUNDS_DIR / "Plot_Metrics_Table.png"


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
    plt.xlabel("Time, ms")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_shifted_wavelet_plot(time_axis, original, signals, output_path: Path):
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, original, label=f"Оригінал (fs={SAMPLE_RATE} Гц)")
    plt.plot(time_axis, signals[0], label="Wavelet Shifted: no shift")
    plt.plot(time_axis, signals[1], label="Wavelet Shifted: 1x2")
    plt.plot(time_axis, signals[2], label="Wavelet Shifted: 1x4")
    plt.plot(time_axis, signals[3], label="Wavelet Shifted: 1x6")
    plt.title("Порівняння сигналів у часовій області, вейвлет-фільтр, модифікований")
    plt.xlabel("Час (мс)")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_gaussian_plot(time_axis, original, filtered_signal, output_path: Path):
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, original, label=f"Оригінал (fs={SAMPLE_RATE} Гц)")
    plt.plot(time_axis, filtered_signal, label="Gaussian Filter")
    plt.title("Порівняння сигналів у часовій області, фільтр Гаусса")
    plt.xlabel("Час (мс)")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_practical_2_plot():
    original, fs_original = load_mono_signal(NAME_ORIGINAL_WAV)
    resampled, fs_resampled = load_mono_signal(NAME_RESAMPLED_WAV)
    filtered, fs_filtered = load_mono_signal(NAME_FILTERED_WAV)

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(original)) / fs_original * 1000, original, label=f"Original (fs={fs_original} Hz)")
    plt.plot(
        np.arange(len(resampled)) / fs_resampled * 1000,
        resampled,
        label=f"Resampled (fs={fs_resampled} Hz)",
    )
    plt.plot(
        np.arange(len(filtered)) / fs_filtered * 1000,
        filtered,
        label=f"Filtered (LPF 4000 Hz)",
    )
    plt.title("Comparison of Signals in Time Domain")
    plt.xlabel("Time, ms")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PRACTICAL_2, dpi=200)
    plt.close()


def wavelet_denoiser(signal, level=5, mode="hard", wavelet="db4"):
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


def gaussian_kernel(size, sigma):
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


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
    time_ms = np.arange(len(data)) / fs_original * 1000
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

    sf.write(NAME_INVARIANCE_WAV, invariance, fs_original)
    sf.write(NAME_TOTAL_VARIATION_WAV, total_variation, fs_original)
    sf.write(NAME_BILATERAL_WAV, bilateral, fs_original)
    sf.write(NAME_WAVELET_WAV, wavelet, fs_original)

    build_practical_2_plot()
    save_plot(time_ms, data, invariance, "J-Invariance", "J-Invariance", PLOT_INVARIANCE)
    save_plot(
        time_ms,
        data,
        total_variation,
        "Total Variation",
        "Total Variation",
        PLOT_TOTAL_VARIATION,
    )
    save_plot(time_ms, data, bilateral, "Bilateral", "Bilateral", PLOT_BILATERAL)
    save_plot(time_ms, data, wavelet, "Wavelet", "Wavelet", PLOT_WAVELET)


def wavelet_shifted_filter():
    if not NAME_ORIGINAL_WAV.exists():
        raise FileNotFoundError(
            f"Missing source file: {NAME_ORIGINAL_WAV}. Run practical work 2 recording first."
        )

    SOUNDS_DIR.mkdir(exist_ok=True)

    data, fs_original = load_mono_signal(NAME_ORIGINAL_WAV)
    time_ms = np.arange(len(data)) / fs_original * 1000

    max_shifts = [0, 1, 3, 5]
    signals = []
    for index, shift in enumerate(max_shifts):
        sig_filtered = cycle_spin(
            data,
            func=wavelet_denoiser,
            max_shifts=shift,
            shift_steps=5,
            workers=1,
        ).astype(np.float32)
        sf.write(SOUNDS_DIR / f"Filtered_Shifted_Wavelet_{index}.wav", sig_filtered, fs_original)
        signals.append(sig_filtered)

    kernel = gaussian_kernel(size=11, sigma=2)
    filtered_signal = convolve(data, kernel, mode="same").astype(np.float32)
    sf.write(NAME_GAUSSIAN_WAV, filtered_signal, fs_original)

    save_shifted_wavelet_plot(time_ms, data, signals, PLOT_SHIFTED_WAVELET)
    save_gaussian_plot(time_ms, data, filtered_signal, PLOT_GAUSSIAN)


def calculate_metrics(reference_signal, processed_signal):
    mse = mean_squared_error(reference_signal, processed_signal)
    mae = mean_absolute_error(reference_signal, processed_signal)
    rmse = np.sqrt(mse)
    r2 = r2_score(reference_signal, processed_signal)
    variance = np.var(reference_signal - processed_signal)
    return mse, mae, rmse, r2, variance


def to_scientific_pretty(x, precision=2):
    superscripts = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    mantissa, exponent = f"{x:.{precision}e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa} · 10{str(int(exponent)).translate(superscripts)}"


def save_metrics_table(results, row_labels, headers, output_path: Path):
    n_rows = len(row_labels)
    n_cols = len(headers)
    fig, ax = plt.subplots(figsize=(n_cols * 2.8, max(1, n_rows) * 0.4 + 1.0))
    ax.axis("off")
    table = ax.table(
        cellText=results,
        rowLabels=row_labels,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0.08, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def prettify_filter_name(sound_path):
    type_filter = sound_path.replace("./Sounds/Filtered_", "")
    type_filter = type_filter.replace(".wav", "")
    type_filter = type_filter.replace("_", " ")
    if type_filter == "4000[Hz] 2[byte]":
        return "Linear Filter 4 kHz"
    return type_filter


if __name__ == "__main__":
    # sound_filter()
    # wavelet_shifted_filter()
    results = []
    row = []
    headers = ["MSE", "MAE", "RMSE", "R2", "D"]

    data_original, fs_original = load_mono_signal(NAME_ORIGINAL_WAV)
    wav_files = sorted(glob.glob("./Sounds/*.wav"))
    original_sound = f"./{NAME_ORIGINAL_WAV.as_posix()}"
    resampled_sound = f"./{NAME_RESAMPLED_WAV.as_posix()}"

    for sounds in wav_files:
        sounds = sounds.replace("\\", "/")

        if sounds == original_sound:
            continue

        if sounds == resampled_sound:
            row.append("Resample 4 kHz")
            data, fs = load_mono_signal(Path(sounds))
            data = resample(data, len(data_original)).astype(np.float32)
        else:
            row.append(prettify_filter_name(sounds))
            data, fs = load_mono_signal(Path(sounds))
            if len(data) != len(data_original):
                data = resample(data, len(data_original)).astype(np.float32)

        mse, mae, rmse, r2, variance = calculate_metrics(data_original, data)
        results.append(
            [
                to_scientific_pretty(mse),
                to_scientific_pretty(mae),
                to_scientific_pretty(rmse),
                round(r2, 2),
                to_scientific_pretty(variance),
            ]
        )

    save_metrics_table(results, row, headers, PLOT_METRICS)
