import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt
from pathlib import Path


N_SAMPLES = 500
FS_HZ = 1000
F_MAX_HZ = 19
F_FILTER_HZ = F_MAX_HZ
RANDOM_SEED = 42
DISCRETIZATION_STEPS = [2, 4, 8, 16]
QUANTIZATION_LEVELS = [4, 16, 64, 256]


def generate_signal(n=N_SAMPLES, mean=0.0, std=10.0, seed=RANDOM_SEED):
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(mean, std, n)


def plot_signal(y, x, title, xlabel, ylabel):
    width_cm = 21
    height_cm = 14
    font_size = 14
    line_width = 1

    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    ax.plot(x, y, linewidth=line_width)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    plt.title(title, fontsize=font_size)

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{title}.png", dpi=600)


def discretize_signal(signal_in, dt):
    n = len(signal_in)
    discrete_signal = np.zeros(n)
    max_i = round(n / dt)
    for i in range(0, max_i):
        idx = i * dt
        if idx < n:
            discrete_signal[idx] = signal_in[idx]
    return discrete_signal


def plot_discrete_signals_grid(signals, time, dt_values):
    width_cm = 21
    height_cm = 14
    font_size = 14
    line_width = 1

    fig, ax = plt.subplots(2, 2, figsize=(width_cm / 2.54, height_cm / 2.54))
    s = 0
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i][j].plot(time, signals[s], linewidth=line_width)
            ax[i][j].set_title(f"Dt = {dt_values[s]}", fontsize=font_size)
            s += 1

    fig.supxlabel("Час (секунди)", fontsize=font_size)
    fig.supylabel("Амплітуда сигналу", fontsize=font_size)
    fig.suptitle("Дискретизовані сигнали", fontsize=font_size)

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "discrete_signals_grid.png", dpi=600)


def plot_discrete_spectrums_grid(spectrums, freqs_shifted, dt_values):
    width_cm = 21
    height_cm = 14
    font_size = 14
    line_width = 1

    fig, ax = plt.subplots(2, 2, figsize=(width_cm / 2.54, height_cm / 2.54))
    s = 0
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i][j].plot(freqs_shifted, spectrums[s], linewidth=line_width)
            ax[i][j].set_title(f"Dt = {dt_values[s]}", fontsize=font_size)
            s += 1

    fig.supxlabel("Частота (Гц)", fontsize=font_size)
    fig.supylabel("Амплітуда спектра", fontsize=font_size)
    fig.suptitle("Спектри дискретизованих сигналів", fontsize=font_size)

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "discrete_spectrums_grid.png", dpi=600)


def plot_restored_signals_grid(signals, time, dt_values):
    width_cm = 21
    height_cm = 14
    font_size = 14
    line_width = 1

    fig, ax = plt.subplots(2, 2, figsize=(width_cm / 2.54, height_cm / 2.54))
    s = 0
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i][j].plot(time, signals[s], linewidth=line_width)
            ax[i][j].set_title(f"Dt = {dt_values[s]}", fontsize=font_size)
            s += 1

    fig.supxlabel("Час (секунди)", fontsize=font_size)
    fig.supylabel("Амплітуда сигналу", fontsize=font_size)
    fig.suptitle("Відновлені сигнали (ФНЧ)", fontsize=font_size)

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "restored_signals_grid.png", dpi=600)


def plot_metric(dt_values, values, title, ylabel, filename):
    width_cm = 21
    height_cm = 14
    font_size = 14
    line_width = 1

    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    ax.plot(dt_values, values, linewidth=line_width, marker="o")
    ax.set_xlabel("Крок дискретизації Dt", fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.grid(True, linewidth=0.4, alpha=0.6)

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=600)


def plot_metric_levels(levels, values, title, ylabel, filename):
    width_cm = 21
    height_cm = 14
    font_size = 14
    line_width = 1

    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    ax.plot(levels, values, linewidth=line_width, marker="o")
    ax.set_xlabel("Кількість рівнів квантування M", fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.grid(True, linewidth=0.4, alpha=0.6)

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=600)


def plot_quantization_table(quantize_levels, quantize_bits, m_levels):
    fig, ax = plt.subplots(figsize=(14 / 2.54, m_levels / 2.54))
    table_data = np.c_[np.round(quantize_levels[:m_levels], 6), quantize_bits[:m_levels]]
    table = ax.table(
        cellText=table_data,
        colLabels=["Значення сигналу", "Кодова послідовність"],
        loc="center",
    )
    table.set_fontsize(14)
    table.scale(1, 2)
    ax.axis("off")

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"Таблиця квантування для {m_levels} рівнів.png", dpi=600)


def plot_bit_sequence(bits, m_levels):
    width_cm = 21
    height_cm = 14
    font_size = 14
    line_width = 0.1

    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    x = np.arange(0, len(bits))
    ax.step(x, bits, linewidth=line_width)
    ax.set_xlabel("Відліки", fontsize=font_size)
    ax.set_ylabel("Бітова послідовність", fontsize=font_size)
    ax.set_title(f"Кодова послідовність для M = {m_levels}", fontsize=font_size)

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"Кодова послідовність для M = {m_levels}.png", dpi=600)


def plot_quantized_signals_grid(signals, time, levels):
    width_cm = 21
    height_cm = 14
    font_size = 14
    line_width = 1

    fig, ax = plt.subplots(2, 2, figsize=(width_cm / 2.54, height_cm / 2.54))
    s = 0
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i][j].plot(time, signals[s], linewidth=line_width)
            ax[i][j].set_title(f"M = {levels[s]}", fontsize=font_size)
            s += 1

    fig.supxlabel("Час (секунди)", fontsize=font_size)
    fig.supylabel("Амплітуда сигналу", fontsize=font_size)
    fig.suptitle("Цифрові сигнали для різних рівнів квантування", fontsize=font_size)

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "quantized_signals_grid.png", dpi=600)


if __name__ == "__main__":
    time = np.arange(N_SAMPLES) / FS_HZ
    random_signal = generate_signal()
    w = F_FILTER_HZ / (FS_HZ / 2)
    sos = signal.butter(3, w, "low", output="sos")
    filtered_signal = signal.sosfiltfilt(sos, random_signal)

    spectrum = fft.fft(filtered_signal)
    spectrum_shifted = np.abs(fft.fftshift(spectrum))
    freqs = fft.fftfreq(N_SAMPLES, 1 / FS_HZ)
    freqs_shifted = fft.fftshift(freqs)

    plot_signal(
        filtered_signal,
        time,
        title=f"Сигнал з максимальною частотою F_max = {F_MAX_HZ} Гц",
        xlabel="Час (секунди)",
        ylabel="Амплітуда сигналу",
    )

    plot_signal(
        spectrum_shifted,
        freqs_shifted,
        title=f"Спектр сигналу з максимальною частотою F_max = {F_MAX_HZ} Гц",
        xlabel="Частота (Гц)",
        ylabel="Амплітуда спектра",
    )

    discrete_signals = []
    discrete_spectrums = []
    restored_signals = []
    variances = []
    snrs = []
    w_restore = F_FILTER_HZ / (FS_HZ / 2)
    sos_restore = signal.butter(3, w_restore, "low", output="sos")
    variance_signal = np.var(filtered_signal)
    for dt in DISCRETIZATION_STEPS:
        discrete_signal = discretize_signal(filtered_signal, dt)
        discrete_signals += [list(discrete_signal)]
        spectrum_discrete = fft.fft(discrete_signal)
        spectrum_discrete_shifted = np.abs(fft.fftshift(spectrum_discrete))
        discrete_spectrums += [list(spectrum_discrete_shifted)]
        restored_signal = signal.sosfiltfilt(sos_restore, discrete_signal)
        restored_signals += [list(restored_signal)]
        error = restored_signal - filtered_signal
        variance_error = np.var(error)
        variances += [variance_error]
        snrs += [variance_signal / variance_error if variance_error != 0 else np.inf]

    plot_discrete_signals_grid(discrete_signals, time, DISCRETIZATION_STEPS)
    plot_discrete_spectrums_grid(discrete_spectrums, freqs_shifted, DISCRETIZATION_STEPS)
    plot_restored_signals_grid(restored_signals, time, DISCRETIZATION_STEPS)
    plot_metric(DISCRETIZATION_STEPS, variances, "Дисперсія похибки", "Дисперсія", "variance_vs_dt.png")
    plot_metric(DISCRETIZATION_STEPS, snrs, "Співвідношення сигнал/шум", "SNR", "snr_vs_dt.png")

    quantized_signals = []
    quant_variances = []
    quant_snrs = []
    variance_signal = np.var(filtered_signal)
    for m_levels in QUANTIZATION_LEVELS:
        min_signal = np.min(filtered_signal)
        max_signal = np.max(filtered_signal)
        delta = (max_signal - min_signal) / (m_levels - 1)

        quantize_signal = delta * np.round(filtered_signal / delta)
        quantized_signals.append(list(quantize_signal))

        levels = np.arange(np.min(quantize_signal), np.max(quantize_signal) + delta / 2, delta)
        if len(levels) < m_levels:
            levels = np.linspace(np.min(quantize_signal), np.max(quantize_signal), m_levels)

        bits_per_sample = int(np.log2(m_levels))
        quantize_bit = [format(bits, "0" + str(bits_per_sample) + "b") for bits in range(m_levels)]

        plot_quantization_table(levels, quantize_bit, m_levels)

        level_indices = np.round((quantize_signal - levels[0]) / delta).astype(int)
        level_indices = np.clip(level_indices, 0, m_levels - 1)
        bit_strings = [quantize_bit[index] for index in level_indices]
        bits = [int(item) for item in list("".join(bit_strings))]
        plot_bit_sequence(bits, m_levels)

        error = quantize_signal - filtered_signal
        variance_error = np.var(error)
        quant_variances.append(variance_error)
        quant_snrs.append(variance_signal / variance_error if variance_error != 0 else np.inf)

    plot_quantized_signals_grid(quantized_signals, time, QUANTIZATION_LEVELS)
    plot_metric_levels(
        QUANTIZATION_LEVELS,
        quant_variances,
        "Залежність дисперсії від кількості рівнів квантування",
        "Дисперсія",
        "variance_vs_levels.png",
    )
    plot_metric_levels(
        QUANTIZATION_LEVELS,
        quant_snrs,
        "Залежність SNR від кількості рівнів квантування",
        "SNR",
        "snr_vs_levels.png",
    )
