import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt
from pathlib import Path


N_SAMPLES = 500
FS_HZ = 1000
F_MAX_HZ = 19
F_FILTER_HZ = 26
RANDOM_SEED = 42
DISCRETIZATION_STEPS = [2, 4, 8, 16]


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
