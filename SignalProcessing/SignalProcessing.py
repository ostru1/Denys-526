import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt
from pathlib import Path


N_SAMPLES = 500
FS_HZ = 1000
F_MAX_HZ = 19
RANDOM_SEED = 42


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


if __name__ == "__main__":
    time = np.arange(N_SAMPLES) / FS_HZ
    random_signal = generate_signal()
    w = F_MAX_HZ / (FS_HZ / 2)
    sos = signal.butter(3, w, "low", output="sos")
    filtered_signal = signal.sosfiltfilt(sos, random_signal)

    spectrum = fft.fft(filtered_signal)
    spectrum_shifted = np.abs(fft.fftshift(spectrum))
    freqs = fft.fftfreq(N_SAMPLES, 1 / FS_HZ)
    freqs_shifted = fft.fftshift(freqs)

    plot_signal(
        filtered_signal,
        time,
        title=f"Сигнал с максимальной частотой F_max = {F_MAX_HZ} Гц",
        xlabel="Час (секунды)",
        ylabel="Амплитуда сигнала",
    )

    plot_signal(
        spectrum_shifted,
        freqs_shifted,
        title=f"Спектр сигнала с максимальной частотой F_max = {F_MAX_HZ} Гц",
        xlabel="Частота (Гц)",
        ylabel="Амплитуда спектра",
    )
