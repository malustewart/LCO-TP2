from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks, windows
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def get_frequency(A, fs):
    a = np.abs(fft(A))
    f = fftfreq(len(A), 1/fs)
    
    pos_freqs = f >=0
    f = f[pos_freqs]
    a = a[pos_freqs]

    prom = 0.5 * np.max(a)
    peaks = find_peaks(a, prominence=prom)  # prominence evita detectar falsos picos por ruido en el espectro

    return f[peaks[0][0]]

def create_gaussian_pulse(std, length, fs, amplitude):
    """
        std: std deviation in ps
        length: pulse length in ps
        fs: sampling frequency, in THz
    """
    n_samples = int(length * fs)
    std_in_samples = std * fs
    length_in_s = length

    t = np.linspace(0, length_in_s, n_samples)

    pulse = windows.gaussian(n_samples, std_in_samples)

    return amplitude * pulse, t

def plot_signals(signals, fs, labels=None):
    """
        All elements in signal array must be of the same lengt
    """
    n = len(signals[0])
    t = np.arange(n) / fs  # in ps

    ###### in time (Real part) #######
    plt.figure(figsize=(10,4))
    # Plot real part of signal
    for i, sig in enumerate(signals):
        label = labels[i] if labels is not None else f"Signal {i+1}"
        plt.plot(t, np.abs(sig), label=label)
    plt.xlabel("Time [ps]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    ###### in frequency #######
    f = np.fft.fftfreq(n, d=1/fs) / 1e12 # Thz
    f_shifted = np.fft.fftshift(f)
    fft_signals = [np.fft.fftshift(np.fft.fft(sig)) for sig in signals]

    # Create figure and axes objects
    # plt.figure()
    fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot amplitude
    for i, fft_sig in enumerate(fft_signals):
        label = labels[i] if labels is not None else f"Signal {i+1}"
        ax_amp.plot(f_shifted, np.abs(fft_sig), label=label)
        ax_phase.plot(f_shifted, np.angle(fft_sig), label=label)
    ax_amp.set_title("Amplitude Spectrum")
    ax_amp.set_ylabel("Amplitude")
    ax_amp.grid(True)
    ax_amp.legend()
    ax_phase.set_title("Phase Spectrum")
    ax_phase.set_xlabel("Frequency [THz]")
    ax_phase.set_ylabel("Phase [rad]")
    ax_phase.grid(True)
    ax_phase.legend()

    plt.tight_layout()

def plot_signal_timemap(signal_evolution, t, z, labels=None):
    plt.figure()
    plt.imshow(np.abs(signal_evolution)**2, aspect="auto",
            extent=[t[0], t[-1], z[-1], z[0]], cmap="inferno", norm=LogNorm(vmin=0.00000000000001))
    plt.xlabel("Time [ps]")
    plt.ylabel("Distance [km]")
    plt.title("Pulse evolution in fiber")
    plt.colorbar(label="log10(|A|)")

def show_plots():
    plt.show()