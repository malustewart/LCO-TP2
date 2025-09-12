from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import find_peaks, windows, welch
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

def create_nrz_signal(bit_rate, length, fs, amplitude):
    """
        bit_rate: Gbps
        length: signal length in ps
        fs: THz
    """
    n_samples = int(length * fs)
    samples_per_bit = int(fs / (bit_rate * 1e-3))
    n_bits = int(n_samples / samples_per_bit) + 1
    bits = np.random.randint(0, 2, n_bits)
    signal = np.repeat(bits, samples_per_bit)[:n_samples]

    return amplitude * signal, np.linspace(0, length, n_samples) / fs

def plot_signals(signals, fs, labels=None):
    """
        All elements in signal array must be of the same lengt
    """
    n = len(signals[0])
    t = np.arange(n) / fs  # in ps

    ###### in time (absolute) #######
    plt.figure(figsize=(10,4))
    for i, sig in enumerate(signals):
        label = labels[i] if labels is not None else f"Signal {i+1}"
        plt.plot(t, np.abs(sig), label=label)
    plt.xlabel("Time [ps]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()

    ###### in frequency #######
    f = np.fft.fftfreq(n, d=1/fs)
    f_shifted = np.fft.fftshift(f)
    fft_signals = [np.fft.fftshift(np.fft.fft(sig)) for sig in signals]

    # Create figure and axes objects
    # plt.figure()
    fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot amplitude
    for i, fft_sig in enumerate(fft_signals):
        label = labels[i] if labels is not None else f"Signal {i+1}"
        ax_amp.plot(f_shifted, np.abs(fft_sig), label=label)
        ax_phase.plot(f_shifted, np.unwrap(np.angle(fft_sig)), label=label)
    ax_amp.set_title("Amplitude Spectrum")
    ax_amp.set_ylabel("Amplitude")
    ax_amp.grid(True)
    ax_phase.set_title("Phase Spectrum")
    ax_phase.set_xlabel("Frequency [THz]")
    ax_phase.set_ylabel("Phase [rad]")
    ax_phase.grid(True)
    ax_phase.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)

    plt.tight_layout()

def plot_MZM(E_out, E_in, u, Vpi, Vbias):
    trans = E_out / E_in
    plt.figure(figsize=(10, 4))
    plt.plot(u, np.abs(trans), label="E_out / E_in")
    plt.plot(u, np.abs(trans**2), label="P_out / P_in")
    plt.xlabel("u [V]")
    plt.title("MZM Modulation")
    plt.grid(True)
    # Plot vertical line where u + Vbias = Vpi
    u_min = np.min(u + Vbias)
    u_max = np.max(u + Vbias)
    n_min = int(np.floor(u_min / Vpi))
    n_max = int(np.ceil(u_max / Vpi))
    for n in range(n_min, n_max + 1):
        x = n * Vpi - Vbias
        plt.axvline(x, color='r', linestyle='--', alpha=0.7, label=f"n·Vpi - Vbias" if n == 1 else None)    
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()

def plot_PSD(PSDs, labels=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, PSD in enumerate(PSDs):
        label = labels[i] if labels is not None else f"Signal {i+1}"
        f = fftshift(PSD[0])
        psd = fftshift(PSD[1])
        ax.semilogy(f, np.abs(psd) + 1e-20, label=label)  # log scale, avoid log(0)
    ax.set_title("Power Spectral Density")
    ax.set_ylabel("|PSD|")
    ax.set_xlabel("Frequency [THz]")
    ax.grid(True, which="both")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig.tight_layout()

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