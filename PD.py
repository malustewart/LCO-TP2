import numpy as np
import scipy.signal as sg
import scipy.constants as sc

def pd(
    Ein: np.array,  # Input optical signal
    B: float,       # Bandwidth of the photodetector in Hz.
    fs: float,      # Sampling frequency in Hz.
    r: float = 1.0, # Responsivity in A/W.
    T: float = 300.0, # Temperature in Kelvin.
    Rf: float = 50.0, # Load resistance in Ohms.
    i_d: float = 10e-9, # Dark current in Amperes.
    Fn: float = 0.0, # Noise figure of amplifier in dB.
):
    """
    Wrapper para simular un fotodetector PIN con ruido térmico y de disparo.

    Parameters
    ----------
    Ein : np.array
        Señal óptica a ser fotodetectada
    B : float
        Ancho de banda del detector en [Hz]
    r : float, optional
        Responsividad del detector en [A/W]
    T : float, optional
        Temperatura del detector en [K]
    Rf : float, optional
        Resistencia de carga del detector en [Ohms]
    i_d : float, optional
        Corriente oscura del fotodetector en [A]
    Fn : float, optional
        Figura de ruido del amplificador de transimpedancia, en [dB]

    Returns
    -------
    v : np.array
        La señal eléctrica detectada, en [v].
    """
    n_samples = len(Ein)

    # Definición del filtro pasa bajos del fotodetector
    sos = sg.bessel(N=5, Wn=2*B/fs, btype="low", output="sos", norm="mag")

    i_s = r * np.abs(Ein)**2

    var_sh = sc.e * (i_s + i_d)*fs
    i_sh = np.random.normal(0, np.sqrt(var_sh), n_samples)
    var_th = 2*sc.k*T*Fn*fs/Rf
    i_th = np.random.normal(0, np.sqrt(var_th), n_samples)
    
    i = i_s + i_d + i_sh + i_th

    v_out = sg.sosfiltfilt(sos, i) * Rf  # Voltage signal
    return v_out

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 20e9
    B = 5e9
    t = np.arange(0, 1e-6, 1/fs)
    f = 1e9
    P_dbm = -20
    r = 0.8
    i_d = 1e-9
    Rf = 50
    T = 300

    P = 10**(P_dbm/10) / 1000
    Ein = np.sqrt(P) * np.exp(1j*2*np.pi*f*t)

    v = pd(Ein, B=B, fs=fs, r=r, i_d=i_d, T=T, Rf=Rf)

    plt.figure()
    plt.plot(t*1e9, v)
    plt.xlim(0, 2/f*1e9)
    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [V]")
    plt.grid()
    plt.show()