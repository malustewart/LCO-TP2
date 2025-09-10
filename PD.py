import numpy as np
import scipy.signal as sg

def pd(
    Ein: np.array,  # Input optical signal
    B: float,       # Bandwidth of the photodetector in Hz.
    fs: float,      # Sampling frequency in Hz.
    r: float = 1.0, # Responsivity in A/W.
    T: float = 300.0, # Temperature in Kelvin.
    Rf: float = 50.0, # Load resistance in Ohms.
    id: float = 10e-9, # Dark current in Amperes.
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
    id : float, optional
        Corriente oscura del fotodetector en [A]
    Fn : float, optional
        Figura de ruido del amplificador de transimpedancia, en [dB]

    Returns
    -------
    v : np.array
        La señal eléctrica detectada, en [v].
    """
    # Definición del filtro pasa bajos del fotodetector
    sos = sg.bessel(N=5, Wn=2*B/fs, btype="low", output="sos", norm="mag")

    i_s = ... # Signal current

    var_sh = ...  # Shot noise variance
    n_sh = ...  # Shot noise signal
    var_th = ...  # Thermal noise variance
    n_th = ...  # Thermal noise signal
    
    i = ...    # Total current
    v_out = sg.sosfiltfilt(sos, i) * Rf  # Voltage signal
    return v_out

