import numpy as np
import scipy.signal as sg
import scipy.constants as sc
from dataclasses import dataclass


@dataclass
class PdSystem:
    Ein: np.array       # Señal óptica a ser fotodetectada
    B: float            # Ancho de banda del detector en [Hz]
    fs: float           # Sampling frequency in Hz.
    r: float = 1.0      # Responsividad del detector en [A/W].
    T: float = 300.0    # Temperatura del detector en [K]
    Rf: float = 50.0    # Resistencia de carga del detector en [Ohms]
    i_d: float = 10e-9  # Corriente oscura del fotodetector en [A]
    Fn_dB: float = 0.0  # Figura de ruido del amplificador de transimpedancia, en [dB]
    disable_shot_noise: bool = False  # Deshabilitar ruido de disparo
    name: str = ""

def calc_real_SNR(v_signal: np.ndarray, v_noise: np.ndarray) -> float:
    """
    Calcula la SNR real a partir de la señal y el ruido medidos.

    Parameters
    ----------
    v_signal : np.ndarray
        Señal eléctrica detectada sin ruido, en [v].
    v_noise : np.ndarray
        Señal de ruido eléctrico detectada, en [v].

    Returns
    -------
    SNR : float
        SNR real calculada, en veces.
    """
    P_signal = np.sum(v_signal**2)
    P_noise = np.sum(v_noise**2)

    if P_noise == 0:
        return np.inf

    SNR_dB = 10*np.log10(P_signal / P_noise)
    return SNR_dB

def calc_expected_SNR(Pin, r, B, T, Rf, i_d, disable_shot_noise) -> float :
    """
    Calcula la SNR esperada en un fotodetector PIN con ruido térmico, de disparo, y de oscuridad.

    Parameters
    ----------
    Pin : float
        Potencia óptica de entrada en [W].
    r : float
        Responsividad del detector en [A/W].
    B : float
        Ancho de banda del detector en [Hz].
    T : float
        Temperatura del detector en [K].
    Rf : float
        Resistencia de carga del detector en [Ohms].
    i_d : float
        Corriente oscura del fotodetector en [A].

    Returns
    -------
    SNR_dB : float
        SNR esperada en db.
    """
    P_signal = (r * Pin)**2
    P_noise = 4*sc.k*T*B/Rf
    if not disable_shot_noise:
        P_noise += 2*sc.e*(r*Pin + i_d)*B
    
    if P_noise == 0:
        return np.inf
    
    SNR = P_signal / P_noise
    SNR_dB = 10 * np.log10(SNR)
    return SNR_dB


def pd(params: PdSystem) -> np.ndarray:
    """
    Wrapper para simular un fotodetector PIN con ruido térmico y de disparo.

    Parameters
    ----------
    params : PdSystem
        Parámetros del fotodetector + señal de entrada.
        
    Returns
    -------
    v : np.array
        La señal eléctrica detectada, en [v].
    """
    Ein, B, fs, r, T, Rf, i_d, Fn_dB, disable_shot_noise = params.Ein, params.B, params.fs, params.r, params.T, params.Rf, params.i_d, params.Fn_dB, params.disable_shot_noise
    n_samples = len(Ein)
    Fn = 10**(Fn_dB/10)
    # Definición del filtro pasa bajos del fotodetector
    sos = sg.bessel(N=5, Wn=2*B/fs, btype="low", output="sos", norm="mag")

    i_s = r * np.real(Ein)**2

    var_sh = 0 if disable_shot_noise else sc.e * (i_s + i_d)*fs
    i_sh = np.random.normal(0, np.sqrt(var_sh), n_samples)
    var_th = 2*sc.k*T*Fn*fs/Rf
    i_th = np.random.normal(0, np.sqrt(var_th), n_samples)
    
    v_signal = sg.sosfiltfilt(sos, i_s) * Rf  # Voltage signal without noise
    v_noise = sg.sosfiltfilt(sos, i_d + i_sh + i_th) * Rf  # Noise voltage signal
    v_out = v_signal + v_noise

    SNR_dB = calc_real_SNR(v_signal, v_noise)

    return v_out, SNR_dB

if __name__ == "__main__":
    from utils import plot_signals, show_plots, plot_SNR_vs_P

    np.random.seed(12345)

    fs = 20e9
    B = 5e9
    f = 1e9
    t = np.arange(0, 4/f, 1/fs) # 4 periods
    P_dbm = -20
    r = 0.8
    i_d = 1e-9
    Rf = 50
    T = 300

    # Ejercicio B

    P = 10**(P_dbm/10) / 1000
    Ein = np.sqrt(P * np.exp(1j*2*np.pi*f*t) )

    systems =  [
        PdSystem(Ein, B=B, fs=fs, r=r, i_d=i_d, T=T, Rf=Rf, disable_shot_noise=False, name="Ruido térmico, disparo, y oscuridad"),
        PdSystem(Ein, B=B, fs=fs, r=r, i_d=i_d, T=T, Rf=Rf, disable_shot_noise=True,  name="Ruido térmico y oscuridad"),
        PdSystem(Ein, B=B, fs=fs, r=r, i_d=i_d, T=0, Rf=Rf, disable_shot_noise=False, name="Ruido de disparo y oscuridad"),
        PdSystem(Ein, B=B, fs=fs, r=r, i_d=0,   T=0, Rf=Rf, disable_shot_noise=True,  name="Sin ruido"),
    ]

    out = [pd(s) for s in systems]
    signals = [o[0] for o in out]
    SNRs_dB_sim = [o[1] for o in out]

    SNRs_dB_calc = [calc_expected_SNR(P, s.r, s.B, s.T, s.Rf, s.i_d, s.disable_shot_noise) for s in systems]
    labels = [f"{s.name} (SNR sim: {SNR_dB_sim:.2f}dB - SNR calc: {SNR_dB_calc:.2f}dB)" for s, SNR_dB_sim, SNR_dB_calc in zip(systems, SNRs_dB_sim, SNRs_dB_calc)]
    plot_signals(signals, fs*1e-12, labels=labels)  # fs en THz

    # Ejercicio C

    Ps_dbm = range(-40, 60, 5)
    Ps = [10**(P_dbm/10) / 1000 for P_dbm in Ps_dbm]

    out_C = [pd(PdSystem(Ein=np.sqrt(P * np.exp(1j*2*np.pi*f*t)), B=B, fs=fs, r=r, i_d=i_d, T=T, Rf=Rf, disable_shot_noise=False)) for P in Ps]
    SNRs_dB_sim_C = [o[1] for o in out_C]
    SNRs_dB_calc_C = [calc_expected_SNR(P, r, B, T, Rf, i_d, disable_shot_noise=False) for P in Ps]

    Pin_limit = -(i_d/r + 2 * sc.k * T / (sc.e * r * Rf))
    plot_SNR_vs_P(Ps_dbm, SNRs_dB_sim_C, SNRs_dB_calc_C)

    show_plots()