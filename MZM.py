import numpy as np
from dataclasses import dataclass
from utils import plot_signals, show_plots, plot_MZM, create_nrz_signal

@dataclass
class MzmSystem:
    E_in: np.ndarray
    u: np.ndarray
    Vpi: float
    Vbias: float = 0.0
    K: float = 0.0
    ER_dB: float = np.inf

def mzm(params: MzmSystem) -> np.ndarray:
    """
    Wrapper para simular el modulador Mach-Zehnder (MZM).

    Parameters
    ----------
    params : MzmSystem
        Parámetros del modulador.

    Returns
    -------
    :E_out: np.ndarray
        Optical signal output after modulation.
    """
    ER = 10**(params.ER_dB/10)

    real = np.cos(np.pi * (params.u + params.Vbias) / (2*params.Vpi))
    imag = np.sin(np.pi * (params.u + params.Vbias) / (2*params.Vpi))/np.sqrt(ER)

    Eout = np.sqrt(params.K) * (real + 1j*imag) * params.E_in

    return Eout

if __name__ == "__main__":
    # punto b
    n = 1000
    # E_in = np.ones(n)
    # V_pi = 5  # V
    # V_bias = 0  # V
    # u = np.linspace(V_bias - V_pi*2, V_bias + V_pi*2, n)  # V
    # K = 1
    # ER_db = np.inf  # dB
    # s = MzmSystem(E_in=E_in, u=u, Vpi=V_pi, Vbias=V_bias, K_dB=0, ER_dB=20)
    # E_out = mzm(s)
    # plot_MZM(E_out, E_in, u, V_pi, V_bias)

    # punto c
    E_in = np.ones(n)
    amplitude = 5
    V_pi = amplitude  # V
    V_bias = -V_pi  # V
    K = 0.8
    ER_db = 8  # dB

    ## plot de transferencia
    u = np.linspace(V_bias - V_pi, V_bias + V_pi, n)  # V
    s = MzmSystem(E_in=E_in, u=u, Vpi=V_pi, Vbias=V_bias, K=K, ER_dB=ER_db)
    E_out = mzm(s)
    plot_MZM(E_out, E_in, u, V_pi, V_bias)

    ## plot de la salida en tiempo del MZM
    nrz_signal = create_nrz_signal(length=100000, bit_rate=10, fs=10, amplitude=amplitude)
    E_in = np.ones_like(nrz_signal[0])
    u = nrz_signal[0]

    s = MzmSystem(E_in=E_in, u=u, Vpi=V_pi, Vbias=V_bias, K=K, ER_dB=ER_db)
    mzm_signal = mzm(s)
    plot_signals([u, mzm_signal], fs=10, labels=["Input NRZ signal (u)", "Output MZM signal"])


    show_plots()