import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fft as fft
from utils import plot_signals, plot_PSD, show_plots
from dataclasses import dataclass

@dataclass
class CWLaserParams:
    t: np.ndarray
    P0: float
    lw: float = 0.0 # Hz
    rin: float = -np.inf # dB/Hz
    df: float = 0.0 # Hz
    name: str = ""  # Custom name for each test case

def cw_laser(params: CWLaserParams) -> np.ndarray:
    r"""
    Wrapper para simular un láser de onda continua (CW) utilizando el equivalente en banda base (envolvente compleja).

    Parameters
    ----------
    params : CWLaserParams
        Parámetros del láser.

    Returns
    -------
    Eout : np.ndarray
        Envolvente compleja de la señal óptica del láser.
    """
    t = params.t
    P0 = params.P0
    lw = params.lw * 1e-12  # THz
    rin = params.rin
    df = params.df * 1e-12  # THz

    dt = t[1] - t[0]    #ps
    fs = 1/dt  # THz

    n_samples = len(t)

    p0 = 10**(P0/10) / 1000

    delta_phase_sigma = np.sqrt(2*np.pi*lw*dt)
    delta_phase = np.random.normal(0, delta_phase_sigma, n_samples)
    phase_noise = np.cumsum(delta_phase)

    rin_sigma = np.sqrt(10**(rin/10)*fs)
    rin_noise = np.random.normal(0, rin_sigma, n_samples)

    Eout = np.sqrt(p0*(1+rin_noise))*np.exp(1j*(phase_noise - 2*np.pi*df*t))

    return Eout


if __name__ == "__main__":
    fs = 1  # THz
    length = 2**16 # ps
    t = np.linspace(0, length, int(length*fs))

    RINs = [-np.inf, -140, -120, -100]  # dB/Hz
    lws = [0, 100e6, 1e9]  # Hz
    dfs = [0, 1e3, 1e6, 1e9]  # Hz


    params_lists = [
    # [   # Punto b
    #     CWLaserParams(t=t, P0=1, rin=-140, df=0, lw=1e6, name="B real "),
    #     CWLaserParams(t=t, P0=1, rin=-np.inf, df=0, lw=0, name="B ideal"),
    # ], 
    [
        CWLaserParams(t=t, P0=1, rin=-np.inf, df=0, lw=lw, name=f"LW={lw}") for lw in lws
    ], 
    # [
    #     CWLaserParams(t=t, P0=1, rin=rin, df=0, lw=0, name=f"RIN={rin}") for rin in RINs
    # ], 
    [
        CWLaserParams(t=t, P0=1, rin=-140, df=df, lw=1e3, name=f"df={df}") for df in dfs
    ],
    ]
    for params_list in params_lists:
        labels = [
            f"{params.name}: P0={params.P0} dBm, RIN={params.rin} dB/Hz, df={params.df} Hz, lw={params.lw} Hz"
            for params in params_list
        ]

        Eouts = [cw_laser(params) for params in params_list]
        PSDs = [signal.welch(Eout, return_onesided=False, fs=fs, detrend=False, nperseg=2**13, scaling='density') for Eout in Eouts]

        print(np.sum(np.abs(fft.fft(Eouts[0]))**2)/len(Eouts[0]))
        print(np.sum(np.abs(fft.fft(Eouts[1]))**2)/len(Eouts[1]))
        print(np.sum(np.abs(fft.fft(Eouts[2]))**2)/len(Eouts[2]))

        plot_signals(Eouts, fs, labels)
        plot_PSD(PSDs, labels)

    show_plots()