import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from utils import plot_signals, plot_PSD, show_plots
from dataclasses import dataclass

@dataclass
class CWLaserParams:
    t: np.ndarray
    P0: float
    lw: float = 0.0
    rin: float = -np.inf
    df: float = 0.0

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
    lw = params.lw
    rin = params.rin
    df = params.df

    dt = t[1] - t[0]
    fs = 1/dt

    n_samples = len(t)

    p0 = 10**(P0/10) / 1000

    delta_phase_sigma = np.sqrt(2*np.pi*lw*dt)
    delta_phase = np.random.normal(0, delta_phase_sigma, n_samples)
    phase_noise = np.cumsum(delta_phase)

    rin_sigma = np.sqrt(10**(rin/10)*fs)
    rin_noise = np.random.normal(0, rin_sigma, n_samples)

    Eout = np.sqrt(p0*(1+rin_noise))*np.exp(1j*(phase_noise - 2*np.pi*df*t))

    return Eout

fs = 0.5  # THz
length = 1000 # ps
P0 = 0 # dBm
rin = -np.inf # dB
df = 0 # Hz
lw = 1000 # Hz
t = np.linspace(0, length, int(length*fs))

params_list = [
    CWLaserParams(t=t, P0=0, rin=-np.inf, df=0, lw=10000),
    CWLaserParams(t=t, P0=0, rin=-np.inf, df=10000, lw=0),
    CWLaserParams(t=t, P0=0, rin=-np.inf, df=10000, lw=10000),
    # CWLaserParams(t=t, P0=-3, rin=-100, df=2e6, lw=200),
    # CWLaserParams(t=t, P0=5, rin=-90, df=0, lw=10000),
]

Eouts = [cw_laser(params) for params in params_list]
PSDs = [signal.welch(Eout, return_onesided=False, fs=fs) for Eout in Eouts]

plot_signals(Eouts, fs)
plot_PSD(PSDs)

show_plots()