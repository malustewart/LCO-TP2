import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def cw_laser(
    t: np.ndarray, 
    P0: float, 
    lw: float = 0.0, 
    rin: float = -np.inf,  
    df: float = 0.0
) -> np.ndarray:
    r"""
    Wrapper para simular un láser de onda continua (CW) utilizando el equivalente en banda base (envolvente compleja).

    Parameters
    ----------
    t : np.array
        Vector de tiempo.
    P0 : float
        Potencia óptica del láser, en dBm.
    lw : float
        Ancho de línea del láser, en Hz.
    rin : float
        Densidad de potencia del Ruido de Intensidad Relativa, en dB/Hz.
    df : float
        Desplazamiento de frecuencia del láser, en Hz.

    Returns
    -------
    Eout : np.ndarray
        Envolvente compleja de la señal óptica del láser.
    """
    dt = t[1] - t[0]
    fs = 1/dt

    n_samples = len(t)

    p0 = 10**(P0/10) / 1000

    delta_phase_sigma = np.sqrt(2*np.pi*df*dt)
    delta_phase = np.random.normal(0, delta_phase_sigma, n_samples)
    phase_noise = np.cumsum(delta_phase)

    rin_sigma = np.sqrt(10**(rin/10)*fs)
    rin_noise = np.random.normal(0, rin_sigma, n_samples)

    Eout = np.sqrt(p0*(1+rin_noise))*np.exp(1j*(phase_noise - 2*np.pi*dt*t))

    return Eout


fs = 10  # THz
length = 10000 # ps
P0 = 0 # dBm
rin_std = 1
t = np.linspace(0, length, length*fs)
Eout = cw_laser(t, P0, rin=-140, df = 1000)


f, psd = signal.welch(Eout)

plt.plot(t, np.abs(Eout))
plt.figure()
plt.plot(f,psd)
plt.show()