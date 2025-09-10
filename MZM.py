import numpy as np

def mzm(
    E_in: np.ndarray,
    u: np.ndarray,
    Vpi: float,
    Vbias: float = 0.0,
    K_dB: float = 0.0,
    ER_dB: float = np.inf,
) -> np.ndarray:
    """
    Wrapper para simular el modulador Mach-Zehnder (MZM).

    Parameters
    ----------
    E_in : np.ndarray
        Optical signal input to be modulated.
    u : np.ndarray
        Driver voltage, with zero bias.
    Vpi : float
        Voltage at which the device switches from on-state to off-state.
    Vbias : float, optional
        Modulator bias voltage. Default is 0.0.
    K : float, optional
        Propagation losses < 0 dB. Default is 0.0 dB.
    ER_dB : float, optional
        Extinction ratio of the modulator, in dB. Default is np.inf.

    Returns
    -------
    :E_out: np.ndarray
        Optical signal output after modulation.
    """
    K = ...
    ER = ...

    Eout = ...

    return Eout