
import numpy as np
import CW_LASER as laser
import MZM as mzm
from utils import plot_MZM
from utils_catedra import eyediagram
from enum import Enum
from matplotlib import pyplot as plt

class Modulation(Enum):
    NRZ = 1
    RZ = 2

def generate_signal(bits, sps, modulation=Modulation.NRZ):
    if modulation == Modulation.NRZ:
        signal = np.repeat(bits, sps)
        return signal
    elif modulation == Modulation.RZ:
        raise NotImplementedError("RZ modulation not implemented yet.")
    raise ValueError("Unknown modulation type.")


def ej_1(n_bits=5000, bitrate=10e9, sps=16, P_dbm=10, RIN_db=-150, df=10e6, Vpi=5, r=1, T=300, Rl=50, Vbias = -2.5, Vpp_over_Vpi=[1.2, 1.0, 0.8, 0.5], K=0.8, ER_db=30):
    # laser
    n_samples = n_bits * sps
    fs = bitrate * sps
    t = np.arange(n_samples) / fs
    P0 = 10 ** (P_dbm / 10) / 1e3
    laser_system = laser.CWLaserParams(t, P0=P0, rin=RIN_db, df=df, lw=0)
    laser_out = laser.cw_laser(laser_system)

    # MZM
    bits = np.random.randint(0, 2, n_bits)
    signal = generate_signal(bits, sps, modulation=Modulation.NRZ)

    for alpha in Vpp_over_Vpi:
        Vpp = alpha * Vpi
        u = (Vpp) * signal - (Vpp / 2)

        mzm_system = mzm.MzmSystem(E_in=laser_out, Vpi=Vpi, u=u, Vbias=Vbias, K=K, ER_dB=ER_db)
        mzm_E_out = mzm.mzm(mzm_system)
        mzm_P_out = (mzm_E_out**2) * 1000  # mW
        
        n_traces = 1000
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax = eyediagram(mzm_P_out, sps, n_traces=n_traces, ylabel='Amplitude (mW)', title=f'MZM Eye Diagram (Vpp/Vpi={alpha} - {n_traces} traces)', alpha=0.5, ax=ax, show=False)
        
        fig.savefig(f'figs/ej1_mzm_eye_VppOverVpi_{alpha}.svg')

    n=1000
    u = np.linspace(Vbias - Vpi, Vbias + Vpi, n)
    E_in = np.ones(n)
    s = mzm.MzmSystem(E_in=E_in, u=u, Vpi=Vpi, Vbias=Vbias, K=K, ER_dB=ER_db)
    E_out = mzm.mzm(s)
    plot_MZM(E_out, E_in, u, Vpi, Vbias)
    plt.savefig('figs/ej1_mzm_transfer_function.svg')

    return bits

def ej_2(n_bits=5000, bitrate=10e9, sps=16, P_dbm=10, RIN_db=-150, df=10e6, Vpi=5, r=1, T=300, Rl=50, Vbias = -2.5, Vpp_over_Vpi=1.0, K=0.8, ER_db=30, L=[40, 60, 80, 100]):
    # laser
    n_samples = n_bits * sps
    fs = bitrate * sps
    t = np.arange(n_samples) / fs
    P0 = 10 ** (P_dbm / 10) / 1e3
    laser_system = laser.CWLaserParams(t, P0=P0, rin=RIN_db, df=df, lw=0)
    laser_out = laser.cw_laser(laser_system)

    # MZM
    bits = np.random.randint(0, 2, n_bits)
    signal = generate_signal(bits, sps, modulation=Modulation.NRZ)

    Vpp = Vpp_over_Vpi * Vpi
    u = (Vpp) * signal - (Vpp / 2)

    mzm_system = mzm.MzmSystem(E_in=laser_out, Vpi=Vpi, u=u, Vbias=Vbias, K=K, ER_dB=ER_db)
    mzm_E_out = mzm.mzm(mzm_system)

    # Fibra optica


if __name__ == "__main__":
    np.random.seed(12345)
    ej_1()
    ej_2()
