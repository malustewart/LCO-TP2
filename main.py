
import numpy as np
import CW_LASER as laser
import MZM as mzm
import FO as of
import PD as pd
from utils import plot_MZM, plot_signals
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
        signal = np.zeros(len(bits) * sps)
        for i, bit in enumerate(bits):
            if bit:
                signal[i * sps : i * sps + sps // 2] = 1
        signal = np.roll(signal, sps // 4)
        return signal
    raise ValueError("Unknown modulation type.")


def ej_1(n_bits=5000, bitrate=10e9, sps=16, P_dbm=-10, RIN_db=-150, df=10e6, Vpi=5, r=1, T=300, Rl=50, Vbias = -2.5, Vpp_over_Vpi=[1.2, 1.0, 0.8, 0.5], K=0.8, ER_db=30):
    # laser
    n_samples = n_bits * sps
    fs = bitrate * sps
    t = np.arange(n_samples) / fs
    laser_system = laser.CWLaserParams(t, P0=P_dbm, rin=RIN_db, df=df, lw=0)
    laser_out = laser.cw_laser(laser_system)

    # MZM
    bits = np.random.randint(0, 2, n_bits)
    signal = generate_signal(bits, sps, modulation=Modulation.NRZ)

    for alpha in Vpp_over_Vpi:
        Vpp = alpha * Vpi
        u = (Vpp) * signal - (Vpp / 2)

        mzm_system = mzm.MzmSystem(E_in=laser_out, Vpi=Vpi, u=u, Vbias=Vbias, K=K, ER_dB=ER_db)
        mzm_E_out = mzm.mzm(mzm_system)
        mzm_P_out = np.abs(mzm_E_out)**2 * 1000  # mW
        
        n_traces = 1000
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_ylim(0, np.max(mzm_P_out)*1.1)
        ax = eyediagram(mzm_P_out, sps, n_traces=n_traces, ylabel='Amplitude (mW)', title=f'MZM Eye Diagram (Vpp/Vpi={alpha} - {n_traces} traces)', alpha=0.5, ax=ax, show=False)
        fig.savefig(f'figs/ej1_mzm_eye_VppOverVpi_{alpha}.svg')

        pd_system = pd.PdSystem(Ein=mzm_E_out, B=20e9, fs=fs, r=r, T=T, Rf=Rl) # TODO: poner bien B de acuerdo a la consigna

        v, _ = pd.pd(pd_system)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_ylim(0, np.max(v)*1.1)
        ax = eyediagram(v, sps, n_traces=n_traces, ylabel='Amplitude (V)', title=f'PD Eye Diagram ({n_traces} traces)', alpha=0.5, ax=ax, show=False)
        fig.savefig(f'figs/ej1_pd_eye_VppOverVpi_{alpha}.svg')

    n=1000
    u = np.linspace(Vbias - Vpi, Vbias + Vpi, n)
    E_in = np.ones(n)
    s = mzm.MzmSystem(E_in=E_in, u=u, Vpi=Vpi, Vbias=Vbias, K=K, ER_dB=ER_db)
    E_out = mzm.mzm(s)
    plot_MZM(E_out, E_in, u, Vpi, Vbias)
    plt.savefig('figs/ej1_mzm_transfer_function.svg')
    plt.close('all')

def ej_2(n_bits=5000, bitrate=10e9, sps=16, P_dbm=-10, RIN_db=-150, df=10e6, Vpi=5, r=1, T=300, Rl=50, Vbias = -2.5, Vpp_over_Vpi=1.0, K=0.8, ER_db=30, L=[0, 40, 60, 80, 100], beta_2=-20, beta_3=0.1, gamma=0, alpha_of_db=0):
    # laser
    n_samples = n_bits * sps
    fs = bitrate * sps
    fs_THz = fs * 1e-12
    t = np.arange(n_samples) / fs
    laser_system = laser.CWLaserParams(t, P0=P_dbm, rin=RIN_db, df=df, lw=0)
    laser_out = laser.cw_laser(laser_system)

    # MZM
    bits = np.random.randint(0, 2, n_bits)
    signal = generate_signal(bits, sps, modulation=Modulation.NRZ)

    Vpp = Vpp_over_Vpi * Vpi
    u = (Vpp) * signal - (Vpp / 2)

    mzm_system = mzm.MzmSystem(E_in=laser_out, Vpi=Vpi, u=u, Vbias=Vbias, K=K, ER_dB=ER_db)
    mzm_E_out = mzm.mzm(mzm_system)
    mzm_P_out = np.abs(mzm_E_out)**2 * 1000  # mW

    n_traces = 1000
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax = eyediagram(mzm_P_out, sps, n_traces=n_traces, ylabel='Amplitude (mW)', title=f'MZM Eye Diagram ({n_traces} traces)', alpha=0.5, ax=ax, show=False)
    fig.savefig(f'figs/ej2_mzm_eye.svg')

    # Fibra optica
    for l in L:
        of_system = of.OpticalFiberSystem(A=mzm_E_out, fs=fs_THz, length=l, beta_2=beta_2, beta_3=beta_3, gamma=gamma, alpha=alpha_of_db)
        of_E_out, _, _ = of.optical_fiber(of_system, dz_save=l)
        
        pd_system = pd.PdSystem(Ein=of_E_out, B=20e9, fs=fs, r=r, T=T, Rf=Rl) # TODO: poner bien B de acuerdo a la consigna
        v, _ = pd.pd(pd_system)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_ylim(np.min(v), np.max(v)*1.1)
        ax = eyediagram(v, sps, n_traces=n_traces, ylabel='Amplitude (V)', title=f'PD Eye Diagram ({n_traces} traces)', alpha=0.5, ax=ax, show=False)
        fig.savefig(f'figs/ej2_pd_eye_P_{P_dbm}_a_{alpha_of_db}_b2_{beta_2}_b3_{beta_3}_gamma_{gamma}_L_{l}.svg')
    plt.close('all')

def ej_3(n_bits=5000, bitrate=10e9, sps=16, P_dbm=[-10, 0, 10, 20, 30], RIN_db=-150, df=10e6, Vpi=5, r=1, T=300, Rl=50, Vbias = -2.5, Vpp_over_Vpi=1.0, K=0.8, ER_db=30, L=50, beta_2=-20, beta_3=0.1, gamma=1.5, alpha_of_db=0.2):
    n_samples = n_bits * sps
    fs = bitrate * sps
    fs_THz = fs * 1e-12
    t = np.arange(n_samples) / fs

    # MZM
    bits = np.random.randint(0, 2, n_bits)
    signal = generate_signal(bits, sps, modulation=Modulation.NRZ)

    Vpp = Vpp_over_Vpi * Vpi
    u = (Vpp) * signal - (Vpp / 2)

    for P in P_dbm:
        # laser
        laser_system = laser.CWLaserParams(t, P0=P, rin=RIN_db, df=df, lw=0)
        laser_out = laser.cw_laser(laser_system)

        mzm_system = mzm.MzmSystem(E_in=laser_out, Vpi=Vpi, u=u, Vbias=Vbias, K=K, ER_dB=ER_db)
        mzm_E_out = mzm.mzm(mzm_system)
        mzm_P_out = np.abs(mzm_E_out)**2 * 1000  # mW

        n_traces = 1000
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax = eyediagram(mzm_P_out, sps, n_traces=n_traces, ylabel='Amplitude (mW)', title=f'MZM Eye Diagram (P={P}dBm - {n_traces} traces)', alpha=0.5, ax=ax, show=False)
        fig.savefig(f'figs/ej3_mzm_eye_P_{P}.svg')

        # Optical Fiber
        of_system = of.OpticalFiberSystem(A=mzm_E_out, fs=fs_THz, length=L, beta_2=beta_2, beta_3=beta_3, gamma=gamma, alpha=alpha_of_db)
        of_E_out, _, _ = of.optical_fiber(of_system)
        
        pd_system = pd.PdSystem(Ein=of_E_out, B=20e9, fs=fs, r=r, T=T, Rf=Rl) # TODO: poner bien B de acuerdo a la consigna
        v, _ = pd.pd(pd_system)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_ylim(np.min(v), np.max(v)*1.1)
        ax = eyediagram(v, sps, n_traces=n_traces, ylabel='Amplitude (V)', title=f'PD Eye Diagram (P={P}dBm - {n_traces} traces)', alpha=0.5, ax=ax, show=False)
        fig.savefig(f'figs/ej3_pd_eye_P_{P}_a_{alpha_of_db}_b2_{beta_2}_b3_{beta_3}_gamma_{gamma}_L_{L}.svg')
    plt.close('all')

def ej_4(n_bits=5000, bitrates=[5e9, 10e9, 15e9, 20e9], sps=16, P_dbm=10, RIN_db=-150, df=10e6, Vpi=5, r=1, T=300, Rl=50, Vbias = -2.5, Vpp_over_Vpi=1.0, K=0.8, ER_db=30, L=50, beta_2=-20, beta_3=0.1, gamma=0, alpha_of_db=0):
    for bitrate in bitrates:
        # laser
        n_samples = n_bits * sps
        fs = bitrate * sps
        fs_THz = fs * 1e-12
        t = np.arange(n_samples) / fs
        laser_system = laser.CWLaserParams(t, P0=P_dbm, rin=RIN_db, df=df, lw=0)
        laser_out = laser.cw_laser(laser_system)

        # MZM
        bits = np.random.randint(0, 2, n_bits)
        signal = generate_signal(bits, sps, modulation=Modulation.NRZ)

        Vpp = Vpp_over_Vpi * Vpi
        u = (Vpp) * signal - (Vpp / 2)

        mzm_system = mzm.MzmSystem(E_in=laser_out, Vpi=Vpi, u=u, Vbias=Vbias, K=K, ER_dB=ER_db)
        mzm_E_out = mzm.mzm(mzm_system)

        # Fibra optica
        of_system = of.OpticalFiberSystem(A=mzm_E_out, fs=fs_THz, length=L, beta_2=beta_2, beta_3=beta_3, gamma=gamma, alpha=alpha_of_db)
        of_E_out, _, _ = of.optical_fiber(of_system)
        
        pd_system = pd.PdSystem(Ein=of_E_out, B=20e9, fs=fs, r=r, T=T, Rf=Rl) # TODO: poner bien B de acuerdo a la consigna
        v, _ = pd.pd(pd_system)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_ylim(np.min(v), np.max(v)*1.1)
        n_traces = 1000
        ax = eyediagram(v, sps, n_traces=n_traces, ylabel='Amplitude (V)', title=f'PD Eye Diagram ({n_traces} traces)', alpha=0.5, ax=ax, show=False)
        fig.savefig(f'figs/ej4_pd_eye_P_{P_dbm}_a_{alpha_of_db}_b2_{beta_2}_b3_{beta_3}_gamma_{gamma}_L_{L}_Br_{bitrate}.svg')
    plt.close('all')

def ej_5(n_bits=5000, bitrate=10e9, sps=16, P_dbm=10, RIN_db=-150, df=10e6, Vpi=5, r=1, T=300, Rl=50, Vbias = -2.5, Vpp_over_Vpi=1.0, K=0.8, ER_db=30, L=50, beta_2=-20, beta_3=0.1, gamma=0, alpha_of_db=0):
    # laser
    n_samples = n_bits * sps
    fs = bitrate * sps
    fs_THz = fs * 1e-12
    t = np.arange(n_samples) / fs
    laser_system = laser.CWLaserParams(t, P0=P_dbm, rin=RIN_db, df=df, lw=0)
    laser_out = laser.cw_laser(laser_system)

    # MZM
    bits = np.random.randint(0, 2, n_bits)
    for modulation in [Modulation.NRZ, Modulation.RZ]:
        signal = generate_signal(bits, sps, modulation=modulation)
        Vpp = Vpp_over_Vpi * Vpi
        u = (Vpp) * signal - (Vpp / 2)

        mzm_system = mzm.MzmSystem(E_in=laser_out, Vpi=Vpi, u=u, Vbias=Vbias, K=K, ER_dB=ER_db)
        mzm_E_out = mzm.mzm(mzm_system)

        # Fibra optica
        of_system = of.OpticalFiberSystem(A=mzm_E_out, fs=fs_THz, length=L, beta_2=beta_2, beta_3=beta_3, gamma=gamma, alpha=alpha_of_db)
        of_E_out, _, _ = of.optical_fiber(of_system)
        
        pd_system = pd.PdSystem(Ein=of_E_out, B=bitrate, fs=fs, r=r, T=T, Rf=Rl)
        v, _ = pd.pd(pd_system)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_ylim(np.min(v), np.max(v)*1.1)
        n_traces = 1000
        ax = eyediagram(v, sps, n_traces=n_traces, ylabel='Amplitude (V)', title=f'PD Eye Diagram ({n_traces} traces)', alpha=0.5, ax=ax, show=False)
        fig.savefig(f'figs/ej5_pd_eye_P_{P_dbm}_a_{alpha_of_db}_b2_{beta_2}_b3_{beta_3}_gamma_{gamma}_L_{L}_mod_{modulation.name}.svg')
    plt.close('all')

if __name__ == "__main__":
    np.random.seed(12345)
    ej_1()

    np.random.seed(12345)
    ej_2(P_dbm=-10)
    ej_2(P_dbm=0)
    ej_2(P_dbm=10)
    ej_2(P_dbm=20)

    np.random.seed(12345)
    ej_3(beta_2=0,   beta_3=0,    gamma=0,   alpha_of_db=0.2, L=20)
    ej_3(beta_2=0,   beta_3=0,    gamma=0,   alpha_of_db=0.2, L=40)
    ej_3(beta_2=-20, beta_3=0,    gamma=0,   alpha_of_db=0,   L=20)
    ej_3(beta_2=-20, beta_3=0,    gamma=0,   alpha_of_db=0,   L=40)
    ej_3(beta_2=0,   beta_3=0.1,  gamma=0,   alpha_of_db=0,   L=20)
    ej_3(beta_2=0,   beta_3=0.1,  gamma=0,   alpha_of_db=0,   L=40)
    ej_3(beta_2=0,   beta_3=0,    gamma=1.5, alpha_of_db=0,   L=20)
    ej_3(beta_2=0,   beta_3=0,    gamma=1.5, alpha_of_db=0,   L=40)
    ej_3(beta_2=-20, beta_3=0,    gamma=1.5, alpha_of_db=0,   L=20)
    ej_3(beta_2=-20, beta_3=0,    gamma=1.5, alpha_of_db=0,   L=40)

    np.random.seed(12345)
    ej_4()

    np.random.seed(12345)
    ej_5(P_dbm=-10, L=20,  beta_2=-20, beta_3=0.1, gamma=1.5, alpha_of_db=0)
    ej_5(P_dbm=-10, L=40,  beta_2=-20, beta_3=0.1, gamma=1.5, alpha_of_db=0)
    ej_5(P_dbm=-10, L=60,  beta_2=-20, beta_3=0.1, gamma=1.5, alpha_of_db=0)
    ej_5(P_dbm=-10, L=80,  beta_2=-20, beta_3=0.1, gamma=1.5, alpha_of_db=0)
    ej_5(P_dbm=-10, L=100, beta_2=-20, beta_3=0.1, gamma=1.5, alpha_of_db=0)
    ej_5(P_dbm=-7, L=20,  beta_2=-20, beta_3=0.1, gamma=1.5, alpha_of_db=0)
    ej_5(P_dbm=-7, L=40,  beta_2=-20, beta_3=0.1, gamma=1.5, alpha_of_db=0)
    ej_5(P_dbm=-7, L=60,  beta_2=-20, beta_3=0.1, gamma=1.5, alpha_of_db=0)
    ej_5(P_dbm=-7, L=80,  beta_2=-20, beta_3=0.1, gamma=1.5, alpha_of_db=0)
    ej_5(P_dbm=-7, L=100, beta_2=-20, beta_3=0.1, gamma=1.5, alpha_of_db=0)

