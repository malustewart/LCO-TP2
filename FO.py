import numpy as np
from scipy.fft import fft, ifft, fftfreq
from utils import create_gaussian_pulse, plot_signals, plot_signal_timemap, show_plots
from tqdm.auto import tqdm
from dataclasses import dataclass

@dataclass
class OpticalFiberSystem:
    A: np.ndarray
    fs: float
    length: float
    alpha: float = 0.0
    beta_2: float = 0.0
    beta_3: float = 0.0
    gamma: float = 0.0
    phi_max: float = 0.05

def optical_fiber(sys: OpticalFiberSystem, dz_save: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper para simular la propagación en fibra óptica monomodo usando SSFM adaptativo.
    
    Esta función resuelve la NLSE numéricamente, considerando efectos lineales (atenuación y dispersión)
    y no lineales (Kerr). Usa paso adaptativo basado en la fase no lineal para optimizar precisión y eficiencia.
    
    Parameters
    ----------
    A : np.ndarray
        Señal óptica de entrada (array complejo).
    fs : float
        Frecuencia de muestreo (para calcular frecuencias angulares).
    length : float
        Longitud de la fibra.
    alpha : float, optional
        Atenuación [dB/km].
    beta_2 : float, optional
        Dispersión de segundo orden [ps²/km].
    beta_3 : float, optional
        Dispersión de tercer orden [ps³/km].
    gamma : float, optional
        Coeficiente no lineal [(W·km)⁻¹].
    phi_max : float, optional
        Umbral para paso adaptativo [rad]. Reemplazar este argumento por el número de pasos en caso de optar por una implementación de paso fijo.
    dz_save : float
        Intervalo [km] para guardar capturas de la señal.
    Returns
    -------
    A_out : np.ndarray
        Señal óptica de salida (array complejo).
    A_evolution : np.ndarray
        Array de shape (n_saves, n_samples) con las capturas de la señal a lo largo de la fibra.
    z_positions : np.ndarray
        Array de distancias [km] donde se guardaron las capturas.
    """
    
    A = np.copy(sys.A)
    fs = sys.fs
    length = sys.length
    alpha = sys.alpha
    beta_2 = sys.beta_2
    beta_3 = sys.beta_3
    gamma = sys.gamma
    phi_max= sys.phi_max

    # Paso 1: Convertir alpha a unidades neperianas (1/km)
    alpha_np = alpha * np.log(10) / 10
    
    # Paso 2: Calcular frecuencias angulares ω (en rad/ps)
    w = fftfreq(len(A), 1/fs) * 2 * np.pi

    # Paso 3: Operador lineal D en dominio de frecuencia
    D_op = -alpha_np/2 + 1j*beta_2*w*w/2 + 1j*beta_3*w*w*w/6
    
    # Paso 4: Inicializar paso adaptativo h
    if gamma == 0 or (beta_2 == 0 and beta_3 == 0):
        h = length  # Caso sin dispersión o no linealidad, la solución es exacta
    else:
        P_max = np.max(np.abs(A*A))
        h = phi_max/gamma/P_max
    
    h = min(h, length)  # Asegurar que h no exceda la longitud total

    # Prealoco memoria para guardar capturas
    z_positions = np.arange(0, length + 1e-12, dz_save) # sumo un epsilon a length porque el rango es no inclusivo al final
    n_saves = len(z_positions)
    n_samples = len(A)
    A_evolution = np.zeros((n_saves, n_samples), dtype=np.complex128)
    
    # Guardo primer captura
    A_evolution[0, :] = A
    next_save_idx = 1

    # Inicializar posición actual
    steps = 0
    z = 0.0
    prev_z = 0.0

    # ignorar esta linea..
    progress_bar = tqdm(total=100, desc="Progreso de propagación", bar_format="{l_bar}{bar}|[{elapsed}{postfix}]", postfix={"steps": 0})
    
    # Paso 5: Bucle principal de SSFM adaptativo
    while z < length:
        z = prev_z + h

        A_squared = np.abs(A)**2

        progress_bar.set_postfix({"steps": steps, "z": z}) # ignorar esta linea..

        ## Implementar SSFM simétrico
        N_op = gamma * A_squared * 1j
        
        # Paso 5.1: Operador N en h/2
        A = np.exp(N_op*h/2)*A

        # Paso 5.2: Operados D en h
        A = ifft(np.exp(D_op*h)*fft(A))

        # Paso 5.3: Operador N en h/2
        A = np.exp(N_op*h/2)*A

        # Si corresponde, guardo captura
        while next_save_idx < n_saves and z >= z_positions[next_save_idx]:
            A_evolution[next_save_idx, :] = A
            next_save_idx += 1

        progress_bar.update(100 * h / length) # ignorar esta linea..

        # Paso 5.4: Si se optó por la implementación adaptiva, calcular nueva h
        if gamma == 0 or (beta_2 == 0 and beta_3 == 0):
            pass
        else:
            P_max = np.max(A_squared)
            h = phi_max/gamma/P_max
            h = min(h, length - z)  # Asegurar que h no exceda la longitud restante

        prev_z = z
        z += h # Avanzar posición
        steps += 1 # Actualizar número de pasos

    progress_bar.close() # ignorar esta linea..
    
    return A, A_evolution, z_positions

if __name__ == "__main__":

    N = 1000
    pulse_std = 10  # in ps
    length = pulse_std * 100 # in ps
    dt = length / N
    fs = 1/dt # Thz
    amplitude = 1
    A0, t = create_gaussian_pulse(pulse_std, length, fs, amplitude)

    systems = [
        OpticalFiberSystem(A0, fs, length=20, alpha=0.2, beta_2=0,   beta_3=0,    gamma=0),
        OpticalFiberSystem(A0, fs, length=20, alpha=0,   beta_2=-20, beta_3=0,    gamma=0),
        OpticalFiberSystem(A0, fs, length=20, alpha=0,   beta_2=0,   beta_3=0.15, gamma=0),
        OpticalFiberSystem(A0, fs, length=20, alpha=0,   beta_2=0,   beta_3=0,    gamma=1.5),
        OpticalFiberSystem(A0, fs, length=20, alpha=0,   beta_2=-20, beta_3=0,    gamma=1.5),
    ]

    for i, sys in enumerate(systems):
        result = optical_fiber(sys)
        filename = f"FO_a_{sys.alpha}_b2_{sys.beta_2}_b3_{sys.beta_3}_gamma_{sys.gamma}_L_{sys.length}"
        plot_signals([A0, result[0]], fs, ["Initial signal", f"Final signal ({i})"], time_filename=f"figs/{filename}_time.svg", freq_filename=f"figs/{filename}_freq.svg") 
        # plot_signal_timemap(result[1], t, result[2])

    show_plots()