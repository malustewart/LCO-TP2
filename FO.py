import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def get_frequency(A, fs):
    a = np.abs(fft(A))
    f = fftfreq(len(A), 1/fs)
    
    pos_freqs = f >=0
    f = f[pos_freqs]
    a = a[pos_freqs]

    prom = 0.5 * np.max(a)
    peaks = find_peaks(a, prominence=prom)  # prominence evita detectar falsos picos por ruido en el espectro

    return f[peaks[0][0]]

def optical_fiber(
    A: np.ndarray,          # Señal óptica de entrada (array complejo, shape: (n_samples,))
    fs: float,       # Frecuencia de muestreo [Hz] (ajustar según la señal)
    length: float,          # Longitud de la fibra [km]
    alpha: float = 0.0,     # Atenuación [dB/km]
    beta_2: float = 0.0,    # Dispersión de segundo orden [ps²/km]
    beta_3: float = 0.0,    # Dispersión de tercer orden [ps³/km]
    gamma: float = 0.0,     # Coeficiente no lineal [(W·km)⁻¹]
    phi_max: float = 0.05,  # Máxima rotación de fase no lineal permitida [rad]
    dz_save: float = 0.1,   # [km] interval at which to save intermediate results
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    
    # Paso 1: Convertir alpha a unidades neperianas (1/km)
    alpha_np = alpha * np.log(10) / 10
    
    # Paso 2: Calcular frecuencias angulares ω (en rad/ps)
    w = get_frequency(A, fs) * 2 * np.pi

    # Paso 3: Operador lineal D en dominio de frecuencia
    D_op = -alpha_np/2 + beta_2*w*w/2*1j + beta_3*w*w*w/6*1j
    
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

        A_squared = abs(A*A)

        progress_bar.set_postfix({"steps": steps, "z": z}) # ignorar esta linea..

        ## Implementar SSFM simétrico
        N_op = gamma * (np.abs(A_squared)) * 1j
        
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


fs = 6000
t = np.linspace(0, 1, fs, endpoint=False)
A = np.exp(2j*np.pi*123*t) + 0.5*np.exp(2j*np.pi*300*t)

A_final, A_snapshots, z = optical_fiber(A, fs, 0.2, alpha=20, beta_2=0, beta_3=0, gamma=0)

plt.plot(t, A)
plt.plot(t, A_final)
plt.show()


t = np.arange(A.size) / fs

plt.imshow(np.abs(A_snapshots)**2, aspect="auto",
           extent=[t[0], t[-1], z[-1], z[0]], cmap="inferno")
plt.xlabel("Time [s]")
plt.ylabel("Distance [km]")
plt.title("Pulse evolution in fiber")
plt.colorbar(label="|A|²")
plt.show()