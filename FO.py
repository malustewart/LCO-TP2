import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from tqdm.auto import tqdm

def get_frequency(A, fs):
    a = np.abs(fft(A))
    f = fftfreq(len(A), 1/fs)
    
    pos_freqs = f >=0
    f = f[pos_freqs]
    a = a[pos_freqs]

    prom = 0.5 * np.max(a)
    peaks = find_peaks(a, prominence=prom)  # prominence evita detectar falsos picos por ruido en el espectro
    
    return f[peaks[0]]

def optical_fiber(
    A: np.ndarray,          # Señal óptica de entrada (array complejo, shape: (n_samples,))
    fs: float,       # Frecuencia de muestreo [Hz] (ajustar según la señal)
    length: float,          # Longitud de la fibra [km]
    alpha: float = 0.0,     # Atenuación [dB/km]
    beta_2: float = 0.0,    # Dispersión de segundo orden [ps²/km]
    beta_3: float = 0.0,    # Dispersión de tercer orden [ps³/km]
    gamma: float = 0.0,     # Coeficiente no lineal [(W·km)⁻¹]
    phi_max: float = 0.05,  # Máxima rotación de fase no lineal permitida [rad]
) -> np.ndarray:
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
        
    Returns
    -------
    np.ndarray
        Señal óptica de salida (array complejo).
    """
    
    # Paso 1: Convertir alpha a unidades neperianas (1/km)
    alpha_np = alpha * np.log(10) / 10
    
    # Paso 2: Calcular frecuencias angulares ω (en rad/ps)
    w = get_frequency(A, fs)
    
    # Paso 3: Operador lineal D en dominio de frecuencia
    D_op = ...
    
    # Paso 4: Inicializar paso adaptativo h
    if gamma == 0 or (beta_2 == 0 and beta_3 == 0):
        h = length  # Caso sin dispersión o no linealidad, la solución es exacta
    else:
        h = ...  # Calcular paso inicial basado en P_max y phi_max
    
    h = min(h, length)  # Asegurar que h no exceda la longitud total

    # Inicializar posición actual
    z = 0.0
    steps = 0

    # ignorar esta linea..
    progress_bar = tqdm(total=100, desc="Progreso de propagación", bar_format="{l_bar}{bar}|[{elapsed}{postfix}]", postfix={"steps": 0})
    
    # Paso 5: Bucle principal de SSFM adaptativo
    while z < length:
        z += h # Avanzar posición
        step += 1 # Actualizar número de pasos
        progress_bar.set_postfix({"steps": steps, "z": z}) # ignorar esta linea..


        ## Implementar SSFM simétrico
        N_op = ...
        # Paso 5.1: Operador N en h/2
        ...

        # Paso 5.2: Operados D en h
        ...

        # Paso 5.3: Operador N en h/2
        ...


        progress_bar.update(100 * h / length) # ignorar esta linea..

        # Paso 5.4: Si se optó por la impleemntación adaptiva, calcular nueva h, sino eliminar estas lineas
        h = ...
        h = min(h, length - z)  # Asegurar que h no exceda la longitud restante
    
    progress_bar.close() # ignorar esta linea..
    
    return A


fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)
x = np.exp(2j*np.pi*123*t) + 0.5*np.exp(2j*np.pi*300*t)
