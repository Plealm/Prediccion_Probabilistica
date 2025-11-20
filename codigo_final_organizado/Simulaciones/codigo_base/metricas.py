# metricas.py

import numpy as np
from typing import Union

def crps(F_samples: np.ndarray, x: float) -> float:
    """
    Calcula el Continuous Ranked Probability Score (CRPS) para una observación puntual.
    
    Implementa la fórmula (21) del paper Gneiting & Raftery (2007):
    CRPS(F, x) = E_F|X - x| - (1/2) * E_F|X - X'|
    
    donde X y X' son copias independientes de la distribución F.
    
    Args:
        F_samples: Muestras de la distribución predictiva F (puede ser de cualquier tamaño)
        x: Valor observado (observación real que materializó)
    
    Returns:
        Valor del CRPS (negativo, donde valores más cercanos a 0 son mejores)
    
    Referencias:
        Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, 
        prediction, and estimation. Journal of the American Statistical Association, 
        102(477), 359-378. Ecuación (21).
    """
    F_samples = np.asarray(F_samples).flatten()
    
    if len(F_samples) == 0:
        return np.nan
    
    # Término 1: E_F|X - x| (distancia esperada entre predicción y observación)
    term1 = np.mean(np.abs(F_samples - x))
    
    # Término 2: (1/2) * E_F|X - X'| (nitidez/dispersión de la predicción)
    # Usamos broadcasting para calcular todas las diferencias
    pairwise_diffs = np.abs(F_samples[:, np.newaxis] - F_samples)
    term2 = 0.5 * np.mean(pairwise_diffs)
    
    # CRPS = término1 - término2 (en orientación negativa: valores más negativos = peor)
    return term1 - term2


def ecrps(samples_F: np.ndarray, samples_G: np.ndarray) -> float:
    """
    Calcula el Expected CRPS (ECRPS) como el promedio de múltiples CRPS.
    
    El ECRPS es simplemente el promedio del CRPS calculado para cada muestra
    de la distribución verdadera G. Es decir:
    
    ECRPS(F, G) = (1/n_g) * Σ_i CRPS(F, g_i)
    
    donde g_i son las muestras de G.
    
    Args:
        samples_F: Muestras de la distribución predictiva (pronóstico)
        samples_G: Muestras de la distribución real (ground truth)
    
    Returns:
        Valor escalar del ECRPS (promedio de CRPS individuales)
    
    Nota:
        Esta métrica se usa cuando la distribución verdadera también es empírica
        (representada por muestras), y queremos promediar el error de predicción
        sobre todas las posibles observaciones de esa distribución.
    """
    forecast_samples = np.asarray(samples_F).flatten()
    ground_truth_samples = np.asarray(samples_G).flatten()

    n_g = len(ground_truth_samples)

    if len(forecast_samples) == 0 or n_g == 0:
        return np.nan

    # Calcular CRPS para cada muestra de G y promediar
    crps_values = []
    for g_i in ground_truth_samples:
        crps_i = crps(forecast_samples, g_i)
        crps_values.append(crps_i)
    
    # ECRPS es simplemente el promedio de todos los CRPS
    return np.mean(crps_values)