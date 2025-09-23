import numpy as np
from scipy.integrate import simpson

def crps(samples, observation):
    """
    Calcula el Continuous Ranked Probability Score (CRPS) para una distribución
    de pronóstico empírica (samples) y una única observación.
    Esta es una implementación eficiente para muestras.
    """
    samples = np.asarray(samples).flatten()
    observation = np.asarray(observation).flatten()
    
    if len(samples) == 0:
        return np.nan
    if len(observation) != 1:
        raise ValueError("La función crps_single_obs espera una única observación.")
    
    y = observation[0] # Obtener la única observación
    
    samples_sorted = np.sort(samples)
    n = len(samples_sorted)
    
    # Calcular el CRPS usando la fórmula de Gneiting et al. (2005) para muestras
    # CRPS(F, y) = 1/N * sum(|X_i - y|) - 1/(2N^2) * sum(|X_i - X_j|)
    
    term1 = np.sum(np.abs(samples_sorted - y)) / n
    
    # Calcular el segundo término de forma más eficiente (O(N))
    # E[|X - X'|] = sum_{i,j} |X_i - X_j| / N^2
    # Equivalente a: 2/N^2 * sum_{i=1 to N} (2i-N-1) * X_i (para X ordenados)
    
    term2_sum = 0.0
    for i in range(n):
        term2_sum += (2*i + 1 - n) * samples_sorted[i]
    term2 = term2_sum / n

    return term1 - 0.5 * term2

def ecrps(samples_F, samples_G, n_subsample_G=2000):
    """
    Calcula el Expected Continuous Ranked Probability Score (ECRPS) = E_{y ~ G}[CRPS(F, y)].
    Para mejorar el rendimiento, se puede submuestrear samples_G si es muy grande.
    """
    samples_F = np.asarray(samples_F).flatten()
    samples_G = np.asarray(samples_G).flatten()
    
    if len(samples_F) == 0 or len(samples_G) == 0:
        return np.nan

    # Submuestrear samples_G si es demasiado grande para acelerar el cálculo
    if len(samples_G) > n_subsample_G:
        local_rng = np.random.default_rng() # Usar un RNG local para el subsampling
        samples_G_sub = local_rng.choice(samples_G, size=n_subsample_G, replace=False)
    else:
        samples_G_sub = samples_G
        
    # Calcular CRPS para cada observación subsampleada de G y promediar
    return np.mean([crps(samples_F, y) for y in samples_G_sub])