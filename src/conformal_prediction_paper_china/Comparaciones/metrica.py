# metrica.py
import numpy as np
from scipy.integrate import simpson

def crps(samples, observations):
    samples = np.asarray(samples).flatten()
    observations = np.asarray(observations).flatten()
    
    # Asegurar que tengan al menos un elemento
    if len(samples) == 0 or len(observations) == 0:
        return np.nan
    
    samples_sorted = np.sort(samples)
    n = len(samples_sorted)
    crps_sum = 0.0
    
    for obs in observations:
        x = np.concatenate([samples_sorted, [obs]])
        x.sort()
        
        cdf = np.searchsorted(samples_sorted, x, side='right') / n
        heaviside = (x >= obs).astype(float)
        
        # Integración numérica usando np.trapz
        integral = np.trapz((cdf - heaviside)**2, x)
        crps_sum += integral
    
    return crps_sum / len(observations)

def ecrps(samples_F, samples_G):
    """Calcula ECRPS(F, G) = E_{y ~ G}[CRPS(F, y)]."""
    return np.mean([crps(samples_F, y) for y in samples_G])
