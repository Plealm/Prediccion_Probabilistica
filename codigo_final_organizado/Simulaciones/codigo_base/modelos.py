# modelos.py
import pandas as pd
import numpy as np
from typing import Union, Dict
class CircularBlockBootstrapModel:
    """
    Circular Block Bootstrap (Politis & Romano, 1992)
    tal y como se describe en Lahiri (2003), sección 2.7.1 "Circular Block Bootstrap".
    
    Envuelve la serie circularmente para reducir sesgo de bordes. Hace la serie bootstrap estacionaria.
    Eficiente como MBB (ARE=1), ideal para predicción. n starts posibles.
    Totalmente no paramétrico.
    """

    def __init__(self, 
                 block_length: Union[int, str] = 'auto',
                 n_boot: int = 1000, 
                 random_state: int = 42, 
                 verbose: bool = False,
                 hyperparam_ranges: Dict = None
                 ):
        self.block_length = block_length
        self.n_boot = n_boot
        self.random_state = random_state
        self.verbose = verbose
        self.hyperparam_ranges = hyperparam_ranges or {'block_length': [2, 50]}
        self.rng = np.random.default_rng(random_state)
        self.best_params = {}

    def _count_model_parameters(self, params: Dict) -> int:
        return 2

    def _determine_block_length(self, n: int) -> int:
        if self.block_length == 'auto':
            default_l = max(2, int(round(n ** (1.0 / 3.0))))
            min_l, max_l = self.hyperparam_ranges.get('block_length', [2, n//2])
            return min(max(default_l, min_l), max_l)
        return max(2, int(self.block_length))

    def _get_prediction_samples(self, history: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        series = np.asarray(history).flatten()
        n = len(series)

        if n < 10:
            mean_recent = np.mean(series[-8:] if len(series) >= 8 else series)
            return np.full(n_samples, mean_recent)

        l = self._determine_block_length(n)
        
        # OPTIMIZACIÓN: Generar todos los starts de una vez
        num_blocks = int(np.ceil((n + 1) / l))
        start_indices = self.rng.integers(0, n, size=(n_samples, num_blocks))
        
        # OPTIMIZACIÓN: Vectorizar construcción de bloques
        # Para cada muestra bootstrap, tomamos el valor en posición n de la serie extendida
        predictive_samples = np.empty(n_samples)
        
        for k in range(n_samples):
            # Posición n en la serie extendida (0-indexed)
            block_idx = n // l
            within_block_idx = n % l
            
            if block_idx < num_blocks:
                start = start_indices[k, block_idx]
                predictive_samples[k] = series[(start + within_block_idx) % n]
            else:
                # Caso borde: usar último bloque
                start = start_indices[k, -1]
                predictive_samples[k] = series[(start + within_block_idx) % n]

        return predictive_samples

    def fit_predict(self, history: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        series = history['valor'].values if isinstance(history, pd.DataFrame) else np.asarray(history).flatten()
        return self._get_prediction_samples(series, n_samples=self.n_boot)


class SieveBootstrapModel:
    """
    Sieve Bootstrap (Bühlmann, 1997)
    tal y como se describe en Lahiri (2003), sección 2.10 "Sieve Bootstrap".
    
    Aproxima con AR(p) creciente (sieve). Resamplea residuales IID, genera serie recursiva.
    Más preciso para procesos lineales (AR(∞)), pero restringido vs. block. Trade-off precisión-robustez.
    Semiparamétrico.
    """

    def __init__(self, 
                 order: Union[int, str] = 'auto',
                 n_boot: int = 1000, 
                 random_state: int = 42, 
                 verbose: bool = False,
                 hyperparam_ranges: Dict = None
                 ):
        self.order = order
        self.n_boot = n_boot
        self.random_state = random_state
        self.verbose = verbose
        self.hyperparam_ranges = hyperparam_ranges or {'order': [1, 20]}
        self.rng = np.random.default_rng(random_state)
        self.best_params = {}

    def _count_model_parameters(self, params: Dict) -> int:
        return params.get('order', 1) + 1

    def _determine_order(self, n: int) -> int:
        if self.order == 'auto':
            default_p = max(1, int(np.log(n)))
            min_p, max_p = self.hyperparam_ranges.get('order', [1, 20])
            return min(max(default_p, min_p), max_p)
        return max(1, int(self.order))

    def _get_prediction_samples(self, history: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        from statsmodels.tsa.ar_model import AutoReg
        series = np.asarray(history).flatten()
        n = len(series)

        if n < 10:
            mean_recent = np.mean(series[-8:] if len(series) >= 8 else series)
            return np.full(n_samples, mean_recent)

        p = self._determine_order(n)
        if p >= n - 1:
            p = max(1, n // 2)

        # OPTIMIZACIÓN: Ajustar modelo solo una vez
        model = AutoReg(series, lags=p).fit()
        residuals = model.resid - np.mean(model.resid)
        
        # OPTIMIZACIÓN: Vectorizar bootstrap de residuales
        boot_residuals = self.rng.choice(residuals, size=n_samples)
        
        # OPTIMIZACIÓN: Vectorizar predicción
        last_p = series[-p:]
        ar_prediction = model.params[0] + np.dot(model.params[1:], last_p)
        predictive_samples = ar_prediction + boot_residuals

        return predictive_samples

    def fit_predict(self, history: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        series = history['valor'].values if isinstance(history, pd.DataFrame) else np.asarray(history).flatten()
        return self._get_prediction_samples(series, n_samples=self.n_boot)
