# simulacion.py
import pandas as pd
import numpy as np
from scipy.stats import t
from typing import List, Dict, Tuple
import pandas as pd


class ARMASimulation:
    """
    Genera series temporales ARMA con diferentes tipos de ruido y puede proporcionar
    la distribución teórica del siguiente paso, dado el historial completo.
    """
    def __init__(self, model_type: str = 'AR(1)', phi: List[float] = [], theta: List[float] = [], 
                 noise_dist: str = 'normal', sigma: float = 1.0, seed: int = None, verbose: bool = False):
        """
        Inicializa el simulador ARMA.
        """
        self.model_type = model_type
        self.phi = np.array(phi)
        self.theta = np.array(theta)
        self.noise_dist = noise_dist
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.series = None
        self.errors = None

    def model_params(self) -> Dict:
        """Devuelve los parámetros del modelo en un diccionario."""
        return {
            'model_type': self.model_type, 
            'phi': self.phi.tolist(), 
            'theta': self.theta.tolist(), 
            'sigma': self.sigma
        }

    def simulate(self, n: int = 250, burn_in: int = 50, return_just_series: bool = False) -> Tuple[np.ndarray, np.ndarray]:
            """
            Simula una serie temporal ARMA.
            """
            total_length = n + burn_in
            p, q = len(self.phi), len(self.theta)
            errors = self._generate_errors(total_length + q) # Generar suficientes errores para el MA
            series = np.zeros(total_length)
            
            initial_values = self.rng.normal(0, self.sigma, max(p, q))
            if len(initial_values) > 0:
                series[:len(initial_values)] = initial_values

            for t in range(max(p, q), total_length):
                ar_part = np.dot(self.phi, series[t-p:t][::-1]) if p > 0 else 0
                ma_part = np.dot(self.theta, errors[t-q:t][::-1]) if q > 0 else 0
                series[t] = ar_part + ma_part + errors[t]
                
            if return_just_series:
                # Opción para devolver la serie completa (con burn-in) y None para los errores
                return series, None

            self.series = series[burn_in:]
            self.errors = errors[burn_in:burn_in + n] # Asegurarse de que los errores coincidan con la serie
            return self.series, self.errors
    
    def get_true_next_step_samples(self, series_history: np.ndarray, errors_history: np.ndarray, 
                                        n_samples: int = 10000) -> np.ndarray:
        """
        Calcula una muestra grande de la distribución real para el siguiente paso (X_{n+1}).
        
        Args:
            series_history (np.ndarray): El historial de valores de la serie (X_1, ..., X_n).
            errors_history (np.ndarray): El historial de errores que generaron la serie (e_1, ..., e_n).
            n_samples (int): El número de muestras a generar para estimar la densidad.

        Returns:
            Un array de numpy con las muestras de la distribución del siguiente paso.
        """
        p, q = len(self.phi), len(self.theta)

        if len(series_history) < p:
            raise ValueError(f"El historial de la serie es insuficiente para el orden AR(p={p}).")
        if len(errors_history) < q:
            raise ValueError(f"El historial de errores es insuficiente para el orden MA(q={q}).")

        deterministic_part = np.dot(self.phi, series_history[-p:][::-1]) if p > 0 else 0
        ma_part = np.dot(self.theta, errors_history[-q:][::-1]) if q > 0 else 0
        conditional_mean = deterministic_part + ma_part

        future_errors = self._generate_errors(n_samples)
        
        # Devolvemos las muestras directamente para que puedan ser usadas en un gráfico KDE
        return conditional_mean + future_errors

    def _generate_errors(self, n: int) -> np.ndarray:
        """Genera el término de error según la distribución especificada."""
        if self.noise_dist == 'normal':
            return self.rng.normal(0, self.sigma, n)
        if self.noise_dist == 'uniform':
            limit = np.sqrt(3) * self.sigma
            return self.rng.uniform(-limit, limit, size=n)
        if self.noise_dist == 'exponential':
            return self.rng.exponential(scale=self.sigma, size=n) - self.sigma
        if self.noise_dist == 't-student':
            df = 5
            scale_factor = self.sigma * np.sqrt((df - 2) / df)
            return t.rvs(df, scale=scale_factor, size=n, random_state=self.rng)
        elif self.noise_dist == 'mixture':
            n1 = int(n * 0.75)
            n2 = n - n1
            variance_of_means = 0.75 * (-0.25 * self.sigma * 2)**2 + 0.25 * (0.75 * self.sigma * 2)**2
            if self.sigma**2 < variance_of_means:
                raise ValueError("La varianza de la mezcla no puede ser la sigma deseada.")
            component_std = np.sqrt(self.sigma**2 - variance_of_means)
            comp1 = self.rng.normal(-0.25 * self.sigma * 2, component_std, n1)
            comp2 = self.rng.normal(0.75 * self.sigma * 2, component_std, n2)
            mixture = np.concatenate([comp1, comp2])
            self.rng.shuffle(mixture)
            return mixture - np.mean(mixture)
        else:
            raise ValueError(f"Distribución de ruido no soportada: {self.noise_dist}")
    
    def generate_series(self, n_total: int = 505, seed: int = None) -> np.ndarray:
        """
        MÉTODO NUEVO - 100% compatible con el pipeline honesto
        Genera una serie ARMA limpia de n_total puntos útiles (ya sin burn-in)
        Este es el que usa PipelineOptimizado
        """
        if seed is not None:
            old_rng_state = self.rng.bit_generator.state  # guardamos estado actual
            self.rng = np.random.default_rng(seed)
        
        # Usamos tu método simulate existente (¡genial!)
        series, _ = self.simulate(n=n_total, burn_in=200, return_just_series=False)
        
        # Restauramos el estado del RNG si se pasó seed
        if seed is not None:
            self.rng.bit_generator.state = old_rng_state
        
        return series