## Simulación
import numpy as np
import pandas as pd
from scipy.stats import t
from typing import List, Dict, Union, Tuple
import os



# ============================================================================
# 1. LIMPIEZA AGRESIVA DE TENSORFLOW
# ============================================================================
import tensorflow as tf
import torch
import gc
import os
from typing import Dict, Tuple
import pandas as pd

# Configuración inicial más restrictiva
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forzar CPU

# Limitar threads
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def clear_all_sessions():
    """Limpieza completa de todas las sesiones."""
    tf.keras.backend.clear_session()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ============================================================================
# 2. WORKER CON LIMPIEZA FORZADA
# ============================================================================
def run_single_scenario_robust(scenario_with_seed: Dict) -> Tuple[Dict, pd.DataFrame]:
    """Worker con limpieza estricta para modelos no lineales."""
    scenario_config = scenario_with_seed['config']
    seed = scenario_with_seed['seed']
    plot_flag = scenario_with_seed.get('plot', False)
    
    clear_all_sessions()
    
    try:
        pipeline = PipelineOptimizado(
            **{k: v for k, v in scenario_config.items() if k != 'scenario_id'},
            scenario_id=scenario_config.get('scenario_id', 'N/A'),
            seed=seed,
            verbose=False
        )
        
        results_df = pipeline.execute(show_intermediate_plots=plot_flag)
        
        return (scenario_config, results_df)
    
    except Exception as e:
        print(f"ERROR escenario {scenario_config.get('scenario_id', 'N/A')}: {e}")
        return (scenario_config, None)
    
    finally:
        clear_all_sessions()
        import sys
        sys.stdout.flush()
        sys.stderr.flush()

class NonlinearARIMASimulation:
    """
    Genera series temporales no lineales con diferentes tipos de ruido.
    Implementa modelos SETAR (Self-Exciting Threshold Autoregressive) y TAR (Threshold AR).
    """
    def __init__(self, model_type: str = 'SETAR(2,1)', 
                 phi_low: List[float] = [0.5], 
                 phi_high: List[float] = [-0.5],
                 threshold: float = 0.0,
                 delay: int = 1,
                 noise_dist: str = 'normal', 
                 sigma: float = 1.0, 
                 seed: int = None, 
                 verbose: bool = False):
        """
        Inicializa el simulador no lineal.
        
        Args:
            model_type: Tipo de modelo ('SETAR', 'TAR', 'EXPAR', 'BILINEAR')
            phi_low: Coeficientes AR para régimen bajo
            phi_high: Coeficientes AR para régimen alto
            threshold: Umbral para cambio de régimen
            delay: Retardo para la variable umbral
            noise_dist: Distribución del ruido
            sigma: Desviación estándar del ruido
            seed: Semilla aleatoria
        """
        self.model_type = model_type
        self.phi_low = np.array(phi_low)
        self.phi_high = np.array(phi_high)
        self.threshold = threshold
        self.delay = delay
        self.noise_dist = noise_dist
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.series = None
        self.errors = None
        self.regime_history = None

    def model_params(self) -> Dict:
        return {
            'model_type': self.model_type,
            'phi_low': self.phi_low.tolist(),
            'phi_high': self.phi_high.tolist(),
            'threshold': self.threshold,
            'delay': self.delay,
            'sigma': self.sigma,
            'noise_dist': self.noise_dist
        }

    def simulate(self, n: int = 250, burn_in: int = 50, return_just_series: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Simula una serie temporal no lineal."""
        total_length = n + burn_in
        p = max(len(self.phi_low), len(self.phi_high))
        
        errors = self._generate_errors(total_length)
        series = np.zeros(total_length)
        regime_history = np.zeros(total_length, dtype=int)
        
        # Valores iniciales
        series[:p] = self.rng.normal(0, self.sigma, p)
        
        for t in range(p, total_length):
            # Determinar régimen según el modelo
            if 'SETAR' in self.model_type or 'TAR' in self.model_type:
                threshold_var = series[t - self.delay] if t >= self.delay else 0
                regime = 1 if threshold_var > self.threshold else 0
            elif 'EXPAR' in self.model_type:
                # Exponential AR: transición suave entre regímenes
                threshold_var = series[t - self.delay] if t >= self.delay else 0
                weight = 1 / (1 + np.exp(-10 * (threshold_var - self.threshold)))
                regime = 1 if self.rng.random() < weight else 0
            elif 'BILINEAR' in self.model_type:
                # Modelo bilineal: interacción entre serie y ruido
                regime = t  # Marca especial para bilinear
            else:
                regime = 0
            
            regime_history[t] = regime
            
            # Calcular predicción según régimen
            if regime == 0:
                phi = self.phi_low
            elif regime == 1:
                phi = self.phi_high
            else:  # Bilinear
                phi = self.phi_low
            
            # Componente autorregresivo
            ar_part = 0
            for lag in range(len(phi)):
                if t - lag - 1 >= 0:
                    ar_part += phi[lag] * series[t - lag - 1]
            
            # Componente bilineal (si aplica)
            if 'BILINEAR' in self.model_type and t > 0:
                bilinear_term = 0.3 * series[t-1] * errors[t-1]
                series[t] = ar_part + errors[t] + bilinear_term
            else:
                series[t] = ar_part + errors[t]

        if return_just_series:
            return series, None

        self.series = series[burn_in:]
        self.errors = errors[burn_in:]
        self.regime_history = regime_history[burn_in:]
        return self.series, self.errors

    def get_true_next_step_samples(self, series_history: np.ndarray, 
                                   errors_history: np.ndarray,
                                   n_samples: int = 10000) -> np.ndarray:
        """
        Calcula muestras de la distribución real para el siguiente paso.
        Para modelos no lineales, esto requiere simulación Monte Carlo.
        """
        p = max(len(self.phi_low), len(self.phi_high))
        
        if len(series_history) < p:
            return self.rng.normal(np.mean(series_history), self.sigma, n_samples)
        
        samples = np.zeros(n_samples)
        future_errors = self._generate_errors(n_samples)
        
        for i in range(n_samples):
            # Determinar régimen para la predicción
            if 'SETAR' in self.model_type or 'TAR' in self.model_type:
                threshold_var = series_history[-self.delay] if len(series_history) >= self.delay else 0
                regime = 1 if threshold_var > self.threshold else 0
            elif 'EXPAR' in self.model_type:
                threshold_var = series_history[-self.delay] if len(series_history) >= self.delay else 0
                weight = 1 / (1 + np.exp(-10 * (threshold_var - self.threshold)))
                regime = 1 if self.rng.random() < weight else 0
            else:
                regime = 0
            
            phi = self.phi_high if regime == 1 else self.phi_low
            
            # Calcular predicción
            ar_part = np.dot(phi, series_history[-len(phi):][::-1])
            
            if 'BILINEAR' in self.model_type:
                bilinear_term = 0.3 * series_history[-1] * errors_history[-1]
                samples[i] = ar_part + future_errors[i] + bilinear_term
            else:
                samples[i] = ar_part + future_errors[i]
        
        return samples

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


## Plot
# plot.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict
import os


class PlotManager:
    """Clase de utilidad para generar los gráficos del análisis."""
    _STYLE = {'figsize': (14, 6), 'grid_style': {'alpha': 0.3, 'linestyle': ':'},
              'default_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#000000']}

    @classmethod
    def _base_plot(cls, title, xlabel, ylabel, figsize=None):
        """Crea la base para un gráfico estándar y devuelve el objeto figura."""
        fig_size = figsize if figsize else cls._STYLE['figsize']
        fig = plt.figure(figsize=fig_size)
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(**cls._STYLE['grid_style'])
        plt.gca().spines[['top', 'right']].set_visible(False)
        return fig

    @classmethod
    def plot_series_split(cls, series: np.ndarray, burn_in_len: int, test_len: int, title: str = "Serie Temporal Simulada", save_path: str = None):
        """
        Grafica la serie temporal mostrando las divisiones de burn-in, train y test.
        Si se proporciona save_path, guarda el gráfico en lugar de mostrarlo.
        """
        # CORRECCIÓN: Se añade el argumento 'save_path' y se usa un título dinámico
        fig = cls._base_plot(title, "Tiempo", "Valor")
        
        # La serie que se pasa aquí ya no tiene el burn-in, por lo que el eje x debe ajustarse
        total_visible_len = len(series)
        train_len = total_visible_len - test_len
        
        # Ajustar los índices para reflejar el tiempo original, incluyendo burn-in
        time_axis = np.arange(burn_in_len, burn_in_len + total_visible_len)
        
        plt.plot(time_axis, series, label='Serie de Entrenamiento y Test', color=cls._STYLE['default_colors'][0])
        
        # Las regiones coloreadas deben corresponder al eje de tiempo ajustado
        plt.axvspan(burn_in_len, burn_in_len + train_len, color='green', alpha=0.2, label=f'Entrenamiento Inicial ({train_len} puntos)')
        plt.axvspan(burn_in_len + train_len, burn_in_len + total_visible_len, color='red', alpha=0.2, label=f'Test (Ventana Rodante) ({test_len} puntos)')
        
        plt.legend(loc='upper left')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100)
        else:
            plt.show()
            
        plt.close(fig)

    @classmethod
    def plot_density_comparison(cls, distributions: Dict[str, np.ndarray], metric_values: Dict[str, float], title: str, colors: Dict[str, str] = None, save_path: str = None):
        """
        Compara las densidades de las distribuciones predictivas para un único paso de tiempo.
        Si se proporciona save_path, guarda el gráfico en lugar de mostrarlo.
        """
        fig = cls._base_plot(title, "Valor", "Densidad")
        
        if colors is None:
            colors = {name: cls._STYLE['default_colors'][i % len(cls._STYLE['default_colors'])] for i, name in enumerate(distributions.keys())}

        for name, data in distributions.items():
            color = colors.get(name, '#333333') 
            linestyle = '-' if name == 'Teórica' else '--'
            linewidth = 3.0 if name == 'Teórica' else 2.0
            clean_data = data[np.isfinite(data)]
            if len(clean_data) > 1 and np.std(clean_data) > 1e-9:
                sns.kdeplot(clean_data, color=color, label=name, linestyle=linestyle, linewidth=linewidth, warn_singular=False)
            else:
                point_prediction = np.mean(clean_data)
                plt.axvline(point_prediction, color=color, linestyle=linestyle, linewidth=linewidth, label=f'{name} (Puntual)')

        sorted_metrics = sorted(metric_values.items(), key=lambda x: x[1])
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in sorted_metrics])
        
        plt.text(0.98, 0.98, f'ECRPS vs Teórica:\n{metrics_text}', transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        
        plt.legend(loc='upper left', frameon=True)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100)
        else:
            plt.show()
        
        plt.close(fig)


## Metricas

import numpy as np

def ecrps(samples_F: np.ndarray, samples_G: np.ndarray, batch_size: int = 256) -> float:
    """
    Calcula el Energy-based Continuous Ranked Probability Score (ECRPS)
    de manera eficiente en memoria, usando procesamiento por lotes.

    Esta implementación sigue la lógica de promediar el "CRPS" de cada muestra
    del pronóstico contra la distribución empírica real, evitando la creación
    de matrices intermedias muy grandes.

    Args:
        samples_F (np.ndarray): Muestras de la distribución predictiva (F).
        samples_G (np.ndarray): Muestras de la distribución real (G).
        batch_size (int): El tamaño del lote para controlar el uso de memoria. 
                          Un valor más pequeño usa menos memoria pero puede ser un poco más lento.

    Returns:
        El valor escalar del ECRPS.
    """
    # 1. Preparación de los datos
    forecast_samples = np.asarray(samples_F).flatten()
    ground_truth_samples = np.asarray(samples_G).flatten()

    n_f = len(forecast_samples)
    n_g = len(ground_truth_samples)

    if n_f == 0 or n_g == 0:
        return np.nan

    # -----------------------------------------------------------------------------------
    # 2. Cálculo del Término 1: E|X - Y| (Distancia entre pronóstico y realidad)
    #    Aquí implementamos tu lógica de forma óptima en memoria.
    #    En lugar de un bucle `for` muestra por muestra, lo hacemos en lotes para mayor velocidad.
    # -----------------------------------------------------------------------------------
    
    term1_sum = 0.0
    for i in range(0, n_f, batch_size):
        # Tomamos un "lote" de muestras del pronóstico F
        batch_f = forecast_samples[i:i + batch_size]
        
        # Para este lote, calculamos la diferencia absoluta contra TODAS las muestras de G.
        # El broadcasting de NumPy crea una matriz temporal de tamaño (tamaño_lote, n_g),
        # por ejemplo, (256, 20000), que es totalmente manejable en memoria.
        abs_diff_chunk = np.abs(batch_f[:, np.newaxis] - ground_truth_samples)
        
        # Sumamos todas las diferencias de este lote. Esto corresponde a la suma de los
        # "CRPS parciales" para cada muestra `f` en el lote.
        term1_sum += np.sum(abs_diff_chunk)
    
    # Finalmente, calculamos el promedio dividiendo la suma total por el número total de pares (n_f * n_g)
    term1 = term1_sum / (n_f * n_g)

    # -----------------------------------------------------------------------------------
    # 3. Cálculo del Término 2: 0.5 * E|X - X'| (Nitidez del pronóstico)
    #    Este cálculo suele ser seguro para la memoria, ya que la matriz (n_f, n_f),
    #    por ejemplo (5000, 5000), es mucho más pequeña que la matriz (n_f, n_g).
    #    Por tanto, lo mantenemos vectorizado para un rendimiento óptimo.
    # -----------------------------------------------------------------------------------
    abs_diff_forecast_forecast = np.abs(forecast_samples[:, np.newaxis] - forecast_samples)
    term2 = 0.5 * np.mean(abs_diff_forecast_forecast)

    # 4. Devolver el resultado final
    return term1 - term2
## Modelos
import numpy as np
import pandas as pd 
from typing import List, Dict, Union
from statsmodels.tsa.ar_model import AutoReg
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import toeplitz
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge


import numpy as np
import pandas as pd
from typing import Union
from statsmodels.tsa.ar_model import AutoReg

class EnhancedBootstrappingModel:
    """Modelo con Block Bootstrap corregido y evaluación ECRPS para optimización."""
    
    def __init__(self, arma_simulator, random_state=42, verbose=False):
        self.n_lags = None
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(self.random_state)
        self.arma_simulator = arma_simulator
        self.train = None
        self.test = None
        self.mean_val = None
        self.std_val = None

    def prepare_data(self, series):
        """Prepara y normaliza los datos."""
        self.mean_val = np.mean(series)
        self.std_val = np.std(series)
        return (series - self.mean_val) / (self.std_val + 1e-8)

    def denormalize(self, values): 
        """Desnormaliza valores usando media y desviación estándar del entrenamiento."""
        return (values * self.std_val) + self.mean_val

    def _block_bootstrap_predict(self, fitted_model, n_boot):
        """
        Implementación de Block Bootstrap para pronóstico.
        
        Preserva la estructura de dependencia temporal remuestreando bloques
        completos de residuos.
        """
        residuals = fitted_model.resid
        n_resid = len(residuals)
        
        if n_resid < 2: 
            forecast = fitted_model.forecast(steps=1)
            if n_resid == 1:
                return forecast + self.rng.choice(residuals, size=n_boot, replace=True)
            else:
                return np.full(n_boot, forecast)

        block_length = max(1, int(n_resid ** (1/3)))
        forecast = fitted_model.forecast(steps=1)
        bootstrap_forecasts = []
        
        for _ in range(n_boot):
            resampled_residuals = []
            while len(resampled_residuals) < n_resid:
                start_idx = self.rng.integers(0, n_resid - block_length + 1)
                block = residuals[start_idx:start_idx + block_length]
                resampled_residuals.extend(block)
            
            resampled_residuals = np.array(resampled_residuals[:n_resid])
            bootstrap_forecasts.append(forecast + resampled_residuals[-1])
        
        return np.array(bootstrap_forecasts)

    def fit_predict(self, data, n_boot=1000):
        """
        Ajusta el modelo y genera una predicción bootstrap para el siguiente paso.
        """
        normalized_data = self.prepare_data(data)
        
        if len(normalized_data) <= self.n_lags * 2 + 1:
            pred = np.mean(data)
            return np.full(n_boot, pred)
        
        try: 
            fitted_model = AutoReg(normalized_data, lags=self.n_lags, old_names=False).fit()
        except (np.linalg.LinAlgError, ValueError):
            pred = np.mean(data)
            return np.full(n_boot, pred)

        boot_preds_normalized = self._block_bootstrap_predict(fitted_model, n_boot)
        
        return self.denormalize(boot_preds_normalized)

    def optimize_hyperparameters(self, df: Union[pd.DataFrame, np.ndarray], reference_noise: np.ndarray):
        """
        Encuentra el número óptimo de retardos (n_lags) minimizando el ECRPS.
        """
        series = df['valor'].values if isinstance(df, pd.DataFrame) else df
        
        self.mean_val = np.mean(series)
        self.std_val = np.std(series)
        normalized_data = (series - self.mean_val) / (self.std_val + 1e-8)
        
        best_ecrps = float('inf')
        best_lag = 1
        lags_range = range(1, 13)

        for n_lags in lags_range:
            if len(normalized_data) <= 2 * n_lags + 1:
                continue
            
            try:
                fitted_model = AutoReg(normalized_data, lags=n_lags, old_names=False).fit()
                boot_preds_normalized = self._block_bootstrap_predict(fitted_model, n_boot=2000)
                boot_preds = self.denormalize(boot_preds_normalized)
                
                # Aquí se asume que la función ecrps está definida globalmente
                current_ecrps = ecrps(boot_preds, reference_noise)

                if current_ecrps < best_ecrps:
                    best_ecrps = current_ecrps
                    best_lag = n_lags
                    
            except (np.linalg.LinAlgError, ValueError):
                continue
        
        self.n_lags = best_lag
        
        if self.verbose:
            print(f"✅ Opt. Block Bootstrap: Mejor n_lags={best_lag} (ECRPS: {best_ecrps:.4f})")
            
        return best_lag, best_ecrps
    
class LSPM:
    """Least Squares Prediction Machine (LSPM) - Versión Studentized."""
    def __init__(self, random_state=42, verbose=False):
        self.version, self.random_state, self.verbose = 'studentized', random_state, verbose
        self.rng, self.n_lags = np.random.default_rng(random_state), None

    def optimize_hyperparameters(self, df, reference_noise):
        if self.verbose: print(f"✅ Opt. LSPM: Usando '{self.version}'. No se requiere optimización.")
        return None, -1.0
    
    def _calculate_critical_values(self, df):
        values = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        p = self.n_lags if self.n_lags and self.n_lags > 0 else max(1, int(len(values)**(1/3)))
        if len(values) < 2 * p + 2: return []
        y_full, X_full = values[p:], np.array([values[i:i+p] for i in range(len(values) - p)])
        X_train, y_train, x_test, n = X_full[:-1], y_full[:-1], X_full[-1], len(X_full)
        X_train_b, x_test_b = np.c_[np.ones(n - 1), X_train], np.r_[1, x_test]
        X_bar = np.vstack([X_train_b, x_test_b])
        try: H_bar = X_bar @ np.linalg.pinv(X_bar.T @ X_bar) @ X_bar.T
        except np.linalg.LinAlgError: return []
        
        # --- CORRECCIÓN DEL UnboundLocalError ---
        # 1. Asignar h_ii primero
        h_ii = np.diag(H_bar)
        # 2. Luego usar h_ii para asignar las demás variables
        h_n_vec, h_in_vec, h_n = H_bar[-1, :-1], H_bar[:-1, -1], h_ii[-1]
        # ----------------------------------------
        
        critical_values = []
        for i in range(n - 1):
            h_i = h_ii[i]
            if 1 - h_n < 1e-8 or 1 - h_i < 1e-8: continue
            B_i = np.sqrt(1 - h_n) + h_in_vec[i] / np.sqrt(1 - h_i)
            term1 = np.dot(h_n_vec, y_train) / np.sqrt(1 - h_n)
            term2 = (y_train[i] - np.dot(H_bar[i, :-1], y_train)) / np.sqrt(1 - h_i)
            if abs(B_i) > 1e-8: critical_values.append((term1 + term2) / B_i)
        return critical_values

    def fit_predict(self, df):
        critical_values = self._calculate_critical_values(df)
        if not critical_values:
            mean_pred = np.mean(df['valor'].values if isinstance(df, pd.DataFrame) else df)
            return [{'value': mean_pred, 'probability': 1.0}]
        counts = pd.Series(critical_values).value_counts(normalize=True)
        return [{'value': val, 'probability': prob} for val, prob in counts.items()]
    
class LSPMW(LSPM):
    """
    Weighted Least Squares Prediction Machine (LSPM Ponderado).
    
    Extiende el LSPM estándar incorporando pesos exponenciales para proporcionar
    robustez frente a la deriva de la distribución (distribution drift), basándose en
    la teoría de cuantiles ponderados de la predicción conforme no-intercambiable.
    
    Esta implementación es teóricamente más fiel a Barber et al. (2022).
    """
    def __init__(self, rho: float = 0.99, **kwargs):
        """
        Inicializa el LSPM Ponderado.
        
        Args:
            rho (float): Factor de decaimiento para los pesos. Un valor cercano a 1
                         da pesos más uniformes. Un valor más bajo prioriza
                         fuertemente las observaciones recientes. Debe estar en (0, 1).
            **kwargs: Argumentos para la clase base LSPM.
        """
        super().__init__(**kwargs)
        if not (0 < rho < 1):
            raise ValueError("El factor de decaimiento 'rho' debe estar entre 0 y 1.")
        self.rho = rho
        self.best_params = {'rho': rho} # Para consistencia con flujos de optimización

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray]) -> List[Dict[str, float]]:
        """
        Calcula la distribución predictiva usando cuantiles ponderados.

        Los pesos se aplican directamente al construir la distribución de probabilidad
        a partir de los valores críticos, formando una wECDF (weighted Empirical
        Cumulative Distribution Function).
        """
        # 1. Calcular los valores críticos C_i de la manera estándar. Esta parte no cambia.
        critical_values = self._calculate_critical_values(df)
        
        if not critical_values:
            # Si no se pueden calcular, se recurre a la predicción simple de la clase base.
            mean_pred = np.mean(df['valor'].values if isinstance(df, pd.DataFrame) else df)
            return [{'value': mean_pred, 'probability': 1.0}]
            
        n_crit = len(critical_values)


        weights = self.rho ** np.arange(n_crit - 1, -1, -1)

  
        dist_df = pd.DataFrame({
            'value': critical_values,
            'weight': weights
        })

        # Agrupar por valores críticos idénticos y sumar sus pesos.
        weighted_dist = dist_df.groupby('value')['weight'].sum()
        
        # Normalizar los pesos para que sumen 1, creando una distribución de probabilidad.
        total_weight = weighted_dist.sum()
        if total_weight > 1e-9:
            weighted_dist /= total_weight
        else:

            if self.verbose: print("⚠️ WeightedLSPM: La suma de pesos es cero. Usando predicción de media.")
            return super().fit_predict(df)

        return [{'value': val, 'probability': prob} for val, prob in weighted_dist.items()]

    def optimize_hyperparameters(self, df, reference_noise):
        def objective(rho):
            try:
                self.rho = np.clip(rho, 0.5, 0.999)
                dist = self.fit_predict(df)
                if not dist: return -1e10
                values, probs = [d['value'] for d in dist], [d['probability'] for d in dist]
                samples = self.rng.choice(values, size=1000, p=probs, replace=True)
                return -ecrps(samples, reference_noise)
            except Exception: return -1e10
        optimizer = BayesianOptimization(f=objective, pbounds={'rho': (0.7, 0.999)}, random_state=self.random_state, verbose=0)
        try: optimizer.maximize(init_points=5, n_iter=10)
        except Exception: pass
        if optimizer.max and optimizer.max['target'] > -1e9:
            self.rho, best_ecrps = optimizer.max['params']['rho'], -optimizer.max['target']
            self.best_params = {'rho': self.rho}
        else: self.rho, best_ecrps = 0.95, -1
        if self.verbose: print(f"✅ Opt. LSPMW: Rho={self.rho:.3f}, ECRPS={best_ecrps:.4f}")
        return self.rho, best_ecrps
 
 
class DeepARModel:
    def __init__(self, hidden_size=20, n_lags=5, num_layers=1, dropout=0.1, lr=0.01, 
                 batch_size=32, epochs=50, num_samples=1000, random_state=42, verbose=False):
        self.hidden_size, self.n_lags, self.num_layers, self.dropout, self.lr = hidden_size, n_lags, num_layers, dropout, lr
        self.batch_size, self.epochs, self.num_samples = batch_size, epochs, num_samples
        self.model, self.scaler_mean, self.scaler_std, self.random_state, self.verbose, self.best_params = None, None, None, random_state, verbose, {}
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    class _DeepARNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            dropout_to_apply = dropout if num_layers > 1 else 0
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_to_apply)
            self.fc_mu, self.fc_sigma = nn.Linear(hidden_size, 1), nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            mu = self.fc_mu(lstm_out[:, -1, :])
            sigma = torch.exp(self.fc_sigma(lstm_out[:, -1, :])) + 1e-6
            return mu, sigma
    
    def _create_sequences(self, series):
        X, y = [], []
        for i in range(len(series) - self.n_lags):
            X.append(series[i:i + self.n_lags])
            y.append(series[i + self.n_lags])
        return np.array(X), np.array(y)
    
    def optimize_hyperparameters(self, df, reference_noise):
        def objective(n_lags, hidden_size, num_layers, dropout, lr):
            try:
                self.n_lags = max(1, int(n_lags))
                self.hidden_size = max(5, int(hidden_size))
                self.num_layers = max(1, int(num_layers))
                self.dropout = min(0.5, max(0.0, dropout))
                self.lr = max(0.0001, lr)
                
                series = df['valor'].values if isinstance(df, pd.DataFrame) else df
                self.scaler_mean = np.nanmean(series)
                self.scaler_std = np.nanstd(series) + 1e-8
                normalized_series = (series - self.scaler_mean) / self.scaler_std
                
                if len(normalized_series) <= self.n_lags:
                    return -float('inf')
                
                X_train, y_train = self._create_sequences(normalized_series)
                if len(X_train) == 0:
                    return -float('inf')
                
                mu = np.nanmean(y_train)
                sigma = np.nanstd(y_train) if np.nanstd(y_train) > 1e-6 else 1e-6
                predictions = (np.random.normal(mu, sigma, self.num_samples) * self.scaler_std + self.scaler_mean)
                
                return -ecrps(predictions, reference_noise)
            except Exception:
                return -float('inf')
            finally:
                # Limpieza de PyTorch
                if hasattr(self, 'model') and self.model is not None:
                    del self.model
                    self.model = None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        
        optimizer = BayesianOptimization(
            f=objective, 
            pbounds={'n_lags': (1, 10), 'hidden_size': (5, 30), 'num_layers': (1, 3), 
                     'dropout': (0.0, 0.5), 'lr': (0.001, 0.1)}, 
            random_state=self.random_state, 
            verbose=0
        )
        
        try:
            optimizer.maximize(init_points=3, n_iter=7)
        except Exception:
            pass
        
        best_ecrps = -1
        if optimizer.max and optimizer.max['target'] > -float('inf'):
            best_ecrps = -optimizer.max['target']
            self.best_params = {k: v for k, v in optimizer.max['params'].items()}
            self.best_params.update({
                'n_lags': int(self.best_params['n_lags']),
                'hidden_size': int(self.best_params['hidden_size']),
                'num_layers': int(self.best_params['num_layers'])
            })
        else:
            self.best_params = {'n_lags': 5, 'hidden_size': 20, 'num_layers': 1, 'dropout': 0.1, 'lr': 0.01}
        
        if self.verbose:
            print(f"✅ Opt. DeepAR (ECRPS: {best_ecrps:.4f}): {self.best_params}")
        
        return self.best_params, best_ecrps
    
    def fit_predict(self, df):
        try:
            series = df['valor'].values if isinstance(df, pd.DataFrame) else df
            self.scaler_mean = np.nanmean(series)
            self.scaler_std = np.nanstd(series) + 1e-8
            normalized_series = (series - self.scaler_mean) / self.scaler_std
            
            if self.best_params:
                self.__dict__.update(self.best_params)
            
            if len(normalized_series) <= self.n_lags:
                return np.full(self.num_samples, self.scaler_mean)
            
            X_train, y_train = self._create_sequences(normalized_series)
            if X_train.shape[0] < self.batch_size:
                return np.full(self.num_samples, self.scaler_mean)
            
            X_tensor = torch.FloatTensor(X_train.reshape(-1, self.n_lags, 1))
            y_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
            
            self.model = self._DeepARNN(1, self.hidden_size, self.num_layers, self.dropout)
            criterion = nn.GaussianNLLLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            self.model.train()
            for _ in range(self.epochs):
                perm = torch.randperm(X_tensor.size(0))
                for i in range(0, X_tensor.size(0), self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    mu, sigma = self.model(X_tensor[idx])
                    loss = criterion(mu, y_tensor[idx], sigma)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                last_sequence = torch.FloatTensor(normalized_series[-self.n_lags:].reshape(1, self.n_lags, 1))
                mu, sigma = self.model(last_sequence)
            
            result = np.nan_to_num(
                (np.random.normal(mu.item(), sigma.item(), self.num_samples) * self.scaler_std + self.scaler_mean)
            )
            
            return result
        
        except Exception:
            return np.full(self.num_samples, np.nanmean(df))
        
        finally:
            # LIMPIEZA CRÍTICA
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

    def __del__(self):
        """Destructor con limpieza explícita."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        clear_all_sessions()


class SieveBootstrap:
    """
    Implementa el Sieve Bootstrap como un modelo predictivo.
    """
    def __init__(self, p_order: int = 2, n_bootstrap: int = 1000, random_state: int = 42, verbose: bool = False):
        self.p_order = p_order
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)
        self.best_params = {'p_order': self.p_order}
        self.residuals = None
        self.phi_hat = None

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray], h: int = 1) -> List[Dict[str, float]]:
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        if len(series) <= self.p_order:
            if self.verbose: print("⚠️ SieveBootstrap: No hay suficientes datos para ajustar el modelo AR.")
            return [{'value': np.mean(series), 'probability': 1.0}]
        n = len(series)
        X = series - np.mean(series)
        r = [np.mean(X[k:] * X[:n - k]) for k in range(self.p_order + 1)]
        R = toeplitz(r[:self.p_order])
        rho = r[1:self.p_order + 1]
        try: self.phi_hat = np.linalg.solve(R, rho)
        except np.linalg.LinAlgError:
            if self.verbose: print("⚠️ SieveBootstrap: La matriz es singular, usando pseudo-inversa.")
            self.phi_hat = np.linalg.pinv(R) @ rho
        residuals = [X[t] - np.dot(self.phi_hat, X[t-self.p_order:t][::-1]) for t in range(self.p_order, n)]
        self.residuals = np.array(residuals) - np.mean(residuals)
        bootstrap_samples = []
        for _ in range(self.n_bootstrap):
            eps_star = self.rng.choice(self.residuals, size=n + h, replace=True)
            X_star = np.zeros(n + h)
            X_star[:n] = X
            for t in range(n, n + h):
                X_star[t] = np.dot(self.phi_hat, X_star[t-self.p_order:t][::-1]) + eps_star[t]
            bootstrap_samples.append(X_star[-h:] + np.mean(series))
        flat_samples = np.array(bootstrap_samples).flatten()
        counts = pd.Series(flat_samples).value_counts(normalize=True)
        return [{'value': val, 'probability': prob} for val, prob in counts.items()]

    def optimize_hyperparameters(self, df, reference_noise):
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        best_p_order = self.p_order
        best_ecrps = float('inf')
        max_p = min(20, len(series) // 4)
        p_range = range(1, max_p + 1)
        if self.verbose: print(f"✅ Opt. SieveBootstrap: Buscando p_order óptimo en el rango [1, {max_p}]...")

        for p in p_range:
            self.p_order = p
            try:
                dist = self.fit_predict(df, h=1)
                if not dist: continue
                values, probs = [d['value'] for d in dist], [d['probability'] for d in dist]
                samples = self.rng.choice(values, size=1000, p=probs, replace=True)
                
                # --- CAMBIO 2: Usar la función ecrps correcta en lugar del placeholder ---
                current_ecrps = ecrps(samples, reference_noise)

                if current_ecrps < best_ecrps:
                    best_ecrps = current_ecrps
                    best_p_order = p
            except Exception as e:
                if self.verbose: print(f"Error optimizando para p_order={p}: {e}")
                continue
        
        self.p_order = best_p_order
        self.best_params = {'p_order': self.p_order}
        if self.verbose: print(f"✅ Opt. SieveBootstrap: Mejor p_order={self.p_order}, ECRPS={best_ecrps:.4f}")
        return self.p_order, best_ecrps

class AREPD:
    """Autoregressive Encompassing Predictive Distribution."""
    def __init__(self, n_lags=5, rho=0.95, alpha=0.1, poly_degree=2, random_state=42, verbose=False):
        self.n_lags, self.rho, self.alpha, self.poly_degree = n_lags, rho, alpha, poly_degree
        self.mean_val, self.std_val, self.random_state, self.verbose = None, None, random_state, verbose
        self.rng = check_random_state(random_state)
        np.random.seed(random_state)

    def optimize_hyperparameters(self, df, reference_noise):
        def objective(n_lags, rho, poly_degree):
            try:
                self.n_lags, self.rho, self.poly_degree = max(1, int(round(n_lags))), min(0.999, max(0.5, float(rho))), max(1, int(round(poly_degree)))
                if len(df) < self.n_lags * 2: return -1e12
                dist = self.fit_predict(df)
                if not dist: return -1e12
                values, probs = np.array([d['value'] for d in dist]), np.array([d['probability'] for d in dist])
                if probs.sum() <= 0: return -1e12
                samples = self.rng.choice(values, size=1000, p=probs / probs.sum())
                return -ecrps(samples, reference_noise)
            except Exception: return -1e12
        optimizer = BayesianOptimization(f=objective, pbounds={'n_lags': (1, 8), 'rho': (0.6, 0.99), 'poly_degree': (1, 3)}, random_state=self.random_state, allow_duplicate_points=True, verbose=0)
        try: optimizer.maximize(init_points=5, n_iter=10)
        except Exception: pass
        if optimizer.max and optimizer.max['target'] > -1e11:
            best, best_ecrps = optimizer.max['params'], -optimizer.max['target']
            self.n_lags, self.rho, self.poly_degree = int(round(best['n_lags'])), best['rho'], int(round(best['poly_degree']))
        else: self.n_lags, self.rho, self.poly_degree, best_ecrps = 3, 0.85, 2, -1
        if self.verbose: print(f"✅ Opt. AREPD: Lags={self.n_lags}, Rho={self.rho:.3f}, Pol={self.poly_degree}, ECRPS={best_ecrps:.4f}")
        return self.n_lags, self.rho, self.poly_degree, best_ecrps
        
    def _create_lag_matrix(self, values, n_lags, degree=2):
        n = len(values) - n_lags
        if n <= 0: return np.array([]), np.array([])
        y, X_list = values[n_lags:], [np.ones((n, 1))]
        for lag in range(n_lags):
            lagged = values[lag:lag + n].reshape(-1, 1)
            for d in range(1, degree + 1): X_list.append(np.power(lagged, d))
        return np.hstack(X_list), y

    def _Qn_distribution(self, C):
        sorted_C, n, distribution = np.sort(C), len(C), []
        for i in range(n):
            denorm = (sorted_C[i] * self.std_val) + self.mean_val
            lower, upper = i / (n + 1), (i + 1) / (n + 1)
            if i > 0 and np.isclose(sorted_C[i], sorted_C[i-1]):
                distribution[-1]['probability'], distribution[-1]['upper'] = upper - distribution[-1]['lower'], upper
            else:
                distribution.append({'value': denorm, 'lower': lower, 'upper': upper, 'probability': upper - lower})
        return distribution

    def fit_predict(self, df):
        try:
            values = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
            if len(values) < self.n_lags * 2: return []
            self.mean_val, self.std_val = np.nanmean(values), np.nanstd(values) + 1e-8
            normalized = (values - self.mean_val) / self.std_val
            X, y = self._create_lag_matrix(normalized, self.n_lags, self.poly_degree)
            if X.shape[0] == 0: return []
            weights = self.rho ** np.arange(len(y))[::-1]
            model = Ridge(alpha=self.alpha, fit_intercept=False).fit(X, y, sample_weight=weights / (weights.sum() + 1e-8))
            return self._Qn_distribution(model.predict(X))
        except Exception: return []

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Union
from bayes_opt import BayesianOptimization

class MondrianCPSModel:
    """
    Implementación corregida y DETERMINISTA del Mondrian Conformal Predictive System (MCPS)
    siguiendo la teoría del paper original.
    """
    
    def __init__(self, n_lags: int = 10, n_bins: int = 10, test_size: float = 0.25,
                 random_state: int = 42, verbose: bool = False):
        
        self.n_lags = n_lags
        self.n_bins = n_bins
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        # El rng ahora solo se usaría en la optimización, pero la mantenemos por consistencia
        self.rng = np.random.default_rng(random_state) 
        
        self.base_model = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=100, 
            learning_rate=0.05,
            max_depth=4, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        self.best_params = {}

    def _create_lag_matrix(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crea matriz de lags para series temporales."""
        X, y = [], []
        for i in range(len(series) - self.n_lags):
            X.append(series[i:(i + self.n_lags)])
            y.append(series[i + self.n_lags])
        return np.array(X), np.array(y)
    
    def _compute_cdf(self, scores: np.ndarray, y_value: float) -> float:
        """
        Computa la CDF según la fórmula teórica del MCPS de forma DETERMINISTA.
        
        F̂(y|x) = (n + τ)/(N_c + 1) si y ∈ (C_(n), C_(n+1))
        donde τ se fija en 0.5 para garantizar la replicabilidad.
        """
        if len(scores) == 0:
            return 0.5
            
        sorted_scores = np.sort(scores)
        n = np.searchsorted(sorted_scores, y_value, side='right')
        
        # Suavizado DETERMINISTA usando el valor esperado de la variable uniforme
        tau = 0.5 # CORRECCIÓN CLAVE
        
        cdf_value = (n + tau) / (len(scores) + 1)
        
        return np.clip(cdf_value, 0, 1)
    
    def _create_distribution_from_scores(self, scores: np.ndarray) -> List[Dict[str, float]]:
        """
        Crea distribución discreta a partir de scores de calibración.
        """
        if len(scores) == 0:
            return [{'value': 0.0, 'probability': 1.0}]
        
        unique_scores = np.unique(scores)
        
        if len(unique_scores) == 1:
            return [{'value': float(unique_scores[0]), 'probability': 1.0}]
        
        distribution = []
        
        for i, score in enumerate(unique_scores):
            if i == 0:
                prob = self._compute_cdf(scores, score)
            else:
                prob = (self._compute_cdf(scores, score) - 
                       self._compute_cdf(scores, unique_scores[i-1]))
            
            if prob > 1e-10:
                distribution.append({
                    'value': float(score),
                    'probability': float(prob)
                })
        
        total_prob = sum(d['probability'] for d in distribution)
        if total_prob > 0:
            for d in distribution:
                d['probability'] /= total_prob
        
        return distribution if distribution else [{'value': float(np.mean(unique_scores)), 'probability': 1.0}]

    def optimize_hyperparameters(self, df: Union[pd.DataFrame, np.ndarray], 
                                reference_noise: np.ndarray) -> Tuple[Dict, float]:
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df).flatten()

        def objective(n_lags, n_bins):
            try:
                old_lags, old_bins = self.n_lags, self.n_bins
                self.n_lags = max(5, int(n_lags))
                self.n_bins = max(3, int(n_bins))
                
                if len(series) <= self.n_lags * 2:
                    self.n_lags, self.n_bins = old_lags, old_bins
                    return -1e10
                
                dist = self.fit_predict(series)
                if not dist:
                    self.n_lags, self.n_bins = old_lags, old_bins
                    return -1e10
                
                values = [d['value'] for d in dist]
                probs = [d['probability'] for d in dist]
                samples = self.rng.choice(values, size=2000, p=probs, replace=True)
                
                ecrps_score = ecrps(samples, reference_noise)
                
                self.n_lags, self.n_bins = old_lags, old_bins
                
                return -ecrps_score
                
            except Exception:
                return -1e10

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={'n_lags': (5, 20.99), 'n_bins': (3, 15.99)},
            random_state=self.random_state, 
            verbose=0
        )
        
        try:
            optimizer.maximize(init_points=4, n_iter=8)
            if optimizer.max:
                self.best_params = {
                    'n_lags': int(optimizer.max['params']['n_lags']),
                    'n_bins': int(optimizer.max['params']['n_bins'])
                }
                best_ecrps = -optimizer.max['target']
            else:
                best_ecrps = float('inf')
        except Exception:
            best_ecrps = float('inf')
        
        if self.verbose:
            print(f"Optimización MCPS (ECRPS: {best_ecrps:.4f}): {self.best_params}")
            
        return self.best_params, best_ecrps

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray]) -> List[Dict[str, float]]:
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df).flatten()
        
        if self.best_params:
            self.n_lags = self.best_params.get('n_lags', self.n_lags)
            self.n_bins = self.best_params.get('n_bins', self.n_bins)

        if len(series) < self.n_lags * 2:
            mean_val = np.mean(series) if series.size > 0 else 0
            return [{'value': mean_val, 'probability': 1.0}]
        
        X, y = self._create_lag_matrix(series)
        x_test = series[-self.n_lags:].reshape(1, -1)
        
        n_calib = max(10, int(len(X) * self.test_size))
        if n_calib >= len(X):
            return [{'value': np.mean(series), 'probability': 1.0}]

        X_train, X_calib = X[:-n_calib], X[-n_calib:]
        y_train, y_calib = y[:-n_calib], y[-n_calib:]
        
        self.base_model.fit(X_train, y_train)
        
        point_prediction = self.base_model.predict(x_test)[0]
        calib_preds = self.base_model.predict(X_calib)
        
        try:
            _, bin_edges = pd.qcut(calib_preds, self.n_bins, retbins=True, duplicates='drop')
            bin_indices = np.digitize(calib_preds, bins=bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        except (ValueError, IndexError):
            bin_indices = np.zeros(len(calib_preds), dtype=int)
            bin_edges = [-np.inf, np.inf]
        
        test_bin = np.digitize(point_prediction, bins=bin_edges) - 1
        test_bin = np.clip(test_bin, 0, len(bin_edges) - 2)
        
        local_mask = (bin_indices == test_bin)
        
        if not np.any(local_mask) or np.sum(local_mask) < 5:
            local_y = y_calib
            local_preds = calib_preds
        else:
            local_y = y_calib[local_mask]
            local_preds = calib_preds[local_mask]
        
        calibration_scores = point_prediction + (local_y - local_preds)
        
        return self._create_distribution_from_scores(calibration_scores)




import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple
from bayes_opt import BayesianOptimization

class AdaptiveVolatilityMondrianCPS:
    """
    Adaptive Volatility Mondrian Conformal Predictive System (AV-MCPS).

    Este modelo extiende el MCPS estándar creando categorías Mondrian bidimensionales.
    En lugar de agrupar los datos solo por el nivel de la predicción puntual,
    los agrupa simultáneamente por:
    1. El nivel de la predicción (cuantil de la predicción).
    2. La volatilidad local de la serie (cuantil de la desviación estándar reciente).

    Esto permite que el modelo genere distribuciones predictivas mucho más adaptativas,
    produciendo intervalos más amplios en períodos de alta inestabilidad y más estrechos
    en períodos de calma, incluso para el mismo valor de predicción.
    """

    def __init__(self,
                 n_lags: int = 15,
                 n_pred_bins: int = 10,
                 n_vol_bins: int = 5,
                 volatility_window: int = 20,
                 test_size: float = 0.25,
                 random_state: int = 42,
                 verbose: bool = False):
        """
        Inicializa el modelo AV-MCPS.

        Args:
            n_lags (int): Número de observaciones pasadas a usar como características.
            n_pred_bins (int): Número de cuantiles para categorizar las predicciones.
            n_vol_bins (int): Número de cuantiles para categorizar la volatilidad.
            volatility_window (int): Ventana temporal para calcular la volatilidad local.
            test_size (float): Proporción del dataset a usar para calibración.
            random_state (int): Semilla para reproducibilidad.
            verbose (bool): Si es True, imprime información durante la optimización.
        """
        self.n_lags = n_lags
        self.n_pred_bins = n_pred_bins
        self.n_vol_bins = n_vol_bins
        self.volatility_window = volatility_window
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)
        
        # Modelo base robusto y rápido
        self.base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.best_params = {}

    def _create_lag_matrix(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crea la matriz de características (lags) y el vector objetivo."""
        X, y = [], []
        for i in range(len(series) - self.n_lags):
            X.append(series[i:(i + self.n_lags)])
            y.append(series[i + self.n_lags])
        return np.array(X), np.array(y)
    
    def _calculate_volatility(self, series: np.ndarray) -> np.ndarray:
        """Calcula la volatilidad local para cada punto que puede ser predicho."""
        # Usamos pandas rolling para un cálculo eficiente y limpio
        volatility = pd.Series(series).rolling(
            window=self.volatility_window
        ).std().bfill().values
        
        # Devolvemos la volatilidad correspondiente a cada muestra en X
        return volatility[self.n_lags - 1 : -1]

    def _create_distribution_from_scores(self, scores: np.ndarray) -> List[Dict[str, float]]:
        """Crea una distribución de probabilidad discreta a partir de los scores de conformidad."""
        if len(scores) == 0:
            return [{'value': 0.0, 'probability': 1.0}]
        
        counts = pd.Series(scores).value_counts(normalize=True)
        return [{'value': val, 'probability': prob} for val, prob in counts.items()]

    def optimize_hyperparameters(self, df: Union[pd.DataFrame, np.ndarray],
                                 reference_noise: np.ndarray) -> Tuple[Dict, float]:
        """
        Optimiza los hiperparámetros clave del modelo usando optimización Bayesiana.
        """
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df).flatten()

        def objective(n_lags, n_pred_bins, n_vol_bins, volatility_window):
            try:
                # Guardar estado anterior para restaurarlo después
                old_params = (self.n_lags, self.n_pred_bins, self.n_vol_bins, self.volatility_window)
                
                # Asignar nuevos hiperparámetros (asegurando que sean enteros)
                self.n_lags = int(n_lags)
                self.n_pred_bins = int(n_pred_bins)
                self.n_vol_bins = int(n_vol_bins)
                self.volatility_window = int(volatility_window)
                
                # Validar que los parámetros son factibles
                if len(series) <= self.n_lags * 2 or self.volatility_window < 2:
                    self.n_lags, self.n_pred_bins, self.n_vol_bins, self.volatility_window = old_params
                    return -1e10

                dist = self.fit_predict(series)
                if not dist:
                    self.n_lags, self.n_pred_bins, self.n_vol_bins, self.volatility_window = old_params
                    return -1e10
                
                values = [d['value'] for d in dist]
                probs = [d['probability'] for d in dist]
                samples = self.rng.choice(values, size=2000, p=probs, replace=True)
                
                # Asumiendo que ecrps está definida globalmente
                ecrps_score = ecrps(samples, reference_noise)
                
                # Restaurar estado
                self.n_lags, self.n_pred_bins, self.n_vol_bins, self.volatility_window = old_params
                
                return -ecrps_score
            except Exception:
                return -1e10

        # Rangos de búsqueda para los hiperparámetros
        pbounds = {
            'n_lags': (5, 30.99),
            'n_pred_bins': (3, 15.99),
            'n_vol_bins': (2, 10.99),
            'volatility_window': (5, 40.99)
        }
        
        optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=self.random_state, verbose=0)
        
        try:
            optimizer.maximize(init_points=5, n_iter=15)
            if optimizer.max:
                self.best_params = {k: int(v) for k, v in optimizer.max['params'].items()}
                best_ecrps = -optimizer.max['target']
            else:
                best_ecrps = float('inf')
        except Exception:
            best_ecrps = float('inf')
        
        if self.verbose:
            print(f"✅ Opt. AV-MCPS (ECRPS: {best_ecrps:.4f}): {self.best_params}")
            
        return self.best_params, best_ecrps

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray]) -> List[Dict[str, float]]:
        """
        Ajusta el modelo y genera la distribución predictiva para el siguiente paso.
        """
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df).flatten()
        
        # Usar hiperparámetros optimizados si están disponibles
        if self.best_params:
            self.__dict__.update(self.best_params)

        if len(series) < self.n_lags * 2 or len(series) < self.volatility_window:
            mean_val = np.mean(series) if series.size > 0 else 0
            return [{'value': mean_val, 'probability': 1.0}]
        
        # 1. Crear lags y calcular volatilidad
        X, y = self._create_lag_matrix(series)
        volatility_features = self._calculate_volatility(series)
        
        # 2. Preparar datos de prueba y dividir en entrenamiento/calibración
        x_test = series[-self.n_lags:].reshape(1, -1)
        test_volatility = np.std(series[-self.volatility_window:])
        
        n_calib = max(10, int(len(X) * self.test_size))
        if n_calib >= len(X):
            return [{'value': np.mean(series), 'probability': 1.0}]

        X_train, X_calib = X[:-n_calib], X[-n_calib:]
        y_train, y_calib = y[:-n_calib], y[-n_calib:]
        vol_calib = volatility_features[-n_calib:]
        
        # 3. Entrenar modelo base y hacer predicciones
        self.base_model.fit(X_train, y_train)
        point_prediction = self.base_model.predict(x_test)[0]
        calib_preds = self.base_model.predict(X_calib)
        
        # 4. Categorización Mondrian 2D
        try:
            # Bins para las predicciones
            _, pred_bin_edges = pd.qcut(calib_preds, self.n_pred_bins, retbins=True, duplicates='drop')
            # Bins para la volatilidad
            _, vol_bin_edges = pd.qcut(vol_calib, self.n_vol_bins, retbins=True, duplicates='drop')
        except ValueError: # No hay suficientes puntos únicos para crear bins
            return [{'value': float(point_prediction), 'probability': 1.0}]

        # Asignar cada punto de calibración a su bin 2D
        calib_pred_indices = np.digitize(calib_preds, bins=pred_bin_edges[:-1]) -1
        calib_vol_indices = np.digitize(vol_calib, bins=vol_bin_edges[:-1]) -1
        
        # Encontrar el bin 2D para el punto de prueba
        test_pred_bin = np.digitize(point_prediction, bins=pred_bin_edges[:-1]) -1
        test_vol_bin = np.digitize(test_volatility, bins=vol_bin_edges[:-1]) -1

        # 5. Seleccionar scores de conformidad del bin correspondiente
        local_mask = (calib_pred_indices == test_pred_bin) & (calib_vol_indices == test_vol_bin)
        
        # Lógica de fallback robusta
        if np.sum(local_mask) < 5: # Si el bin 2D está casi vacío...
            local_mask = (calib_pred_indices == test_pred_bin) # ...usar solo el bin de predicción (1D)
            if np.sum(local_mask) < 5: # Si incluso ese está vacío...
                local_mask = np.ones_like(calib_preds, dtype=bool) # ...usar todos los datos (conformal estándar)

        local_y = y_calib[local_mask]
        local_preds = calib_preds[local_mask]
        
        # 6. Calcular scores y construir la distribución final
        calibration_scores = point_prediction + (local_y - local_preds)
        
        return self._create_distribution_from_scores(calibration_scores)


import tensorflow as tf
import warnings
# --- Configuración para un entorno limpio ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
import numpy as np
import pandas as pd
from typing import Union

from tensorflow.keras import layers, optimizers, Model
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization
import gc
import tensorflow as tf
import warnings
import os

# Configuración global
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Limitar threads de TensorFlow
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Deshabilitar GPU si causa problemas
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

from tensorflow.keras import layers, optimizers, Model
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization
import gc

class EnCQR_LSTM_Model:
    """
    EnCQR-LSTM con limpieza completa de memoria de TensorFlow.
    """
    def __init__(self, n_lags: int = 24, B: int = 3, units: int = 50, n_layers: int = 2,
                 lr: float = 0.005, batch_size: int = 16, epochs: int = 25, 
                 num_samples: int = 5000, random_state: int = 42, verbose: bool = False):
        self.B, self.n_lags, self.units = B, n_lags, units
        self.n_layers, self.lr, self.batch_size, self.epochs = n_layers, lr, batch_size, epochs
        self.num_samples, self.random_state, self.verbose = num_samples, random_state, verbose
        self.scaler, self.best_params = MinMaxScaler(), {}
        self.rng = np.random.default_rng(random_state)
        self.quantiles = np.round(np.arange(0.1, 1.0, 0.1), 2)
        self.median_idx = np.where(self.quantiles == 0.5)[0][0]
        
        # Configurar semillas
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    def _pin_loss(self, y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.quantiles * error, (self.quantiles - 1) * error), axis=-1)

    def _build_lstm(self):
        """Construye el modelo LSTM."""
        x_in = layers.Input(shape=(self.n_lags, 1))
        x = x_in
        for _ in range(self.n_layers - 1):
            x = layers.LSTM(self.units, return_sequences=True)(x)
        x = layers.LSTM(self.units, return_sequences=False)(x)
        x = layers.Dense(len(self.quantiles))(x)
        model = Model(inputs=x_in, outputs=x)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr), loss=self._pin_loss)
        return model

    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.n_lags):
            X.append(data[i:(i + self.n_lags)])
            y.append(data[i + self.n_lags])
        return np.array(X), np.array(y)

    def _prepare_data(self, series: np.ndarray):
        series_scaled = self.scaler.fit_transform(series.reshape(-1, 1))
        X, y = self._create_sequences(series_scaled)
        n_samples = X.shape[0]
        
        if n_samples < self.B:
            raise ValueError(f"No hay suficientes muestras ({n_samples}) para crear {self.B} lotes.")
        
        batch_size = n_samples // self.B
        train_data_batches = []
        
        for b in range(self.B):
            start = b * batch_size
            end = (b + 1) * batch_size if b < self.B - 1 else n_samples
            train_data_batches.append({'X': X[start:end], 'y': y[start:end]})
        
        return train_data_batches
    
    def optimize_hyperparameters(self, df: pd.DataFrame, reference_noise: np.ndarray):
        series = df['valor'].values
        
        def objective(n_lags, units, B):
            # Limpiar sesión ANTES de cada evaluación
            tf.keras.backend.clear_session()
            gc.collect()
            
            try:
                self.n_lags = max(10, int(n_lags))
                self.units = max(16, int(units))
                self.B = max(2, int(B))
                
                if len(series) <= self.n_lags * self.B:
                    return -1e10
                
                predictions = self.fit_predict(series)
                
                if predictions is not None and len(predictions) > 0:
                    return -ecrps(predictions, reference_noise)
                else:
                    return -1e10
            
            except Exception:
                return -1e10
            
            finally:
                # Limpieza DESPUÉS de cada evaluación
                tf.keras.backend.clear_session()
                gc.collect()

        optimizer = BayesianOptimization(
            f=objective, 
            pbounds={'n_lags': (10, 50), 'units': (20, 80), 'B': (2, 5.99)}, 
            random_state=self.random_state, 
            verbose=0
        )
        
        try:
            optimizer.maximize(init_points=4, n_iter=8)
            best_ecrps = -optimizer.max['target'] if optimizer.max else -1
            if optimizer.max:
                self.best_params = {k: int(v) for k, v in optimizer.max['params'].items()}
        except Exception:
            best_ecrps = -1
            
        if self.verbose:
            print(f"✅ Opt. EnCQR-LSTM (ECRPS: {best_ecrps:.4f}): {self.best_params}")
            
        return self.best_params, best_ecrps

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # Limpiar sesión ANTES de empezar
        tf.keras.backend.clear_session()
        gc.collect()
        
        ensemble_models = []
        
        try:
            series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
            if self.best_params:
                self.__dict__.update(self.best_params)

            if len(series) <= self.n_lags + self.B:
                if self.verbose:
                    print("⚠️ EnCQR-LSTM: No hay suficientes datos.")
                return np.full(self.num_samples, np.mean(series))

            try:
                train_batches = self._prepare_data(series)
            except ValueError as e:
                if self.verbose:
                    print(f"⚠️ EnCQR-LSTM: Error en preparación - {e}")
                return np.full(self.num_samples, np.mean(series))

            loo_preds_median = [[] for _ in range(self.B)]
            
            for b in range(self.B):
                model = self._build_lstm()
                model.fit(
                    train_batches[b]['X'], 
                    train_batches[b]['y'], 
                    epochs=self.epochs, 
                    batch_size=self.batch_size, 
                    verbose=0, 
                    shuffle=False
                )
                ensemble_models.append(model)
                
                for i in range(self.B):
                    if i != b:
                        preds = model.predict(train_batches[i]['X'], verbose=0)
                        loo_preds_median[i].append(preds[:, self.median_idx])

            conformity_scores = []
            for i in range(self.B):
                avg_loo_pred = np.mean(loo_preds_median[i], axis=0).reshape(-1, 1)
                scores = train_batches[i]['y'] - avg_loo_pred
                conformity_scores.extend(scores.flatten())
            conformity_scores = np.array(conformity_scores)

            last_window_scaled = self.scaler.transform(
                series[-self.n_lags:].reshape(-1, 1)
            ).reshape(1, self.n_lags, 1)

            final_preds_median = [
                model.predict(last_window_scaled, verbose=0)[0, self.median_idx] 
                for model in ensemble_models
            ]
            point_prediction_scaled = np.mean(final_preds_median)
            
            predictive_dist_scaled = point_prediction_scaled + self.rng.choice(
                conformity_scores, size=self.num_samples, replace=True
            )
            
            final_samples = self.scaler.inverse_transform(
                predictive_dist_scaled.reshape(-1, 1)
            ).flatten()
            
            return final_samples
            
        except Exception as e:
            if self.verbose:
                print(f"Error en EnCQR-LSTM: {e}")
            return np.full(self.num_samples, np.nanmean(df))
            
        finally:
            # LIMPIEZA CRÍTICA
            for model in ensemble_models:
                del model
            ensemble_models.clear()
            
            tf.keras.backend.clear_session()
            gc.collect()
    
    def __del__(self):
        """Destructor con limpieza explícita."""
        clear_all_sessions()
                
## Pipeline
# pipeline.py
import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from typing import Dict, Any, List, Tuple
import tensorflow as tf

# ============================================================================
# CONFIGURACIÓN GLOBAL PARA LIMITAR RECURSOS Y EVITAR MEMORY LEAKS
# ============================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Limitar threads de TensorFlow para evitar sobrecarga
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ============================================================================
# FUNCIÓN AUXILIAR MEJORADA PARA PREDICCIÓN (SIN PARALELIZACIÓN INTERNA)
# ============================================================================
# ============================================================================
# FUNCIÓN AUXILIAR MEJORADA PARA PREDICCIÓN (SIN PARALELIZACIÓN INTERNA)
# ============================================================================
def _predict_with_model_safe(model_name: str, model_instance: Any, 
                             history_df: pd.DataFrame, history_series: np.ndarray,
                             sim_config: Dict, seed: int) -> Dict:
    """Versión corregida para modelos no lineales."""
    try:
        np.random.seed(seed)
        local_rng = np.random.default_rng(seed)
        
        if model_name == 'Block Bootstrapping':
            # CORRECCIÓN: Usar NonlinearARIMASimulation
            local_simulator = NonlinearARIMASimulation(**sim_config)
            bb_model_step = EnhancedBootstrappingModel(local_simulator, random_state=seed)
            bb_model_step.n_lags = model_instance.n_lags
            bb_model_step.arma_simulator.series = history_series
            
            # Retorna array completo de muestras
            prediction_output = bb_model_step.fit_predict(history_series)
        else:
            prediction_output = model_instance.fit_predict(history_df)
        
        # Procesar salida
        if isinstance(prediction_output, list) and prediction_output and isinstance(prediction_output[0], dict):
            values = [d['value'] for d in prediction_output]
            probs = np.array([d['probability'] for d in prediction_output])
            if np.sum(probs) > 1e-9:
                probs /= np.sum(probs)
            else:
                probs = np.ones(len(values)) / len(values) if values else []
            samples = local_rng.choice(values, size=5000, p=probs, replace=True) if values else np.array([])
        else:
            samples = np.array(prediction_output).flatten()
        
        return {'name': model_name, 'samples': samples, 'error': None}
    
    except Exception as e:
        return {'name': model_name, 'samples': np.array([]), 'error': str(e)}
    
    finally:
        if model_name in ['DeepAR', 'EnCQR-LSTM']:
            tf.keras.backend.clear_session()
        gc.collect()


# ============================================================================
# PIPELINE OPTIMIZADO (SIN PARALELIZACIÓN INTERNA)
# ============================================================================
class PipelineOptimizado:
    """Pipeline adaptado para modelos no lineales."""
    N_TEST_STEPS = 5
    BURN_IN_PERIOD = 50

    def __init__(self, model_type='SETAR(2,1)', phi_low=[0.5], phi_high=[-0.5],
                 threshold=0.0, delay=1, sigma=1.2, noise_dist='t-student', 
                 n_samples=250, scenario_id=None, seed=42, verbose=False):
        self.config = {
            'model_type': model_type, 
            'phi_low': phi_low, 
            'phi_high': phi_high,
            'threshold': threshold,
            'delay': delay,
            'sigma': sigma,
            'noise_dist': noise_dist, 
            'n_samples': n_samples, 
            'seed': seed,
            'verbose': verbose
        }
        self.scenario_id = scenario_id
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.simulator, self.full_series, self.full_errors = None, None, None
        self.rolling_ecrps: List[Dict] = []

    def _setup_models(self) -> Dict:
        """Inicializa modelos."""
        seed = self.config['seed']
        p_order = max(len(self.config['phi_low']), len(self.config['phi_high']))
        
        return {
            'LSPM': LSPM(random_state=seed, verbose=False),
            'LSPMW': LSPMW(random_state=seed, verbose=False),
            'AREPD': AREPD(random_state=seed, verbose=False),
            'DeepAR': DeepARModel(random_state=seed, verbose=False, epochs=15),
            'Block Bootstrapping': EnhancedBootstrappingModel(self.simulator, random_state=seed, verbose=False),
            'Sieve Bootstrap': SieveBootstrap(p_order=p_order, random_state=seed, verbose=False),
            'MCPS': MondrianCPSModel(random_state=seed, verbose=False),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(random_state=seed, verbose=False),
            'EnCQR-LSTM': EnCQR_LSTM_Model(random_state=seed, verbose=False, epochs=6, B=2, units=24, n_layers=1)
        }
    
    def execute(self, show_intermediate_plots=False):
        """Ejecuta pipeline con simulación no lineal."""
        sim_config = {k: v for k, v in self.config.items() 
                     if k not in ['n_samples', 'verbose']}
        
        # CAMBIO CLAVE: Usar NonlinearARIMASimulation
        self.simulator = NonlinearARIMASimulation(**sim_config)
        self.full_series, self.full_errors = self.simulator.simulate(
            n=self.config['n_samples'], burn_in=self.BURN_IN_PERIOD
        )
        
        if show_intermediate_plots:
            plots_directory = "plots_densidades_por_escenario"
            series_plot_filename = os.path.join(
                plots_directory,
                f"escenario_{self.scenario_id}_series_completa.png"
            )
            plot_title = f"Serie Temporal No Lineal - Escenario {self.scenario_id} ({self.config['model_type']})"
            
            PlotManager.plot_series_split(
                series=self.full_series,
                burn_in_len=self.BURN_IN_PERIOD,
                test_len=self.N_TEST_STEPS,
                title=plot_title,
                save_path=series_plot_filename
            )

        initial_train_len = len(self.full_series) - self.N_TEST_STEPS
        initial_train_series = self.full_series[:initial_train_len]
        df_initial_train = pd.DataFrame({'valor': initial_train_series})
        
        all_models = self._setup_models()
        
        initial_errors = self.full_errors[:initial_train_len]
        reference_noise_for_opt = self.simulator.get_true_next_step_samples(
            initial_train_series, initial_errors, 5000
        )

        for name, model in all_models.items():
            if hasattr(model, 'optimize_hyperparameters'):
                try:
                    model.optimize_hyperparameters(df_initial_train, reference_noise_for_opt)
                except Exception as e:
                    if self.verbose:
                        print(f"Error optimizando {name}: {e}")
        
        for t in range(self.N_TEST_STEPS):
            step_t = initial_train_len + t
            
            history_series = self.full_series[:step_t]
            history_errors = self.full_errors[:step_t]
            df_history = pd.DataFrame({'valor': history_series})
            
            theoretical_samples = self.simulator.get_true_next_step_samples(
                history_series, history_errors, 20000
            )

            step_ecrps = {'Paso': t + 1}
            step_distributions = {'Teórica': theoretical_samples}
            
            for name, model in all_models.items():
                result = _predict_with_model_safe(
                    name, model, df_history, history_series,
                    sim_config, self.config['seed'] + t
                )
                
                if result['samples'].size > 0 and result['error'] is None:
                    step_distributions[name] = result['samples']
                    step_ecrps[name] = ecrps(result['samples'], theoretical_samples)
                else:
                    step_ecrps[name] = np.nan
            
            self.rolling_ecrps.append(step_ecrps)
            
            if show_intermediate_plots:
                plots_directory = "plots_densidades_por_escenario"
                plot_filename = os.path.join(
                    plots_directory, 
                    f"escenario_{self.scenario_id}_paso_{t + 1}.png"
                )
                title = f"Densidades Predictivas (No Lineal) - Escenario {self.scenario_id} - Paso {t + 1}"
                metrics_for_plot = {k: v for k, v in step_ecrps.items() if k != 'Paso'}
                
                PlotManager.plot_density_comparison(
                    step_distributions, 
                    metrics_for_plot, 
                    title, 
                    save_path=plot_filename
                )
            
            clear_all_sessions()

        del all_models
        clear_all_sessions()
        
        return self._prepare_results_df()
        
    def _prepare_results_df(self):
        if not self.rolling_ecrps:
            return pd.DataFrame()
        ecrps_df = pd.DataFrame(self.rolling_ecrps).set_index('Paso')
        model_cols = [col for col in ecrps_df.columns if col not in ['Paso', 'Mejor Modelo']]
        ecrps_df['Mejor Modelo'] = ecrps_df[model_cols].idxmin(axis=1)
        averages = ecrps_df[model_cols].mean(numeric_only=True)
        best_overall_model = averages.idxmin()
        ecrps_df.loc['Promedio'] = averages
        ecrps_df.loc['Promedio', 'Mejor Modelo'] = best_overall_model
        return ecrps_df

class ScenarioRunnerMejorado:
    """Runner para escenarios no lineales."""
    def __init__(self, seed=420):
        self.seed = seed
        self.model_names = []
        # Configuración de modelos no lineales ESTACIONARIOS
        self.models_config = [
            # Modelos ya estacionarios
            {'model_type': 'SETAR(2,1)', 'phi_low': [0.6], 'phi_high': [-0.5], 'threshold': 0.0, 'delay': 1},
            {'model_type': 'TAR(2,1)', 'phi_low': [0.7], 'phi_high': [-0.7], 'threshold': 0.0, 'delay': 2},
            {'model_type': 'EXPAR(2,1)', 'phi_low': [0.6], 'phi_high': [-0.4], 'threshold': 0.0, 'delay': 1},
            {'model_type': 'BILINEAR(1)', 'phi_low': [0.5], 'phi_high': [0.5], 'threshold': 0.0, 'delay': 1},
            {'model_type': 'SETAR(2,2)', 'phi_low': [0.5, -0.2], 'phi_high': [-0.3, 0.1], 'threshold': 0.5, 'delay': 1},
            {'model_type': 'TAR(2,2)', 'phi_low': [0.3, 0.1], 'phi_high': [-0.2, -0.1], 'threshold': 1.0, 'delay': 1},
            {'model_type': 'SETAR(2,3)', 'phi_low': [0.4, -0.1, 0.05], 'phi_high': [-0.3, 0.1, -0.05], 'threshold': 0.0, 'delay': 1}
        ]
        self.distributions = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
        self.variances = [0.2, 0.5, 1.0, 3.0]
        self.plots_directory = "plots_densidades_por_escenario"


    def _generate_scenarios(self, n_scenarios: int) -> List[Dict]:
        """Genera escenarios no lineales."""
        all_possible_scenarios = []
        for model in self.models_config:
            for dist in self.distributions:
                for var in self.variances:
                    all_possible_scenarios.append({
                        **model, 
                        'noise_dist': dist, 
                        'sigma': np.sqrt(var)
                    })

        num_to_generate = min(n_scenarios, len(all_possible_scenarios))
        final_scenarios = all_possible_scenarios[:num_to_generate]
        
        return [{'config': {**sc, 'scenario_id': i + 1}, 'seed': self.seed + i} 
                for i, sc in enumerate(final_scenarios)]

    def _prepare_rows_from_result(self, scenario_config: Dict, results_df: pd.DataFrame) -> List[Dict]:
        """Convierte resultados a formato Excel."""
        rows = []
        if results_df is None or results_df.empty:
            return rows
            
        if not self.model_names:
            self.model_names = [col for col in results_df.columns 
                               if col not in ['Paso', 'Mejor Modelo']]
             
        for step, data_row in results_df.iterrows():
            row = {
                'Paso': step,
                'Tipo de Modelo': scenario_config['model_type'],
                'Phi Régimen Bajo': str(scenario_config['phi_low']),
                'Phi Régimen Alto': str(scenario_config['phi_high']),
                'Umbral': scenario_config.get('threshold', 0.0),
                'Retardo': scenario_config.get('delay', 1),
                'Distribución': scenario_config['noise_dist'],
                'Varianza error': np.round(scenario_config['sigma'] ** 2, 2),
                'Mejor Modelo': data_row.get('Mejor Modelo', 'Error')
            }
            for model_name in self.model_names:
                row[model_name] = data_row.get(model_name, np.nan)
            rows.append(row)
        return rows

    def _run_batch(self, batch_scenarios: List[Dict], restart_every: int, plot: bool) -> List[Tuple]:
        """
        Ejecuta un lote de escenarios, reiniciando el pool de procesos periódicamente
        para mantener la estabilidad del sistema.
        """
        batch_results = []
        max_workers = 1 # Se ejecuta secuencialmente dentro del lote, pero los lotes son paralelos
        
        # Dividir el lote en sublotes más pequeños para el reinicio
        sublotes = [batch_scenarios[i:i+restart_every] 
                   for i in range(0, len(batch_scenarios), restart_every)]
        
        for sublote_idx, sublote in enumerate(sublotes):
            print(f"    Sublote {sublote_idx + 1}/{len(sublotes)}")
            
            # Añadir el flag 'plot' a cada escenario antes de enviarlo al worker
            for sc in sublote:
                sc['plot'] = plot

            # Crear un nuevo executor para cada sublote garantiza un entorno limpio
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_single_scenario_robust, sc): sc 
                    for sc in sublote
                }
                
                # Procesar resultados a medida que se completan
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                  total=len(futures), desc="      Escenarios"):
                    try:
                        # Obtener resultado con un timeout para evitar bloqueos indefinidos
                        scenario_config, results_df = future.result(timeout=600)
                        if results_df is not None:
                            batch_results.append((scenario_config, results_df))
                    except concurrent.futures.TimeoutError:
                        sc_id = futures[future]['config'].get('scenario_id', 'N/A')
                        print(f"      Timeout en escenario {sc_id}")
                    except Exception as e:
                        sc_id = futures[future]['config'].get('scenario_id', 'N/A')
                        print(f"      Error en escenario {sc_id}: {e}")
            
            # Limpieza forzada entre sublotes
            clear_all_sessions()
            import time
            time.sleep(2)
        
        return batch_results

    def run(self, n_scenarios: int = 140, excel_filename: str = "resultados_arima_finales.xlsx", 
            batch_size: int = 20, restart_every: int = 5, plot: bool = False):
        """
        Método principal para ejecutar la simulación completa.
        
        Args:
            n_scenarios (int): Número total de escenarios a ejecutar.
            excel_filename (str): Nombre del archivo para guardar los resultados.
            batch_size (int): Número de escenarios a procesar antes de una limpieza mayor.
            restart_every (int): Frecuencia de reinicio del pool de procesos dentro de un lote.
            plot (bool): Si es True, guarda los gráficos de densidad para cada paso de cada escenario.
        """
        if plot:
            os.makedirs(self.plots_directory, exist_ok=True)
            print(f"✅ Gráficos de densidad activados. Se guardarán en la carpeta: '{self.plots_directory}'")
        
        all_scenarios_configs = self._generate_scenarios(n_scenarios)
        all_excel_rows = []
        
        n_batches = (len(all_scenarios_configs) + batch_size - 1) // batch_size
        
        print(f"Ejecutando {len(all_scenarios_configs)} escenarios en {n_batches} lotes.")
        
        for batch_num in range(n_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(all_scenarios_configs))
            batch = all_scenarios_configs[start_idx:end_idx]
            
            print(f"\n{'='*60}")
            print(f"LOTE {batch_num + 1}/{n_batches} (Escenarios {start_idx + 1}-{end_idx})")
            print(f"{'='*60}")
            
            batch_results = self._run_batch(batch, restart_every=restart_every, plot=plot)
            
            for scenario_config, results_df in batch_results:
                new_rows = self._prepare_rows_from_result(scenario_config, results_df)
                all_excel_rows.extend(new_rows)
            
            print(f"  Limpiando memoria del lote...")
            clear_all_sessions()
            import time
            time.sleep(3)
            
            # Guardar un checkpoint periódico para no perder el progreso
            if all_excel_rows and (batch_num + 1) % 3 == 0:
                temp_filename = f"checkpoint_lote_{batch_num + 1}.xlsx"
                pd.DataFrame(all_excel_rows).to_excel(temp_filename, index=False)
                print(f"  ✅ Checkpoint guardado: {temp_filename}")

        if all_excel_rows:
            df_final = pd.DataFrame(all_excel_rows)
            # Definir el orden de las columnas para una mejor legibilidad
            first_cols = ['Paso', 'Tipo de Modelo', 'Valores de AR', 'Valores MA', 'Distribución', 'Varianza error']
            last_col = ['Mejor Modelo']
            
            if not self.model_names and not df_final.empty:
                self.model_names = [col for col in df_final.columns 
                                   if col not in first_cols + last_col]

            model_cols = sorted([m for m in self.model_names if m in df_final.columns])
            ordered_columns = first_cols + model_cols + last_col
            df_final = df_final[[col for col in ordered_columns if col in df_final.columns]]
            
            df_final.to_excel(excel_filename, index=False)
            print(f"✅ {len(all_excel_rows)} filas de resultados guardadas en '{excel_filename}'")
            
            # Generar los gráficos resumen al final de toda la ejecución
            self.plot_results_from_excel(excel_filename)
        else:
            print("❌ No se generaron resultados. Revise la configuración o posibles errores.")

    def plot_results_from_excel(self, filename: str):
        """
        Genera y muestra gráficos resumen (boxplots y diagramas de tarta) a partir
        del archivo de resultados final.
        """
        try:
            df_total = pd.read_excel(filename)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo de resultados '{filename}'. No se pueden generar los gráficos.")
            return

        if df_total.empty:
            print("El archivo de resultados está vacío. No se pueden generar gráficos.")
            return

        # Identificar dinámicamente las columnas de los modelos
        if not self.model_names:
            non_model_cols = ['Paso', 'Tipo de Modelo', 'Valores de AR', 'Valores MA', 'Distribución', 'Varianza error', 'Mejor Modelo']
            self.model_names = [col for col in df_total.columns if col not in non_model_cols]

        pasos_disponibles = df_total['Paso'].unique()

        for step_name in pasos_disponibles:
            df_step = df_total[df_total['Paso'] == step_name].copy()

            # Asegurarse de que las columnas de los modelos sean numéricas
            for col in self.model_names:
                df_step[col] = pd.to_numeric(df_step[col], errors='coerce')
            
            df_melted = df_step.melt(value_vars=self.model_names, var_name='Modelo', value_name='ECRPS').dropna()
            wins = df_step['Mejor Modelo'].value_counts()
            
            title_prefix = (f"Resultados para el Paso {step_name}" if step_name != "Promedio" 
                          else "Resultados Generales (Promedio de Pasos)")
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(title_prefix, fontsize=18)

            # Boxplot de distribución de ECRPS
            sns.boxplot(ax=axes[0], data=df_melted, x='Modelo', y='ECRPS', palette='viridis')
            axes[0].set_title('Distribución de ECRPS por Modelo', fontsize=14)
            axes[0].set_xlabel('')
            axes[0].set_ylabel('ECRPS (menor es mejor)')
            axes[0].tick_params(axis='x', rotation=45, labelsize=11)
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Gráfico de tarta de victorias
            if not wins.empty:
                pie_colors_map = {name: PlotManager._STYLE['default_colors'][i % len(PlotManager._STYLE['default_colors'])] for i, name in enumerate(self.model_names)}
                pie_colors = [pie_colors_map.get(label, '#CCCCCC') for label in wins.index]
                axes[1].pie(wins, labels=wins.index, autopct='%1.1f%%', startangle=140, colors=pie_colors, wedgeprops={"edgecolor":"k",'linewidth': 0.5, 'antialiased': True})
                axes[1].set_title('Porcentaje de Victorias por Modelo', fontsize=14)
            else:
                axes[1].text(0.5, 0.5, 'No hay datos de victorias', ha='center', va='center')
                axes[1].set_title('Porcentaje de Victorias por Modelo', fontsize=14)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            plt.close(fig)