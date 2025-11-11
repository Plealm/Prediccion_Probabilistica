# ============================================================================
# PARTE 1 DE 3
# Contenido:
# - Importaciones y configuración global.
# - Funciones de limpieza de memoria.
# - Clases base: ARMASimulation, PlotManager.
# - Métrica de evaluación: ecrps.
# - Modelos Predictivos (Grupo 1): EnhancedBootstrappingModel, LSPM, LSPMW,
#   SieveBootstrap, AREPD.
# ============================================================================

# ============================================================================
# 0. IMPORTACIONES Y CONFIGURACIÓN INICIAL
# ============================================================================
import numpy as np
import pandas as pd
from scipy.stats import t
from typing import List, Dict, Union, Tuple, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import concurrent.futures
import warnings
import sys
import time
import gc

# Dependencias de modelos
from statsmodels.tsa.ar_model import AutoReg
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import toeplitz
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# ============================================================================
# 1. LIMPIEZA AGRESIVA DE MEMORIA (ESPECIALMENTE PARA TENSORFLOW)
# ============================================================================
# Configuración inicial más restrictiva
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forzar CPU para evitar problemas de concurrencia en multiprocessing
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning)

# Limitar threads de TensorFlow para estabilidad
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def clear_all_sessions():
    """Limpieza completa de todas las sesiones de Keras/TF y caché de CUDA."""
    tf.keras.backend.clear_session()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ============================================================================
# 2. CLASE DE SIMULACIÓN (ARMASimulation)
# ============================================================================
class ARMASimulation:
    """
    Genera series temporales ARMA y proporciona la distribución teórica
    para cualquier horizonte de pronóstico 'h'.
    """
    def __init__(self, model_type: str = 'AR(1)', phi: List[float] = [], theta: List[float] = [],
                 noise_dist: str = 'normal', sigma: float = 1.0, seed: int = None, verbose: bool = False):
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
        return {'model_type': self.model_type, 'phi': self.phi.tolist(),
                'theta': self.theta.tolist(), 'sigma': self.sigma}

    def simulate(self, n: int = 250, burn_in: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        total_length = n + burn_in
        p, q = len(self.phi), len(self.theta)
        errors = self._generate_errors(total_length + q)
        series = np.zeros(total_length)

        initial_values = self.rng.normal(0, self.sigma, max(p, q))
        if len(initial_values) > 0:
            series[:len(initial_values)] = initial_values

        for t_step in range(max(p, q), total_length):
            ar_part = np.dot(self.phi, series[t_step-p:t_step][::-1]) if p > 0 else 0
            ma_part = np.dot(self.theta, errors[t_step-q:t_step][::-1]) if q > 0 else 0
            series[t_step] = ar_part + ma_part + errors[t_step]

        self.series = series[burn_in:]
        self.errors = errors[burn_in:burn_in + n]
        return self.series, self.errors

    def get_true_future_step_distribution(self, series_history: np.ndarray, errors_history: np.ndarray,
                                          horizon: int, n_samples: int = 10000) -> np.ndarray:
        """
        Calcula la distribución teórica para el paso futuro X_{n+h}.
        Para h > 1, calcula la predicción puntual de forma recursiva y luego añade el ruido.
        """
        p, q = len(self.phi), len(self.theta)
        if len(series_history) < p:
            raise ValueError(f"El historial de la serie es insuficiente para el orden AR(p={p}).")

        series_future = list(series_history)
        
        # Predicción puntual recursiva hasta el horizonte h-1
        for h_step in range(1, horizon):
            ar_part = np.dot(self.phi, series_future[-p:][::-1]) if p > 0 else 0
            # Los errores futuros son desconocidos, su expectativa es 0.
            # Solo usamos errores pasados si q >= h_step
            known_errors_for_ma = errors_history[-(q-h_step+1):] if q >= h_step else []
            ma_part = np.dot(self.theta[:len(known_errors_for_ma)], known_errors_for_ma[::-1]) if len(known_errors_for_ma) > 0 else 0
            
            point_prediction = ar_part + ma_part
            series_future.append(point_prediction)

        # Cálculo final para el horizonte h
        ar_part_final = np.dot(self.phi, series_future[-p:][::-1]) if p > 0 else 0
        known_errors_final = errors_history[-(q-horizon+1):] if q >= horizon else []
        ma_part_final = np.dot(self.theta[:len(known_errors_final)], known_errors_final[::-1]) if len(known_errors_final) > 0 else 0
        conditional_mean = ar_part_final + ma_part_final

        # La incertidumbre del pronóstico viene dada por el ruido en el paso h y el efecto
        # acumulado de los errores de los pasos intermedios.
        # Una simplificación común (y la que usamos aquí) es centrar la distribución
        # de ruido en la media condicional.
        future_errors = self._generate_errors(n_samples)
        
        return conditional_mean + future_errors

    def _generate_errors(self, n: int) -> np.ndarray:
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

# ============================================================================
# 3. CLASE DE GRÁFICOS (PlotManager)
# ============================================================================
class PlotManager:
    _STYLE = {'figsize': (14, 6), 'grid_style': {'alpha': 0.3, 'linestyle': ':'},
              'default_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#000000']}

    @classmethod
    def _base_plot(cls, title, xlabel, ylabel, figsize=None):
        fig_size = figsize if figsize else cls._STYLE['figsize']
        fig = plt.figure(figsize=fig_size)
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(**cls._STYLE['grid_style'])
        plt.gca().spines[['top', 'right']].set_visible(False)
        return fig

    @classmethod
    def plot_density_comparison(cls, distributions: Dict[str, np.ndarray], metric_values: Dict[str, float], title: str, save_path: str = None):
        fig = cls._base_plot(title, "Valor", "Densidad")
        colors = {name: cls._STYLE['default_colors'][i % len(cls._STYLE['default_colors'])] for i, name in enumerate(distributions.keys())}

        for name, data in distributions.items():
            color = colors.get(name, '#333333')
            linestyle = '-' if name == 'Teórica' else '--'
            linewidth = 3.0 if name == 'Teórica' else 2.0
            clean_data = data[np.isfinite(data)]
            if len(clean_data) > 1 and np.std(clean_data) > 1e-9:
                sns.kdeplot(clean_data, color=color, label=name, linestyle=linestyle, linewidth=linewidth, warn_singular=False)
            else:
                point_prediction = np.mean(clean_data) if len(clean_data) > 0 else np.nan
                plt.axvline(point_prediction, color=color, linestyle=linestyle, linewidth=linewidth, label=f'{name} (Puntual)')

        sorted_metrics = sorted([item for item in metric_values.items() if np.isfinite(item[1])], key=lambda x: x[1])
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

# ============================================================================
# 4. MÉTRICA DE EVALUACIÓN Y FUNCIÓN AUXILIAR
# ============================================================================
def ecrps(samples_F: np.ndarray, samples_G: np.ndarray, batch_size: int = 256) -> float:
    """Calcula el Energy-based Continuous Ranked Probability Score (ECRPS)."""
    forecast_samples = np.asarray(samples_F).flatten()
    ground_truth_samples = np.asarray(samples_G).flatten()
    n_f, n_g = len(forecast_samples), len(ground_truth_samples)
    if n_f == 0 or n_g == 0: return np.nan

    term1_sum = 0.0
    for i in range(0, n_f, batch_size):
        batch_f = forecast_samples[i:i + batch_size]
        abs_diff_chunk = np.abs(batch_f[:, np.newaxis] - ground_truth_samples)
        term1_sum += np.sum(abs_diff_chunk)
    term1 = term1_sum / (n_f * n_g)

    abs_diff_forecast_forecast = np.abs(forecast_samples[:, np.newaxis] - forecast_samples)
    term2 = 0.5 * np.mean(abs_diff_forecast_forecast)
    return term1 - term2

def _get_point_pred_from_dist(dist: Union[List[Dict], np.ndarray], rng) -> float:
    """Calcula la media de una distribución de predicción para la estrategia recursiva."""
    if isinstance(dist, np.ndarray):
        return np.mean(dist) if len(dist) > 0 else 0.0
    if isinstance(dist, list) and dist:
        values = np.array([d['value'] for d in dist])
        probs = np.array([d['probability'] for d in dist])
        if np.sum(probs) > 1e-9:
            return np.average(values, weights=probs)
        else:
            return rng.choice(values) if len(values) > 0 else 0.0
    return 0.0

# ============================================================================
# 5. CLASES DE MODELOS PREDICTIVOS (GRUPO 1 - REFRACTORIZADAS)
# ============================================================================
class EnhancedBootstrappingModel:
    """
    Modelo con Block Bootstrap CORRECTAMENTE IMPLEMENTADO para pronóstico multi-step.
    Simula trayectorias completas del proceso AR para capturar la incertidumbre acumulada.
    """
    def __init__(self, arma_simulator, random_state=42, verbose=False):
        self.n_lags = None
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(self.random_state)
        self.arma_simulator = arma_simulator
        self.mean_val = None
        self.std_val = None

    def prepare_data(self, series):
        self.mean_val = np.mean(series)
        self.std_val = np.std(series)
        return (series - self.mean_val) / (self.std_val + 1e-8)

    def denormalize(self, values):
        return (values * self.std_val) + self.mean_val

    def fit_predict(self, df, h: int = 1, n_boot: int = 1000) -> np.ndarray:
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        
        # Determinar número de lags si no está definido
        if self.n_lags is None:
            self.n_lags = max(1, int(len(series) ** (1/3)))

        normalized_data = self.prepare_data(series)

        if len(normalized_data) <= self.n_lags * 2 + 1:
            return np.full(n_boot, np.mean(series))

        try:
            # 1. Ajustar modelo AR a los datos históricos
            fitted_model = AutoReg(normalized_data, lags=self.n_lags, old_names=False).fit()
            residuals = fitted_model.resid
            n_resid = len(residuals)
            
            # 2. Calcular longitud de bloque óptima
            block_length = max(1, int(n_resid ** (1/3)))
            
            # 3. Extraer los coeficientes AR estimados
            ar_coeffs = fitted_model.params[1:]  # Excluir el intercepto
            
            bootstrap_forecasts = []
            
            for _ in range(n_boot):
                # --- SIMULACIÓN CORRECTA DE TRAYECTORIAS MULTI-STEP ---
                
                # 4. Crear una trayectoria de errores remuestreando bloques
                resampled_errors = []
                while len(resampled_errors) < h:
                    start_idx = self.rng.integers(0, n_resid - block_length + 1)
                    block = residuals[start_idx : start_idx + block_length]
                    resampled_errors.extend(block)
                
                # Truncar al tamaño exacto del horizonte
                resampled_errors = resampled_errors[:h]
                
                # 5. Simular la trayectoria completa del proceso AR
                # Inicializar con los últimos p valores observados
                simulated_path = list(normalized_data[-self.n_lags:])
                
                # Generar cada paso futuro de forma recursiva
                for step in range(h):
                    # Predicción AR usando los últimos p valores
                    ar_part = np.dot(ar_coeffs, simulated_path[-self.n_lags:][::-1])
                    
                    # Agregar el error correspondiente de este paso
                    next_value = ar_part + resampled_errors[step]
                    simulated_path.append(next_value)
                
                # 6. La predicción final es el último valor simulado
                final_prediction_at_h = simulated_path[-1]
                bootstrap_forecasts.append(final_prediction_at_h)

            # 7. Desnormalizar y retornar
            return self.denormalize(np.array(bootstrap_forecasts))

        except (np.linalg.LinAlgError, ValueError) as e:
            if self.verbose:
                print(f"Error en Block Bootstrap: {e}")
            return np.full(n_boot, np.mean(series))

    def optimize_hyperparameters(self, df, reference_noise: np.ndarray):
        """Optimiza el número de lags usando validación cruzada."""
        series = df['valor'].values if isinstance(df, pd.DataFrame) else df
        best_ecrps, best_lag = float('inf'), 1
        lags_range = range(1, min(13, len(series) // 4))

        for n_lags in lags_range:
            if len(series) <= 2 * n_lags + 1: 
                continue
            try:
                self.n_lags = n_lags
                boot_preds = self.fit_predict(df, h=1, n_boot=2000)
                current_ecrps = ecrps(boot_preds, reference_noise)
                
                if current_ecrps < best_ecrps:
                    best_ecrps, best_lag = current_ecrps, n_lags
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error optimizando con lag={n_lags}: {e}")
                continue
        
        self.n_lags = best_lag
        return best_lag, best_ecrps

class LSPM:
    """Least Squares Prediction Machine (LSPM) - Versión Studentized con pronóstico multi-step."""
    def __init__(self, random_state=42, verbose=False):
        self.random_state, self.verbose = random_state, verbose
        self.rng = np.random.default_rng(random_state)
        self.n_lags = None

    def optimize_hyperparameters(self, df, reference_noise):
        return None, -1.0
    
    def _calculate_critical_values(self, df):
        values = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        p = self.n_lags if self.n_lags and self.n_lags > 0 else max(1, int(len(values)**(1/3)))
        if len(values) < 2 * p + 2: return []
        y_full = values[p:]
        X_full = np.array([values[i:i+p] for i in range(len(values) - p)])
        X_train, y_train, x_test, n = X_full[:-1], y_full[:-1], X_full[-1], len(X_full)
        X_train_b, x_test_b = np.c_[np.ones(n - 1), X_train], np.r_[1, x_test]
        X_bar = np.vstack([X_train_b, x_test_b])
        try:
            H_bar = X_bar @ np.linalg.pinv(X_bar.T @ X_bar) @ X_bar.T
        except np.linalg.LinAlgError:
            return []
        
        h_ii = np.diag(H_bar)
        h_n_vec, h_in_vec, h_n = H_bar[-1, :-1], H_bar[:-1, -1], h_ii[-1]
        
        critical_values = []
        for i in range(n - 1):
            h_i = h_ii[i]
            if 1 - h_n < 1e-8 or 1 - h_i < 1e-8: continue
            B_i = np.sqrt(1 - h_n) + h_in_vec[i] / np.sqrt(1 - h_i)
            term1 = np.dot(h_n_vec, y_train) / np.sqrt(1 - h_n)
            term2 = (y_train[i] - np.dot(H_bar[i, :-1], y_train)) / np.sqrt(1 - h_i)
            if abs(B_i) > 1e-8:
                critical_values.append((term1 + term2) / B_i)
        return critical_values

    def _fit_predict_one_step(self, series_data: np.ndarray) -> List[Dict[str, float]]:
        """Lógica original para predecir un solo paso."""
        critical_values = self._calculate_critical_values(pd.DataFrame({'valor': series_data}))
        if not critical_values:
            mean_pred = np.mean(series_data)
            return [{'value': mean_pred, 'probability': 1.0}]
        counts = pd.Series(critical_values).value_counts(normalize=True)
        return [{'value': val, 'probability': prob} for val, prob in counts.items()]

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray], h: int = 1) -> List[Dict[str, float]]:
        """Implementa estrategia recursiva para pronóstico multi-step."""
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        current_series = np.copy(series)
        final_dist = None

        for step in range(h):
            dist = self._fit_predict_one_step(current_series)
            if not dist or not dist[0].get('value', None):
                return [{'value': np.mean(series), 'probability': 1.0}]

            if step < h - 1:
                point_pred = _get_point_pred_from_dist(dist, self.rng)
                current_series = np.append(current_series, point_pred)
            else:
                final_dist = dist
        
        return final_dist if final_dist is not None else [{'value': np.mean(series), 'probability': 1.0}]

class LSPMW(LSPM):
    """LSPM Ponderado con pronóstico multi-step."""
    def __init__(self, rho: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        if not (0 < rho < 1):
            raise ValueError("El factor de decaimiento 'rho' debe estar entre 0 y 1.")
        self.rho = rho
        self.best_params = {'rho': rho}

    def _fit_predict_one_step(self, series_data: np.ndarray) -> List[Dict[str, float]]:
        """Sobrescribe la lógica de un paso para incluir pesos."""
        critical_values = self._calculate_critical_values(pd.DataFrame({'valor': series_data}))
        if not critical_values:
            return [{'value': np.mean(series_data), 'probability': 1.0}]
        
        n_crit = len(critical_values)
        weights = self.rho ** np.arange(n_crit - 1, -1, -1)
        dist_df = pd.DataFrame({'value': critical_values, 'weight': weights})
        weighted_dist = dist_df.groupby('value')['weight'].sum()
        
        total_weight = weighted_dist.sum()
        if total_weight > 1e-9:
            weighted_dist /= total_weight
        else:
            return super()._fit_predict_one_step(series_data)

        return [{'value': val, 'probability': prob} for val, prob in weighted_dist.items()]
    
    # El método fit_predict(h) se hereda de LSPM y usará automáticamente
    # el _fit_predict_one_step sobreescrito, por lo que no es necesario re-implementarlo.

    def optimize_hyperparameters(self, df, reference_noise):
        def objective(rho):
            try:
                self.rho = np.clip(rho, 0.5, 0.999)
                dist = self.fit_predict(df, h=1)
                if not dist: return -1e10
                values, probs = [d['value'] for d in dist], [d['probability'] for d in dist]
                samples = self.rng.choice(values, size=1000, p=probs, replace=True)
                return -ecrps(samples, reference_noise)
            except Exception:
                return -1e10
        optimizer = BayesianOptimization(f=objective, pbounds={'rho': (0.7, 0.999)}, random_state=self.random_state, verbose=0)
        try:
            optimizer.maximize(init_points=5, n_iter=10)
        except Exception:
            pass
        if optimizer.max and optimizer.max['target'] > -1e9:
            self.rho, best_ecrps = optimizer.max['params']['rho'], -optimizer.max['target']
            self.best_params = {'rho': self.rho}
        else:
            self.rho, best_ecrps = 0.95, -1
        return self.rho, best_ecrps

class SieveBootstrap:
    """Sieve Bootstrap con pronóstico multi-step directo."""
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
        
        try:
            self.phi_hat = np.linalg.solve(R, rho)
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
            for t_step in range(n, n + h):
                X_star[t_step] = np.dot(self.phi_hat, X_star[t_step-self.p_order:t_step][::-1]) + eps_star[t_step]
            bootstrap_samples.append(X_star[-1] + np.mean(series)) # El valor en el horizonte h
        
        flat_samples = np.array(bootstrap_samples).flatten()
        counts = pd.Series(flat_samples).value_counts(normalize=True)
        return [{'value': val, 'probability': prob} for val, prob in counts.items()]

    def optimize_hyperparameters(self, df, reference_noise):
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        best_p_order = self.p_order
        best_ecrps = float('inf')
        max_p = min(20, len(series) // 4)
        p_range = range(1, max_p + 1)

        for p in p_range:
            self.p_order = p
            try:
                dist = self.fit_predict(df, h=1)
                if not dist: continue
                values, probs = [d['value'] for d in dist], [d['probability'] for d in dist]
                samples = self.rng.choice(values, size=1000, p=probs, replace=True)
                current_ecrps = ecrps(samples, reference_noise)
                if current_ecrps < best_ecrps:
                    best_ecrps, best_p_order = current_ecrps, p
            except Exception:
                continue
        
        self.p_order = best_p_order
        self.best_params = {'p_order': self.p_order}
        return self.p_order, best_ecrps

class AREPD:
    """Autoregressive Encompassing Predictive Distribution con pronóstico multi-step."""
    def __init__(self, n_lags=5, rho=0.95, alpha=0.1, poly_degree=2, random_state=42, verbose=False):
        self.n_lags, self.rho, self.alpha, self.poly_degree = n_lags, rho, alpha, poly_degree
        self.mean_val, self.std_val, self.random_state, self.verbose = None, None, random_state, verbose
        self.rng = check_random_state(random_state)

    def _create_lag_matrix(self, values, n_lags, degree=2):
        n = len(values) - n_lags
        if n <= 0: return np.array([]), np.array([])
        y = values[n_lags:]
        X_list = [np.ones((n, 1))]
        for lag in range(n_lags):
            lagged = values[lag:lag + n].reshape(-1, 1)
            for d in range(1, degree + 1):
                X_list.append(np.power(lagged, d))
        return np.hstack(X_list), y

    def _Qn_distribution(self, C):
        sorted_C = np.sort(C)
        n = len(C)
        distribution = []
        for i in range(n):
            denorm = (sorted_C[i] * self.std_val) + self.mean_val
            lower = i / (n + 1)
            upper = (i + 1) / (n + 1)
            if i > 0 and np.isclose(sorted_C[i], sorted_C[i-1]):
                distribution[-1]['probability'] = upper - distribution[-1]['lower']
                distribution[-1]['upper'] = upper
            else:
                distribution.append({'value': denorm, 'lower': lower, 'upper': upper, 'probability': upper - lower})
        return distribution

    def _fit_predict_one_step(self, series_data: np.ndarray) -> List[Dict[str, float]]:
        """Lógica original para predecir un solo paso."""
        try:
            if len(series_data) < self.n_lags * 2: return []
            self.mean_val, self.std_val = np.nanmean(series_data), np.nanstd(series_data) + 1e-8
            normalized = (series_data - self.mean_val) / self.std_val
            X, y = self._create_lag_matrix(normalized, self.n_lags, self.poly_degree)
            if X.shape[0] == 0: return []
            weights = self.rho ** np.arange(len(y))[::-1]
            model = Ridge(alpha=self.alpha, fit_intercept=False).fit(X, y, sample_weight=weights / (weights.sum() + 1e-8))
            return self._Qn_distribution(model.predict(X))
        except Exception:
            return []

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray], h: int = 1) -> List[Dict[str, float]]:
        """Implementa estrategia recursiva para pronóstico multi-step."""
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        current_series = np.copy(series)
        final_dist = None

        for step in range(h):
            dist = self._fit_predict_one_step(current_series)
            if not dist:
                return [{'value': np.mean(series), 'probability': 1.0}]

            if step < h - 1:
                point_pred = _get_point_pred_from_dist(dist, self.rng)
                current_series = np.append(current_series, point_pred)
            else:
                final_dist = dist
                
        return final_dist if final_dist is not None else [{'value': np.mean(series), 'probability': 1.0}]

    def optimize_hyperparameters(self, df, reference_noise):
        def objective(n_lags, rho, poly_degree):
            try:
                self.n_lags, self.rho, self.poly_degree = max(1, int(round(n_lags))), min(0.999, max(0.5, float(rho))), max(1, int(round(poly_degree)))
                if len(df) < self.n_lags * 2: return -1e12
                dist = self.fit_predict(df, h=1)
                if not dist: return -1e12
                values, probs = np.array([d['value'] for d in dist]), np.array([d['probability'] for d in dist])
                if probs.sum() <= 0: return -1e12
                probs /= probs.sum()
                samples = self.rng.choice(values, size=1000, p=probs)
                return -ecrps(samples, reference_noise)
            except Exception:
                return -1e12
        optimizer = BayesianOptimization(f=objective, pbounds={'n_lags': (1, 8), 'rho': (0.6, 0.99), 'poly_degree': (1, 3)}, random_state=self.random_state, allow_duplicate_points=True, verbose=0)
        try:
            optimizer.maximize(init_points=5, n_iter=10)
        except Exception:
            pass
        if optimizer.max and optimizer.max['target'] > -1e11:
            best, best_ecrps = optimizer.max['params'], -optimizer.max['target']
            self.n_lags, self.rho, self.poly_degree = int(round(best['n_lags'])), best['rho'], int(round(best['poly_degree']))
        else:
            self.n_lags, self.rho, self.poly_degree, best_ecrps = 3, 0.85, 2, -1
        return self.n_lags, self.rho, self.poly_degree, best_ecrps
    
# ============================================================================
# PARTE 2 DE 3
# Contenido:
# - Modelos Predictivos (Grupo 2): Modelos más complejos basados en
#   machine learning y deep learning.
#   - MondrianCPSModel (MCPS)
#   - AdaptiveVolatilityMondrianCPS (AV-MCPS)
#   - DeepARModel
#   - EnCQR_LSTM_Model
# ============================================================================

class MondrianCPSModel:
    """
    Mondrian Conformal Predictive System (MCPS) con pronóstico multi-step.
    """
    def __init__(self, n_lags: int = 10, n_bins: int = 10, test_size: float = 0.25,
                 random_state: int = 42, verbose: bool = False):
        self.n_lags = n_lags
        self.n_bins = n_bins
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)
        self.base_model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=100, learning_rate=0.05,
            max_depth=4, random_state=self.random_state, n_jobs=1,
            # Limitar recursos para estabilidad
            nthread=1
        )
        self.best_params = {}

    def _create_lag_matrix(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        if len(series) < self.n_lags + 1:
            return np.array([]), np.array([])
        for i in range(len(series) - self.n_lags):
            X.append(series[i:(i + self.n_lags)])
            y.append(series[i + self.n_lags])
        return np.array(X), np.array(y)

    def _create_distribution_from_scores(self, scores: np.ndarray) -> List[Dict[str, float]]:
        if len(scores) == 0:
            return [{'value': 0.0, 'probability': 1.0}]
        counts = pd.Series(scores).value_counts(normalize=True)
        return [{'value': val, 'probability': prob} for val, prob in counts.items()]

    def _fit_predict_one_step(self, series: np.ndarray) -> List[Dict[str, float]]:
        """Lógica original para predecir un solo paso."""
        if self.best_params:
            self.n_lags = self.best_params.get('n_lags', self.n_lags)
            self.n_bins = self.best_params.get('n_bins', self.n_bins)

        if len(series) < self.n_lags * 2:
            mean_val = np.mean(series) if series.size > 0 else 0
            return [{'value': mean_val, 'probability': 1.0}]

        X, y = self._create_lag_matrix(series)
        if X.shape[0] == 0:
             return [{'value': np.mean(series), 'probability': 1.0}]

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
        except (ValueError, IndexError):
            bin_edges = [-np.inf, np.inf]
        
        bin_indices = np.digitize(calib_preds, bins=bin_edges[:-1]) - 1
        test_bin = np.digitize(point_prediction, bins=bin_edges[:-1]) - 1

        local_mask = (bin_indices == test_bin)
        if np.sum(local_mask) < 5:  # Fallback si el bin tiene muy pocas muestras
            local_y, local_preds = y_calib, calib_preds
        else:
            local_y, local_preds = y_calib[local_mask], calib_preds[local_mask]

        calibration_scores = point_prediction + (local_y - local_preds)
        return self._create_distribution_from_scores(calibration_scores)

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray], h: int = 1) -> List[Dict[str, float]]:
        """Implementa estrategia recursiva para pronóstico multi-step."""
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        current_series = np.copy(series)
        final_dist = None

        for step in range(h):
            dist = self._fit_predict_one_step(current_series)
            if not dist:
                return [{'value': np.mean(series), 'probability': 1.0}]
            
            if step < h - 1:
                point_pred = _get_point_pred_from_dist(dist, self.rng)
                current_series = np.append(current_series, point_pred)
            else:
                final_dist = dist
        
        return final_dist if final_dist is not None else [{'value': np.mean(series), 'probability': 1.0}]

    def optimize_hyperparameters(self, df: Union[pd.DataFrame, np.ndarray], reference_noise: np.ndarray) -> Tuple[Dict, float]:
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df).flatten()

        def objective(n_lags, n_bins):
            try:
                self.n_lags = max(5, int(n_lags))
                self.n_bins = max(3, int(n_bins))
                if len(series) <= self.n_lags * 2: return -1e10
                
                dist = self.fit_predict(series, h=1)
                if not dist: return -1e10
                
                values = [d['value'] for d in dist]
                probs = [d['probability'] for d in dist]
                samples = self.rng.choice(values, size=2000, p=probs, replace=True)
                return -ecrps(samples, reference_noise)
            except Exception:
                return -1e10

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={'n_lags': (5, 20.99), 'n_bins': (3, 15.99)},
            random_state=self.random_state, verbose=0)
        
        try:
            optimizer.maximize(init_points=4, n_iter=8)
            if optimizer.max:
                self.best_params = {'n_lags': int(optimizer.max['params']['n_lags']), 'n_bins': int(optimizer.max['params']['n_bins'])}
                best_ecrps = -optimizer.max['target']
            else: best_ecrps = float('inf')
        except Exception: best_ecrps = float('inf')
        
        # Reset to optimized params
        if self.best_params:
            self.n_lags = self.best_params.get('n_lags')
            self.n_bins = self.best_params.get('n_bins')

        return self.best_params, best_ecrps

class AdaptiveVolatilityMondrianCPS:
    """AV-MCPS con pronóstico multi-step."""
    def __init__(self, n_lags: int = 15, n_pred_bins: int = 10, n_vol_bins: int = 5,
                 volatility_window: int = 20, test_size: float = 0.25,
                 random_state: int = 42, verbose: bool = False):
        self.n_lags, self.n_pred_bins, self.n_vol_bins = n_lags, n_pred_bins, n_vol_bins
        self.volatility_window, self.test_size = volatility_window, test_size
        self.random_state, self.verbose = random_state, verbose
        self.rng = np.random.default_rng(random_state)
        self.base_model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=150, learning_rate=0.05,
            max_depth=4, subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_state, n_jobs=1, nthread=1)
        self.best_params = {}

    def _create_lag_matrix(self, series: np.ndarray):
        X, y = [], []
        if len(series) < self.n_lags + 1: return np.array([]), np.array([])
        for i in range(len(series) - self.n_lags):
            X.append(series[i:(i + self.n_lags)])
            y.append(series[i + self.n_lags])
        return np.array(X), np.array(y)
    
    def _calculate_volatility(self, series: np.ndarray):
        if len(series) < self.n_lags: return np.array([])
        volatility = pd.Series(series).rolling(window=self.volatility_window).std().bfill().values
        return volatility[self.n_lags - 1 : -1]

    def _create_distribution_from_scores(self, scores: np.ndarray):
        if len(scores) == 0: return [{'value': 0.0, 'probability': 1.0}]
        counts = pd.Series(scores).value_counts(normalize=True)
        return [{'value': val, 'probability': prob} for val, prob in counts.items()]

    def _fit_predict_one_step(self, series: np.ndarray) -> List[Dict[str, float]]:
        """Lógica original para predecir un solo paso."""
        if self.best_params: self.__dict__.update(self.best_params)

        if len(series) < self.n_lags * 2 or len(series) < self.volatility_window:
            return [{'value': np.mean(series) if series.size > 0 else 0, 'probability': 1.0}]
        
        X, y = self._create_lag_matrix(series)
        if X.shape[0] == 0: return [{'value': np.mean(series), 'probability': 1.0}]
        
        volatility_features = self._calculate_volatility(series)
        
        x_test = series[-self.n_lags:].reshape(1, -1)
        test_volatility = np.std(series[-self.volatility_window:])
        
        n_calib = max(10, int(len(X) * self.test_size))
        if n_calib >= len(X): return [{'value': np.mean(series), 'probability': 1.0}]

        X_train, X_calib = X[:-n_calib], X[-n_calib:]
        y_train, y_calib = y[:-n_calib], y[-n_calib:]
        vol_calib = volatility_features[-n_calib:]
        
        self.base_model.fit(X_train, y_train)
        point_prediction = self.base_model.predict(x_test)[0]
        calib_preds = self.base_model.predict(X_calib)
        
        try:
            _, pred_bin_edges = pd.qcut(calib_preds, self.n_pred_bins, retbins=True, duplicates='drop')
            _, vol_bin_edges = pd.qcut(vol_calib, self.n_vol_bins, retbins=True, duplicates='drop')
        except ValueError:
            return [{'value': float(point_prediction), 'probability': 1.0}]

        calib_pred_indices = np.digitize(calib_preds, bins=pred_bin_edges[:-1]) -1
        calib_vol_indices = np.digitize(vol_calib, bins=vol_bin_edges[:-1]) -1
        
        test_pred_bin = np.digitize(point_prediction, bins=pred_bin_edges[:-1]) -1
        test_vol_bin = np.digitize(test_volatility, bins=vol_bin_edges[:-1]) -1

        local_mask = (calib_pred_indices == test_pred_bin) & (calib_vol_indices == test_vol_bin)
        
        if np.sum(local_mask) < 5:
            local_mask = (calib_pred_indices == test_pred_bin)
            if np.sum(local_mask) < 5:
                local_mask = np.ones_like(calib_preds, dtype=bool)

        local_y, local_preds = y_calib[local_mask], calib_preds[local_mask]
        calibration_scores = point_prediction + (local_y - local_preds)
        return self._create_distribution_from_scores(calibration_scores)

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray], h: int = 1) -> List[Dict[str, float]]:
        """Implementa estrategia recursiva para pronóstico multi-step."""
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        current_series = np.copy(series)
        final_dist = None

        for step in range(h):
            dist = self._fit_predict_one_step(current_series)
            if not dist: return [{'value': np.mean(series), 'probability': 1.0}]
            
            if step < h - 1:
                point_pred = _get_point_pred_from_dist(dist, self.rng)
                current_series = np.append(current_series, point_pred)
            else:
                final_dist = dist
        
        return final_dist if final_dist is not None else [{'value': np.mean(series), 'probability': 1.0}]

    def optimize_hyperparameters(self, df: Union[pd.DataFrame, np.ndarray], reference_noise: np.ndarray) -> Tuple[Dict, float]:
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df).flatten()

        def objective(n_lags, n_pred_bins, n_vol_bins, volatility_window):
            try:
                # Asignar nuevos hiperparámetros
                self.n_lags = int(n_lags)
                self.n_pred_bins = int(n_pred_bins)
                self.n_vol_bins = int(n_vol_bins)
                self.volatility_window = int(volatility_window)
                
                if len(series) <= self.n_lags * 2 or self.volatility_window < 2: return -1e10

                dist = self.fit_predict(series, h=1)
                if not dist: return -1e10
                
                values = [d['value'] for d in dist]
                probs = [d['probability'] for d in dist]
                samples = self.rng.choice(values, size=2000, p=probs, replace=True)
                return -ecrps(samples, reference_noise)
            except Exception:
                return -1e10

        pbounds = {'n_lags': (5, 30.99), 'n_pred_bins': (3, 15.99),
                   'n_vol_bins': (2, 10.99), 'volatility_window': (5, 40.99)}
        
        optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=self.random_state, verbose=0)
        
        try:
            optimizer.maximize(init_points=5, n_iter=15)
            if optimizer.max:
                self.best_params = {k: int(v) for k, v in optimizer.max['params'].items()}
                best_ecrps = -optimizer.max['target']
            else: best_ecrps = float('inf')
        except Exception: best_ecrps = float('inf')

        if self.best_params: self.__dict__.update(self.best_params)
        return self.best_params, best_ecrps

class DeepARModel:
    """DeepAR con LSTM y pronóstico multi-step."""
    def __init__(self, hidden_size=20, n_lags=5, num_layers=1, dropout=0.1, lr=0.01,
                 batch_size=32, epochs=50, num_samples=1000, random_state=42, verbose=False):
        self.hidden_size, self.n_lags, self.num_layers = hidden_size, n_lags, num_layers
        self.dropout, self.lr, self.batch_size, self.epochs = dropout, lr, batch_size, epochs
        self.num_samples, self.random_state, self.verbose = num_samples, random_state, verbose
        self.model, self.scaler_mean, self.scaler_std, self.best_params = None, None, None, {}
        torch.manual_seed(random_state)
        self.rng = np.random.default_rng(random_state)

    class _DeepARNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc_mu = nn.Linear(hidden_size, 1)
            self.fc_sigma = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            mu = self.fc_mu(lstm_out[:, -1, :])
            sigma = torch.exp(self.fc_sigma(lstm_out[:, -1, :])) + 1e-6
            return mu, sigma

    def _create_sequences(self, series):
        X, y = [], []
        if len(series) < self.n_lags + 1: return np.array([]), np.array([])
        for i in range(len(series) - self.n_lags):
            X.append(series[i:i + self.n_lags])
            y.append(series[i + self.n_lags])
        return np.array(X), np.array(y)

    def _fit_predict_one_step(self, series: np.ndarray) -> np.ndarray:
        """Lógica original para predecir un solo paso."""
        try:
            self.scaler_mean, self.scaler_std = np.nanmean(series), np.nanstd(series) + 1e-8
            normalized_series = (series - self.scaler_mean) / self.scaler_std
            
            if self.best_params: self.__dict__.update(self.best_params)
            
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
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                last_sequence = torch.FloatTensor(normalized_series[-self.n_lags:].reshape(1, self.n_lags, 1))
                mu, sigma = self.model(last_sequence)
            
            return np.nan_to_num((self.rng.normal(mu.item(), sigma.item(), self.num_samples) * self.scaler_std + self.scaler_mean))
        except Exception:
            return np.full(self.num_samples, np.nanmean(series))
        finally:
            if hasattr(self, 'model') and self.model is not None: del self.model; self.model = None
            clear_all_sessions()

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray], h: int = 1) -> np.ndarray:
        """Implementa estrategia recursiva para pronóstico multi-step."""
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        current_series = np.copy(series)
        final_dist = None

        for step in range(h):
            dist = self._fit_predict_one_step(current_series)
            if step < h - 1:
                point_pred = _get_point_pred_from_dist(dist, self.rng)
                current_series = np.append(current_series, point_pred)
            else:
                final_dist = dist
        return final_dist if final_dist is not None else np.full(self.num_samples, np.mean(series))

    def optimize_hyperparameters(self, df, reference_noise):
        # La optimización es costosa, se puede simplificar o usar valores por defecto
        self.best_params = {'n_lags': 10, 'hidden_size': 25, 'num_layers': 1, 'dropout': 0.1, 'lr': 0.01}
        self.__dict__.update(self.best_params)
        return self.best_params, -1.0


class EnCQR_LSTM_Model:
    """EnCQR-LSTM con pronóstico multi-step y limpieza de memoria."""
    def __init__(self, n_lags: int = 24, B: int = 3, units: int = 50, n_layers: int = 2,
                 lr: float = 0.005, batch_size: int = 16, epochs: int = 25,
                 num_samples: int = 5000, random_state: int = 42, verbose: bool = False):
        self.B, self.n_lags, self.units, self.n_layers = B, n_lags, units, n_layers
        self.lr, self.batch_size, self.epochs = lr, batch_size, epochs
        self.num_samples, self.random_state, self.verbose = num_samples, random_state, verbose
        self.scaler, self.best_params = MinMaxScaler(), {}
        self.rng = np.random.default_rng(random_state)
        self.quantiles = np.round(np.arange(0.1, 1.0, 0.1), 2)
        self.median_idx = np.where(self.quantiles == 0.5)[0][0]
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    def _pin_loss(self, y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.quantiles * error, (self.quantiles - 1) * error), axis=-1)

    def _build_lstm(self):
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
        if len(data) < self.n_lags + 1: return np.array([]), np.array([])
        for i in range(len(data) - self.n_lags):
            X.append(data[i:(i + self.n_lags)])
            y.append(data[i + self.n_lags])
        return np.array(X), np.array(y)

    def _prepare_data(self, series: np.ndarray):
        series_scaled = self.scaler.fit_transform(series.reshape(-1, 1))
        X, y = self._create_sequences(series_scaled)
        n_samples = X.shape[0]
        if n_samples < self.B: raise ValueError(f"No hay suficientes muestras ({n_samples}) para crear {self.B} lotes.")
        
        batch_size = n_samples // self.B
        train_data_batches = [{'X': X[b*batch_size:(b+1)*batch_size if b < self.B-1 else n_samples],
                               'y': y[b*batch_size:(b+1)*batch_size if b < self.B-1 else n_samples]}
                              for b in range(self.B)]
        return train_data_batches

    def _fit_predict_one_step(self, series: np.ndarray) -> np.ndarray:
        """Lógica original para predecir un solo paso."""
        clear_all_sessions()
        ensemble_models = []
        try:
            if self.best_params: self.__dict__.update(self.best_params)

            if len(series) <= self.n_lags + self.B:
                return np.full(self.num_samples, np.mean(series))

            train_batches = self._prepare_data(series)
            loo_preds_median = [[] for _ in range(self.B)]
            
            for b in range(self.B):
                model = self._build_lstm()
                model.fit(train_batches[b]['X'], train_batches[b]['y'], epochs=self.epochs,
                          batch_size=self.batch_size, verbose=0, shuffle=False)
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

            last_window_scaled = self.scaler.transform(series[-self.n_lags:].reshape(-1, 1)).reshape(1, self.n_lags, 1)
            final_preds_median = [model.predict(last_window_scaled, verbose=0)[0, self.median_idx] for model in ensemble_models]
            point_prediction_scaled = np.mean(final_preds_median)
            
            predictive_dist_scaled = point_prediction_scaled + self.rng.choice(conformity_scores, size=self.num_samples, replace=True)
            return self.scaler.inverse_transform(predictive_dist_scaled.reshape(-1, 1)).flatten()
        except Exception:
            return np.full(self.num_samples, np.nanmean(series))
        finally:
            for model in ensemble_models: del model
            clear_all_sessions()

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray], h: int = 1) -> np.ndarray:
        """Implementa estrategia recursiva para pronóstico multi-step."""
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        current_series = np.copy(series)
        final_dist = None

        for step in range(h):
            dist = self._fit_predict_one_step(current_series)
            if step < h - 1:
                point_pred = _get_point_pred_from_dist(dist, self.rng)
                current_series = np.append(current_series, point_pred)
            else:
                final_dist = dist
        
        return final_dist if final_dist is not None else np.full(self.num_samples, np.mean(series))

    def optimize_hyperparameters(self, df: pd.DataFrame, reference_noise: np.ndarray):
        # La optimización es demasiado costosa, usamos valores fijos robustos
        self.best_params = {'n_lags': 30, 'units': 32, 'B': 3}
        self.__dict__.update(self.best_params)
        return self.best_params, -1.0

# ============================================================================
# PARTE 3 DE 3
# Contenido:
# - Motores de la simulación:
#   - _predict_with_model_safe (Función auxiliar)
#   - PipelineOptimizado (Orquestador de la evaluación multi-step)
#   - run_single_scenario_robust (Worker para el procesamiento en paralelo)
#   - ScenarioRunnerMejorado (Controlador principal del experimento)
# - Punto de entrada para la ejecución del script (`if __name__ == '__main__'`).
# ============================================================================

# ============================================================================
# 6. WORKER Y PIPELINE (MODIFICADOS PARA MULTI-STEP)
# ============================================================================

def _predict_with_model_safe(model_name: str, model_instance: Any,
                             history_df: pd.DataFrame, h: int, seed: int) -> Dict:
    """
    Función wrapper segura que llama a un modelo, pasando el horizonte 'h'.
    Maneja la memoria y los errores explícitamente.
    """
    try:
        # Asegurar que cada llamada tenga una semilla predecible pero única
        np.random.seed(seed)
        if hasattr(model_instance, 'rng'):
            model_instance.rng = np.random.default_rng(seed)
        
        # Pasar el dataframe y el horizonte 'h' a la función fit_predict del modelo
        prediction_output = model_instance.fit_predict(history_df, h=h)
        
        # Procesar la salida del modelo para obtener un array de muestras
        if isinstance(prediction_output, list) and prediction_output and isinstance(prediction_output[0], dict):
            values = [d['value'] for d in prediction_output]
            probs = np.array([d['probability'] for d in prediction_output])
            # Normalizar probabilidades por si no suman 1
            if np.sum(probs) > 1e-9:
                probs /= np.sum(probs)
            else:
                probs = np.ones(len(values)) / len(values) if values else []
            samples = np.random.choice(values, size=5000, p=probs, replace=True) if values else np.array([])
        else:
            samples = np.array(prediction_output).flatten()
        
        return {'name': model_name, 'samples': samples, 'error': None}
    
    except Exception as e:
        # Capturar cualquier error durante la predicción
        return {'name': model_name, 'samples': np.array([]), 'error': str(e)}
    finally:
        # Limpieza de memoria crítica después de cada llamada, especialmente para TF/Keras
        clear_all_sessions()


class PipelineOptimizado:
    """
    Pipeline que ejecuta la simulación, optimización y evaluación multi-step para un único escenario.
    """
    N_TEST_STEPS = 5
    N_FORECAST_HORIZONS = 5

    def __init__(self, model_type='ARMA(1,1)', phi=[0.7], theta=[0.3],
                 sigma=1.2, noise_dist='t-student', n_samples=250, scenario_id=None,
                 seed=42, verbose=False):
        self.config = {
            'model_type': model_type, 'phi': phi, 'theta': theta, 'sigma': sigma,
            'noise_dist': noise_dist, 'n_samples': n_samples, 'seed': seed, 'verbose': verbose
        }
        self.scenario_id = scenario_id
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.simulator = None
        self.full_series = None
        self.full_errors = None
        self.rolling_ecrps: List[Dict] = []

    def _setup_models(self) -> Dict:
        """Inicializa todas las clases de modelo que se van a evaluar."""
        seed = self.config['seed']
        p_order = len(self.config['phi']) if self.config['phi'] else 2
        
        return {
            'LSPM': LSPM(random_state=seed, verbose=False),
            'LSPMW': LSPMW(random_state=seed, verbose=False),
            'AREPD': AREPD(random_state=seed, verbose=False),
            'DeepAR': DeepARModel(random_state=seed, verbose=False, epochs=15),
            'Enhanced Bootstrapping': EnhancedBootstrappingModel(self.simulator, random_state=seed, verbose=False),
            'Sieve Bootstrap': SieveBootstrap(p_order=p_order, random_state=seed, verbose=False),
            'MCPS': MondrianCPSModel(random_state=seed, verbose=False),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(random_state=seed, verbose=False),
            'EnCQR-LSTM': EnCQR_LSTM_Model(random_state=seed, verbose=False, epochs=6, B=2, units=24, n_layers=1)
        }

    def execute(self, show_intermediate_plots=False):
        """Ejecuta el pipeline completo: simulación, optimización única y evaluación en ventana rodante multi-step."""
        sim_config = {k: v for k, v in self.config.items() if k not in ['n_samples', 'verbose']}
        self.simulator = ARMASimulation(**sim_config)
        
        # Simular suficientes datos para tener valores reales con los que comparar en todos los horizontes
        total_simulation_length = self.config['n_samples'] + self.N_FORECAST_HORIZONS
        self.full_series, self.full_errors = self.simulator.simulate(n=total_simulation_length, burn_in=50)
        
        initial_train_len = self.config['n_samples'] - self.N_TEST_STEPS
        initial_train_series = self.full_series[:initial_train_len]
        df_initial_train = pd.DataFrame({'valor': initial_train_series})
        
        all_models = self._setup_models()
        
        # Optimización de hiperparámetros UNA SOLA VEZ (usando pronóstico a h=1 como referencia)
        reference_noise_for_opt = self.simulator.get_true_future_step_distribution(
            initial_train_series, self.full_errors[:initial_train_len], horizon=1, n_samples=5000)

        for name, model in all_models.items():
            if hasattr(model, 'optimize_hyperparameters'):
                try:
                    model.optimize_hyperparameters(df_initial_train, reference_noise_for_opt)
                except Exception as e:
                    if self.verbose: print(f"Error optimizando {name}: {e}")
        
        # --- BUCLE DE EVALUACIÓN MULTI-STEP ---
        for t in range(self.N_TEST_STEPS):
            step_t = initial_train_len + t
            history_series = self.full_series[:step_t]
            history_errors = self.full_errors[:step_t]
            df_history = pd.DataFrame({'valor': history_series})

            for h in range(1, self.N_FORECAST_HORIZONS + 1):
                theoretical_samples = self.simulator.get_true_future_step_distribution(
                    history_series, history_errors, horizon=h, n_samples=20000)

                step_ecrps = {'Paso': t + 1, 'Horizonte': h}
                step_distributions = {'Teórica': theoretical_samples}
                
                # Predecir secuencialmente con cada modelo para el horizonte h
                for name, model in all_models.items():
                    result = _predict_with_model_safe(name, model, df_history, h, self.config['seed'] + t + h)
                    
                    if result['samples'].size > 0 and result['error'] is None:
                        step_distributions[name] = result['samples']
                        step_ecrps[name] = ecrps(result['samples'], theoretical_samples)
                    else:
                        step_ecrps[name] = np.nan
                
                self.rolling_ecrps.append(step_ecrps)
                
                if show_intermediate_plots:
                    plots_dir = "plots_densidades_por_escenario"
                    plot_filename = os.path.join(plots_dir, f"escenario_{self.scenario_id}_paso_{t+1}_h_{h}.png")
                    title = f"Densidades - Escenario {self.scenario_id} - Origen {t+1} - Horizonte {h}"
                    metrics = {k: v for k, v in step_ecrps.items() if k not in ['Paso', 'Horizonte']}
                    PlotManager.plot_density_comparison(step_distributions, metrics, title, save_path=plot_filename)
        
        clear_all_sessions()
        return self._prepare_results_df()
        
    def _prepare_results_df(self) -> pd.DataFrame:
        """Agrega los resultados ECRPS por horizonte y calcula un promedio general."""
        if not self.rolling_ecrps: return pd.DataFrame()
        
        ecrps_df = pd.DataFrame(self.rolling_ecrps)
        model_cols = [col for col in ecrps_df.columns if col not in ['Paso', 'Horizonte', 'Mejor Modelo']]
        
        # Agrupar por horizonte y calcular el promedio de ECRPS para cada modelo
        avg_by_horizon = ecrps_df.groupby('Horizonte')[model_cols].mean()
        
        # Calcular el promedio general a través de todos los pasos y horizontes
        overall_avg = ecrps_df[model_cols].mean(numeric_only=True).to_frame('Promedio General').T
        
        # Combinar resultados en un único DataFrame de resumen
        summary_df = pd.concat([avg_by_horizon, overall_avg])
        
        # Encontrar el mejor modelo para cada fila (cada horizonte y el promedio general)
        summary_df['Mejor Modelo'] = summary_df[model_cols].idxmin(axis=1)
        
        return summary_df.reset_index().rename(columns={'index': 'Horizonte'})


def run_single_scenario_robust(scenario_with_seed: Dict) -> Tuple[Dict, pd.DataFrame]:
    """Worker para el pool de procesos. Ejecuta un escenario y limpia la memoria."""
    scenario_config = scenario_with_seed['config']
    seed = scenario_with_seed['seed']
    plot_flag = scenario_with_seed.get('plot', False)
    
    clear_all_sessions()
    try:
        pipeline = PipelineOptimizado(
            **{k: v for k, v in scenario_config.items() if k != 'scenario_id'},
            scenario_id=scenario_config.get('scenario_id', 'N/A'),
            seed=seed, verbose=False)
        
        results_df = pipeline.execute(show_intermediate_plots=plot_flag)
        return (scenario_config, results_df)
    except Exception as e:
        print(f"ERROR escenario {scenario_config.get('scenario_id', 'N/A')}: {e}")
        return (scenario_config, None)
    finally:
        clear_all_sessions()
        sys.stdout.flush()
        sys.stderr.flush()

# ============================================================================
# 7. SCENARIO RUNNER (CONTROLADOR PRINCIPAL)
# ============================================================================
class ScenarioRunnerMejorado:
    """Gestiona la ejecución de múltiples escenarios, el procesamiento por lotes y la presentación de resultados."""
    def __init__(self, seed=420):
        self.seed = seed
        self.model_names = []
        # Configuración de escenarios a probar
        self.models_config = [
            {'model_type': 'AR(1)', 'phi': [0.9], 'theta': []},
            {'model_type': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
            {'model_type': 'MA(1)', 'phi': [], 'theta': [0.7]},
            {'model_type': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]}
        ]
        self.distributions = ['normal', 't-student', 'mixture']
        self.variances = [0.5, 1.0, 2.0]
        self.plots_directory = "plots_densidades_por_escenario"

    def _generate_scenarios(self, n_scenarios):
        scenarios, count = [], 0
        for model in self.models_config:
            for dist in self.distributions:
                for var in self.variances:
                    if count < n_scenarios:
                        scenarios.append({**model, 'noise_dist': dist, 'sigma': np.sqrt(var), 'scenario_id': count + 1})
                        count += 1
        return [{'config': sc, 'seed': self.seed + i} for i, sc in enumerate(scenarios)]

    def _prepare_rows_from_result(self, scenario_config, results_df):
        """Formatea el DataFrame de resumen de un escenario para agregarlo al archivo Excel final."""
        rows = []
        if results_df is None or results_df.empty: return rows
            
        if not self.model_names:
            self.model_names = [col for col in results_df.columns if col not in ['Horizonte', 'Mejor Modelo']]
             
        for _, data_row in results_df.iterrows():
            row = {
                'Horizonte': data_row['Horizonte'],
                'Valores de AR': str(scenario_config['phi']),
                'Valores MA': str(scenario_config['theta']),
                'Distribución': scenario_config['noise_dist'],
                'Varianza error': np.round(scenario_config['sigma'] ** 2, 2),
                'Mejor Modelo': data_row.get('Mejor Modelo', 'Error')
            }
            for model_name in self.model_names:
                row[model_name] = data_row.get(model_name, np.nan)
            rows.append(row)
        return rows

    def _run_batch(self, batch_scenarios, restart_every=10, plot=False):
        batch_results = []
        max_workers = max(1, os.cpu_count() // 2 - 1) # Usar la mitad de los cores disponibles menos uno
        
        sublotes = [batch_scenarios[i:i+restart_every] for i in range(0, len(batch_scenarios), restart_every)]
        
        for sublote_idx, sublote in enumerate(sublotes):
            print(f"    Sublote {sublote_idx + 1}/{len(sublotes)}")
            
            for sc in sublote: sc['plot'] = plot

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_single_scenario_robust, sc): sc for sc in sublote}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="      Escenarios"):
                    try:
                        scenario_config, results_df = future.result(timeout=900) # Timeout de 15 mins
                        if results_df is not None:
                            batch_results.append((scenario_config, results_df))
                    except concurrent.futures.TimeoutError:
                        sc_id = futures[future]['config'].get('scenario_id', 'N/A')
                        print(f"      Timeout en escenario {sc_id}")
                    except Exception as e:
                        print(f"      Error en worker: {e}")
            
            clear_all_sessions()
            time.sleep(2)
        
        return batch_results

    def run(self, n_scenarios=120, excel_filename="resultados_finales_multistep.xlsx",
            batch_size=20, restart_every=5, plot: bool = False):
        """Ejecuta todos los escenarios y guarda los resultados."""
        if plot:
            os.makedirs(self.plots_directory, exist_ok=True)
            print(f"✅ Gráficos de densidad activados. Se guardarán en: '{self.plots_directory}'")
        
        all_scenarios_configs = self._generate_scenarios(n_scenarios)
        all_excel_rows = []
        n_batches = (len(all_scenarios_configs) + batch_size - 1) // batch_size
        
        print(f"Ejecutando {len(all_scenarios_configs)} escenarios en {n_batches} lotes.")
        
        for batch_num in range(n_batches):
            start_idx, end_idx = batch_num * batch_size, min((batch_num + 1) * batch_size, len(all_scenarios_configs))
            batch = all_scenarios_configs[start_idx:end_idx]
            
            print(f"\n{'='*60}\nLOTE {batch_num + 1}/{n_batches} (Escenarios {start_idx + 1}-{end_idx})\n{'='*60}")
            
            batch_results = self._run_batch(batch, restart_every=restart_every, plot=plot)
            
            for scenario_config, results_df in batch_results:
                new_rows = self._prepare_rows_from_result(scenario_config, results_df)
                all_excel_rows.extend(new_rows)
            
            print(f"  Limpiando memoria del lote...")
            clear_all_sessions()
            
            if all_excel_rows and (batch_num + 1) % 2 == 0:
                temp_filename = f"checkpoint_lote_{batch_num + 1}.xlsx"
                pd.DataFrame(all_excel_rows).to_excel(temp_filename, index=False)
                print(f"  ✅ Checkpoint guardado: {temp_filename}")

        if all_excel_rows:
            df_final = pd.DataFrame(all_excel_rows)
            df_final.to_excel(excel_filename, index=False)
            print(f"\n✅ {len(all_excel_rows)} filas de resultados guardadas en '{excel_filename}'")
            self.plot_results_from_excel(excel_filename)
        else:
            print("\n❌ No se generaron resultados.")

    def plot_results_from_excel(self, filename):
        """Genera boxplots y gráficos de torta para cada horizonte a partir del archivo Excel final."""
        try:
            df_total = pd.read_excel(filename)
        except FileNotFoundError:
            print(f"No se pudo generar el gráfico: archivo '{filename}' no encontrado.")
            return
        if df_total.empty:
            print("No se pudo generar el gráfico: el archivo de resultados está vacío.")
            return

        if not self.model_names:
            self.model_names = [col for col in df_total.columns if col not in ['Horizonte', 'Valores de AR', 'Valores MA', 'Distribución', 'Varianza error', 'Mejor Modelo']]

        horizontes_disponibles = df_total['Horizonte'].unique()

        for step_name in horizontes_disponibles:
            df_step = df_total[df_total['Horizonte'] == step_name].copy()
            df_melted = df_step.melt(id_vars=['Horizonte'], value_vars=self.model_names, var_name='Modelo', value_name='ECRPS')
            wins = df_step['Mejor Modelo'].value_counts()
            
            title_prefix = f"Resultados para el Horizonte '{step_name}'"
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(title_prefix, fontsize=18)

            sns.boxplot(ax=axes[0], data=df_melted, x='Modelo', y='ECRPS', palette='viridis')
            axes[0].set_title('Distribución de ECRPS por Modelo', fontsize=14)
            axes[0].set_xlabel('')
            axes[0].tick_params(axis='x', rotation=45, labelsize=10)
            
            if not wins.empty:
                pie_colors_map = {name: PlotManager._STYLE['default_colors'][i % len(PlotManager._STYLE['default_colors'])] for i, name in enumerate(self.model_names)}
                pie_colors = [pie_colors_map.get(label, '#CCCCCC') for label in wins.index]
                axes[1].pie(wins, labels=wins.index, autopct='%1.1f%%', startangle=140, colors=pie_colors,
                           wedgeprops={"edgecolor":"k", 'linewidth': 0.5, 'antialiased': True})
                axes[1].set_title('Distribución de Victorias por Modelo', fontsize=14)
            else:
                axes[1].text(0.5, 0.5, 'No hay datos de victorias', ha='center', va='center')
                axes[1].set_title('Distribución de Victorias por Modelo', fontsize=14)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            plt.close(fig)

