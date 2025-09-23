## Simulación
import numpy as np
import pandas as pd
from scipy.stats import t
from typing import List, Dict, Union, Tuple

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
## Plot
# plot.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict

class PlotManager:
    """Clase de utilidad para generar los gráficos del análisis."""
    # Estilos actualizados según lo solicitado
    _STYLE = {'figsize': (14, 6), 'grid_style': {'alpha': 0.3, 'linestyle': ':'},
              'default_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#000000']}

    @classmethod
    def _base_plot(cls, title, xlabel, ylabel, figsize=None):
        """Crea la base para un gráfico estándar."""
        fig_size = figsize if figsize else cls._STYLE['figsize']
        plt.figure(figsize=fig_size)
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(**cls._STYLE['grid_style'])
        plt.gca().spines[['top', 'right']].set_visible(False)

    @classmethod
    def plot_series_split(cls, series: np.ndarray, burn_in_len: int, test_len: int):
        """Grafica la serie temporal mostrando las divisiones de burn-in, train y test."""
        cls._base_plot("Serie Temporal Simulada con Divisiones", "Tiempo", "Valor")
        
        train_len = len(series) - burn_in_len - test_len
        
        plt.plot(series, label='Serie Completa', color=cls._STYLE['default_colors'][0])
        
        plt.axvspan(0, burn_in_len, color='gray', alpha=0.3, label=f'Burn-in ({burn_in_len} puntos)')
        plt.axvspan(burn_in_len, burn_in_len + train_len, color='green', alpha=0.2, label=f'Entrenamiento Inicial ({train_len} puntos)')
        plt.axvspan(burn_in_len + train_len, len(series), color='red', alpha=0.2, label=f'Test (Ventana Rodante) ({test_len} puntos)')
        
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_density_comparison(cls, distributions: Dict[str, np.ndarray], metric_values: Dict[str, float], title: str, colors: Dict[str, str] = None):
        """
        [NUEVA FUNCIÓN]
        Compara las densidades de las distribuciones predictivas para un único paso de tiempo.
        """
        cls._base_plot(title, "Valor", "Densidad")
        
        if colors is None:
            colors = {name: cls._STYLE['default_colors'][i % len(cls._STYLE['default_colors'])] for i, name in enumerate(distributions.keys())}

        # Dibuja cada distribución
        for name, data in distributions.items():
            color = colors.get(name, '#333333') 
            
            # Estilo diferencial para la distribución teórica (real)
            linestyle = '-' if name == 'Teórica' else '--'
            linewidth = 3.0 if name == 'Teórica' else 2.0

            clean_data = data[np.isfinite(data)]
            if len(clean_data) > 1 and np.std(clean_data) > 1e-9:
                sns.kdeplot(clean_data, color=color, label=name, linestyle=linestyle, linewidth=linewidth, warn_singular=False)
            else: # Maneja el caso de predicciones puntuales
                point_prediction = np.mean(clean_data)
                plt.axvline(point_prediction, color=color, linestyle=linestyle, linewidth=linewidth, label=f'{name} (Puntual)')

        # Ordena las métricas y las muestra en una caja de texto
        sorted_metrics = sorted(metric_values.items(), key=lambda x: x[1])
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in sorted_metrics])
        
        plt.text(0.98, 0.98, f'ECRPS vs Teórica:\n{metrics_text}', transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        
        plt.legend(loc='upper left', frameon=True)
        plt.tight_layout()
        plt.show()
## Metricas

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



class EnhancedBootstrappingModel:
    """Modelo con Block Bootstrap y evaluación CRPS."""
    def __init__(self, arma_simulator, random_state=42, verbose=False):
        self.n_lags, self.random_state, self.verbose = None, random_state, verbose
        self.rng = np.random.default_rng(self.random_state)
        self.arma_simulator = arma_simulator
        self.train, self.test, self.mean_val, self.std_val = None, None, None, None

    def prepare_data(self, series):
        split_idx = int(len(series)*0.9)
        self.train, self.test = series[:split_idx], series[split_idx:]
        self.mean_val, self.std_val = np.mean(self.train), np.std(self.train)
        return (self.train - self.mean_val) / (self.std_val + 1e-8)

    def denormalize(self, values): return (values * self.std_val) + self.mean_val

    def _block_bootstrap_predict(self, fitted_model, n_boot):
        residuals, n_resid = fitted_model.resid, len(fitted_model.resid)
        if n_resid < 2: return fitted_model.forecast(steps=1) + self.rng.choice(residuals, size=n_boot, replace=True)
        block_length = max(1, int(n_resid**(1/3)))
        blocks = [residuals[i:i+block_length] for i in range(n_resid - block_length + 1)]
        if not blocks: return fitted_model.forecast(steps=1)
        resampled_indices = self.rng.choice(len(blocks), size=(n_boot // block_length) + 1, replace=True)
        bootstrap_error_pool = np.concatenate([blocks[i] for i in resampled_indices])
        return fitted_model.forecast(steps=1) + self.rng.choice(bootstrap_error_pool, size=n_boot, replace=True)

    def fit_predict(self, data, n_boot=1000):
        normalized_train = self.prepare_data(data)
        test_predictions, current_data = [], normalized_train.copy()
        for _ in range(len(self.test)):
            if len(current_data) <= self.n_lags * 2 + 1:
                pred = self.denormalize(np.mean(current_data))
                test_predictions.append(np.full(n_boot, pred))
                current_data = np.append(current_data, np.mean(current_data))
                continue
            try: fitted_model = AutoReg(current_data, lags=self.n_lags, old_names=False).fit()
            except (np.linalg.LinAlgError, ValueError):
                pred = self.denormalize(np.mean(current_data))
                test_predictions.append(np.full(n_boot, pred))
                current_data = np.append(current_data, np.mean(current_data))
                continue
            boot_preds = self._block_bootstrap_predict(fitted_model, n_boot)
            current_data = np.append(current_data, np.mean(boot_preds))
            test_predictions.append(self.denormalize(boot_preds))
        return test_predictions

    def grid_search(self, lags_range=range(1, 13), n_boot=500):
        best_crps, best_lag = float('inf'), 1
        normalized_train = self.prepare_data(self.arma_simulator.series)
        for n_lags in lags_range:
            self.n_lags = n_lags
            if len(normalized_train) <= n_lags + 10: continue
            crps_values = []
            try:
                for t in range(n_lags + 10, len(normalized_train)):
                    train_subset = normalized_train[:t]
                    if len(train_subset) <= 2 * n_lags + 1: continue
                    try: fitted_model = AutoReg(train_subset, lags=n_lags, old_names=False).fit()
                    except (np.linalg.LinAlgError, ValueError): continue
                    boot_preds = self._block_bootstrap_predict(fitted_model, n_boot)
                    crps_values.append(crps(boot_preds, normalized_train[t]))
                if crps_values and (avg_crps := np.nanmean(crps_values)) < best_crps:
                    best_crps, best_lag = avg_crps, n_lags
            except Exception: continue
        self.n_lags = best_lag
        if self.verbose: print(f"✅ Mejor n_lags (Block Bootstrap): {best_lag} (CRPS: {best_crps:.4f})")
        return best_lag, best_crps

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
    def __init__(self, hidden_size=20, n_lags=5, num_layers=1, dropout=0.1, lr=0.01, batch_size=32, epochs=50, num_samples=1000, random_state=42, verbose=False):
        self.hidden_size, self.n_lags, self.num_layers, self.dropout, self.lr, self.batch_size, self.epochs, self.num_samples = hidden_size, n_lags, num_layers, dropout, lr, batch_size, epochs, num_samples
        self.model, self.scaler_mean, self.scaler_std, self.random_state, self.verbose, self.best_params = None, None, None, random_state, verbose, {}
        np.random.seed(random_state), torch.manual_seed(random_state)
    class _DeepARNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__(); dropout_to_apply = dropout if num_layers > 1 else 0; self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_to_apply); self.fc_mu, self.fc_sigma = nn.Linear(hidden_size, 1), nn.Linear(hidden_size, 1)
        def forward(self, x): lstm_out, _ = self.lstm(x); mu, sigma = self.fc_mu(lstm_out[:, -1, :]), torch.exp(self.fc_sigma(lstm_out[:, -1, :])) + 1e-6; return mu, sigma
    def _create_sequences(self, series):
        X, y = [], []; [X.append(series[i:i + self.n_lags]) for i in range(len(series) - self.n_lags)]; [y.append(series[i + self.n_lags]) for i in range(len(series) - self.n_lags)]; return np.array(X), np.array(y)
    def optimize_hyperparameters(self, df, reference_noise):
        def objective(n_lags, hidden_size, num_layers, dropout, lr):
            try:
                self.n_lags, self.hidden_size, self.num_layers, self.dropout, self.lr = max(1, int(n_lags)), max(5, int(hidden_size)), max(1, int(num_layers)), min(0.5, max(0.0, dropout)), max(0.0001, lr)
                series = df['valor'].values if isinstance(df, pd.DataFrame) else df
                self.scaler_mean, self.scaler_std = np.nanmean(series), np.nanstd(series) + 1e-8
                normalized_series = (series - self.scaler_mean) / self.scaler_std
                if len(normalized_series) <= self.n_lags: return -float('inf')
                X_train, y_train = self._create_sequences(normalized_series)
                if len(X_train) == 0: return -float('inf')
                mu, sigma = np.nanmean(y_train), np.nanstd(y_train) if np.nanstd(y_train) > 1e-6 else 1e-6
                predictions = (np.random.normal(mu, sigma, self.num_samples) * self.scaler_std + self.scaler_mean)
                return -ecrps(predictions, reference_noise)
            except Exception: return -float('inf')
        optimizer = BayesianOptimization(f=objective, pbounds={'n_lags': (1, 10), 'hidden_size': (5, 30), 'num_layers': (1, 3), 'dropout': (0.0, 0.5), 'lr': (0.001, 0.1)}, random_state=self.random_state, verbose=0)
        try: optimizer.maximize(init_points=3, n_iter=7)
        except Exception: pass
        best_ecrps = -1
        if optimizer.max and optimizer.max['target'] > -float('inf'):
            best_ecrps, self.best_params = -optimizer.max['target'], {k: v for k, v in optimizer.max['params'].items()}
            self.best_params.update({'n_lags': int(self.best_params['n_lags']), 'hidden_size': int(self.best_params['hidden_size']), 'num_layers': int(self.best_params['num_layers'])})
        else: self.best_params = {'n_lags': 5, 'hidden_size': 20, 'num_layers': 1, 'dropout': 0.1, 'lr': 0.01}
        if self.verbose: print(f"✅ Opt. DeepAR (ECRPS: {best_ecrps:.4f}): {self.best_params}")
        return self.best_params, best_ecrps
    def fit_predict(self, df):
        try:
            series = df['valor'].values if isinstance(df, pd.DataFrame) else df
            self.scaler_mean, self.scaler_std = np.nanmean(series), np.nanstd(series) + 1e-8
            normalized_series = (series - self.scaler_mean) / self.scaler_std
            if self.best_params: self.__dict__.update(self.best_params)
            if len(normalized_series) <= self.n_lags: return np.full(self.num_samples, self.scaler_mean)
            X_train, y_train = self._create_sequences(normalized_series)
            if X_train.shape[0] < self.batch_size: return np.full(self.num_samples, self.scaler_mean)
            X_tensor, y_tensor = torch.FloatTensor(X_train.reshape(-1, self.n_lags, 1)), torch.FloatTensor(y_train.reshape(-1, 1))
            self.model = self._DeepARNN(1, self.hidden_size, self.num_layers, self.dropout)
            criterion, optimizer = nn.GaussianNLLLoss(), optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            for _ in range(self.epochs):
                perm = torch.randperm(X_tensor.size(0))
                for i in range(0, X_tensor.size(0), self.batch_size):
                    idx = perm[i:i + self.batch_size]; mu, sigma = self.model(X_tensor[idx]); loss = criterion(mu, y_tensor[idx], sigma); optimizer.zero_grad(), loss.backward(), optimizer.step()
            self.model.eval()
            with torch.no_grad(): mu, sigma = self.model(torch.FloatTensor(normalized_series[-self.n_lags:].reshape(1, self.n_lags, 1)))
            return np.nan_to_num((np.random.normal(mu.item(), sigma.item(), self.num_samples) * self.scaler_std + self.scaler_mean))
        except Exception: return np.full(self.num_samples, np.nanmean(df))

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

# Coloca esta importación al principio de tu archivo arma_pipeline.py si no la tienes
import xgboost as xgb

# ===================================================================
# CLASE 11 (NUEVA): Mondrian Conformal Predictive System
# ===================================================================
class MondrianCPSModel:
    """
    Implementa el Mondrian Conformal Predictive System (MCPS) inspirado en el paper.

    Este método genera distribuciones predictivas localmente calibradas. Particiona
    el conjunto de calibración en 'bins' basados en la magnitud de la predicción
    del modelo subyacente, y luego calibra usando solo los residuos del bin
    correspondiente. Esto permite intervalos más estrechos para predicciones seguras
    y más anchos para las inciertas.
    """
    def __init__(self, n_lags: int = 10, n_bins: int = 10, test_size: float = 0.25, 
                 random_state: int = 42, verbose: bool = False):
        
        self.n_lags = n_lags
        self.n_bins = n_bins
        self.test_size = test_size # Proporción de datos para calibración
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)
        
        # Usamos XGBoost como el regresor subyacente, como en el paper.
        self.base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.best_params = {}

    def _create_lag_matrix(self, series: np.ndarray):
        """Crea una matriz de características y un vector objetivo a partir de una serie temporal."""
        X, y = [], []
        for i in range(len(series) - self.n_lags):
            X.append(series[i:(i + self.n_lags)])
            y.append(series[i + self.n_lags])
        return np.array(X), np.array(y)

    def optimize_hyperparameters(self, df: pd.DataFrame, reference_noise: np.ndarray):
        """Optimiza el número de bins para MCPS usando optimización Bayesiana."""
        series = df['valor'].values
        
        def objective(n_bins):
            try:
                self.n_bins = max(2, int(n_bins))
                if len(series) <= self.n_lags * 2: return -1e10
                
                # Generamos una predicción con los hiperparámetros actuales
                predictions = self.fit_predict(df)
                if predictions is None or len(predictions) == 0: return -1e10
                
                # El objetivo es minimizar el ECRPS (por eso el negativo)
                return -ecrps(predictions, reference_noise)
            except Exception:
                return -1e10

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={'n_bins': (3, 20.99)}, # n_bins es un entero
            random_state=self.random_state,
            verbose=0
        )
        try:
            optimizer.maximize(init_points=4, n_iter=8)
            if optimizer.max:
                self.best_params = {'n_bins': int(optimizer.max['params']['n_bins'])}
                best_ecrps = -optimizer.max['target']
            else:
                best_ecrps = -1
        except Exception:
            best_ecrps = -1
        
        if self.verbose:
            print(f"✅ Opt. MCPS (ECRPS: {best_ecrps:.4f}): {self.best_params}")
            
        return self.best_params, best_ecrps

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Ejecuta el flujo completo de MCPS y devuelve una distribución de muestras.
        """
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        
        # Aplicar hiperparámetros optimizados si existen
        if self.best_params:
            self.__dict__.update(self.best_params)

        if len(series) < self.n_lags * 2:
            if self.verbose: print("⚠️ MCPS: No hay suficientes datos para el proceso.")
            return np.full(5000, np.mean(series))

        # 1. Crear características y objetivo
        X, y = self._create_lag_matrix(series)
        
        # El punto para el que queremos predecir
        x_test = series[-self.n_lags:].reshape(1, -1)

        # 2. Dividir en entrenamiento propio y calibración
        n_calib = int(len(X) * self.test_size)
        if n_calib < self.n_bins * 2: # Asegurarse de que haya suficientes datos para calibrar
            if self.verbose: print("⚠️ MCPS: No hay suficientes datos para la calibración.")
            return np.full(5000, np.mean(series))
            
        X_train, X_calib = X[:-n_calib], X[-n_calib:]
        y_train, y_calib = y[:-n_calib], y[-n_calib:]

        # 3. Entrenar el modelo base
        self.base_model.fit(X_train, y_train)

        # 4. Obtener predicciones y residuos para el conjunto de calibración
        calib_preds = self.base_model.predict(X_calib)
        calib_residuals = y_calib - calib_preds

        # 5. Particionar el conjunto de calibración (Lógica Mondrian)
        # Usamos qcut para crear bins con aproximadamente el mismo número de puntos
        try:
            bin_indices = pd.qcut(calib_preds, self.n_bins, labels=False, duplicates='drop')
            # Obtenemos los bordes de los bins para saber dónde cae nuestra predicción de test
            _, bin_edges = pd.qcut(calib_preds, self.n_bins, retbins=True, duplicates='drop')
        except ValueError: # Si no se pueden crear los bins (pocos puntos únicos)
             if self.verbose: print("⚠️ MCPS: No se pudieron crear los bins. Usando un solo bin global.")
             bin_indices = np.zeros(len(calib_preds), dtype=int)
             bin_edges = [-np.inf, np.inf]


        # 6. Realizar la predicción para el punto de test
        point_prediction = self.base_model.predict(x_test)[0]

        # 7. Encontrar el bin correspondiente para la predicción de test
        test_bin = np.digitize(point_prediction, bins=bin_edges) - 1
        test_bin = np.clip(test_bin, 0, len(bin_edges) - 2) # Asegurar que el índice esté en rango

        # 8. Seleccionar los residuos del bin correspondiente
        local_residuals = calib_residuals[bin_indices == test_bin]
        
        if len(local_residuals) < 10: # Si el bin tiene muy pocos puntos, usar todos los residuos
            if self.verbose: print(f"⚠️ MCPS: El bin {test_bin} tiene solo {len(local_residuals)} puntos. Usando calibración global.")
            local_residuals = calib_residuals

        # 9. Construir la distribución predictiva final
        # La distribución son las muestras de (predicción puntual + errores históricos del bin)
        final_samples = point_prediction + self.rng.choice(local_residuals, size=5000, replace=True)
        
        return final_samples

# Añade estas importaciones al principio de tu archivo de modelos si no las tienes
import numpy as np
import pandas as pd
from typing import List, Dict, Union
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization
import warnings
import os

# --- Configuración para un entorno limpio ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')


class EnCQR_LSTM_Model:
    """
    Implementa el método Ensemble Conformalized Quantile Regression (EnCQR)
    utilizando un predictor LSTM interno.

    Esta clase es compatible con el pipeline de backtesting, manejando el
    preprocesamiento de datos, el entrenamiento del ensamble, el cálculo de
    scores de no conformidad y la generación de una distribución predictiva
    final en forma de muestras.
    """
    def __init__(self, n_lags: int = 24, B: int = 3, units: int = 50, n_layers: int = 2,
                 lr: float = 0.005, batch_size: int = 16, epochs: int = 25, 
                 num_samples: int = 5000, random_state: int = 42, verbose: bool = False):
        
        # --- Hiperparámetros EnCQR ---
        self.B = B  # Número de modelos en el ensamble

        # --- Hiperparámetros del LSTM Interno ---
        self.n_lags = n_lags
        self.units = units
        self.n_layers = n_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
        # --- Configuración del Pipeline ---
        self.num_samples = num_samples
        self.random_state = random_state
        self.verbose = verbose
        
        # --- Atributos Internos ---
        self.scaler = MinMaxScaler()
        self.best_params = {}
        self.rng = np.random.default_rng(random_state)
        
        # --- Grilla de Cuantiles ---
        self.quantiles = np.round(np.arange(0.05, 1.0, 0.05), 2)
        self.median_idx = np.where(self.quantiles == 0.5)[0][0] # Índice del cuantil 0.5
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    # --- Funciones Auxiliares del Modelo Keras ---
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

    # --- Funciones de Preprocesamiento de Datos ---
    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.n_lags):
            X.append(data[i:(i + self.n_lags)])
            y.append(data[i + self.n_lags])
        return np.array(X), np.array(y)

    def _prepare_data(self, series: np.ndarray):
        """Encapsula todo el preprocesamiento: escalar, crear ventanas y dividir para EnCQR."""
        series_scaled = self.scaler.fit_transform(series.reshape(-1, 1))
        X, y = self._create_sequences(series_scaled)
        
        # Dividir los datos de entrenamiento en B conjuntos para el ensamble
        n_samples = X.shape[0]
        if n_samples < self.B:
             raise ValueError(f"No hay suficientes muestras ({n_samples}) para crear {self.B} lotes.")
        
        batch_size = n_samples // self.B
        train_data_batches = []
        for b in range(self.B):
            start, end = b * batch_size, (b + 1) * batch_size
            if b == self.B - 1: end = n_samples # Asegurarse de que el último lote llegue hasta el final
            train_data_batches.append({'X': X[start:end], 'y': y[start:end]})
            
        return train_data_batches

    # --- Métodos Principales del Pipeline ---
    def optimize_hyperparameters(self, df: pd.DataFrame, reference_noise: np.ndarray):
        """Optimiza hiperparámetros clave usando optimización bayesiana."""
        series = df['valor'].values

        def objective(n_lags, units, B):
            try:
                self.n_lags, self.units, self.B = max(10, int(n_lags)), max(16, int(units)), max(2, int(B))
                if len(series) <= self.n_lags * self.B: return -1e10
                
                predictions = self.fit_predict(series)
                if predictions is None or len(predictions) == 0: return -1e10
                
                return -ecrps(predictions, reference_noise)
            except Exception:
                return -1e10

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={'n_lags': (10, 50), 'units': (20, 80), 'B': (2, 5.99)}, # B es entero
            random_state=self.random_state, verbose=0
        )
        try:
            optimizer.maximize(init_points=4, n_iter=8)
            if optimizer.max:
                self.best_params = {k: v for k, v in optimizer.max['params'].items()}
                for key in ['n_lags', 'units', 'B']: self.best_params[key] = int(self.best_params[key])
                best_ecrps = -optimizer.max['target']
            else: best_ecrps = -1
        except Exception: best_ecrps = -1
        
        if self.verbose: print(f"✅ Opt. EnCQR-LSTM (ECRPS: {best_ecrps:.4f}): {self.best_params}")
        return self.best_params, best_ecrps

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Ejecuta el flujo completo de EnCQR y devuelve una distribución de muestras."""
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        if self.best_params: self.__dict__.update(self.best_params)

        if len(series) <= self.n_lags + self.B:
            if self.verbose: print("⚠️ EnCQR-LSTM: No hay suficientes datos para el proceso.")
            return np.full(self.num_samples, np.mean(series))

        # 1. Preparar los datos (escalar, ventanear, dividir en B lotes)
        try:
            train_batches = self._prepare_data(series)
        except ValueError as e:
            if self.verbose: print(f"⚠️ EnCQR-LSTM: Error en preparación de datos - {e}")
            return np.full(self.num_samples, np.mean(series))

        # 2. Entrenar el ensamble y obtener predicciones "leave-one-out" (LOO)
        ensemble_models, loo_preds_median = [], [[] for _ in range(self.B)]
        for b in range(self.B):
            model = self._build_lstm()
            model.fit(train_batches[b]['X'], train_batches[b]['y'], 
                      epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            ensemble_models.append(model)
            
            # Predecir sobre los otros B-1 lotes
            for i in range(self.B):
                if i != b:
                    preds = model.predict(train_batches[i]['X'], verbose=0)
                    loo_preds_median[i].append(preds[:, self.median_idx]) # Solo guardamos la mediana

        # 3. Calcular los scores de no conformidad (residuos)
        conformity_scores = []
        for i in range(self.B):
            avg_loo_pred = np.mean(loo_preds_median[i], axis=0).reshape(-1, 1)
            true_y = train_batches[i]['y']
            scores = true_y - avg_loo_pred
            conformity_scores.extend(scores.flatten())
        
        conformity_scores = np.array(conformity_scores)

        # 4. Realizar la predicción final sobre los datos más recientes
        last_window_unscaled = series[-self.n_lags:]
        last_window_scaled = self.scaler.transform(last_window_unscaled.reshape(-1, 1)).reshape(1, self.n_lags, 1)

        final_preds_median = []
        for model in ensemble_models:
            pred = model.predict(last_window_scaled, verbose=0)
            final_preds_median.append(pred[0, self.median_idx])
        
        # El punto de predicción es el promedio de las medianas del ensamble
        point_prediction_scaled = np.mean(final_preds_median)
        
        # 5. Generar la distribución predictiva final
        # La distribución es el punto de predicción + los residuos históricos (scores)
        predictive_dist_scaled = point_prediction_scaled + self.rng.choice(
            conformity_scores, size=self.num_samples, replace=True
        )
        
        # 6. Invertir la escala para obtener la distribución en la escala original
        final_samples = self.scaler.inverse_transform(predictive_dist_scaled.reshape(-1, 1)).flatten()

        return final_samples
## Pipeline

import numpy as np
import pandas as pd
from IPython.display import display
from typing import Dict, List, Any
import concurrent.futures # ¡Ya estaba importado, ahora lo usaremos!

# ==============================================================================
# NUEVA FUNCIÓN AUXILIAR PARA PARALELIZACIÓN
# ==============================================================================
def _process_model(model_name: str, model_instance: Any, history_df: pd.DataFrame,
                   history_series: np.ndarray,
                   reference_noise: np.ndarray,
                   sim_config: Dict,
                   seed: int) -> Dict:
    """
    Encapsula toda la lógica para procesar un único modelo.
    Esta función será ejecutada en un proceso separado para cada modelo.
    """
    # --- CAMBIO CLAVE PARA REPLICABILIDAD ---
    # Se establece la semilla de NumPy al inicio de cada proceso hijo.
    # Esto asegura que todas las operaciones aleatorias (ej. en los modelos)
    # sean idénticas en cada ejecución.
    np.random.seed(seed)
    
    local_rng = np.random.default_rng(seed)
    local_simulator = ARMASimulation(**sim_config)

    # 1. Optimización de hiperparámetros
    if hasattr(model_instance, 'optimize_hyperparameters'):
        model_instance.optimize_hyperparameters(history_df, reference_noise)
    elif hasattr(model_instance, 'grid_search'):
        temp_model_for_gs = EnhancedBootstrappingModel(local_simulator, random_state=seed)
        temp_model_for_gs.arma_simulator.series = history_series
        temp_model_for_gs.grid_search()
        model_instance.n_lags = temp_model_for_gs.n_lags

    # 2. Predicción y generación de muestras
    if model_name == 'Block Bootstrapping':
        bb_model_step = EnhancedBootstrappingModel(local_simulator, random_state=seed)
        bb_model_step.n_lags = model_instance.n_lags
        bb_model_step.arma_simulator.series = history_series
        prediction_output = bb_model_step.fit_predict(history_series)
        samples = np.array(prediction_output[0]).flatten()
    else:
        prediction_output = model_instance.fit_predict(history_df)
        if isinstance(prediction_output, list) and prediction_output and isinstance(prediction_output[0], dict):
            values = [d['value'] for d in prediction_output]
            probs = np.array([d['probability'] for d in prediction_output])
            if np.sum(probs) > 1e-9:
                probs /= np.sum(probs)
            else:
                probs = np.ones(len(values)) / len(values)
            samples = local_rng.choice(values, size=5000, p=probs, replace=True)
        else:
            samples = np.array(prediction_output).flatten()

    return {'name': model_name, 'samples': samples}

# ===================================================================
# FUNCIÓN AUXILIAR PARA PARALELIZACIÓN (Requerida por Pipeline)
# ===================================================================
def _process_model(model_name: str, model_instance: Any, history_df: pd.DataFrame,
                   history_series: np.ndarray,
                   reference_noise: np.ndarray,
                   sim_config: Dict,
                   seed: int) -> Dict:
    """
    Encapsula toda la lógica para procesar un único modelo.
    Esta función será ejecutada en un proceso separado para cada modelo.
    """
    local_rng = np.random.default_rng(seed)
    local_simulator = ARMASimulation(**sim_config)

    # 1. Optimización de hiperparámetros
    if hasattr(model_instance, 'optimize_hyperparameters'):
        model_instance.optimize_hyperparameters(history_df, reference_noise)
    elif hasattr(model_instance, 'grid_search'):
        temp_model_for_gs = EnhancedBootstrappingModel(local_simulator, random_state=seed)
        temp_model_for_gs.arma_simulator.series = history_series
        temp_model_for_gs.grid_search()
        model_instance.n_lags = temp_model_for_gs.n_lags

    # 2. Predicción y generación de muestras
    if model_name == 'Block Bootstrapping':
        bb_model_step = EnhancedBootstrappingModel(local_simulator, random_state=seed)
        bb_model_step.n_lags = model_instance.n_lags
        bb_model_step.arma_simulator.series = history_series
        prediction_output = bb_model_step.fit_predict(history_series)
        samples = np.array(prediction_output[0]).flatten()
    else:
        prediction_output = model_instance.fit_predict(history_df)
        if isinstance(prediction_output, list) and prediction_output and isinstance(prediction_output[0], dict):
            values = [d['value'] for d in prediction_output]
            probs = np.array([d['probability'] for d in prediction_output])
            if np.sum(probs) > 1e-9:
                probs /= np.sum(probs)
            else:
                probs = np.ones(len(values)) / len(values)
            samples = local_rng.choice(values, size=5000, p=probs, replace=True)
        else:
            samples = np.array(prediction_output).flatten()

    return {'name': model_name, 'samples': samples}


# ===================================================================
# CLASE Pipeline (Modificada para devolver resultados)
# ===================================================================
class Pipeline:
    """
    Orquesta un backtesting con ventana rodante para múltiples modelos
    y devuelve los resultados para su agregación.
    """
    N_TEST_STEPS = 5 # Reducido a 5 para acelerar las 120 simulaciones

    def __init__(self, model_type='ARMA(1,1)', phi=[0.7], theta=[0.3], sigma=1.2, noise_dist='t-student', n_samples=250, seed=42, verbose=False):
        self.config = {
            'model_type': model_type, 'phi': phi, 'theta': theta, 'sigma': sigma,
            'noise_dist': noise_dist, 'n_samples': n_samples, 'seed': seed,
            'verbose': verbose
        }
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.simulator, self.full_series, self.full_errors = None, None, None
        self.rolling_ecrps: List[Dict] = []

    def _setup_models(self) -> Dict:
        """Inicializa todos los modelos a ser evaluados."""
        seed = self.config['seed']
        p_order = len(self.config['phi']) if self.config['phi'] else 2
        
        return {
            'LSPM': LSPM(random_state=seed, verbose=self.verbose),
            'LSPMW': LSPMW(random_state=seed, verbose=self.verbose),
            'AREPD': AREPD(random_state=seed, verbose=self.verbose),
            'DeepAR': DeepARModel(random_state=seed, verbose=self.verbose, epochs=20),
            'Block Bootstrapping': EnhancedBootstrappingModel(self.simulator, random_state=seed, verbose=self.verbose),
            'Sieve Bootstrap': SieveBootstrap(p_order=p_order, random_state=seed, verbose=self.verbose),
            'MCPS': MondrianCPSModel(random_state=seed, verbose=self.verbose),
            'EnCQR-LSTM': EnCQR_LSTM_Model(random_state=seed, verbose=self.verbose, epochs=8, B=2, units=32, n_layers=1)
        }
    
    def execute(self, show_intermediate_plots=False):
        """Ejecuta el pipeline y devuelve un DataFrame con los resultados."""
        sim_config = {k: v for k, v in self.config.items() if k not in ['n_samples', 'verbose']}
        self.simulator = ARMASimulation(**sim_config)
        self.full_series, self.full_errors = self.simulator.simulate(n=self.config['n_samples'], burn_in=50)
        initial_train_len = len(self.full_series) - self.N_TEST_STEPS
        
        all_models = self._setup_models()
        colors = {name: PlotManager._STYLE['default_colors'][i % len(PlotManager._STYLE['default_colors'])] for i, name in enumerate(all_models.keys())}
        colors['Teórica'] = '#000000'

        for t in range(self.N_TEST_STEPS):
            step_t = initial_train_len + t
            if self.verbose: print(f"\n >> Paso {t+1}/{self.N_TEST_STEPS} (Prediciendo para t={step_t})")
            
            history_series = self.full_series[:step_t]
            history_errors = self.full_errors[:step_t]
            df_history = pd.DataFrame({'valor': history_series})
            reference_noise_for_opt = self.simulator.get_true_next_step_samples(history_series, history_errors, 5000)
            theoretical_samples = self.simulator.get_true_next_step_samples(history_series, history_errors, 20000)

            step_ecrps = {'Paso': t + 1} # Cambiado a Paso 1, 2, 3...
            step_distributions = {'Teórica': theoretical_samples}
            
            max_p_workers = max(1, os.cpu_count() - 2) if os.cpu_count() else 2
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_p_workers) as executor:
                futures = {
                    executor.submit(
                        _process_model, 
                        name, model, df_history, history_series,
                        reference_noise_for_opt, 
                        sim_config,
                        self.config['seed']
                    ): name for name, model in all_models.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        model_name, samples = result['name'], result['samples']
                        step_distributions[model_name] = samples
                        step_ecrps[model_name] = ecrps(samples, theoretical_samples)
                    except Exception as exc:
                        model_name_failed = futures[future]
                        step_ecrps[model_name_failed] = np.nan
            
            self.rolling_ecrps.append(step_ecrps)
            
            if show_intermediate_plots:
                metrics_for_plot = {name: val for name, val in step_ecrps.items() if name != 'Paso'}
                plot_title = f"Comparación de Densidades Predictivas en el Paso t={step_t}"
                PlotManager.plot_density_comparison(step_distributions, metrics_for_plot, plot_title, colors)

        return self._prepare_results_df()

    def _prepare_results_df(self):
        """Convierte la lista de resultados en un DataFrame final con promedios."""
        if not self.rolling_ecrps:
            return pd.DataFrame()
        
        ecrps_df = pd.DataFrame(self.rolling_ecrps).set_index('Paso')
        model_cols = [col for col in ecrps_df.columns if col not in ['Paso', 'Mejor Modelo']]
        
        # Calcular el mejor modelo para cada paso
        ecrps_df['Mejor Modelo'] = ecrps_df[model_cols].idxmin(axis=1)
        
        # Calcular el promedio y el mejor modelo general
        averages = ecrps_df[model_cols].mean(numeric_only=True)
        best_overall_model = averages.idxmin()
        
        # Añadir la fila de promedios al final
        ecrps_df.loc['Promedio'] = averages
        ecrps_df.loc['Promedio', 'Mejor Modelo'] = best_overall_model
        
        return ecrps_df
    

from tqdm import tqdm

# ===================================================================
# CLASE ScenarioRunnerAdaptado (NUEVA CLASE ADAPTADA)
# ===================================================================
class ScenarioRunnerAdaptado:
    """
    Ejecuta múltiples escenarios, agrega los resultados y genera los análisis
    solicitados (Excel y gráficos).
    """
    def __init__(self, seed=420):
        self.seed = seed
        self.results_per_scenario = []
        # Definición de los escenarios a ejecutar
        self.models_config = [
            {'model_type': 'AR(1)', 'phi': [0.9], 'theta': []}, {'model_type': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
            {'model_type': 'MA(1)', 'phi': [], 'theta': [0.7]}, {'model_type': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
            {'model_type': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]}, {'model_type': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]}
        ]
        self.distributions = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
        self.variances = [0.2, 0.5, 1.0, 3.0]
        self.model_names = [] # Se llenará dinámicamente

    def _generate_scenarios(self, n_scenarios):
        """Crea la lista de diccionarios de configuración para cada escenario."""
        scenarios, count = [], 0
        for model in self.models_config:
            for dist in self.distributions:
                for var in self.variances:
                    if count < n_scenarios:
                        scenarios.append({**model, 'noise_dist': dist, 'sigma': np.sqrt(var), 'scenario_id': count + 1})
                        count += 1
        return scenarios

    def _run_single_scenario(self, scenario):
        """Función interna para ejecutar un único pipeline."""
        print(f"\n--- Ejecutando Escenario {scenario['scenario_id']}: {scenario['model_type']} con ruido {scenario['noise_dist']} y σ={scenario['sigma']:.2f} ---")
        try:
            pipeline = Pipeline(**{k: v for k, v in scenario.items() if k != 'scenario_id'}, seed=self.seed, verbose=False)
            results_df = pipeline.execute(show_intermediate_plots=False)
            return {'scenario_config': scenario, 'results_df': results_df}
        except Exception as e:
            print(f"ERROR en escenario {scenario['scenario_id']} ({scenario['model_type']}, {scenario['noise_dist']}): {e}")
            return None

    def run(self, n_scenarios=120):
        """Ejecuta el número especificado de escenarios."""
        all_scenarios = self._generate_scenarios(n_scenarios)
        self.results_per_scenario = [res for scenario in tqdm(all_scenarios, desc="Ejecutando escenarios") if (res := self._run_single_scenario(scenario)) is not None]
        
        if self.results_per_scenario:
             # Obtiene los nombres de los modelos de la primera ejecución exitosa
             first_result = self.results_per_scenario[0]['results_df']
             self.model_names = [col for col in first_result.columns if col not in ['Paso', 'Mejor Modelo']]

    def get_results_for_excel(self) -> pd.DataFrame:
        """
        Prepara el DataFrame final para exportar a Excel, incluyendo una fila
        para cada paso de backtesting y mostrando la VARIANZA del error.
        """
        if not self.results_per_scenario: 
            return pd.DataFrame()
        
        excel_rows = []
        for res in self.results_per_scenario:
            config = res['scenario_config']
            results_df = res['results_df']
            
            for step, data_row in results_df.iterrows():
                row = {
                    'Paso': step,
                    'Valores de AR': str(config['phi']),
                    'Valores MA': str(config['theta']),
                    'Distribución': config['noise_dist'],
                    # --- CAMBIO: Calcular y mostrar la varianza (sigma^2) ---
                    'Varianza error': np.round(config['sigma'] ** 2, 2),
                    'Mejor Modelo': data_row['Mejor Modelo']
                }
                for model_name in self.model_names:
                    row[model_name] = data_row.get(model_name, np.nan)
                
                excel_rows.append(row)
            
        return pd.DataFrame(excel_rows)

    def save_results_to_excel(self, filename="resultados_simulacion.xlsx"):
        """Guarda los resultados detallados por paso en un archivo Excel."""
        df = self.get_results_for_excel()
        if df.empty:
            print("No hay resultados para guardar.")
            return

        # --- CAMBIO: Actualizar el nombre de la columna en el orden final ---
        first_cols = ['Paso', 'Valores de AR', 'Valores MA', 'Distribución', 'Varianza error']
        last_col = ['Mejor Modelo']
        model_cols = sorted([m for m in self.model_names])
        
        ordered_columns = first_cols + model_cols + last_col
        df = df[ordered_columns]
        
        df.to_excel(filename, index=False)
        print(f"\n✅ Resultados detallados por paso guardados en '{filename}'")

    def plot_results(self):
        """Genera los 6 boxplots y 6 diagramas de torta solicitados."""
        if not self.results_per_scenario:
            print("No hay resultados para graficar.")
            return

        # Obtener los índices de los pasos (ej. 1, 2, 3, 4, 5) y el promedio
        result_indices = self.results_per_scenario[0]['results_df'].index
        steps_and_avg = [idx for idx in result_indices if isinstance(idx, (int, np.integer))]
        steps_and_avg.append('Promedio')

        for step_name in steps_and_avg:
            step_results = []
            for res in self.results_per_scenario:
                step_row = res['results_df'].loc[[step_name]]
                step_results.append(step_row)

            df_step = pd.concat(step_results)

            # --- Preparación de datos para los gráficos ---
            df_melted = df_step.melt(
                value_vars=self.model_names,
                var_name='Modelo',
                value_name='ECRPS'
            )
            
            wins = df_step['Mejor Modelo'].value_counts()
            
            # --- Creación de los gráficos ---
            title_prefix = f"Resultados para el Paso {step_name}" if isinstance(step_name, int) else "Resultados Generales (Promedio de Pasos)"
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(title_prefix, fontsize=18)

            # Boxplot
            sns.boxplot(ax=axes[0], data=df_melted, x='Modelo', y='ECRPS', palette='viridis')
            axes[0].set_title('Distribución de ECRPS por Modelo', fontsize=14)
            axes[0].set_xlabel('')
            axes[0].set_ylabel('ECRPS')
            
            # --- CORRECCIÓN APLICADA AQUÍ ---
            # Se rotan las etiquetas y se alinean correctamente por separado.
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
            
            # Pie Chart
            if not wins.empty:
                color_map = {model: PlotManager._STYLE['default_colors'][i % len(PlotManager._STYLE['default_colors'])] for i, model in enumerate(self.model_names)}
                pie_colors = [color_map.get(label, '#CCCCCC') for label in wins.index]
                
                axes[1].pie(wins, labels=wins.index, autopct='%1.1f%%', startangle=140, colors=pie_colors,
                            wedgeprops={"edgecolor":"k",'linewidth': 0.5, 'antialiased': True})
                axes[1].set_title('Distribución de Victorias por Modelo', fontsize=14)
            else:
                axes[1].text(0.5, 0.5, 'No hay datos de victorias', ha='center', va='center')
                axes[1].set_title('Distribución de Victorias por Modelo', fontsize=14)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()