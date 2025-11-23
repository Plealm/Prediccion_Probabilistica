# modelos.py (JUSTO Y OPTIMIZADO - 9 modelos sin sesgos)

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple
from dataclasses import dataclass
import gc
import os
import warnings

# Configurar TensorFlow ANTES de importar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.filterwarnings("ignore", category=UserWarning)

# Imports condicionales
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from bayes_opt import BayesianOptimization
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.utils import check_random_state
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    try:
        tf.config.set_visible_devices([], 'GPU')
    except:
        pass
    from tensorflow.keras import layers, optimizers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

@dataclass
class CVResult:
    param_value: Union[int, float]
    mean_crps: float
    std_crps: float
    fold_scores: List[float]


class CircularBlockBootstrapModel:
    """CBB con congelamiento funcional."""

    def __init__(self, block_length: Union[int, str] = 'auto', n_boot: int = 1000,
                 random_state: int = 42, verbose: bool = False,
                 hyperparam_ranges: Dict = None, optimize: bool = True):
        self.block_length = block_length
        self.n_boot = n_boot
        self.random_state = random_state
        self.verbose = verbose
        self.hyperparam_ranges = hyperparam_ranges or {'block_length': [2, 50]}
        self.optimize = optimize
        self.rng = np.random.default_rng(random_state)
        self.best_params = {}
        self._frozen_block_length = None  # ← NUEVO: almacena valor congelado

    def _determine_block_length_optimal(self, n: int) -> int:
        """Heurística Politis-White (2004): l ≈ 1.5 × n^(1/3)"""
        if isinstance(self.block_length, (int, np.integer)):
            return max(2, int(self.block_length))
        
        l_opt = max(2, int(round(1.5 * (n ** (1/3)))))
        min_l, max_l = self.hyperparam_ranges.get('block_length', [2, min(50, n//2)])
        return min(max(l_opt, min_l), max_l)

    def fit_predict(self, history: Union['pd.DataFrame', np.ndarray]) -> np.ndarray:
        """
        CORREGIDO: Respeta block_length congelado.
        """
        import pandas as pd
        series = history['valor'].values if isinstance(history, pd.DataFrame) else np.asarray(history).flatten()
        n = len(series)
        
        if n < 10:
            return np.full(self.n_boot, np.mean(series[-min(8, n):]))
        
        # FIX BUG #14: Verificar si hay valor congelado
        if self._frozen_block_length is not None:
            l = self._frozen_block_length
        elif isinstance(self.block_length, (int, np.integer)):
            l = max(2, int(self.block_length))
        else:
            l = self._determine_block_length_optimal(n)
            if self.optimize and not isinstance(self.block_length, (int, np.integer)):
                self.best_params = {'block_length': l}
        
        # CBB estándar
        within_block_pos = n % l
        starts = self.rng.integers(0, n, size=self.n_boot)
        positions = (starts + within_block_pos) % n
        return series[positions]
    
    def freeze_hyperparameters(self, train_data: np.ndarray):
        """Congela block_length basado en datos de entrenamiento."""
        optimal_l = self._determine_block_length_optimal(len(train_data))
        self._frozen_block_length = optimal_l
        self.optimize = False


class SieveBootstrapModel:
    """Sieve Bootstrap con congelamiento funcional y cache correcto."""

    def __init__(self, order: Union[int, str] = 'auto', n_boot: int = 1000,
                 random_state: int = 42, verbose: bool = False,
                 hyperparam_ranges: Dict = None, optimize: bool = True):
        self.order = order
        self.n_boot = n_boot
        self.random_state = random_state
        self.verbose = verbose
        self.hyperparam_ranges = hyperparam_ranges or {'order': [1, 20]}
        self.optimize = optimize
        self.rng = np.random.default_rng(random_state)
        self.best_params = {}
        
        # FIX #5: Cache mejorado con control de invalidación
        self._frozen_order = None
        self._frozen_params = None      # Parámetros AR congelados
        self._frozen_residuals = None   # Residuos congelados
        self._is_frozen = False         # Flag explícito

    def _determine_order_aic(self, series: np.ndarray) -> int:
        """AIC con 5 candidatos estratégicos."""
        from statsmodels.tsa.ar_model import AutoReg
        
        if isinstance(self.order, (int, np.integer)):
            return max(1, int(self.order))
        
        n = len(series)
        min_p, max_p = self.hyperparam_ranges.get('order', [1, min(20, n//3)])
        
        candidates = sorted(set([
            1, 2, max(3, min_p), (min_p + max_p) // 2, min(max_p, n//4)
        ]))
        candidates = [c for c in candidates if min_p <= c <= min(max_p, n//3)]
        
        best_aic, best_p = np.inf, 1
        for p in candidates:
            try:
                aic = AutoReg(series, lags=p).fit().aic
                if aic < best_aic:
                    best_aic, best_p = aic, p
            except:
                continue
        return best_p

    def freeze_hyperparameters(self, train_data: np.ndarray):
        """
        FIX #5: Congela order Y ajusta el modelo AR una sola vez.
        Los parámetros y residuos se calculan aquí y se reutilizan.
        """
        from statsmodels.tsa.ar_model import AutoReg
        
        series = train_data.flatten() if hasattr(train_data, 'flatten') else np.asarray(train_data)
        
        # Determinar orden óptimo
        optimal_p = self._determine_order_aic(series)
        self._frozen_order = optimal_p
        
        # Ajustar modelo AR con los datos de entrenamiento+calibración
        try:
            model = AutoReg(series, lags=optimal_p).fit()
            self._frozen_params = model.params.copy()
            self._frozen_residuals = model.resid - np.mean(model.resid)
            self._is_frozen = True
            
            if self.verbose:
                print(f"  Sieve Bootstrap congelado: order={optimal_p}, "
                      f"n_residuos={len(self._frozen_residuals)}")
        except Exception as e:
            if self.verbose:
                print(f"  Error congelando Sieve: {e}")
            # Fallback: no congelar, se ajustará en cada paso
            self._is_frozen = False
        
        self.optimize = False

    def fit_predict(self, history: Union['pd.DataFrame', np.ndarray]) -> np.ndarray:
        """
        FIX #5: Usa parámetros congelados si existen.
        Solo los últimos p valores de la serie se usan para predecir.
        """
        from statsmodels.tsa.ar_model import AutoReg
        import pandas as pd
        
        series = history['valor'].values if isinstance(history, pd.DataFrame) else np.asarray(history).flatten()
        n = len(series)
        
        if n < 10:
            return np.full(self.n_boot, np.mean(series[-min(8, n):]))
        
        # Determinar orden a usar
        if self._frozen_order is not None:
            p = self._frozen_order
        elif isinstance(self.order, (int, np.integer)):
            p = max(1, int(self.order))
        else:
            p = self._determine_order_aic(series)
            if self.optimize:
                self.best_params = {'order': p}
        
        if p >= n - 1:
            p = max(1, n // 2)
        
        try:
            # FIX #5: Usar parámetros congelados si están disponibles
            if self._is_frozen and self._frozen_params is not None:
                # Usar parámetros y residuos congelados
                params = self._frozen_params
                residuals = self._frozen_residuals
                
                # Verificar que p coincida con los parámetros congelados
                expected_p = len(params) - 1  # params incluye intercepto
                if p != expected_p:
                    p = expected_p  # Usar el orden de los parámetros congelados
            else:
                # No está congelado: ajustar modelo nuevo (comportamiento original)
                model = AutoReg(series, lags=p).fit()
                params = model.params
                residuals = model.resid - np.mean(model.resid)
            
            # Predicción usando los últimos p valores de la serie ACTUAL
            last_p = series[-p:][::-1]
            ar_pred = params[0] + np.dot(params[1:], last_p)
            
            # Bootstrap de residuos (siempre de los residuos guardados/calculados)
            boot_resid = self.rng.choice(residuals, size=self.n_boot, replace=True)
            
            return ar_pred + boot_resid
            
        except Exception as e:
            if self.verbose:
                print(f"  Sieve error: {e}")
            return np.full(self.n_boot, np.mean(series[-min(8, n):]))


class LSPM:
    """
    LSPM Studentizado - Implementación EXACTA del algoritmo original.
    Sin simplificaciones, solo optimizaciones de NumPy.
    """
    
    def __init__(self, random_state=42, verbose=False):
        self.version = 'studentized'
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)
        self.n_lags = None
        self.best_params = {}

    def optimize_hyperparameters(self, df, reference_noise):
        self.best_params = {'version': self.version}
        return None, -1.0

    def _calculate_critical_values(self, values: np.ndarray) -> np.ndarray:
        """
        Algoritmo LSPM studentizado EXACTO.
        Calcula valores críticos usando la matriz hat completa.
        """
        p = self.n_lags if self.n_lags and self.n_lags > 0 else max(1, int(len(values)**(1/3)))
        n = len(values)
        
        if n < 2 * p + 2:
            return np.array([])
        
        # Construir matrices de diseño
        y_full = values[p:]
        n_obs = len(y_full)
        
        # X_full: cada fila son los p lags
        X_full = np.column_stack([values[p-i-1:n-i-1] for i in range(p)])
        
        X_train = X_full[:-1]  # Sin última observación
        y_train = y_full[:-1]
        x_test = X_full[-1]    # Última observación
        
        # Agregar intercepto
        X_train_b = np.column_stack([np.ones(len(X_train)), X_train])
        x_test_b = np.concatenate([[1], x_test])
        X_bar = np.vstack([X_train_b, x_test_b])
        
        try:
            # Matriz hat: H = X(X'X)^{-1}X'
            XtX_inv = np.linalg.pinv(X_bar.T @ X_bar)
            H_bar = X_bar @ XtX_inv @ X_bar.T
        except np.linalg.LinAlgError:
            return np.array([])
        
        n_train = len(y_train)
        h_ii = np.diag(H_bar)[:n_train]
        h_n = H_bar[-1, -1]
        h_in = H_bar[:n_train, -1]
        h_ni = H_bar[-1, :n_train]
        
        # Predicciones leave-one-out
        y_hat_full = H_bar[:n_train, :n_train] @ y_train
        
        # Filtrar casos válidos
        valid = (np.abs(1 - h_ii) > 1e-10) & (np.abs(1 - h_n) > 1e-10)
        
        if not np.any(valid):
            return np.array([])
        
        # Calcular B_i para casos válidos
        sqrt_1_minus_h_n = np.sqrt(np.maximum(1 - h_n, 1e-10))
        sqrt_1_minus_h_ii = np.sqrt(np.maximum(1 - h_ii[valid], 1e-10))
        
        B_i = sqrt_1_minus_h_n + h_in[valid] / sqrt_1_minus_h_ii
        
        # Término 1: suma ponderada de y_train
        term1 = np.dot(h_ni, y_train) / sqrt_1_minus_h_n
        
        # Término 2: residuos studentizados
        resid_i = y_train[valid] - y_hat_full[valid]
        term2 = resid_i / sqrt_1_minus_h_ii
        
        # Filtrar B_i válidos
        valid_B = np.abs(B_i) > 1e-10
        
        if not np.any(valid_B):
            return np.array([])
        
        critical_values = (term1 + term2[valid_B]) / B_i[valid_B]
        
        return critical_values

    def fit_predict(self, df) -> np.ndarray:
        """Retorna valores críticos como distribución predictiva."""
        values = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        critical = self._calculate_critical_values(values.astype(np.float64))
        
        if len(critical) == 0:
            return np.full(1000, np.mean(values[-min(8, len(values)):]))
        
        return critical


class LSPMW(LSPM):
    """LSPM Ponderado - Pesos exponenciales sobre valores críticos."""
    
    def __init__(self, rho: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        if not (0 < rho < 1):
            raise ValueError("rho debe estar entre 0 y 1")
        self.rho = rho
        self.best_params = {'rho': rho}

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        
        values = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        critical = self._calculate_critical_values(values.astype(np.float64))
        
        if len(critical) == 0:
            return np.full(1000, np.mean(values[-min(8, len(values)):]))
        
        n_crit = len(critical)
        
        # CORRECCIÓN: Pesos basados en índice temporal ANTES de ordenar
        # Los valores críticos más recientes (últimos índices) tienen más peso
        temporal_weights = self.rho ** np.arange(n_crit - 1, -1, -1)
        temporal_weights = temporal_weights / temporal_weights.sum()
        
        # Crear índices ordenados pero mantener asociación con pesos temporales
        sort_indices = np.argsort(critical)
        sorted_critical = critical[sort_indices]
        sorted_weights = temporal_weights[sort_indices]  # ← Pesos viajan con sus valores
        
        # CDF acumulada con pesos temporales
        cum_probs = np.cumsum(sorted_weights)
        
        # Muestreo por inverse CDF
        u = np.linspace(0, 1 - 1e-10, 1000)
        indices = np.searchsorted(cum_probs, u, side='right')
        indices = np.clip(indices, 0, n_crit - 1)
        
        return sorted_critical[indices]

    def optimize_hyperparameters(self, df, reference_noise):
        """Optimización rápida de rho con 3 candidatos."""
        from metricas import ecrps
        
        candidates = [0.90, 0.95, 0.99]
        best_rho, best_score = self.rho, float('inf')
        
        for rho in candidates:
            try:
                self.rho = rho
                samples = self.fit_predict(df)
                if len(samples) == 0:
                    continue
                
                ref = self.rng.choice(reference_noise, 
                                     size=min(1000, len(reference_noise)), 
                                     replace=False)
                score = ecrps(samples, ref)
                
                if score < best_score:
                    best_score, best_rho = score, rho
            except:
                continue
        
        self.rho = best_rho
        self.best_params = {'rho': best_rho}
        return best_rho, best_score


class DeepARModel:
    """DeepAR con early stopping y NO re-entrenamiento en rolling."""
    
    def __init__(self, hidden_size=20, n_lags=5, num_layers=1, dropout=0.1, 
                 lr=0.01, batch_size=32, epochs=30, num_samples=1000,
                 random_state=42, verbose=False, optimize=True,
                 early_stopping_patience=5):  # ← NUEVO
        
        import torch
        
        self.hidden_size = hidden_size
        self.n_lags = n_lags
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_samples = num_samples
        self.random_state = random_state
        self.verbose = verbose
        self.optimize = optimize
        self.early_stopping_patience = early_stopping_patience  # ← NUEVO
        
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.best_params = {}
        self._is_optimized = False
        self._trained_model = None  # ← NUEVO: modelo pre-entrenado
        self._training_history = []  # ← NUEVO: historial de pérdidas
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    class _DeepARNN(torch.nn.Module):
        """Red LSTM que produce mu y sigma."""
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            import torch.nn as nn
            drop = dropout if num_layers > 1 else 0
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=drop)
            self.fc_mu = nn.Linear(hidden_size, 1)
            self.fc_sigma = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            import torch
            out, _ = self.lstm(x)
            mu = self.fc_mu(out[:, -1, :])
            sigma = torch.exp(self.fc_sigma(out[:, -1, :])).clamp(min=1e-6, max=10)
            return mu, sigma
    
    def _train_with_early_stopping(self, X_t, y_t):
        """
        FIX #13: Entrena con early stopping para garantizar convergencia.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Split train/val (80/20)
        n_train = int(0.8 * len(X_t))
        X_train, X_val = X_t[:n_train], X_t[n_train:]
        y_train, y_val = y_t[:n_train], y_t[n_train:]
        
        if len(X_val) < 5:  # Demasiado pequeño para validación
            X_val, y_val = X_train[-10:], y_train[-10:]
        
        self.model = self._DeepARNN(1, self.hidden_size, self.num_layers, self.dropout)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.GaussianNLLLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        self._training_history = []
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            perm = torch.randperm(len(X_train))
            
            for i in range(0, len(X_train), self.batch_size):
                idx = perm[i:i+self.batch_size]
                if len(idx) < 2:
                    continue
                mu, sig = self.model(X_train[idx])
                loss = crit(mu, y_train[idx], sig**2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                mu_val, sig_val = self.model(X_val)
                val_loss = crit(mu_val, y_val, sig_val**2).item()
            
            self._training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / max(1, len(X_train) // self.batch_size),
                'val_loss': val_loss
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restaurar mejor modelo
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.model
    
    def fit_predict(self, df) -> np.ndarray:
        """
        FIX #7: NO re-entrena en ventana rodante, usa modelo pre-entrenado.
        """
        import torch
        import pandas as pd
        
        try:
            series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
            self.scaler_mean = np.nanmean(series)
            self.scaler_std = np.nanstd(series) + 1e-8
            norm_series = (series - self.scaler_mean) / self.scaler_std
            
            # Aplicar hiperparámetros optimizados
            if self.best_params:
                self.n_lags = self.best_params.get('n_lags', self.n_lags)
                self.hidden_size = self.best_params.get('hidden_size', self.hidden_size)
                self.num_layers = self.best_params.get('num_layers', self.num_layers)
                self.dropout = self.best_params.get('dropout', self.dropout)
                self.lr = self.best_params.get('lr', self.lr)
            
            if len(norm_series) <= self.n_lags:
                return np.full(self.num_samples, self.scaler_mean)
            
            # FIX #7: Solo entrenar si no hay modelo previo
            if self._trained_model is None:
                X, y = [], []
                for i in range(len(norm_series) - self.n_lags):
                    X.append(norm_series[i:i + self.n_lags])
                    y.append(norm_series[i + self.n_lags])
                X, y = np.array(X), np.array(y)
                
                if len(X) < self.batch_size:
                    return np.full(self.num_samples, self.scaler_mean)
                
                X_t = torch.FloatTensor(X.reshape(-1, self.n_lags, 1))
                y_t = torch.FloatTensor(y.reshape(-1, 1))
                
                self._train_with_early_stopping(X_t, y_t)
                self._trained_model = self.model  # Guardar modelo entrenado
            else:
                # Usar modelo pre-entrenado
                self.model = self._trained_model
            
            # Predicción con modelo fijo
            self.model.eval()
            with torch.no_grad():
                last_seq = torch.FloatTensor(norm_series[-self.n_lags:].reshape(1, self.n_lags, 1))
                mu, sig = self.model(last_seq)
            
            samples = np.random.normal(mu.item(), sig.item(), self.num_samples)
            samples = samples * self.scaler_std + self.scaler_mean
            
            return np.nan_to_num(samples, nan=self.scaler_mean)
            
        except Exception as e:
            if self.verbose:
                print(f"    DeepAR error: {e}")
            return np.full(self.num_samples, np.nanmean(df) if hasattr(df, '__len__') else 0)
    
    def get_training_history(self):
        """Retorna historial de entrenamiento para diagnosticar convergencia."""
        return self._training_history


# =============================================================================
# AREPD - Autoregressive Encompassing Predictive Distribution
# =============================================================================

class AREPD:
    """
    Autoregressive Encompassing Predictive Distribution.
    Usa Ridge regression con pesos exponenciales y features polinomiales.
    """
    
    def __init__(self, n_lags=5, rho=0.95, alpha=0.1, poly_degree=2,
                 random_state=42, verbose=False, optimize=True):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn requerido: pip install scikit-learn")
        
        self.n_lags = n_lags
        self.rho = rho
        self.alpha = alpha
        self.poly_degree = poly_degree
        self.random_state = random_state
        self.verbose = verbose
        self.optimize = optimize
        
        self.mean_val = None
        self.std_val = None
        self.rng = check_random_state(random_state)
        self.best_params = {}
        self._is_optimized = False
        
        np.random.seed(random_state)
    
    def _create_lag_matrix(self, values: np.ndarray, n_lags: int, degree: int = 2):
        """Crea matriz de features con lags polinomiales."""
        n = len(values) - n_lags
        if n <= 0:
            return np.array([]), np.array([])
        
        y = values[n_lags:]
        X_list = [np.ones((n, 1))]
        
        for lag in range(n_lags):
            lagged = values[lag:lag + n].reshape(-1, 1)
            for d in range(1, degree + 1):
                X_list.append(np.power(lagged, d))
        
        return np.hstack(X_list), y
    
    def _Qn_distribution(self, C: np.ndarray) -> np.ndarray:
        """Convierte predicciones en muestras de distribución."""
        sorted_C = np.sort(C)
        # Desnormalizar
        samples = (sorted_C * self.std_val) + self.mean_val
        return samples
    
    def optimize_hyperparameters(self, df, reference_noise):
        """Optimización Bayesiana rápida."""
        from metricas import ecrps
        
        if not BAYESOPT_AVAILABLE:
            self.best_params = {'n_lags': 5, 'rho': 0.9, 'poly_degree': 2}
            self._is_optimized = True
            return self.best_params, -1
        
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        
        def objective(n_lags, rho, poly_degree):
            try:
                self.n_lags = max(1, int(round(n_lags)))
                self.rho = min(0.999, max(0.5, float(rho)))
                self.poly_degree = max(1, int(round(poly_degree)))
                
                if len(series) < self.n_lags * 2:
                    return -1e12
                
                samples = self.fit_predict(df)
                if len(samples) == 0:
                    return -1e12
                
                ref_sub = self.rng.choice(reference_noise, 
                                         size=min(500, len(reference_noise)), 
                                         replace=False)
                return -ecrps(samples, ref_sub)
            except:
                return -1e12
        
        optimizer = BayesianOptimization(
            f=objective,
            pbounds={'n_lags': (2, 8), 'rho': (0.7, 0.99), 'poly_degree': (1, 3)},
            random_state=self.random_state,
            verbose=0
        )
        
        try:
            optimizer.maximize(init_points=2, n_iter=3)
        except:
            pass
        
        best_ecrps = -1
        if optimizer.max and optimizer.max['target'] > -1e11:
            p = optimizer.max['params']
            best_ecrps = -optimizer.max['target']
            self.best_params = {
                'n_lags': int(round(p['n_lags'])),
                'rho': p['rho'],
                'poly_degree': int(round(p['poly_degree']))
            }
        else:
            self.best_params = {'n_lags': 5, 'rho': 0.9, 'poly_degree': 2}
        
        self._is_optimized = True
        
        if self.verbose:
            print(f"  AREPD: {self.best_params}")
        
        return self.best_params, best_ecrps
    
    def fit_predict(self, df) -> np.ndarray:
        """Ajusta y retorna muestras de la distribución predictiva."""
        try:
            values = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
            
            if len(values) < self.n_lags * 2:
                return np.full(1000, np.mean(values))
            
            # Aplicar hiperparámetros optimizados
            if self.best_params:
                self.n_lags = self.best_params.get('n_lags', self.n_lags)
                self.rho = self.best_params.get('rho', self.rho)
                self.poly_degree = self.best_params.get('poly_degree', self.poly_degree)
            
            self.mean_val = np.nanmean(values)
            self.std_val = np.nanstd(values) + 1e-8
            normalized = (values - self.mean_val) / self.std_val
            
            X, y = self._create_lag_matrix(normalized, self.n_lags, self.poly_degree)
            if X.shape[0] == 0:
                return np.full(1000, self.mean_val)
            
            # Pesos exponenciales
            weights = self.rho ** np.arange(len(y))[::-1]
            weights = weights / (weights.sum() + 1e-8)
            
            model = Ridge(alpha=self.alpha, fit_intercept=False)
            model.fit(X, y, sample_weight=weights)
            
            predictions = model.predict(X)
            samples = self._Qn_distribution(predictions)
            
            return samples
            
        except Exception as e:
            if self.verbose:
                print(f"    AREPD error: {e}")
            return np.full(1000, np.nanmean(df) if hasattr(df, '__len__') else 0)


# =============================================================================
# MondrianCPS - Mondrian Conformal Predictive System
# =============================================================================

class MondrianCPSModel:
    """
    Mondrian Conformal Predictive System con τ ~ Uniform(0, 1).
    Usa XGBoost como modelo base con estratificación por nivel de predicción.
    
    OPTIMIZACIÓN: Implementa lógica para entrenar solo una vez al inicio de la 
    ventana de prueba y reutilizar los artefactos de calibración.
    """
    
    def __init__(self, n_lags: int = 10, n_bins: int = 10, test_size: float = 0.25,
                 random_state: int = 42, verbose: bool = False, optimize: bool = True):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost requerido: pip install xgboost")
        
        self.n_lags = n_lags
        self.n_bins = n_bins
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.optimize = optimize
        self.rng = np.random.default_rng(random_state)
        
        self.base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=self.random_state,
            n_jobs=1,  # Importante: 1 hilo para no interferir con paralelización externa
            verbosity=0
        )
        self.best_params = {}
        self._is_optimized = False
        
        # VARIABLE CLAVE PARA OPTIMIZACIÓN DE TIEMPO
        # Aquí guardaremos el modelo entrenado y los datos de calibración
        self._fitted_artifacts = None 
    
    def _create_lag_matrix(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(series) - self.n_lags):
            X.append(series[i:(i + self.n_lags)])
            y.append(series[i + self.n_lags])
        return np.array(X), np.array(y)
    
    def _create_samples_from_scores(self, scores: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """Genera muestras usando los scores de calibración."""
        if len(scores) == 0:
            return np.zeros(n_samples)
        
        # Añadir ruido aleatorio para diversificar muestras
        base_samples = self.rng.choice(scores, size=n_samples, replace=True)
        noise = self.rng.normal(0, np.std(scores) * 0.1, n_samples)
        
        return base_samples + noise
    
    def optimize_hyperparameters(self, df, reference_noise):
        """Optimización rápida de hiperparámetros."""
        from metricas import ecrps
        
        if not BAYESOPT_AVAILABLE:
            self.best_params = {'n_lags': 10, 'n_bins': 8}
            self._is_optimized = True
            return self.best_params, -1
        
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        
        def objective(n_lags, n_bins):
            try:
                old_lags, old_bins = self.n_lags, self.n_bins
                self.n_lags = max(5, int(n_lags))
                self.n_bins = max(3, int(n_bins))
                
                if len(series) <= self.n_lags * 2:
                    self.n_lags, self.n_bins = old_lags, old_bins
                    return -1e10
                
                # Nota: fit_predict usará logic de caché si _fitted_artifacts existe.
                # Para optimización, queremos forzar re-entrenamiento o limpiar caché,
                # pero fit_predict maneja series cambiantes.
                # Para simplificar en optimización bayesiana rápida, limpiamos caché temporalmente:
                saved_artifacts = self._fitted_artifacts
                self._fitted_artifacts = None
                
                samples = self.fit_predict(series)
                
                # Restaurar caché si existía (aunque en optimización suele estar vacío)
                self._fitted_artifacts = saved_artifacts
                
                if len(samples) == 0:
                    self.n_lags, self.n_bins = old_lags, old_bins
                    return -1e10
                
                ref_sub = self.rng.choice(reference_noise,
                                         size=min(500, len(reference_noise)),
                                         replace=False)
                score = ecrps(samples, ref_sub)
                
                self.n_lags, self.n_bins = old_lags, old_bins
                return -score
            except:
                return -1e10
        
        optimizer = BayesianOptimization(
            f=objective,
            pbounds={'n_lags': (5, 15), 'n_bins': (3, 12)},
            random_state=self.random_state,
            verbose=0
        )
        
        try:
            optimizer.maximize(init_points=2, n_iter=3)
        except:
            pass
        
        best_ecrps = -1
        if optimizer.max and optimizer.max['target'] > -1e9:
            self.best_params = {
                'n_lags': int(optimizer.max['params']['n_lags']),
                'n_bins': int(optimizer.max['params']['n_bins'])
            }
            best_ecrps = -optimizer.max['target']
        else:
            self.best_params = {'n_lags': 10, 'n_bins': 8}
        
        self._is_optimized = True
        
        if self.verbose:
            print(f"  MCPS: {self.best_params}")
        
        return self.best_params, best_ecrps
    
    def fit_predict(self, df) -> np.ndarray:
        """
        Ajusta y retorna muestras de la distribución predictiva.
        OPTIMIZACIÓN: Si _fitted_artifacts ya existe, salta el entrenamiento 
        y usa el modelo guardado para predecir el nuevo paso.
        """
        try:
            series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
            
            if self.best_params:
                self.n_lags = self.best_params.get('n_lags', self.n_lags)
                self.n_bins = self.best_params.get('n_bins', self.n_bins)
            
            # ================================================================
            # FASE 1: ENTRENAMIENTO (Solo se ejecuta la primera vez)
            # ================================================================
            if self._fitted_artifacts is None:
                if len(series) < self.n_lags * 2:
                    return np.full(1000, np.mean(series))
                
                X, y = self._create_lag_matrix(series)
                
                # Usamos los últimos datos disponibles hasta el momento para calibrar
                n_calib = max(10, int(len(X) * self.test_size))
                
                if n_calib >= len(X):
                    # Fallback simple
                    self._fitted_artifacts = {
                        'fallback_mean': np.mean(series),
                        'is_fallback': True
                    }
                    return np.full(1000, np.mean(series))
                
                X_train, X_calib = X[:-n_calib], X[-n_calib:]
                y_train, y_calib = y[:-n_calib], y[-n_calib:]
                
                # Entrenar modelo base
                self.base_model.fit(X_train, y_train)
                
                # Generar predicciones de calibración
                calib_preds = self.base_model.predict(X_calib)
                
                # Calcular bordes de los bins (Mondrian)
                bin_edges = None
                try:
                    _, bin_edges = pd.qcut(calib_preds, self.n_bins, retbins=True, duplicates='drop')
                except:
                    pass # Fallback a sin bins si falla qcut
                
                # GUARDAR TODO EN ARTEFACTOS
                self._fitted_artifacts = {
                    'calib_preds': calib_preds,
                    'y_calib': y_calib,
                    'bin_edges': bin_edges,
                    'is_fallback': False
                }

            # ================================================================
            # FASE 2: PREDICCIÓN (Se ejecuta en cada paso usando lo guardado)
            # ================================================================
            
            # Verificar si estamos en modo fallback
            if self._fitted_artifacts.get('is_fallback', False):
                return np.full(1000, self._fitted_artifacts['fallback_mean'])

            # Preparar el vector de entrada para el paso actual (últimos lags)
            x_test = series[-self.n_lags:].reshape(1, -1)
            
            # 1. Predicción puntual
            point_pred = self.base_model.predict(x_test)[0]
            
            # Recuperar datos de calibración
            calib_preds = self._fitted_artifacts['calib_preds']
            y_calib = self._fitted_artifacts['y_calib']
            bin_edges = self._fitted_artifacts['bin_edges']
            
            # 2. Estratificación Mondrian (Binning)
            try:
                if bin_edges is not None:
                    # En qué bin cae cada punto de calibración
                    bin_idx = np.digitize(calib_preds, bins=bin_edges) - 1
                    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
                    
                    # En qué bin cae la predicción actual
                    test_bin = np.clip(np.digitize(point_pred, bins=bin_edges) - 1, 0, len(bin_edges) - 2)
                    
                    # Máscara: solo usamos residuos del mismo bin
                    local_mask = (bin_idx == test_bin)
                    
                    # Si hay muy pocos puntos en el bin, usamos todos (fallback local)
                    if np.sum(local_mask) < 5:
                        local_mask = np.ones(len(calib_preds), dtype=bool)
                else:
                    local_mask = np.ones(len(calib_preds), dtype=bool)
            except:
                local_mask = np.ones(len(calib_preds), dtype=bool)
            
            # 3. Calcular scores de no conformidad locales
            local_y = y_calib[local_mask]
            local_preds = calib_preds[local_mask]
            
            # C = y - y_hat  => y = y_hat + C
            # Aplicamos los residuos pasados a la predicción actual
            calibration_scores = point_pred + (local_y - local_preds)
            
            return self._create_samples_from_scores(calibration_scores, 1000)
            
        except Exception as e:
            if self.verbose:
                print(f"    MCPS error: {e}")
            return np.full(1000, np.nanmean(df) if hasattr(df, '__len__') else 0)


# =============================================================================
# AV-MCPS - Adaptive Volatility Mondrian CPS
# =============================================================================

class AdaptiveVolatilityMondrianCPS:
    """
    Adaptive Volatility Mondrian CPS con estratificación bidimensional.
    
    OPTIMIZACIÓN: 
    - Entrena el modelo base (XGBoost) UNA sola vez al inicio del escenario.
    - Calcula los bins de predicción y volatilidad UNA sola vez.
    - Reutiliza todo para los pasos siguientes (N_TEST_STEPS), reduciendo drásticamente el tiempo.
    """
    
    def __init__(self, n_lags: int = 15, n_pred_bins: int = 8, n_vol_bins: int = 4,
                 volatility_window: int = 20, test_size: float = 0.25,
                 random_state: int = 42, verbose: bool = False, optimize: bool = True):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost requerido: pip install xgboost")
        
        self.n_lags = n_lags
        self.n_pred_bins = n_pred_bins
        self.n_vol_bins = n_vol_bins
        self.volatility_window = volatility_window
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.optimize = optimize
        self.rng = np.random.default_rng(random_state)
        
        self.base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=self.random_state,
            n_jobs=1, # 1 hilo para no saturar la paralelización externa
            verbosity=0
        )
        self.best_params = {}
        self._is_optimized = False
        
        # VARIABLE CLAVE PARA OPTIMIZACIÓN DE TIEMPO
        self._fitted_artifacts = None
    
    def _create_lag_matrix(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(series) - self.n_lags):
            X.append(series[i:(i + self.n_lags)])
            y.append(series[i + self.n_lags])
        return np.array(X), np.array(y)
    
    def _calculate_volatility(self, series: np.ndarray) -> np.ndarray:
        volatility = pd.Series(series).rolling(
            window=self.volatility_window, min_periods=1
        ).std().bfill().values
        return volatility[self.n_lags - 1: -1] if len(volatility) > self.n_lags else volatility
    
    def _create_samples_from_scores(self, scores: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        if len(scores) == 0:
            return np.zeros(n_samples)
        
        base_samples = self.rng.choice(scores, size=n_samples, replace=True)
        noise = self.rng.normal(0, np.std(scores) * 0.1, n_samples)
        return base_samples + noise
    
    def optimize_hyperparameters(self, df, reference_noise):
        """Optimización rápida."""
        from metricas import ecrps
        
        if not BAYESOPT_AVAILABLE:
            self.best_params = {'n_lags': 15, 'n_pred_bins': 8, 
                               'n_vol_bins': 4, 'volatility_window': 20}
            self._is_optimized = True
            return self.best_params, -1
        
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        
        def objective(n_lags, n_pred_bins, n_vol_bins, volatility_window):
            try:
                old = (self.n_lags, self.n_pred_bins, self.n_vol_bins, self.volatility_window)
                
                self.n_lags = int(n_lags)
                self.n_pred_bins = int(n_pred_bins)
                self.n_vol_bins = int(n_vol_bins)
                self.volatility_window = int(volatility_window)
                
                if len(series) <= self.n_lags * 2:
                    self.n_lags, self.n_pred_bins, self.n_vol_bins, self.volatility_window = old
                    return -1e10
                
                # Limpiar cache temporalmente para forzar re-evaluación en optimización
                saved_artifacts = self._fitted_artifacts
                self._fitted_artifacts = None
                
                samples = self.fit_predict(series)
                
                self._fitted_artifacts = saved_artifacts
                
                if len(samples) == 0:
                    self.n_lags, self.n_pred_bins, self.n_vol_bins, self.volatility_window = old
                    return -1e10
                
                ref_sub = self.rng.choice(reference_noise,
                                         size=min(500, len(reference_noise)),
                                         replace=False)
                score = ecrps(samples, ref_sub)
                
                self.n_lags, self.n_pred_bins, self.n_vol_bins, self.volatility_window = old
                return -score
            except:
                return -1e10
        
        optimizer = BayesianOptimization(
            f=objective,
            pbounds={
                'n_lags': (5, 20),
                'n_pred_bins': (3, 10),
                'n_vol_bins': (2, 6),
                'volatility_window': (10, 30)
            },
            random_state=self.random_state,
            verbose=0
        )
        
        try:
            optimizer.maximize(init_points=2, n_iter=3)
        except:
            pass
        
        best_ecrps = -1
        if optimizer.max and optimizer.max['target'] > -1e9:
            self.best_params = {k: int(v) for k, v in optimizer.max['params'].items()}
            best_ecrps = -optimizer.max['target']
        else:
            self.best_params = {'n_lags': 15, 'n_pred_bins': 8,
                               'n_vol_bins': 4, 'volatility_window': 20}
        
        self._is_optimized = True
        
        if self.verbose:
            print(f"  AV-MCPS: {self.best_params}")
        
        return self.best_params, best_ecrps
    
    def fit_predict(self, df) -> np.ndarray:
        """Ajusta y retorna muestras con estratificación 2D optimizada."""
        try:
            series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
            
            if self.best_params:
                self.n_lags = self.best_params.get('n_lags', self.n_lags)
                self.n_pred_bins = self.best_params.get('n_pred_bins', self.n_pred_bins)
                self.n_vol_bins = self.best_params.get('n_vol_bins', self.n_vol_bins)
                self.volatility_window = self.best_params.get('volatility_window', self.volatility_window)
            
            # ================================================================
            # FASE 1: ENTRENAMIENTO (Solo primera vez)
            # ================================================================
            if self._fitted_artifacts is None:
                if len(series) < max(self.n_lags * 2, self.volatility_window):
                    return np.full(1000, np.mean(series))
                
                X, y = self._create_lag_matrix(series)
                vol_features = self._calculate_volatility(series)
                
                n_calib = max(10, int(len(X) * self.test_size))
                
                if n_calib >= len(X):
                     self._fitted_artifacts = {'is_fallback': True, 'mean': np.mean(series)}
                     return np.full(1000, np.mean(series))
                
                X_train, X_calib = X[:-n_calib], X[-n_calib:]
                y_train, y_calib = y[:-n_calib], y[-n_calib:]
                
                # Alinear features de volatilidad con calibración
                if len(vol_features) >= n_calib:
                    vol_calib = vol_features[-n_calib:]
                else:
                    vol_calib = np.full(n_calib, np.std(series))
                
                # Ajustar modelo base
                self.base_model.fit(X_train, y_train)
                calib_preds = self.base_model.predict(X_calib)
                
                # Calcular bins 2D
                pred_edges, vol_edges = None, None
                try:
                    _, pred_edges = pd.qcut(calib_preds, self.n_pred_bins, retbins=True, duplicates='drop')
                    _, vol_edges = pd.qcut(vol_calib, self.n_vol_bins, retbins=True, duplicates='drop')
                except:
                    pass
                
                # GUARDAR ARTEFACTOS
                self._fitted_artifacts = {
                    'calib_preds': calib_preds,
                    'y_calib': y_calib,
                    'vol_calib': vol_calib,
                    'pred_edges': pred_edges,
                    'vol_edges': vol_edges,
                    'is_fallback': False
                }
            
            # ================================================================
            # FASE 2: PREDICCIÓN
            # ================================================================
            
            if self._fitted_artifacts.get('is_fallback', False):
                return np.full(1000, self._fitted_artifacts['mean'])

            x_test = series[-self.n_lags:].reshape(1, -1)
            test_vol = np.std(series[-self.volatility_window:])
            
            point_pred = self.base_model.predict(x_test)[0]
            
            # Recuperar
            calib_preds = self._fitted_artifacts['calib_preds']
            y_calib = self._fitted_artifacts['y_calib']
            vol_calib = self._fitted_artifacts['vol_calib']
            pred_edges = self._fitted_artifacts['pred_edges']
            vol_edges = self._fitted_artifacts['vol_edges']
            
            # Estratificación 2D
            try:
                if pred_edges is not None and vol_edges is not None:
                    # Bins de calibración
                    pred_idx = np.clip(np.digitize(calib_preds, pred_edges[:-1]) - 1, 0, len(pred_edges) - 2)
                    vol_idx = np.clip(np.digitize(vol_calib, vol_edges[:-1]) - 1, 0, len(vol_edges) - 2)
                    
                    # Bins del punto actual
                    test_pred_bin = np.clip(np.digitize(point_pred, pred_edges[:-1]) - 1, 0, len(pred_edges) - 2)
                    test_vol_bin = np.clip(np.digitize(test_vol, vol_edges[:-1]) - 1, 0, len(vol_edges) - 2)
                    
                    local_mask = (pred_idx == test_pred_bin) & (vol_idx == test_vol_bin)
                    
                    # Relaxing rules si hay pocos datos
                    if np.sum(local_mask) < 5:
                        local_mask = (pred_idx == test_pred_bin)
                        if np.sum(local_mask) < 5:
                            local_mask = np.ones(len(calib_preds), dtype=bool)
                else:
                    local_mask = np.ones(len(calib_preds), dtype=bool)
            except:
                local_mask = np.ones(len(calib_preds), dtype=bool)
            
            local_y = y_calib[local_mask]
            local_preds = calib_preds[local_mask]
            
            calibration_scores = point_pred + (local_y - local_preds)
            
            return self._create_samples_from_scores(calibration_scores, 1000)
            
        except Exception as e:
            if self.verbose:
                print(f"    AV-MCPS error: {e}")
            return np.full(1000, np.nanmean(df) if hasattr(df, '__len__') else 0)


# =============================================================================
# EnCQR-LSTM - Ensemble Conformalized Quantile Regression with LSTM
# =============================================================================

class EnCQR_LSTM_Model:
    """
    EnCQR-LSTM Optimizado: 
    Ensemble de LSTMs con Conformalized Quantile Regression.
    
    OPTIMIZACIÓN: 
    Entrena el ensemble (B redes neuronales) una sola vez al inicio de la ventana 
    de test y lo reutiliza para generar predicciones en los siguientes pasos.
    """
   
    def __init__(self, n_lags: int = 20, B: int = 3, units: int = 32, n_layers: int = 2,
                 lr: float = 0.005, batch_size: int = 16, epochs: int = 20,
                 num_samples: int = 1000, random_state: int = 42, verbose: bool = False,
                 optimize: bool = True, alpha: float = 0.05):
       
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow requerido: pip install tensorflow")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn requerido: pip install scikit-learn")
       
        self.n_lags = n_lags
        self.B = B
        self.units = units
        self.n_layers = n_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_samples = num_samples
        self.random_state = random_state
        self.verbose = verbose
        self.optimize = optimize
        self.alpha = alpha
       
        self.scaler = MinMaxScaler()
        self.best_params = {}
        self._is_optimized = False
        self.rng = np.random.default_rng(random_state)
       
        self.quantiles = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        
        # VARIABLE CLAVE PARA OPTIMIZACIÓN DE TIEMPO
        self._trained_ensemble = None
       
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
   
    def _pin_loss(self, y_true, y_pred):
        """Pinball loss para regresión cuantílica."""
        error = y_true - y_pred
        return tf.reduce_mean(
            tf.maximum(self.quantiles * error, (self.quantiles - 1) * error),
            axis=-1
        )
   
    def _build_lstm(self):
        """Construye modelo LSTM para regresión cuantílica."""
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
        """Crea secuencias para LSTM."""
        X, y = [], []
        for i in range(len(data) - self.n_lags):
            X.append(data[i:(i + self.n_lags)])
            y.append(data[i + self.n_lags])
        return np.array(X), np.array(y)
   
    def _prepare_data(self, series: np.ndarray):
        """Prepara datos en B batches disjuntos para ensemble."""
        # Nota: el scaler se ajusta aquí una vez
        series_scaled = self.scaler.fit_transform(series.reshape(-1, 1))
        X, y = self._create_sequences(series_scaled)
        n_samples = X.shape[0]
       
        if n_samples < self.B * 5:
            raise ValueError(f"Insuficientes muestras ({n_samples}) para {self.B} batches")
       
        batch_size = n_samples // self.B
        batches = []
       
        for b in range(self.B):
            start = b * batch_size
            end = (b + 1) * batch_size if b < self.B - 1 else n_samples
            batches.append({'X': X[start:end], 'y': y[start:end]})
       
        return batches
   
    def _cleanup(self):
        """Limpia memoria de TensorFlow."""
        tf.keras.backend.clear_session()
        gc.collect()
   
    def optimize_hyperparameters(self, df, reference_noise):
        """Optimización Bayesiana rápida."""
        from metricas import ecrps
       
        if not BAYESOPT_AVAILABLE:
            self.best_params = {'n_lags': 20, 'units': 32, 'B': 3}
            self._is_optimized = True
            return self.best_params, -1
       
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
       
        def objective(n_lags, units, B):
            self._cleanup()
           
            try:
                self.n_lags = max(10, int(n_lags))
                self.units = max(16, int(units))
                self.B = max(2, int(B))
               
                if len(series) <= self.n_lags * self.B:
                    return -1e10
               
                # Limpiar modelo guardado para forzar re-entrenamiento en optimización
                saved_ensemble = self._trained_ensemble
                self._trained_ensemble = None
                
                old_epochs = self.epochs
                self.epochs = 10 # Menos epochs para optimizar
               
                samples = self.fit_predict(series)
               
                self.epochs = old_epochs
                self._trained_ensemble = saved_ensemble # Restaurar
               
                if samples is None or len(samples) == 0:
                    return -1e10
               
                ref_sub = self.rng.choice(reference_noise,
                                         size=min(500, len(reference_noise)),
                                         replace=False)
                return -ecrps(samples, ref_sub)
               
            except:
                return -1e10
            finally:
                self._cleanup()
       
        optimizer = BayesianOptimization(
            f=objective,
            pbounds={'n_lags': (10, 30), 'units': (16, 48), 'B': (2, 4)},
            random_state=self.random_state,
            verbose=0
        )
       
        try:
            optimizer.maximize(init_points=2, n_iter=2)
        except:
            pass
       
        best_ecrps = -1
        if optimizer.max and optimizer.max['target'] > -1e9:
            self.best_params = {k: int(v) for k, v in optimizer.max['params'].items()}
            best_ecrps = -optimizer.max['target']
        else:
            self.best_params = {'n_lags': 20, 'units': 32, 'B': 3}
       
        self._is_optimized = True
        self._cleanup()
        return self.best_params, best_ecrps
   
    def fit_predict(self, df) -> np.ndarray:
        """
        Ajusta y retorna muestras.
        Si _trained_ensemble ya existe, usa las redes ya entrenadas.
        """
        # Limpiar sesión previa para evitar fugas de memoria, 
        # pero CUIDADO: no borrar los modelos que queremos reutilizar.
        # tf.keras.backend.clear_session() borra el grafo global.
        # Como guardamos referencias a los modelos en self._trained_ensemble, 
        # deberíamos estar bien, pero por seguridad solo limpiamos si vamos a re-entrenar.
        
        try:
            series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
           
            if self.best_params:
                self.n_lags = self.best_params.get('n_lags', self.n_lags)
                self.units = self.best_params.get('units', self.units)
                self.B = self.best_params.get('B', self.B)
            
            # ================================================================
            # FASE 1: ENTRENAMIENTO (Solo primera vez)
            # ================================================================
            if self._trained_ensemble is None:
                self._cleanup() # Ahora sí es seguro limpiar
                
                if len(series) <= self.n_lags + self.B * 5:
                    return np.full(self.num_samples, np.mean(series))
                
                try:
                    # Esto ajusta el scaler internamente
                    batches = self._prepare_data(series)
                except ValueError:
                    return np.full(self.num_samples, np.mean(series))
                
                ensemble_models = []
                loo_preds = [[] for _ in range(self.B)]
            
                # Entrenar B modelos
                for b in range(self.B):
                    model = self._build_lstm()
                    model.fit(
                        batches[b]['X'], batches[b]['y'],
                        epochs=self.epochs, batch_size=self.batch_size,
                        verbose=0, shuffle=False
                    )
                    ensemble_models.append(model)
                    
                    # Predicciones Leave-One-Out para calibración
                    for i in range(self.B):
                        if i != b:
                            preds = model.predict(batches[i]['X'], verbose=0)
                            loo_preds[i].append(preds)
            
                # Calcular scores de conformidad
                conformity_scores = [[] for _ in range(len(self.quantiles))]
                for i in range(self.B):
                    if loo_preds[i]:
                        avg_pred = np.mean(loo_preds[i], axis=0)
                        y = batches[i]['y'].reshape(-1, 1)
                        for q_idx, tau in enumerate(self.quantiles):
                            q = avg_pred[:, q_idx].reshape(-1, 1)
                            if tau <= 0.5:
                                score = q - y
                            else:
                                score = y - q
                            conformity_scores[q_idx].extend(score.flatten())
                
                # Guardar todo
                self._trained_ensemble = {
                    'models': ensemble_models,
                    'scores': [np.array(cs) for cs in conformity_scores],
                    'scaler': self.scaler # Guardar el scaler ajustado
                }
           
            # ================================================================
            # FASE 2: PREDICCIÓN
            # ================================================================
            
            scores_list = self._trained_ensemble['scores']
            if any(len(cs) == 0 for cs in scores_list):
                return np.full(self.num_samples, np.mean(series))
           
            # Usar el scaler guardado para transformar la nueva ventana
            # Nota: fit_transform se hizo en FASE 1. Aquí solo transform.
            current_scaler = self._trained_ensemble['scaler']
            
            last_window_scaled = current_scaler.transform(
                series[-self.n_lags:].reshape(-1, 1)
            ).reshape(1, self.n_lags, 1)
           
            # Predicción promediada del ensemble
            final_preds = [
                model.predict(last_window_scaled, verbose=0)[0]
                for model in self._trained_ensemble['models']
            ]
            agg_q = np.mean(final_preds, axis=0)
           
            # Conformalizar
            conf_q = np.zeros_like(agg_q)
            for q_idx, tau in enumerate(self.quantiles):
                omega = np.quantile(scores_list[q_idx], 1 - self.alpha)
                if tau <= 0.5:
                    conf_q[q_idx] = agg_q[q_idx] - omega
                else:
                    conf_q[q_idx] = agg_q[q_idx] + omega
           
            # Invertir escala
            conf_q = current_scaler.inverse_transform(
                conf_q.reshape(-1, 1)
            ).flatten()
           
            # Monotonicidad y muestreo
            conf_q = np.sort(conf_q)
            uniform_samples = self.rng.uniform(0, 1, self.num_samples)
            samples = np.interp(uniform_samples, self.quantiles, conf_q)
           
            return samples
           
        except Exception as e:
            if self.verbose:
                print(f" EnCQR error: {e}")
            return np.full(self.num_samples, np.nanmean(df) if hasattr(df, '__len__') else 0)
   
    def __del__(self):
        # Solo limpiar al destruir el objeto
        self._cleanup()