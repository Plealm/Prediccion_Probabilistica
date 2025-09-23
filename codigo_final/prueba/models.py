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
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

# --- Configuración para un entorno limpio de TensorFlow ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')


class EnhancedBootstrappingModel:
    """Modelo con Block Bootstrap y evaluación CRPS."""
    def __init__(self, arma_simulator_config: Dict, random_state=42, verbose=False):
        self.n_lags, self.random_state, self.verbose = None, random_state, verbose
        self.rng = np.random.default_rng(self.random_state)
        # Recibe la configuración, no el objeto del simulador.
        # El simulador se instanciará localmente cuando sea necesario.
        self.arma_simulator_config = arma_simulator_config 
        self._local_arma_simulator = None # Para mantener una instancia local si se necesita

        self.train, self.test, self.mean_val, self.std_val = None, None, None, None

    def _get_simulator(self):
        """Instancia o devuelve el simulador ARMA local."""
        if self._local_arma_simulator is None:
            from arma_simulation import ARMASimulation # Importar localmente para evitar problemas de serialización
            self._local_arma_simulator = ARMASimulation(**self.arma_simulator_config)
        return self._local_arma_simulator

    def prepare_data(self, series):
        split_idx = int(len(series)*0.9)
        self.train, self.test = series[:split_idx], series[split_idx:]
        self.mean_val, self.std_val = np.mean(self.train), np.std(self.train)
        return (self.train - self.mean_val) / (self.std_val + 1e-8)

    def denormalize(self, values): return (values * self.std_val) + self.mean_val

    def _block_bootstrap_predict(self, fitted_model, n_boot):
        residuals, n_resid = fitted_model.resid, len(fitted_model.resid)
        if n_resid < 2: 
            # Asegurarse de que forecast devuelva un array para consistencia
            forecast_val = fitted_model.forecast(steps=1)[0] if fitted_model.forecast(steps=1).size > 0 else np.mean(fitted_model.model.endog)
            return np.full(n_boot, forecast_val) + self.rng.choice(residuals, size=n_boot, replace=True)
        block_length = max(1, int(n_resid**(1/3)))
        blocks = [residuals[i:i+block_length] for i in range(n_resid - block_length + 1)]
        if not blocks: 
            forecast_val = fitted_model.forecast(steps=1)[0] if fitted_model.forecast(steps=1).size > 0 else np.mean(fitted_model.model.endog)
            return np.full(n_boot, forecast_val)
        resampled_indices = self.rng.choice(len(blocks), size=(n_boot // block_length) + 1, replace=True)
        bootstrap_error_pool = np.concatenate([blocks[i] for i in resampled_indices])
        return fitted_model.forecast(steps=1) + self.rng.choice(bootstrap_error_pool, size=n_boot, replace=True)

    def fit_predict(self, data, n_boot=1000):
        # En este punto, 'data' ya es la serie actual de la ventana rodante.
        # No se necesita self._get_simulator() para obtener 'series'.
        normalized_train = self.prepare_data(data)
        test_predictions, current_data = [], normalized_train.copy()
        
        # Si n_lags no se ha establecido (ej. si no se hizo grid_search), establecer un valor por defecto
        if self.n_lags is None:
            self.n_lags = max(1, int(len(normalized_train)**(1/3))) 

        for _ in range(len(self.test)):
            if len(current_data) <= self.n_lags + 1: # Reducido el umbral para permitir más predicciones
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
        return test_predictions # Devuelve la lista de arrays de muestras

    def grid_search(self, series: np.ndarray, lags_range=range(1, 13), n_boot=500):
        """
        Realiza la búsqueda de la mejor p_order para Block Bootstrapping.
        Recibe la serie directamente en lugar de usar self.arma_simulator.series.
        """
        best_crps, best_lag = float('inf'), 1
        normalized_train = self.prepare_data(series) # Usa la serie pasada, no del simulador
        
        for n_lags in lags_range:
            self.n_lags = n_lags
            if len(normalized_train) <= n_lags + 10: continue
            
            crps_values = []
            try:
                # Usa una ventana rodante más pequeña para la búsqueda de hiperparámetros
                # para que sea más rápido. Por ejemplo, últimos 50 puntos para validación.
                validation_start_idx = max(n_lags + 10, len(normalized_train) - 50) 

                for t in range(validation_start_idx, len(normalized_train)):
                    train_subset = normalized_train[:t]
                    if len(train_subset) <= 2 * n_lags + 1: continue
                    try: fitted_model = AutoReg(train_subset, lags=n_lags, old_names=False).fit()
                    except (np.linalg.LinAlgError, ValueError): continue
                    
                    boot_preds = self._block_bootstrap_predict(fitted_model, n_boot)
                    
                    # Llamada a ecrps para una única observación (valor real)
                    # Aquí usamos la función crps directamente
                    from metrics import crps # Importar localmente
                    crps_values.append(crps(boot_preds, normalized_train[t]))
                
                if crps_values and (avg_crps := np.nanmean(crps_values)) < best_crps:
                    best_crps, best_lag = avg_crps, n_lags
            except Exception: # Captura cualquier error que pueda ocurrir durante el fitting
                continue
        
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
        
        h_ii = np.diag(H_bar)
        h_n_vec, h_in_vec, h_n = H_bar[-1, :-1], H_bar[:-1, -1], h_ii[-1]
        
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
    """
    def __init__(self, rho: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        if not (0 < rho < 1):
            raise ValueError("El factor de decaimiento 'rho' debe estar entre 0 y 1.")
        self.rho = rho
        self.best_params = {'rho': rho}

    def fit_predict(self, df: Union[pd.DataFrame, np.ndarray]) -> List[Dict[str, float]]:
        critical_values = self._calculate_critical_values(df)
        
        if not critical_values:
            mean_pred = np.mean(df['valor'].values if isinstance(df, pd.DataFrame) else df)
            return [{'value': mean_pred, 'probability': 1.0}]
            
        n_crit = len(critical_values)
        weights = self.rho ** np.arange(n_crit - 1, -1, -1)
  
        dist_df = pd.DataFrame({
            'value': critical_values,
            'weight': weights
        })

        weighted_dist = dist_df.groupby('value')['weight'].sum()
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
                from metrics import ecrps # Importar localmente
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
        
        # Usar np.random.default_rng para el sampling y tf.random.set_seed para Keras
        self.rng = np.random.default_rng(random_state)
        tf.random.set_seed(random_state) 

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
                predictions = (self.rng.normal(mu, sigma, self.num_samples) * self.scaler_std + self.scaler_mean) # Usar self.rng
                from metrics import ecrps # Importar localmente
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
            return np.nan_to_num((self.rng.normal(mu.item(), sigma.item(), self.num_samples) * self.scaler_std + self.scaler_mean)) # Usar self.rng
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
                
                from metrics import ecrps # Importar localmente
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
        self.rng = check_random_state(random_state) # Usa sklearn.utils.check_random_state
        
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
                from metrics import ecrps # Importar localmente
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

class MondrianCPSModel:
    """
    Implementa el Mondrian Conformal Predictive System (MCPS).
    """
    def __init__(self, n_lags: int = 10, n_bins: int = 10, test_size: float = 0.25, 
                 random_state: int = 42, verbose: bool = False):
        
        self.n_lags = n_lags
        self.n_bins = n_bins
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state) # Usar self.rng para el sampling
        
        self.base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=self.random_state,
            n_jobs=-1 # Usar todos los cores disponibles
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
                # Usar un número menor de muestras para la optimización si es posible,
                # pero fit_predict ya tiene un num_samples por defecto
                predictions = self.fit_predict(df) 
                if predictions is None or len(predictions) == 0: return -1e10
                
                from metrics import ecrps # Importar localmente
                return -ecrps(predictions, reference_noise)
            except Exception:
                return -1e10

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={'n_bins': (3, 20.99)},
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
        
        if self.best_params:
            self.__dict__.update(self.best_params)

        if len(series) < self.n_lags * 2:
            if self.verbose: print("⚠️ MCPS: No hay suficientes datos para el proceso.")
            return np.full(5000, np.mean(series))

        X, y = self._create_lag_matrix(series)
        
        x_test = series[-self.n_lags:].reshape(1, -1)

        n_calib = int(len(X) * self.test_size)
        if n_calib < self.n_bins * 2 and self.n_bins > 1: # Reducir el número de bins si no hay suficientes datos
            if self.verbose: print(f"⚠️ MCPS: Pocos datos de calibración ({n_calib}). Reduciendo n_bins.")
            self.n_bins = max(1, n_calib // 2) # Asegurarse de tener al menos 1 bin
            if self.n_bins < 1: self.n_bins = 1

        X_train, X_calib = X[:-n_calib], X[-n_calib:]
        y_train, y_calib = y[:-n_calib], y[-n_calib:]

        if len(X_train) == 0 or len(X_calib) == 0:
            if self.verbose: print("⚠️ MCPS: Conjuntos de train o calibración vacíos.")
            return np.full(5000, np.mean(series))

        self.base_model.fit(X_train, y_train)

        calib_preds = self.base_model.predict(X_calib)
        calib_residuals = y_calib - calib_preds

        if len(calib_preds) > 1 and self.n_bins > 1:
            try:
                bin_indices = pd.qcut(calib_preds, self.n_bins, labels=False, duplicates='drop')
                _, bin_edges = pd.qcut(calib_preds, self.n_bins, retbins=True, duplicates='drop')
                # Si 'duplicates' causa menos bins de los esperados, ajustamos n_bins
                if len(bin_edges) - 1 < self.n_bins:
                     if self.verbose: print(f"⚠️ MCPS: Se crearon menos bins ({len(bin_edges)-1}) de los solicitados ({self.n_bins}). Ajustando.")
                     self.n_bins = len(bin_edges) - 1
                     if self.n_bins < 1: self.n_bins = 1 # Fallback a 1 bin si no se puede crear ninguno
            except Exception: # Fallback a 1 bin si qcut falla
                 if self.verbose: print("⚠️ MCPS: Fallo en qcut. Usando un solo bin global.")
                 bin_indices = np.zeros(len(calib_preds), dtype=int)
                 bin_edges = [-np.inf, np.inf]
        else: # Si hay pocos puntos o solo 1 bin
            bin_indices = np.zeros(len(calib_preds), dtype=int)
            bin_edges = [-np.inf, np.inf]

        point_prediction = self.base_model.predict(x_test)[0]
        
        # Encontrar el bin correspondiente para la predicción de test
        if self.n_bins > 0 and len(bin_edges) > 1:
            test_bin = np.digitize(point_prediction, bins=bin_edges) - 1
            test_bin = np.clip(test_bin, 0, self.n_bins - 1)
        else: # Si no hay bins válidos, usar el bin 0 (global)
            test_bin = 0

        local_residuals = calib_residuals[bin_indices == test_bin]
        
        if len(local_residuals) < 10: 
            if self.verbose: print(f"⚠️ MCPS: El bin {test_bin} tiene solo {len(local_residuals)} puntos. Usando calibración global.")
            local_residuals = calib_residuals

        if len(local_residuals) == 0:
            if self.verbose: print("⚠️ MCPS: No hay residuos disponibles para muestreo. Retornando la media.")
            return np.full(5000, point_prediction)

        final_samples = point_prediction + self.rng.choice(local_residuals, size=5000, replace=True)
        
        return final_samples

class EnCQR_LSTM_Model:
    """
    Implementa el método Ensemble Conformalized Quantile Regression (EnCQR)
    utilizando un predictor LSTM interno.
    """
    def __init__(self, n_lags: int = 24, B: int = 3, units: int = 50, n_layers: int = 2,
                 lr: float = 0.005, batch_size: int = 16, epochs: int = 25, 
                 num_samples: int = 5000, random_state: int = 42, verbose: bool = False):
        
        self.B = B
        self.n_lags = n_lags
        self.units = units
        self.n_layers = n_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.num_samples = num_samples
        self.random_state = random_state
        self.verbose = verbose
        
        self.scaler = MinMaxScaler()
        self.best_params = {}
        self.rng = np.random.default_rng(random_state) # Usar self.rng
        
        self.quantiles = np.round(np.arange(0.05, 1.0, 0.05), 2)
        self.median_idx = np.where(self.quantiles == 0.5)[0][0]
        
        tf.random.set_seed(random_state) # Establecer semilla para TensorFlow

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
            start, end = b * batch_size, (b + 1) * batch_size
            if b == self.B - 1: end = n_samples
            train_data_batches.append({'X': X[start:end], 'y': y[start:end]})
            
        return train_data_batches

    def optimize_hyperparameters(self, df: pd.DataFrame, reference_noise: np.ndarray):
        """Optimiza hiperparámetros clave usando optimización bayesiana."""
        series = df['valor'].values

        def objective(n_lags, units, B):
            try:
                self.n_lags, self.units, self.B = max(10, int(n_lags)), max(16, int(units)), max(2, int(B))
                if len(series) <= self.n_lags * self.B: return -1e10
                
                # Para la optimización, se puede reducir el num_samples para fit_predict
                # temporalmente para acelerar. O simplemente dejar el valor por defecto.
                original_num_samples = self.num_samples
                self.num_samples = 1000 # Usar menos muestras para la optimización
                predictions = self.fit_predict(series)
                self.num_samples = original_num_samples # Restaurar
                
                if predictions is None or len(predictions) == 0: return -1e10
                
                from metrics import ecrps # Importar localmente
                return -ecrps(predictions, reference_noise)
            except Exception:
                return -1e10

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={'n_lags': (10, 50), 'units': (20, 80), 'B': (2, 5.99)},
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
        series = df['valor'].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        if self.best_params: self.__dict__.update(self.best_params)

        if len(series) <= self.n_lags + self.B:
            if self.verbose: print("⚠️ EnCQR-LSTM: No hay suficientes datos para el proceso.")
            return np.full(self.num_samples, np.mean(series))

        try:
            train_batches = self._prepare_data(series)
        except ValueError as e:
            if self.verbose: print(f"⚠️ EnCQR-LSTM: Error en preparación de datos - {e}")
            return np.full(self.num_samples, np.mean(series))

        ensemble_models, loo_preds_median = [], [[] for _ in range(self.B)]
        for b in range(self.B):
            model = self._build_lstm() # Construir el modelo Keras dentro del bucle
            
            # Asegurar que haya suficientes datos en el lote para entrenar
            if train_batches[b]['X'].shape[0] < self.batch_size:
                 if self.verbose: print(f"⚠️ EnCQR-LSTM: Lote {b} tiene menos muestras ({train_batches[b]['X'].shape[0]}) que batch_size ({self.batch_size}). Saltando entrenamiento.")
                 ensemble_models.append(None) # Añadir None para mantener la longitud del ensamble
                 continue # Saltar este modelo del ensamble
                 
            model.fit(train_batches[b]['X'], train_batches[b]['y'], 
                      epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            ensemble_models.append(model)
            
            for i in range(self.B):
                if i != b and ensemble_models[b] is not None: # Solo predecir si el modelo fue entrenado
                    preds = model.predict(train_batches[i]['X'], verbose=0)
                    loo_preds_median[i].append(preds[:, self.median_idx])

        # Filtrar None de ensemble_models si algunos no fueron entrenados
        ensemble_models = [model for model in ensemble_models if model is not None]
        if not ensemble_models:
             if self.verbose: print("⚠️ EnCQR-LSTM: Ningún modelo del ensamble pudo ser entrenado.")
             return np.full(self.num_samples, np.mean(series))

        conformity_scores = []
        for i in range(self.B):
            if loo_preds_median[i]: # Si hay predicciones LOO para este lote
                avg_loo_pred = np.mean(loo_preds_median[i], axis=0).reshape(-1, 1)
                true_y = train_batches[i]['y']
                scores = true_y - avg_loo_pred
                conformity_scores.extend(scores.flatten())
        
        conformity_scores = np.array(conformity_scores)
        if len(conformity_scores) == 0:
            if self.verbose: print("⚠️ EnCQR-LSTM: No se pudieron calcular conformity scores. Retornando la media.")
            return np.full(self.num_samples, np.mean(series))

        last_window_unscaled = series[-self.n_lags:]
        last_window_scaled = self.scaler.transform(last_window_unscaled.reshape(-1, 1)).reshape(1, self.n_lags, 1)

        final_preds_median = []
        for model in ensemble_models:
            pred = model.predict(last_window_scaled, verbose=0)
            final_preds_median.append(pred[0, self.median_idx])
        
        point_prediction_scaled = np.mean(final_preds_median)
        
        predictive_dist_scaled = point_prediction_scaled + self.rng.choice(
            conformity_scores, size=self.num_samples, replace=True
        )
        
        final_samples = self.scaler.inverse_transform(predictive_dist_scaled.reshape(-1, 1)).flatten()

        return final_samples