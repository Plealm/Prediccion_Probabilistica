import numpy as np
import pandas as pd
import warnings
import gc
import os
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Union, Any

warnings.filterwarnings("ignore")

from simulacion import ARMASimulation
from modelos import (CircularBlockBootstrapModel, SieveBootstrapModel, LSPM, LSPMW, 
                     DeepARModel, AREPD, MondrianCPSModel, AdaptiveVolatilityMondrianCPS,
                     EnCQR_LSTM_Model, TimeBalancedOptimizer)
from metricas import crps, ecrps
from simulacion import ARIMASimulation, SETARSimulation, ARMASimulation
from figuras import PlotManager

def clear_all_sessions():
    """Limpia memoria de forma agresiva."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except:
        pass


class Pipeline140SinSesgos_ARMA:
    """
    Pipeline para ARMA sin sesgos temporales.
    Comparación de Densidad Predictiva vs Densidad Teórica mediante ECRPS.
    """
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]},
        {'nombre': 'ARMA(2,1)', 'phi': [0.7, 0.2], 'theta': [0.5]}
    ]
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def generate_all_scenarios(self) -> list:
        scenarios = []
        # Generar combinaciones basadas en los atributos actuales de la instancia
        for i, arma in enumerate(self.ARMA_CONFIGS):
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((arma.copy(), dist, var, self.seed + i))
        return scenarios

    def _run_scenario_wrapper(self, args):
        return self.run_single_scenario(*args)

    def run_single_scenario(self, arma_cfg, dist, var, scenario_seed):
        simulator = ARMASimulation(
            phi=arma_cfg['phi'], theta=arma_cfg['theta'],
            noise_dist=dist, sigma=np.sqrt(var), seed=scenario_seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_data = series[:self.N_TRAIN]
        val_data = series[self.N_TRAIN : self.N_TRAIN + self.N_VALIDATION]
        
        models = {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=scenario_seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=scenario_seed),
            'LSPM': LSPM(random_state=scenario_seed),
            'LSPMW': LSPMW(rho=0.95, random_state=scenario_seed),
            'AREPD': AREPD(n_lags=5, rho=0.93, random_state=scenario_seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=scenario_seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=12, random_state=scenario_seed),
            'DeepAR': DeepARModel(hidden_size=20, n_lags=10, epochs=25, num_samples=self.n_boot, random_state=scenario_seed),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=15, B=3, units=24, epochs=20, num_samples=self.n_boot, random_state=scenario_seed)
        }

        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        train_val_full = series[:self.N_TRAIN + self.N_VALIDATION]
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)

        results_rows = []
        plot_data = {}

        for t in range(self.N_TEST_STEPS):
            idx = self.N_TRAIN + self.N_VALIDATION + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
            plot_data[t] = {'true_distribution': true_samples, 'model_predictions': {}}
            # Usar 'Paso' para compatibilidad con run_analysis
            row = {'Paso': t + 1, 'Config': arma_cfg['nombre'], 'Dist': dist, 'Var': var}
            
            for name, model in models.items():
                try:
                    if "Bootstrap" in name: pred = model.fit_predict(h_series)
                    else: pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    pred_array = np.asarray(pred).flatten()
                    plot_data[t]['model_predictions'][name] = pred_array
                    row[name] = ecrps(pred_array, true_samples)
                except:
                    row[name] = np.nan
            results_rows.append(row)

        scen_name = f"{arma_cfg['nombre']}_{dist}_V{var}_S{scenario_seed}"
        df_res = pd.DataFrame(results_rows)
        for m_name in models.keys():
            path = f"reportes/{scen_name}/{m_name.replace(' ', '_')}.png"
            PlotManager.plot_individual_model_evolution(scen_name, m_name, plot_data, df_res, path)

        clear_all_sessions()
        return results_rows

    # CORRECCIÓN: run_all ahora acepta los argumentos del wrapper
    def run_all(self, excel_filename="resultados.xlsx", batch_size=10, n_jobs=2):
        tasks = self.generate_all_scenarios()
        print(f"🚀 Ejecutando {len(tasks)} escenarios en lotes de {batch_size}...")
        
        all_results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            print(f"  -> Procesando lote {i//batch_size + 1}...")
            results = Parallel(n_jobs=n_jobs)(delayed(self._run_scenario_wrapper)(t) for t in batch)
            for r in results:
                all_results.extend(r)
            
            # Guardado intermedio por seguridad
            temp_df = pd.DataFrame(all_results)
            temp_df.to_excel(excel_filename, index=False)

        final_df = pd.DataFrame(all_results)
        return final_df

class Pipeline140SinSesgos_ARIMA:
    """
    Pipeline ARIMA:
    1) Comparación de Densidad Predictiva vs Densidad Teórica mediante ECRPS.
    2) Generación de 9 gráficos (12 pasos) por escenario con colores únicos.
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    SCENARIO_BUDGET = 300  
    OPTIMIZATION_BUDGET = 120  
    FREEZE_BUDGET = 30         
    TEST_BUDGET = 150          

    ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def generate_all_scenarios(self) -> list:
        scenarios = []
        scenario_id = 0
        for arima_cfg in self.ARIMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((arima_cfg.copy(), dist, var, self.seed + scenario_id))
                    scenario_id += 1
        return scenarios

    def _run_scenario_wrapper(self, args):
        return self.run_single_scenario(*args)

    def run_single_scenario(self, arima_config, dist, var, scenario_seed):
        scenario_start = time.time()
        
        # 1. Simulación ARIMA (Serie Integrada + Errores subyacentes)
        simulator = ARIMASimulation(
            phi=arima_config['phi'], theta=arima_config['theta'],
            noise_dist=dist, sigma=np.sqrt(var), seed=scenario_seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_data = series[:self.N_TRAIN]
        val_data = series[self.N_TRAIN : self.N_TRAIN + self.N_VALIDATION]
        
        # 2. Configuración de Modelos Ligeros
        models = self._setup_models(scenario_seed)

        # 3. Optimización y Congelamiento (Budget: 150s)
        prep_start = time.time()
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        train_val_full = series[:self.N_TRAIN + self.N_VALIDATION]
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)
        prep_elapsed = time.time() - prep_start

        # 4. Testing Rolling Window con ECRPS (Budget: 150s)
        test_start = time.time()
        results_rows = []
        plot_data = {}
        time_per_step = self.TEST_BUDGET / self.N_TEST_STEPS

        for t in range(self.N_TEST_STEPS):
            step_start = time.time()
            idx = self.N_TRAIN + self.N_VALIDATION + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            # Obtener Ground Truth Teórico (Dist. de Y_{n+1})
            true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
            plot_data[t] = {'true_distribution': true_samples, 'model_predictions': {}}
            
            row = {'Paso': t + 1, 'Config': arima_config['nombre'], 'Dist': dist, 'Var': var}
            
            for name, model in models.items():
                try:
                    # Check timeout por paso
                    if (time.time() - step_start) > time_per_step and t > 0:
                        row[name] = np.nan
                        continue

                    # Inferencia
                    if "Bootstrap" in name: pred = model.fit_predict(h_series)
                    else: pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    
                    pred_array = np.asarray(pred).flatten()
                    plot_data[t]['model_predictions'][name] = pred_array
                    
                    # Métrica ECRPS (Densidad vs Densidad)
                    row[name] = ecrps(pred_array, true_samples)
                except:
                    row[name] = np.nan
            
            results_rows.append(row)
        
        test_elapsed = time.time() - test_start

        # 5. Generación de Gráficos (9 imágenes por modelo)
        scen_id = f"{arima_config['nombre']}_{dist}_V{var}_S{scenario_seed}"
        df_res = pd.DataFrame(results_rows)
        
        for m_name in models.keys():
            path = f"reportes_arima/{scen_id}/{m_name.replace(' ', '_')}.png"
            PlotManager.plot_individual_model_evolution(scen_id, m_name, plot_data, df_res, path)

        total_elapsed = time.time() - scenario_start
        if self.verbose:
            print(f"✅ Escenario {arima_config['nombre']} fin en {total_elapsed:.1f}s")

        clear_all_sessions()
        return results_rows

    def _setup_models(self, seed):
        """Versión optimizada para velocidad."""
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'AREPD': AREPD(n_lags=5, rho=0.93, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=12, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=16, n_lags=5, epochs=20, num_samples=self.n_boot, 
                                  random_state=seed, early_stopping_patience=3),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=15, B=3, units=24, epochs=15, num_samples=self.n_boot, 
                                          random_state=seed)
        }

    def run_all(self, excel_filename="resultados_arima_ecrps.xlsx", batch_size=10, max_workers=4):
        tasks = self.generate_all_scenarios()
        print(f"🚀 Iniciando Pipeline ARIMA: {len(tasks)} escenarios.")
        
        all_results = []
        num_batches = (len(tasks) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"📦 Procesando Lote {i+1}/{num_batches}...")
            results = Parallel(n_jobs=max_workers, backend='loky')(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for r in results: all_results.extend(r)
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)
            clear_all_sessions()
            gc.collect()

        return pd.DataFrame(all_results)

class Pipeline140SinSesgos_SETAR:
    """
    Pipeline SETAR:
    1) Comparación de Densidad Predictiva vs Densidad Teórica mediante ECRPS.
    2) Generación de 9 gráficos (12 niveles) por escenario con colores únicos.
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    SCENARIO_BUDGET = 300  
    OPTIMIZATION_BUDGET = 120  
    FREEZE_BUDGET = 30         
    TEST_BUDGET = 150          

    SETAR_CONFIGS = [
        {'nombre': 'SETAR-1', 'phi_regime1': [0.6], 'phi_regime2': [-0.5], 'threshold': 0.0, 'delay': 1, 'description': 'SETAR(2;1,1) d=1, r=0'},
        {'nombre': 'SETAR-2', 'phi_regime1': [0.7], 'phi_regime2': [-0.7], 'threshold': 0.0, 'delay': 2, 'description': 'SETAR(2;1,1) d=2, r=0'},
        {'nombre': 'SETAR-3', 'phi_regime1': [0.5, -0.2], 'phi_regime2': [-0.3, 0.1], 'threshold': 0.5, 'delay': 1, 'description': 'SETAR(2;2,2) d=1, r=0.5'},
        {'nombre': 'SETAR-4', 'phi_regime1': [0.8, -0.15], 'phi_regime2': [-0.6, 0.2], 'threshold': 1.0, 'delay': 2, 'description': 'SETAR(2;2,2) d=2, r=1.0'},
        {'nombre': 'SETAR-5', 'phi_regime1': [0.4, -0.1, 0.05], 'phi_regime2': [-0.3, 0.1, -0.05], 'threshold': 0.0, 'delay': 1, 'description': 'SETAR(2;3,3) d=1, r=0'},
        {'nombre': 'SETAR-6', 'phi_regime1': [0.5, -0.3, 0.1], 'phi_regime2': [-0.4, 0.2, -0.05], 'threshold': 0.5, 'delay': 2, 'description': 'SETAR(2;3,3) d=2, r=0.5'},
        {'nombre': 'SETAR-7', 'phi_regime1': [0.3, 0.1], 'phi_regime2': [-0.2, -0.1], 'threshold': 0.8, 'delay': 3, 'description': 'SETAR(2;2,2) d=3, r=0.8'}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def generate_all_scenarios(self) -> list:
        scenarios = []
        scenario_id = 0
        for setar_cfg in self.SETAR_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((setar_cfg.copy(), dist, var, self.seed + scenario_id))
                    scenario_id += 1
        return scenarios

    def _run_scenario_wrapper(self, args):
        return self.run_single_scenario(*args)

    def run_single_scenario(self, setar_config, dist, var, scenario_seed):
        scenario_start = time.time()
        
        # 1. Simulación SETAR
        simulator = SETARSimulation(
            model_type=setar_config['nombre'],
            phi_regime1=setar_config['phi_regime1'],
            phi_regime2=setar_config['phi_regime2'],
            threshold=setar_config['threshold'],
            delay=setar_config['delay'],
            noise_dist=dist,
            sigma=np.sqrt(var),
            seed=scenario_seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_data = series[:self.N_TRAIN]
        val_data = series[self.N_TRAIN : self.N_TRAIN + self.N_VALIDATION]
        
        # 2. Configuración de Modelos
        models = self._setup_models(scenario_seed)

        # 3. Optimización y Congelamiento
        prep_start = time.time()
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        train_val_full = series[:self.N_TRAIN + self.N_VALIDATION]
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)
        prep_elapsed = time.time() - prep_start

        # 4. Testing Rolling Window con ECRPS
        test_start = time.time()
        results_rows = []
        plot_data = {}
        time_per_step = self.TEST_BUDGET / self.N_TEST_STEPS

        for t in range(self.N_TEST_STEPS):
            step_start = time.time()
            idx = self.N_TRAIN + self.N_VALIDATION + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            # Obtener Ground Truth Teórico (Depende del régimen actual en SETAR)
            true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
            plot_data[t] = {'true_distribution': true_samples, 'model_predictions': {}}
            
            # Columna 'Paso' para compatibilidad con run_analysis
            row = {
                'Paso': t + 1, 
                'Config': setar_config['nombre'], 
                'Descripción': setar_config['description'],
                'Dist': dist, 
                'Var': var
            }
            
            for name, model in models.items():
                try:
                    if (time.time() - step_start) > time_per_step and t > 0:
                        row[name] = np.nan
                        continue

                    if "Bootstrap" in name: pred = model.fit_predict(h_series)
                    else: pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    
                    pred_array = np.asarray(pred).flatten()
                    plot_data[t]['model_predictions'][name] = pred_array
                    
                    # ECRPS: Comparación de densidades
                    row[name] = ecrps(pred_array, true_samples)
                except:
                    row[name] = np.nan
            
            results_rows.append(row)
        
        test_elapsed = time.time() - test_start

        # 5. Generación de Gráficos (9 imágenes por modelo)
        scen_id = f"{setar_config['nombre']}_{dist}_V{var}_S{scenario_seed}"
        df_res = pd.DataFrame(results_rows)
        
        for m_name in models.keys():
            path = f"reportes_setar/{scen_id}/{m_name.replace(' ', '_')}.png"
            PlotManager.plot_individual_model_evolution(scen_id, m_name, plot_data, df_res, path)

        total_elapsed = time.time() - scenario_start
        if self.verbose:
            print(f"✅ Escenario SETAR {setar_config['nombre']} fin en {total_elapsed:.1f}s")

        clear_all_sessions()
        return results_rows

    def _setup_models(self, seed):
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'AREPD': AREPD(n_lags=5, rho=0.93, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=12, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=16, n_lags=5, epochs=20, num_samples=self.n_boot, 
                                  random_state=seed, early_stopping_patience=3),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=15, B=3, units=24, epochs=15, num_samples=self.n_boot, 
                                          random_state=seed)
        }

    def run_all(self, excel_filename="resultados_setar_ecrps.xlsx", batch_size=10, max_workers=4):
        tasks = self.generate_all_scenarios()
        print(f"🚀 Iniciando Pipeline SETAR: {len(tasks)} escenarios.")
        
        all_results = []
        num_batches = (len(tasks) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"📦 Procesando Lote {i+1}/{num_batches}...")
            results = Parallel(n_jobs=max_workers, backend='loky')(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for r in results: all_results.extend(r)
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)
            clear_all_sessions()
            gc.collect()

        return pd.DataFrame(all_results)

# ============================================================
# ¿ Es importante diferenciar e integrar en ARIMA?
# ===========================================================

class TransformadorDiferenciacionIntegracion:
    """
    Maneja la transformación diferenciación ↔ integración.
    
    Para ARIMA(p,d,q):
    - d=1: ΔY_t = Y_t - Y_{t-1}
    - Integración: Y_t = Y_{t-1} + ΔY_t
    """
    
    def __init__(self, d: int = 1, verbose: bool = False):
        """
        Args:
            d: Orden de diferenciación (típicamente 1)
            verbose: Mostrar información
        """
        if d not in [1, 2]:
            raise ValueError("Solo se soporta d=1 o d=2")
        self.d = d
        self.verbose = verbose
    
    def diferenciar_serie(self, serie: np.ndarray) -> np.ndarray:
        """
        Aplica diferenciación de orden d.
        
        Args:
            serie: Serie original Y_t
        
        Returns:
            Serie diferenciada ΔY_t
        """
        if self.d == 1:
            # Primera diferencia: ΔY_t = Y_t - Y_{t-1}
            serie_diff = np.diff(serie)
        elif self.d == 2:
            # Segunda diferencia: Δ²Y_t = ΔY_t - ΔY_{t-1}
            serie_diff = np.diff(np.diff(serie))
        else:
            serie_diff = serie
        
        if self.verbose:
            print(f"  Diferenciación d={self.d}: {len(serie)} → {len(serie_diff)} puntos")
        
        return serie_diff
    
    def integrar_predicciones(self, predicciones_diff: np.ndarray,
                              ultimo_valor_observado: float) -> np.ndarray:
        """
        Integra predicciones desde espacio diferenciado.
        
        Para d=1:
            Y_{t+1} = Y_t + ΔY_{t+1}
        
        Args:
            predicciones_diff: Muestras de ΔY_{t+1}
            ultimo_valor_observado: Y_t (último valor conocido)
        
        Returns:
            Muestras de Y_{t+1}
        """
        if self.d == 1:
            # Y_{t+1} = Y_t + ΔY_{t+1}
            predicciones_integradas = ultimo_valor_observado + predicciones_diff
        elif self.d == 2:
            # Para d=2 se necesitaría también Y_{t-1}, por simplicidad solo d=1
            raise NotImplementedError("Integración para d=2 no implementada")
        else:
            predicciones_integradas = predicciones_diff
        
        if self.verbose:
            print(f"  Integración: Y_t={ultimo_valor_observado:.4f}, "
                  f"ΔY_t ∈ [{np.min(predicciones_diff):.4f}, {np.max(predicciones_diff):.4f}] → "
                  f"Y_{{t+1}} ∈ [{np.min(predicciones_integradas):.4f}, {np.max(predicciones_integradas):.4f}]")
        
        return predicciones_integradas


class Pipeline140SinSesgos_ARIMA_ConDiferenciacion:
    """
    Pipeline ARIMA optimizado para generar el formato de Excel solicitado.
    Evalúa cada escenario en dos modalidades: SIN_DIFF y CON_DIFF.
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    # Configuraciones ARIMA (d=1 por defecto para estos nombres)
    ARIMA_CONFIGS = [
        {'nombre': 'RW', 'phi': [], 'theta': []}, # Random Walk es ARIMA(0,1,0)
        {'nombre': 'AR(1)', 'phi': [0.6], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def _setup_models(self, seed: int):
        return {
            'AREPD': AREPD(n_lags=5, rho=0.9, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=15, random_state=seed),
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=15, n_lags=5, epochs=25, num_samples=self.n_boot, random_state=seed),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=20, B=3, units=24, epochs=15, num_samples=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed)
        }

    def _run_modalidad(self, simulator, series_levels, errors, arima_config, dist, var, modalidad, scenario_seed):
        """Ejecuta una modalidad específica (SIN_DIFF o CON_DIFF)"""
        
        # 1. Preparar datos según modalidad
        if modalidad == "CON_DIFF":
            # Los modelos ven incrementos ΔY_t
            series_to_models = np.diff(series_levels, prepend=series_levels[0])
        else:
            # Los modelos ven niveles Y_t
            series_to_models = series_levels

        train_data = series_to_models[:self.N_TRAIN]
        val_data = series_to_models[self.N_TRAIN : self.N_TRAIN + self.N_VALIDATION]
        
        # 2. Setup y Optimización
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=False)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        train_val_full = series_to_models[:self.N_TRAIN + self.N_VALIDATION]
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)

        # 3. Testing Rolling Window
        results_rows = []
        p, q = len(arima_config['phi']), len(arima_config['theta'])
        d = 1 # Por defecto en esta simulación

        for t in range(self.N_TEST_STEPS):
            idx = self.N_TRAIN + self.N_VALIDATION + t
            h_series_levels = series_levels[:idx]
            h_errors = errors[:idx]
            h_to_model = series_to_models[:idx]
            
            # Densidad Teórica (Siguiente paso real)
            true_samples = simulator.get_true_next_step_samples(h_series_levels, h_errors, n_samples=1000)
            
            # Fila base con el formato de la imagen
            row = {
                'Paso': t + 1,
                'Proceso': f"ARMA_I({p},{d},{q})",
                'p': p,
                'd': d,
                'q': q,
                'ARMA_base': arima_config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'Modalidad': modalidad,
                'Valor_Observado': series_levels[idx] # El valor real que ocurrió
            }
            
            for name, model in models.items():
                try:
                    if "Bootstrap" in name: pred = model.fit_predict(h_to_model)
                    else: pred = model.fit_predict(pd.DataFrame({'valor': h_to_model}))
                    
                    pred_array = np.asarray(pred).flatten()
                    
                    # Si predijo incremento, sumar al último nivel para comparar densidades en niveles
                    if modalidad == "CON_DIFF":
                        pred_array = series_levels[idx-1] + pred_array
                    
                    row[name] = ecrps(pred_array, true_samples)
                except:
                    row[name] = np.nan
            
            results_rows.append(row)
            
        return results_rows

    def _run_scenario_wrapper(self, args):
        arima_cfg, dist, var, seed = args
        
        # Simulación (Niveles)
        simulator = ARIMASimulation(
            phi=arima_cfg['phi'], theta=arima_cfg['theta'],
            noise_dist=dist, sigma=np.sqrt(var), seed=seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        series_levels, errors = simulator.simulate(n=total_len, burn_in=100)
        
        # Ejecutar ambas modalidades
        res_sin = self._run_modalidad(simulator, series_levels, errors, arima_cfg, dist, var, "SIN_DIFF", seed)
        res_con = self._run_modalidad(simulator, series_levels, errors, arima_cfg, dist, var, "CON_DIFF", seed + 1)
        
        clear_all_sessions()
        return res_sin + res_con

    def generate_all_scenarios(self) -> list:
        scenarios = []
        s_id = 0
        for arima_cfg in self.ARIMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((arima_cfg.copy(), dist, var, self.seed + s_id))
                    s_id += 1
        return scenarios

    def run_all(self, excel_filename="resultados_arima_completo.xlsx", batch_size=5, n_jobs=2):
        tasks = self.generate_all_scenarios()
        print(f"🚀 Ejecutando {len(tasks)} escenarios ARIMA (Doble Modalidad)...")
        
        all_results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            print(f"  -> Procesando lote {i//batch_size + 1}...")
            results = Parallel(n_jobs=n_jobs)(delayed(self._run_scenario_wrapper)(t) for t in batch)
            for r in results:
                all_results.extend(r)
            
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)

        return pd.DataFrame(all_results)



# ============================================================================
# ¿ Hasta que qué orden de integración d es viable simular ARIMA(p,d,q)?
# ============================================================================


class ARIMAMultiDSimulation:
    """
    Simulador ARIMA(p,d,q) con orden de integración d variable (1 a 10).
    
    Genera: Y_t donde ∇^d Y_t ~ ARMA(p,q)
    Es decir: (1-B)^d Y_t = φ(B) θ(B)^(-1) ε_t
    """
    
    def __init__(self, phi: List[float], theta: List[float], d: int,
                 noise_dist: str = 'normal', sigma: float = 1.0,
                 seed: int = 42, verbose: bool = False):
        """
        Args:
            phi: Coeficientes AR del componente ARMA
            theta: Coeficientes MA del componente ARMA
            d: Orden de integración (1 a 10)
            noise_dist: Distribución del ruido
            sigma: Desviación estándar del ruido
            seed: Semilla aleatoria
            verbose: Mostrar información
        """
        self.phi = np.array(phi) if phi else np.array([])
        self.theta = np.array(theta) if theta else np.array([])
        self.d = d
        self.noise_dist = noise_dist
        self.sigma = sigma
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        
        if d < 1 or d > 10:
            raise ValueError(f"d debe estar entre 1 y 10, recibido: {d}")
    
    def _generate_errors(self, n: int) -> np.ndarray:
        """Genera errores según la distribución especificada (igual que ARMASimulation)."""
        if self.noise_dist == 'normal':
            return self.rng.normal(0, self.sigma, n)
        elif self.noise_dist == 'uniform':
            limit = np.sqrt(3) * self.sigma
            return self.rng.uniform(-limit, limit, size=n)
        elif self.noise_dist == 'exponential':
            return self.rng.exponential(scale=self.sigma, size=n) - self.sigma
        elif self.noise_dist == 't-student':
            from scipy.stats import t
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
    
    def simulate(self, n: int, burn_in: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula serie ARIMA(p,d,q).
        
        Proceso:
        1. Simula W_t ~ ARMA(p,q) estacionario
        2. Integra d veces: Y_t = Σ^d W_t
        
        Returns:
            (serie_integrada, errores)
        """
        total_n = n + burn_in
        p = len(self.phi)
        q = len(self.theta)
        max_lag = max(p, q, 1)
        
        # Paso 1: Simular ARMA(p,q) estacionario
        errors = self._generate_errors(total_n + max_lag)
        w_series = np.zeros(total_n + max_lag)
        
        # Inicialización
        initial_values = self.rng.normal(0, self.sigma, max_lag)
        w_series[:max_lag] = initial_values
        
        # Generar ARMA
        for t in range(max_lag, total_n + max_lag):
            ar_part = 0.0
            if p > 0:
                ar_part = np.dot(self.phi, w_series[t-p:t][::-1])
            
            ma_part = 0.0
            if q > 0:
                ma_part = np.dot(self.theta, errors[t-q:t][::-1])
            
            w_series[t] = ar_part + ma_part + errors[t]
        
        # Remover inicialización
        w_series = w_series[max_lag:]
        errors = errors[max_lag:]
        
        # Paso 2: Integrar d veces
        y_series = w_series.copy()
        for _ in range(self.d):
            y_series = np.cumsum(y_series)
        
        # Remover burn-in
        y_series = y_series[burn_in:]
        errors = errors[burn_in:]
        
        if self.verbose:
            print(f"  Simulado ARIMA(p={p},d={self.d},q={q})")
            print(f"  Serie final: n={len(y_series)}, rango=[{y_series.min():.2f}, {y_series.max():.2f}]")
        
        return y_series, errors
    
    def get_true_next_step_samples(self, series_history: np.ndarray,
                                   errors_history: np.ndarray,
                                   n_samples: int = 5000) -> np.ndarray:
        """
        Genera muestras de la distribución verdadera del siguiente paso.
        
        Para ARIMA(p,d,q):
        - Diferencia d veces la historia para obtener W_t
        - Predice siguiente W usando ARMA
        - Integra de vuelta para obtener Y
        """
        # Diferenciar d veces para obtener el proceso ARMA
        w_history = series_history.copy()
        last_values = []
        
        for _ in range(self.d):
            last_values.append(w_history[-1])
            w_history = np.diff(w_history)
        
        p = len(self.phi)
        q = len(self.theta)
        
        # Predicción en el espacio ARMA
        ar_pred = 0.0
        if p > 0 and len(w_history) >= p:
            ar_pred = np.dot(self.phi, w_history[-p:][::-1])
        
        ma_pred = 0.0
        if q > 0 and len(errors_history) >= q:
            ma_pred = np.dot(self.theta, errors_history[-q:][::-1])
        
        # Generar muestras de W_{t+1}
        noise_samples = self._generate_errors(n_samples)
        w_next_samples = ar_pred + ma_pred + noise_samples
        
        # Integrar d veces de vuelta a Y
        y_next_samples = w_next_samples.copy()
        for i in range(self.d - 1, -1, -1):
            y_next_samples = last_values[i] + y_next_samples
        
        return y_next_samples


class PipelineARIMA_MultiD_DobleModalidad:
    """
    Pipeline Multi-D CORREGIDO para ARIMA(p,d,q) con múltiples órdenes de integración.
    
    CORRECCIONES FUNDAMENTALES (basado en PipelineARIMA_MultiD_SieveOnly):
    1. Usa ARIMASimulation (no ARIMAMultiDSimulation) para d=1
    2. Implementa integración manual para d>1 (como Pipeline140)
    3. Densidades predictivas calculadas en el espacio correcto
    4. Integración coherente para predicciones
    5. Evalúa TODOS los modelos (no solo Sieve Bootstrap)
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION_FOR_OPT = 40
    N_TRAIN_INITIAL = 200

    ARMA_CONFIGS = [
        {'nombre': 'RW', 'phi': [], 'theta': []},
        {'nombre': 'AR(1)', 'phi': [0.6], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    D_VALUES = [1, 2, 3, 4, 5, 6, 7, 10]
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    def _setup_models(self, seed: int):
        """Configura TODOS los modelos (igual que otras pipelines)."""
        return {
            'AREPD': AREPD(n_lags=5, rho=0.9, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=15, random_state=seed),
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=15, n_lags=5, epochs=25, num_samples=self.n_boot, random_state=seed),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=20, B=3, units=24, epochs=15, num_samples=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed)
        }

    def _simulate_arima_manual(self, arma_config: dict, d_value: int, 
                              dist: str, var: float, seed: int, n: int):
        """
        Simula ARIMA EXACTAMENTE como Pipeline140SinSesgos_ARIMA_ConDiferenciacion.
        
        Proceso:
        1. Simula W_t ~ ARMA(p,q) usando ARIMASimulation
        2. Integra manualmente d veces: Y_t = S^d(W_t)
        
        IMPORTANTE: Para d=1, esto es IDÉNTICO a ARIMASimulation directamente.
        """
        from simulacion import ARIMASimulation
        
        # Simular usando ARIMASimulation (siempre con d=1 internamente)
        simulator = ARIMASimulation(
            phi=arma_config['phi'],
            theta=arma_config['theta'],
            noise_dist=dist,
            sigma=np.sqrt(var),
            seed=seed
        )
        
        # Para ARIMASimulation, la serie ya viene con 1 integración
        # Si d=1, usamos directamente. Si d>1, integramos (d-1) veces adicionales
        series_base, errors = simulator.simulate(n=n, burn_in=100)
        
        # Si d=1, ya está integrada correctamente
        if d_value == 1:
            y_series = series_base.copy()
        else:
            # Para d>1, integrar (d-1) veces adicionales
            y_series = series_base.copy()
            for _ in range(d_value - 1):
                y_series = np.cumsum(y_series)
        
        return y_series, series_base, errors, simulator

    def _get_true_density_from_simulator(self, simulator, series_history: np.ndarray,
                                        errors_history: np.ndarray, 
                                        n_samples: int = 1000) -> np.ndarray:
        """
        Obtiene densidad verdadera usando EXACTAMENTE el método de ARIMASimulation.
        
        IDÉNTICO a Pipeline140SinSesgos_ARIMA_ConDiferenciacion.
        """
        return simulator.get_true_next_step_samples(
            series_history, errors_history, n_samples=n_samples
        )

    def _integrate_d_times_for_prediction(self, w_next_samples: np.ndarray,
                                         y_series: np.ndarray, 
                                         current_idx: int,
                                         d_value: int) -> np.ndarray:
        """
        Integra predicciones desde espacio ARMA(d=1) a ARIMA(d>1).
        
        BASADO EN: TransformadorDiferenciacionIntegracion del código original
        
        Para d=1: Y_{t+1} = Y_t + W_{t+1}
        Para d>1: Usar fórmula recursiva
        """
        if d_value == 1:
            # Caso simple: Y_{t+1} = Y_t + ΔY_t donde ΔY_t = W_{t+1}
            return y_series[current_idx - 1] + w_next_samples
        else:
            # Para d>1, necesitamos aplicar integración múltiple
            # Guardamos los últimos d valores de Y
            y_last_values = []
            temp_y = y_series[:current_idx].copy()
            
            for level in range(d_value):
                y_last_values.append(temp_y[-1])
                if level < d_value - 1:
                    temp_y = np.diff(temp_y)
            
            # Integrar desde W_{t+1} hasta Y_{t+1}
            y_next_samples = w_next_samples.copy()
            for level in range(d_value - 1, -1, -1):
                y_next_samples = y_last_values[level] + y_next_samples
            
            return y_next_samples

    def _run_single_modalidad(self, arma_config: dict, d_value: int,
                             dist: str, var: float, scenario_seed: int,
                             y_series: np.ndarray, series_base: np.ndarray,
                             errors: np.ndarray, test_start_idx: int,
                             usar_diferenciacion: bool, simulator) -> list:
        """
        Ejecuta una modalidad EXACTAMENTE como Pipeline140SinSesgos_ARIMA_ConDiferenciacion.
        Pero ahora evalúa TODOS los modelos (no solo Sieve Bootstrap).
        
        MODALIDADES:
        - SIN_DIFF: Modelos ven Y_t (serie integrada de orden d)
        - CON_DIFF: Modelos ven ∇Y_t (serie diferenciada 1 vez)
        """
        modalidad_str = "CON_DIFF" if usar_diferenciacion else "SIN_DIFF"
        
        # Preparar serie según modalidad (IGUAL que Pipeline140)
        if usar_diferenciacion:
            # Los modelos ven incrementos ΔY_t
            series_to_models = np.diff(y_series, prepend=y_series[0])
        else:
            # Los modelos ven niveles Y_t
            series_to_models = y_series.copy()
        
        train_calib_data = series_to_models[:test_start_idx]
        
        # Crear TODOS los modelos
        models = self._setup_models(scenario_seed)
        
        # Optimización (TimeBalancedOptimizer como Pipeline140)
        optimizer = TimeBalancedOptimizer(random_state=self.seed, verbose=self.verbose)
        
        split = min(self.N_VALIDATION_FOR_OPT, len(train_calib_data) // 3)
        best_params = optimizer.optimize_all_models(
            models, 
            train_calib_data[:-split], 
            train_calib_data[-split:]
        )
        
        # Aplicar hiperparámetros óptimos
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_calib_data)

        # Testing rolling window
        results_rows = []
        p = len(arma_config['phi'])
        q = len(arma_config['theta'])

        for t in range(self.N_TEST_STEPS):
            curr_idx = test_start_idx + t
            h_series_levels = y_series[:curr_idx]
            h_to_model = series_to_models[:curr_idx]
            
            # DENSIDAD VERDADERA: Usar el simulador base (ARIMASimulation)
            # Esto da la densidad de Y_{t+1} donde Y tiene 1 integración
            # Si d=1, es directa. Si d>1, necesitamos integrar
            
            if d_value == 1:
                # Para d=1, usar directamente get_true_next_step_samples
                true_samples_base = self._get_true_density_from_simulator(
                    simulator, series_base[:curr_idx], errors[:curr_idx]
                )
                true_samples = true_samples_base
            else:
                # Para d>1, obtener densidad base y luego integrar
                true_samples_base = self._get_true_density_from_simulator(
                    simulator, series_base[:curr_idx], errors[:curr_idx]
                )
                # Integrar las muestras (d-1) veces adicionales
                true_samples = self._integrate_d_times_for_prediction(
                    true_samples_base, y_series, curr_idx, d_value
                )
            
            # Fila de resultados
            row = {
                'Paso': t + 1,
                'Proceso': f"ARMA_I({p},{d_value},{q})",
                'p': p,
                'd': d_value,
                'q': q,
                'ARMA_base': arma_config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'Modalidad': modalidad_str,
                'Valor_Observado': y_series[curr_idx]
            }
            
            # Evaluar TODOS los modelos
            for name, model in models.items():
                try:
                    if "Bootstrap" in name:
                        pred = model.fit_predict(h_to_model)
                    else:
                        pred = model.fit_predict(pd.DataFrame({'valor': h_to_model}))
                    
                    pred_array = np.asarray(pred).flatten()
                    
                    # Integrar predicciones si es necesario (IGUAL que Pipeline140)
                    if usar_diferenciacion:
                        # pred_array son incrementos ΔY_{t+1}
                        # Y_{t+1} = Y_t + ΔY_{t+1}
                        pred_array = y_series[curr_idx - 1] + pred_array
                    
                    # Calcular ECRPS
                    row[name] = ecrps(pred_array, true_samples)
                except Exception as e:
                    if self.verbose:
                        print(f"Error en {name}: {e}")
                    row[name] = np.nan
            
            results_rows.append(row)

        return results_rows

    def _run_scenario_wrapper(self, args):
        """Wrapper para procesamiento paralelo."""
        arma_cfg, d_val, dist, var, seed = args
        
        total_n = self.N_TRAIN_INITIAL + self.N_TEST_STEPS
        
        # Simular ARIMA manualmente (como Pipeline140)
        y_series, series_base, errors, simulator = self._simulate_arima_manual(
            arma_cfg, d_val, dist, var, seed, total_n
        )
        
        # Ejecutar ambas modalidades
        res_sin_diff = self._run_single_modalidad(
            arma_cfg, d_val, dist, var, seed,
            y_series, series_base, errors,
            self.N_TRAIN_INITIAL, False, simulator
        )
        
        res_con_diff = self._run_single_modalidad(
            arma_cfg, d_val, dist, var, seed + 1,
            y_series, series_base, errors,
            self.N_TRAIN_INITIAL, True, simulator
        )
        
        clear_all_sessions()
        return res_sin_diff + res_con_diff

    def run_all(self, excel_filename: str = "RESULTADOS_MULTID_ECRPS_CORREGIDO.xlsx", 
                batch_size: int = 10, n_jobs: int = 3):
        """
        Ejecuta todas las simulaciones con la misma interfaz que la versión original.
        """
        print("="*80)
        print("🚀 PIPELINE MULTI-D CORREGIDO: ARIMA_I(p,d,q) - TODOS LOS MODELOS")
        print("="*80)
        
        # Generar tareas
        tasks = []
        s_id = 0
        for d in self.D_VALUES:
            for cfg in self.ARMA_CONFIGS:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        tasks.append((cfg.copy(), d, dist, var, self.seed + s_id))
                        s_id += 1
        
        print(f"📊 Total de escenarios: {len(tasks)}")
        print(f"   - Valores de d: {self.D_VALUES}")
        print(f"   - ARMA configs: {len(self.ARMA_CONFIGS)}")
        print(f"   - Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   - Varianzas: {len(self.VARIANCES)}")
        print(f"   - Modalidades por escenario: 2 (SIN_DIFF, CON_DIFF)")
        print(f"   - Modelos: TODOS (9 modelos)")
        print(f"   - Simulador base: ARIMASimulation (consistente con Pipeline140)")
        print(f"   - Total filas esperadas: {len(tasks) * 2 * self.N_TEST_STEPS}")
        
        # Procesamiento por lotes
        all_results = []
        num_batches = (len(tasks) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"📦 Procesando lote {i+1}/{num_batches}...")
            
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for r in results: 
                all_results.extend(r)
            
            # Guardar progreso
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)
            print(f"   ✅ {len(all_results)} filas guardadas")
            
            clear_all_sessions()
            gc.collect()
        
        print(f"✅ Simulación completa: {excel_filename}")
        return pd.DataFrame(all_results)


    
# ============================================================================
# ¿La cantidad de datos afecta a la calidad de las densidades predictivas?
# ============================================================================
import numpy as np
import pandas as pd
import gc
from joblib import Parallel, delayed

class Pipeline140_TamanosCrecientes:
    N_TEST_STEPS = 12 
    
    ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]},
        {'nombre': 'ARMA(2,1)', 'phi': [0.7, 0.2], 'theta': [0.5]}
    ]
    
    ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    SETAR_CONFIGS = [
        {'nombre': 'SETAR-1', 'phi_regime1': [0.6], 'phi_regime2': [-0.5], 'threshold': 0.0, 'delay': 1, 'description': 'SETAR(2;1,1) d=1, r=0'},
        {'nombre': 'SETAR-2', 'phi_regime1': [0.7], 'phi_regime2': [-0.7], 'threshold': 0.0, 'delay': 2, 'description': 'SETAR(2;1,1) d=2, r=0'},
        {'nombre': 'SETAR-3', 'phi_regime1': [0.5, -0.2], 'phi_regime2': [-0.3, 0.1], 'threshold': 0.5, 'delay': 1, 'description': 'SETAR(2;2,2) d=1, r=0.5'},
        {'nombre': 'SETAR-4', 'phi_regime1': [0.8, -0.15], 'phi_regime2': [-0.6, 0.2], 'threshold': 1.0, 'delay': 2, 'description': 'SETAR(2;2,2) d=2, r=1.0'},
        {'nombre': 'SETAR-5', 'phi_regime1': [0.4, -0.1, 0.05], 'phi_regime2': [-0.3, 0.1, -0.05], 'threshold': 0.0, 'delay': 1, 'description': 'SETAR(2;3,3) d=1, r=0'},
        {'nombre': 'SETAR-6', 'phi_regime1': [0.5, -0.3, 0.1], 'phi_regime2': [-0.4, 0.2, -0.05], 'threshold': 0.5, 'delay': 2, 'description': 'SETAR(2;3,3) d=2, r=0.5'},
        {'nombre': 'SETAR-7', 'phi_regime1': [0.3, 0.1], 'phi_regime2': [-0.2, -0.1], 'threshold': 0.8, 'delay': 3, 'description': 'SETAR(2;2,2) d=3, r=0.8'}
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.TRAIN_SIZES = [100, 500, 1000]
        self.CALIB_SIZES = [20, 100, 200]

    def _setup_models(self, seed: int):
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'AREPD': AREPD(n_lags=5, rho=0.9, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=15, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=15, n_lags=5, epochs=25, num_samples=self.n_boot, random_state=seed),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=20, B=3, units=24, epochs=15, num_samples=self.n_boot, random_state=seed)
        }

    def _create_simulator(self, config: dict, proceso_tipo: str, dist: str, var: float, seed: int):
        sigma = np.sqrt(var)
        if proceso_tipo == 'ARMA':
            return ARMASimulation(phi=config['phi'], theta=config['theta'], noise_dist=dist, sigma=sigma, seed=seed)
        elif proceso_tipo == 'ARIMA':
            return ARIMASimulation(phi=config['phi'], theta=config['theta'], noise_dist=dist, sigma=sigma, seed=seed)
        else:  # SETAR
            return SETARSimulation(phi_regime1=config['phi_regime1'], phi_regime2=config['phi_regime2'], 
                                   threshold=config['threshold'], delay=config['delay'], noise_dist=dist, sigma=sigma, seed=seed)

    def run_single_scenario(self, config, proceso_tipo, dist, var, n_train, n_calib, size_tag, scenario_seed):
        simulator = self._create_simulator(config, proceso_tipo, dist, var, scenario_seed)
        total_len = n_train + n_calib + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_data = series[:n_train]
        val_data = series[n_train : n_train + n_calib]
        
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        train_val_full = series[:n_train + n_calib]
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)

        results_rows = []
        for t in range(self.N_TEST_STEPS):
            idx = n_train + n_calib + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
            
            row = {
                'Paso': t + 1, 
                'Tipo_Proceso': proceso_tipo,
                'Proceso': config['nombre'], 
                'Distribución': dist,
                'Varianza': var, 
                'N_Train': n_train, 
                'N_Calib': n_calib, 
                'N_Total': n_train + n_calib, 
                'Size': size_tag
            }
            
            for name, model in models.items():
                try:
                    pred = model.fit_predict(h_series) if "Bootstrap" in name else model.fit_predict(pd.DataFrame({'valor': h_series}))
                    row[name] = ecrps(np.asarray(pred).flatten(), true_samples)
                except:
                    row[name] = np.nan
            results_rows.append(row)
        return results_rows

    def _run_scenario_wrapper(self, args):
        return self.run_single_scenario(*args)

    def generate_all_scenarios(self):
        scenarios = []
        s_id = 0
        configs = self.CONFIGS.get(self.proceso_tipo, [])
        
        # CORRECCIÓN: Iterar correctamente sobre TODAS las combinaciones
        for cfg in configs:                    # 7 configuraciones
            for size in self.SIZE_COMBINATIONS: # 5 proporciones
                for dist in self.DISTRIBUTIONS:  # 5 distribuciones
                    for var in self.VARIANCES:    # 4 varianzas
                        scenarios.append((
                            cfg.copy(), 
                            dist, 
                            var, 
                            size['n_train'], 
                            size['n_calib'], 
                            size['prop_tag'], 
                            self.seed + s_id
                        ))
                        s_id += 1
        
        print(f"✅ Generados {len(scenarios)} escenarios para {self.proceso_tipo}")
        print(f"   → {len(configs)} configs × {len(self.SIZE_COMBINATIONS)} props × "
            f"{len(self.DISTRIBUTIONS)} dists × {len(self.VARIANCES)} vars")
        
        return scenarios

    def run_all(self, excel_filename=None, batch_size=10, max_workers=3):
        if excel_filename is None:
            excel_filename = "RESULTADOS_TODOS_PROCESOS.xlsx"
            
        tasks = self.generate_all_scenarios()
        all_results = []
        num_batches = (len(tasks) + batch_size - 1) // batch_size

        print(f"📊 Total de escenarios: {len(tasks)}")
        print(f"📦 Número de batches: {num_batches}")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"🔄 Procesando batch {i+1}/{num_batches}...")
            
            results = Parallel(n_jobs=max_workers, backend='loky')(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for r in results: 
                all_results.extend(r)
            
            # Guardar después de cada batch
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)
            print(f"✅ Guardado progreso: {len(all_results)} filas")
            
            gc.collect()
        
        print(f"\n🎉 ¡Completado! Total de filas: {len(all_results)}")
        return pd.DataFrame(all_results)


# ============================================================================
# ¿La proporción de datos afecta a la calidad de las densidades predictivas?
# ============================================================================

class Pipeline240_ProporcionesVariables:
    """
    Pipeline unificado con Tamaño Fijo (240) y Proporciones Variables.
    1) Compara Densidad Predictiva vs Densidad Teórica mediante ECRPS.
    2) Proporciones de Calibración: 10%, 20%, 30%, 40%, 50%.
    """
    
    N_TOTAL = 240  # Tamaño histórico fijo
    N_TEST_STEPS = 12 
    
    # Configuraciones de Proporciones (N_TRAIN + N_CALIB = 240)
    SIZE_COMBINATIONS = [
        {'prop_tag': '10%', 'n_train': 216, 'n_calib': 24, 'prop_val': 0.10},
        {'prop_tag': '20%', 'n_train': 192, 'n_calib': 48, 'prop_val': 0.20},
        {'prop_tag': '30%', 'n_train': 168, 'n_calib': 72, 'prop_val': 0.30},
        {'prop_tag': '40%', 'n_train': 144, 'n_calib': 96, 'prop_val': 0.40},
        {'prop_tag': '50%', 'n_train': 120, 'n_calib': 120, 'prop_val': 0.50}
    ]
    
    CONFIGS = {
        'ARMA': [
            {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
            {'nombre': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
            {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]},
            {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
            {'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]},
            {'nombre': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]},
            {'nombre': 'ARMA(2,1)', 'phi': [0.7, 0.2], 'theta': [0.5]}
        ],
        'ARIMA': [
            {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
            {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
            {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
            {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
            {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
            {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
            {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
        ],
        'SETAR': [
            {'nombre': 'SETAR-1', 'phi_regime1': [0.6], 'phi_regime2': [-0.5], 'threshold': 0.0, 'delay': 1},
            {'nombre': 'SETAR-2', 'phi_regime1': [0.7], 'phi_regime2': [-0.7], 'threshold': 0.0, 'delay': 2},
            {'nombre': 'SETAR-3', 'phi_regime1': [0.5, -0.2], 'phi_regime2': [-0.3, 0.1], 'threshold': 0.5, 'delay': 1},
            {'nombre': 'SETAR-4', 'phi_regime1': [0.8, -0.15], 'phi_regime2': [-0.6, 0.2], 'threshold': 1.0, 'delay': 2},
            {'nombre': 'SETAR-5', 'phi_regime1': [0.4, -0.1, 0.05], 'phi_regime2': [-0.3, 0.1, -0.05], 'threshold': 0.0, 'delay': 1},
            {'nombre': 'SETAR-6', 'phi_regime1': [0.5, -0.3, 0.1], 'phi_regime2': [-0.4, 0.2, -0.05], 'threshold': 0.5, 'delay': 2},
            {'nombre': 'SETAR-7', 'phi_regime1': [0.3, 0.1], 'phi_regime2': [-0.2, -0.1], 'threshold': 0.8, 'delay': 3}
        ]
    }
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]

    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False, proceso_tipo: str = 'ARMA'):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.proceso_tipo = proceso_tipo.upper()
        self.rng = np.random.default_rng(seed)

    def _setup_models(self, seed: int):
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'LSPM': LSPM(random_state=seed),
            'LSPMW': LSPMW(rho=0.95, random_state=seed),
            'AREPD': AREPD(n_lags=5, rho=0.9, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(n_lags=15, random_state=seed),
            'DeepAR': DeepARModel(hidden_size=15, n_lags=5, epochs=25, num_samples=self.n_boot, random_state=seed),
            'EnCQR-LSTM': EnCQR_LSTM_Model(n_lags=20, B=3, units=24, epochs=15, num_samples=self.n_boot, random_state=seed)
        }

    def _create_simulator(self, config: dict, dist: str, var: float, seed: int):
        sigma = np.sqrt(var)
        if self.proceso_tipo == 'ARMA':
            return ARMASimulation(phi=config['phi'], theta=config['theta'], noise_dist=dist, sigma=sigma, seed=seed)
        elif self.proceso_tipo == 'ARIMA':
            return ARIMASimulation(phi=config['phi'], theta=config['theta'], noise_dist=dist, sigma=sigma, seed=seed)
        else: # SETAR
            return SETARSimulation(phi_regime1=config['phi_regime1'], phi_regime2=config['phi_regime2'], 
                                   threshold=config['threshold'], delay=config['delay'], noise_dist=dist, sigma=sigma, seed=seed)

    def run_single_scenario(self, config, dist, var, n_train, n_calib, prop_tag, scenario_seed):
        # 1. Simulación
        simulator = self._create_simulator(config, dist, var, scenario_seed)
        total_len = self.N_TOTAL + self.N_TEST_STEPS
        series, errors = simulator.simulate(n=total_len, burn_in=100)
        
        train_data = series[:n_train]
        val_data = series[n_train : self.N_TOTAL]
        
        # 2. Optimización y Congelamiento
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        best_params = optimizer.optimize_all_models(models, train_data, val_data)
        
        train_val_full = series[:self.N_TOTAL]
        for name, model in models.items():
            if name in best_params:
                for k, v in best_params[name].items():
                    if hasattr(model, k): setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_full)

        # 3. Test Rolling con ECRPS
        results_rows = []
        model_names = list(models.keys())

        for t in range(self.N_TEST_STEPS):
            idx = self.N_TOTAL + t
            h_series = series[:idx]
            h_errors = errors[:idx]
            
            # Densidad Teórica (Ground Truth)
            true_samples = simulator.get_true_next_step_samples(h_series, h_errors, n_samples=1000)
            
            # FILA RECTIFICADA
            row = {
                'Paso': t + 1, 
                'Proceso': config['nombre'], 
                'Distribución': dist,   # Cambio Dist -> Distribución
                'Varianza': var,        # Cambio Var -> Varianza
                'N_Train': n_train, 
                'N_Calib': n_calib, 
                'Prop_Calib': prop_tag  # Cambio Prop -> Prop_Calib
            }
            
            for name, model in models.items():
                try:
                    if "Bootstrap" in name: pred = model.fit_predict(h_series)
                    else: pred = model.fit_predict(pd.DataFrame({'valor': h_series}))
                    
                    pred_array = np.asarray(pred).flatten()
                    row[name] = ecrps(pred_array, true_samples)
                except:
                    row[name] = np.nan
            
            results_rows.append(row)

        # 4. GENERACIÓN DE FILA "Promedio" (Crucial para analisis_proporciones_240)
        df_temp = pd.DataFrame(results_rows)
        avg_row = {
            'Paso': 'Promedio',
            'Proceso': config['nombre'],
            'Distribución': dist,
            'Varianza': var,
            'N_Train': n_train,
            'N_Calib': n_calib,
            'Prop_Calib': prop_tag
        }
        for m_name in model_names:
            avg_row[m_name] = df_temp[m_name].mean()
        
        results_rows.append(avg_row)

        clear_all_sessions()
        return results_rows

    def _run_scenario_wrapper(self, args):
        return self.run_single_scenario(*args)

    def generate_all_scenarios(self):
        scenarios = []
        s_id = 0
        configs = self.CONFIGS.get(self.proceso_tipo, [])
        
        for size in self.SIZE_COMBINATIONS:
            for cfg in configs:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        scenarios.append((
                            cfg.copy(), 
                            dist, 
                            var, 
                            size['n_train'], 
                            size['n_calib'], 
                            size['prop_tag'], 
                            self.seed + s_id
                        ))
                        s_id += 1
        return scenarios

    def run_all(self, excel_filename=None, batch_size=10, max_workers=3):
        if excel_filename is None:
            excel_filename = f"RESULTADOS_PROPORCIONES_{self.proceso_tipo}.xlsx"
            
        tasks = self.generate_all_scenarios()
        print(f"🚀 Iniciando Pipeline Proporciones ({self.proceso_tipo}): {len(tasks)} escenarios.")
        
        all_results = []
        num_batches = (len(tasks) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tasks))
            batch = tasks[start_idx:end_idx]
            
            print(f"📦 Procesando Lote {i+1}/{num_batches}...")
            results = Parallel(n_jobs=max_workers, backend='loky')(
                delayed(self._run_scenario_wrapper)(t) for t in batch
            )
            
            for r in results: all_results.extend(r)
            pd.DataFrame(all_results).to_excel(excel_filename, index=False)
            clear_all_sessions()
            gc.collect()

        return pd.DataFrame(all_results)