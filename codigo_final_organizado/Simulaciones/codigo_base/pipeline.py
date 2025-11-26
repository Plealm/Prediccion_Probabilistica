# PIPELINE ACTUALIZADO - Garantiza flujo correcto sin sesgos
# Asegura que freeze_hyperparameters() se llame DESPUÉS de optimize_hyperparameters()

import numpy as np
import pandas as pd
import warnings
import gc
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Union, Any

warnings.filterwarnings("ignore")

from simulacion import ARMASimulation, ARIMASimulation, SETARSimulation
from modelos import (CircularBlockBootstrapModel, SieveBootstrapModel, LSPM, LSPMW, 
                     DeepARModel, AREPD, MondrianCPSModel, AdaptiveVolatilityMondrianCPS,
                     EnCQR_LSTM_Model)
from metricas import crps


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
    Pipeline CORREGIDO que garantiza:
    1. Optimización con distribución verdadera
    2. Congelamiento de parámetros (freeze_hyperparameters)
    3. NO re-estimación en ventana rodante
    """
    
    N_TEST_STEPS = 12
    N_CALIBRATION = 40
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
        
    def _setup_models(self, seed: int):
        """Inicializa los 9 modelos con seed específica."""
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(
                block_length='auto', n_boot=self.n_boot,
                random_state=seed, verbose=False,
                hyperparam_ranges={'block_length': [2, 50]}, optimize=True
            ),
            'Sieve Bootstrap': SieveBootstrapModel(
                order='auto', n_boot=self.n_boot,
                random_state=seed, verbose=False,
                hyperparam_ranges={'order': [1, 20]}, optimize=True
            ),
            'LSPM': LSPM(random_state=seed, verbose=False),
            'LSPMW': LSPMW(rho=0.95, random_state=seed, verbose=False),
            'DeepAR': DeepARModel(
                hidden_size=15, n_lags=5, num_layers=1, dropout=0.1,
                lr=0.01, batch_size=32, epochs=30, num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            ),
            'AREPD': AREPD(
                n_lags=5, rho=0.9, alpha=0.1, poly_degree=2,
                random_state=seed, verbose=False, optimize=True
            ),
            'MCPS': MondrianCPSModel(
                n_lags=10, n_bins=8, test_size=0.25,
                random_state=seed, verbose=False, optimize=True
            ),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(
                n_lags=15, n_pred_bins=8, n_vol_bins=4, volatility_window=20,
                test_size=0.25, random_state=seed, verbose=False, optimize=True
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=20, B=3, units=32, n_layers=2, lr=0.005,
                batch_size=16, epochs=20, num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            )
        }
    
    def _generate_true_distribution(self, simulator: ARMASimulation, 
                                    series_history: np.ndarray,
                                    errors_history: np.ndarray,
                                    n_samples: int = 5000) -> np.ndarray:
        """
        Genera la distribución verdadera del siguiente paso usando el simulador ARMA.
        """
        return simulator.get_true_next_step_samples(
            series_history=series_history,
            errors_history=errors_history,
            n_samples=n_samples
        )
    
    def _optimize_models_con_distribucion_verdadera(self, models: dict, 
                                                     train_calib_series: np.ndarray,
                                                     train_calib_errors: np.ndarray,
                                                     simulator: ARMASimulation):
        """
        PASO 1: Optimiza hiperparámetros usando la distribución verdadera.
        """
        true_distribution = self._generate_true_distribution(
            simulator=simulator,
            series_history=train_calib_series,
            errors_history=train_calib_errors,
            n_samples=5000
        )
        
        if self.verbose:
            print(f"  Distribución verdadera: μ={np.mean(true_distribution):.4f}, "
                  f"σ={np.std(true_distribution):.4f}")
        
        for name, model in models.items():
            try:
                if hasattr(model, 'optimize_hyperparameters'):
                    df_tc = pd.DataFrame({'valor': train_calib_series})
                    model.optimize_hyperparameters(df_tc, true_distribution)
                    
                    if hasattr(model, 'optimize'):
                        model.optimize = False
                        
                    if self.verbose and hasattr(model, 'best_params'):
                        print(f"    {name}: {model.best_params}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"    Error optimizando {name}: {e}")

    def _freeze_all_models(self, models: dict, train_calib_series: np.ndarray):
        """
        PASO 2: CRÍTICO - Congela TODOS los modelos después de optimización.
        Esto incluye:
        - Bootstrap: block_length, parámetros AR
        - LSPM/LSPMW: n_lags, rho
        - DeepAR: scaler (mean, std)
        - AREPD: modelo Ridge + scaler
        - MCPS/AV-MCPS: modelo XGBoost + bins
        - EnCQR-LSTM: ensemble + scaler
        """
        if self.verbose:
            print("\n  CONGELANDO MODELOS:")
        
        for name, model in models.items():
            try:
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(train_calib_series)
                    
                    if self.verbose:
                        # Verificar qué se congeló
                        frozen_attrs = []
                        if hasattr(model, '_frozen_block_length'):
                            frozen_attrs.append(f"block_length={model._frozen_block_length}")
                        if hasattr(model, '_frozen_order'):
                            frozen_attrs.append(f"order={model._frozen_order}")
                        if hasattr(model, '_frozen_mean'):
                            frozen_attrs.append(f"mean={model._frozen_mean:.4f}")
                        if hasattr(model, '_frozen_std'):
                            frozen_attrs.append(f"std={model._frozen_std:.4f}")
                        if hasattr(model, '_is_frozen') and model._is_frozen:
                            frozen_attrs.append("✓ frozen")
                        if hasattr(model, '_fitted_artifacts') and model._fitted_artifacts:
                            frozen_attrs.append("✓ artifacts")
                        if hasattr(model, '_trained_ensemble') and model._trained_ensemble:
                            frozen_attrs.append("✓ ensemble")
                        
                        if frozen_attrs:
                            print(f"    {name}: {', '.join(frozen_attrs)}")
                            
            except Exception as e:
                if self.verbose:
                    print(f"    Error congelando {name}: {e}")

    def _run_single_scenario(self, arma_config: dict, distribution: str, 
                            variance: float, scenario_seed: int) -> list:
        """
        Ejecuta un escenario completo con la lógica correcta:
        1. Optimiza hiperparámetros (usa distribución verdadera)
        2. Congela TODOS los parámetros
        3. Predice 10 pasos SIN re-estimar
        """
        try:
            total_needed = self.N_TRAIN + self.N_CALIBRATION + self.N_TEST_STEPS
            
            # Crear simulador
            simulator = ARMASimulation(
                model_type=arma_config['nombre'],
                phi=arma_config['phi'],
                theta=arma_config['theta'],
                noise_dist=distribution,
                sigma=np.sqrt(variance),
                seed=scenario_seed,
                verbose=False
            )
            
            # Simular serie completa con errores
            full_series, full_errors = simulator.simulate(n=total_needed, burn_in=50)
            
            # Dividir datos
            calib_end = self.N_TRAIN + self.N_CALIBRATION
            train_calib_series = full_series[:calib_end]
            train_calib_errors = full_errors[:calib_end]
            
            # Inicializar modelos
            models = self._setup_models(scenario_seed)
            
            # ========================================================
            # FLUJO CORRECTO:
            # PASO 1: Optimizar hiperparámetros con distribución verdadera
            # ========================================================
            self._optimize_models_con_distribucion_verdadera(
                models=models,
                train_calib_series=train_calib_series,
                train_calib_errors=train_calib_errors,
                simulator=simulator
            )
            
            # ========================================================
            # PASO 2: CONGELAR TODOS LOS MODELOS
            # ========================================================
            self._freeze_all_models(models, train_calib_series)
            
            # ========================================================
            # PASO 3: Predicción rodante SIN re-estimación
            # ========================================================
            results_rows = []
            
            for step in range(self.N_TEST_STEPS):
                current_idx = calib_end + step
                history_series = full_series[:current_idx]
                true_value = full_series[current_idx]
                
                step_result = {
                    'Paso': step + 1,
                    'proces_simulacion': arma_config['nombre'],
                    'Distribución': distribution,
                    'Varianza error': variance,
                    'Valor_Observado': true_value
                }
                
                for name, model in models.items():
                    try:
                        # CRÍTICO: fit_predict() NO debe re-estimar
                        # Solo usa history_series para construir ventana
                        if isinstance(model, (CircularBlockBootstrapModel, SieveBootstrapModel)):
                            pred_samples = model.fit_predict(history_series)
                        else:
                            df_hist = pd.DataFrame({'valor': history_series})
                            pred_samples = model.fit_predict(df_hist)
                        
                        pred_samples = np.asarray(pred_samples).flatten()
                        crps_val = crps(pred_samples, true_value)
                        step_result[name] = crps_val
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"    {name} error en paso {step+1}: {e}")
                        step_result[name] = np.nan
                
                results_rows.append(step_result)
            
            # Promedio del escenario
            avg_row = {
                'Paso': 'Promedio',
                'proces_simulacion': arma_config['nombre'],
                'Distribución': distribution,
                'Varianza error': variance,
                'Valor_Observado': np.nan
            }
            
            model_names = list(models.keys())
            for model_name in model_names:
                vals = [r[model_name] for r in results_rows if not pd.isna(r.get(model_name))]
                avg_row[model_name] = np.mean(vals) if vals else np.nan
            
            results_rows.append(avg_row)
            
            del models, simulator
            clear_all_sessions()
            
            return results_rows
            
        except Exception as e:
            if self.verbose:
                print(f"Error en escenario: {e}")
            return []
    
    def generate_all_scenarios(self) -> list:
        """Genera lista de argumentos para ejecución paralela."""
        scenarios = []
        scenario_id = 0
        
        for arma_cfg in self.ARMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((
                        arma_cfg.copy(),
                        dist,
                        var,
                        self.seed + scenario_id,
                        self.N_TEST_STEPS,
                        self.N_TRAIN,
                        self.N_CALIBRATION,
                        self.n_boot
                    ))
                    scenario_id += 1
        
        return scenarios
    
    def run_all(self, excel_filename: str = "resultados_140_CORREGIDO.xlsx", 
                batch_size: int = 10):
        """
        Ejecuta los escenarios con el flujo correcto garantizado.
        """
        print("="*80)
        print("EVALUACIÓN 140 ESCENARIOS - VERSIÓN CORREGIDA SIN SESGOS")
        print("="*80)
        
        cpu_count = os.cpu_count() or 4
        safe_jobs = min(6, max(1, int(cpu_count * 0.75)))
        
        print(f"⚡ Usando {safe_jobs} núcleos en paralelo")
        print(f"⚡ Tamaño del lote: {batch_size}")
        print(f"\n✓ FLUJO CORRECTO:")
        print("  1. Optimización con distribución verdadera")
        print("  2. Congelamiento de TODOS los parámetros")
        print("  3. Predicción rodante SIN re-estimación")
        print("="*80 + "\n")
        
        all_scenarios = self.generate_all_scenarios()
        total_scenarios = len(all_scenarios)
        all_rows = []
        
        num_batches = (total_scenarios + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_scenarios)
            current_batch = all_scenarios[start_idx:end_idx]
            
            print(f"\n🚀 Lote {i+1}/{num_batches} (Escenarios {start_idx+1} a {end_idx})...")
            
            try:
                batch_results = Parallel(n_jobs=safe_jobs, backend='loky', timeout=99999)(
                    delayed(self._run_scenario_wrapper)(args) 
                    for args in tqdm(current_batch, desc=f"  Lote {i+1}", leave=False)
                )
                
                for res in batch_results:
                    all_rows.extend(res)
                
                self._save_checkpoint(all_rows, excel_filename)
                
                del batch_results, current_batch
                clear_all_sessions()
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error en lote {i+1}: {e}")
                self._save_checkpoint(all_rows, f"RECOVERY_{excel_filename}")
                raise e

        print("\n" + "="*80)
        print(f"✅ COMPLETADO - {len(all_rows)} filas generadas")
        print("="*80)
        
        return pd.DataFrame(all_rows)

    def _run_scenario_wrapper(self, args):
        """Wrapper para ejecución paralela."""
        (arma_cfg, dist, var, seed, n_test_steps, 
         n_train, n_calib, n_boot) = args
        
        pipeline = Pipeline140SinSesgos_ARMA(n_boot=n_boot, seed=seed, verbose=False)
        pipeline.N_TEST_STEPS = n_test_steps
        pipeline.N_TRAIN = n_train
        pipeline.N_CALIBRATION = n_calib
        
        return pipeline._run_single_scenario(arma_cfg, dist, var, seed)

    def _save_checkpoint(self, rows, filename):
        """Guarda progreso en Excel."""
        if not rows: 
            return
        
        df_temp = pd.DataFrame(rows)
        
        ordered_cols = [
            'Paso', 'proces_simulacion', 'Distribución', 'Varianza error', 'Valor_Observado',
            'AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 'EnCQR-LSTM',
            'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap'
        ]
        final_cols = [c for c in ordered_cols if c in df_temp.columns]
        remaining_cols = [c for c in df_temp.columns if c not in final_cols]
        final_cols.extend(remaining_cols)
        
        df_temp = df_temp[final_cols]
        df_temp.to_excel(filename, index=False)


class Pipeline140SinSesgos_ARIMA:
    """
    Pipeline para procesos ARIMA(p,1,q) con la misma filosofía que ARMA:
    1. Optimización con distribución verdadera
    2. Congelamiento de parámetros
    3. NO re-estimación en ventana rodante
    """
    
    N_TEST_STEPS = 12
    N_CALIBRATION = 40
    N_TRAIN = 200

    # Configuraciones ARIMA(p,1,q) - el "1" es la diferenciación fija
    ARIMA_CONFIGS = [
       {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},  # Random Walk
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
        
    def _setup_models(self, seed: int):
        """Inicializa los 9 modelos con seed específica."""
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(
                block_length='auto', n_boot=self.n_boot,
                random_state=seed, verbose=False,
                hyperparam_ranges={'block_length': [2, 50]}, optimize=True
            ),
            'Sieve Bootstrap': SieveBootstrapModel(
                order='auto', n_boot=self.n_boot,
                random_state=seed, verbose=False,
                hyperparam_ranges={'order': [1, 20]}, optimize=True
            ),
            'LSPM': LSPM(random_state=seed, verbose=False),
            'LSPMW': LSPMW(rho=0.95, random_state=seed, verbose=False),
            'DeepAR': DeepARModel(
                hidden_size=15, n_lags=5, num_layers=1, dropout=0.1,
                lr=0.01, batch_size=32, epochs=30, num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            ),
            'AREPD': AREPD(
                n_lags=5, rho=0.9, alpha=0.1, poly_degree=2,
                random_state=seed, verbose=False, optimize=True
            ),
            'MCPS': MondrianCPSModel(
                n_lags=10, n_bins=8, test_size=0.25,
                random_state=seed, verbose=False, optimize=True
            ),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(
                n_lags=15, n_pred_bins=8, n_vol_bins=4, volatility_window=20,
                test_size=0.25, random_state=seed, verbose=False, optimize=True
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=20, B=3, units=32, n_layers=2, lr=0.005,
                batch_size=16, epochs=20, num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            )
        }
    
    def _generate_true_distribution(self, simulator: ARIMASimulation, 
                                    series_history: np.ndarray,
                                    errors_history: np.ndarray,
                                    n_samples: int = 5000) -> np.ndarray:
        """
        Genera la distribución verdadera del siguiente paso usando el simulador ARIMA.
        Para ARIMA, esto incluye el componente de integración.
        """
        return simulator.get_true_next_step_samples(
            series_history=series_history,
            errors_history=errors_history,
            n_samples=n_samples
        )
    
    def _optimize_models_con_distribucion_verdadera(self, models: dict, 
                                                     train_calib_series: np.ndarray,
                                                     train_calib_errors: np.ndarray,
                                                     simulator: ARIMASimulation):
        """
        PASO 1: Optimiza hiperparámetros usando la distribución verdadera.
        """
        true_distribution = self._generate_true_distribution(
            simulator=simulator,
            series_history=train_calib_series,
            errors_history=train_calib_errors,
            n_samples=5000
        )
        
        if self.verbose:
            print(f"  Distribución verdadera: μ={np.mean(true_distribution):.4f}, "
                  f"σ={np.std(true_distribution):.4f}")
        
        for name, model in models.items():
            try:
                if hasattr(model, 'optimize_hyperparameters'):
                    df_tc = pd.DataFrame({'valor': train_calib_series})
                    model.optimize_hyperparameters(df_tc, true_distribution)
                    
                    if hasattr(model, 'optimize'):
                        model.optimize = False
                        
                    if self.verbose and hasattr(model, 'best_params'):
                        print(f"    {name}: {model.best_params}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"    Error optimizando {name}: {e}")

    def _freeze_all_models(self, models: dict, train_calib_series: np.ndarray):
        """
        PASO 2: CRÍTICO - Congela TODOS los modelos después de optimización.
        """
        if self.verbose:
            print("\n  CONGELANDO MODELOS:")
        
        for name, model in models.items():
            try:
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(train_calib_series)
                    
                    if self.verbose:
                        frozen_attrs = []
                        if hasattr(model, '_frozen_block_length'):
                            frozen_attrs.append(f"block_length={model._frozen_block_length}")
                        if hasattr(model, '_frozen_order'):
                            frozen_attrs.append(f"order={model._frozen_order}")
                        if hasattr(model, '_frozen_mean'):
                            frozen_attrs.append(f"mean={model._frozen_mean:.4f}")
                        if hasattr(model, '_frozen_std'):
                            frozen_attrs.append(f"std={model._frozen_std:.4f}")
                        if hasattr(model, '_is_frozen') and model._is_frozen:
                            frozen_attrs.append("✓ frozen")
                        if hasattr(model, '_fitted_artifacts') and model._fitted_artifacts:
                            frozen_attrs.append("✓ artifacts")
                        if hasattr(model, '_trained_ensemble') and model._trained_ensemble:
                            frozen_attrs.append("✓ ensemble")
                        
                        if frozen_attrs:
                            print(f"    {name}: {', '.join(frozen_attrs)}")
                            
            except Exception as e:
                if self.verbose:
                    print(f"    Error congelando {name}: {e}")

    def _run_single_scenario(self, arima_config: dict, distribution: str, 
                            variance: float, scenario_seed: int) -> list:
        """
        Ejecuta un escenario completo ARIMA:
        1. Optimiza hiperparámetros (usa distribución verdadera)
        2. Congela TODOS los parámetros
        3. Predice sin re-estimar
        """
        try:
            total_needed = self.N_TRAIN + self.N_CALIBRATION + self.N_TEST_STEPS
            
            # Crear simulador ARIMA
            simulator = ARIMASimulation(
                model_type=arima_config['nombre'],
                phi=arima_config['phi'],
                theta=arima_config['theta'],
                noise_dist=distribution,
                sigma=np.sqrt(variance),
                seed=scenario_seed,
                verbose=False
            )
            
            # Simular serie completa con errores
            full_series, full_errors = simulator.simulate(n=total_needed, burn_in=50)
            
            # Dividir datos
            calib_end = self.N_TRAIN + self.N_CALIBRATION
            train_calib_series = full_series[:calib_end]
            train_calib_errors = full_errors[:calib_end]
            
            # Inicializar modelos
            models = self._setup_models(scenario_seed)
            
            # PASO 1: Optimizar con distribución verdadera
            self._optimize_models_con_distribucion_verdadera(
                models=models,
                train_calib_series=train_calib_series,
                train_calib_errors=train_calib_errors,
                simulator=simulator
            )
            
            # PASO 2: CONGELAR TODOS LOS MODELOS
            self._freeze_all_models(models, train_calib_series)
            
            # PASO 3: Predicción rodante SIN re-estimación
            results_rows = []
            
            for step in range(self.N_TEST_STEPS):
                current_idx = calib_end + step
                history_series = full_series[:current_idx]
                true_value = full_series[current_idx]
                
                step_result = {
                    'Paso': step + 1,
                    'proces_simulacion': arima_config['nombre'],
                    'Distribución': distribution,
                    'Varianza error': variance,
                    'Valor_Observado': true_value
                }
                
                for name, model in models.items():
                    try:
                        if isinstance(model, (CircularBlockBootstrapModel, SieveBootstrapModel)):
                            pred_samples = model.fit_predict(history_series)
                        else:
                            df_hist = pd.DataFrame({'valor': history_series})
                            pred_samples = model.fit_predict(df_hist)
                        
                        pred_samples = np.asarray(pred_samples).flatten()
                        crps_val = crps(pred_samples, true_value)
                        step_result[name] = crps_val
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"    {name} error en paso {step+1}: {e}")
                        step_result[name] = np.nan
                
                results_rows.append(step_result)
            
            # Promedio del escenario
            avg_row = {
                'Paso': 'Promedio',
                'proces_simulacion': arima_config['nombre'],
                'Distribución': distribution,
                'Varianza error': variance,
                'Valor_Observado': np.nan
            }
            
            model_names = list(models.keys())
            for model_name in model_names:
                vals = [r[model_name] for r in results_rows if not pd.isna(r.get(model_name))]
                avg_row[model_name] = np.mean(vals) if vals else np.nan
            
            results_rows.append(avg_row)
            
            del models, simulator
            clear_all_sessions()
            
            return results_rows
            
        except Exception as e:
            if self.verbose:
                print(f"Error en escenario: {e}")
            return []
    
    def generate_all_scenarios(self) -> list:
        """Genera lista de argumentos para ejecución paralela."""
        scenarios = []
        scenario_id = 0
        
        for arima_cfg in self.ARIMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((
                        arima_cfg.copy(),
                        dist,
                        var,
                        self.seed + scenario_id,
                        self.N_TEST_STEPS,
                        self.N_TRAIN,
                        self.N_CALIBRATION,
                        self.n_boot
                    ))
                    scenario_id += 1
        
        return scenarios
    
    def run_all(self, excel_filename: str = "resultados_140_ARIMA.xlsx", 
                batch_size: int = 10):
        """
        Ejecuta los escenarios ARIMA con el flujo correcto garantizado.
        """
        print("="*80)
        print("EVALUACIÓN 140 ESCENARIOS ARIMA - VERSIÓN CORREGIDA SIN SESGOS")
        print("="*80)
        
        cpu_count = os.cpu_count() or 4
        safe_jobs = min(6, max(1, int(cpu_count * 0.75)))
        
        print(f"⚡ Usando {safe_jobs} núcleos en paralelo")
        print(f"⚡ Tamaño del lote: {batch_size}")
        print(f"\n✓ FLUJO CORRECTO:")
        print("  1. Optimización con distribución verdadera")
        print("  2. Congelamiento de TODOS los parámetros")
        print("  3. Predicción rodante SIN re-estimación")
        print("="*80 + "\n")
        
        all_scenarios = self.generate_all_scenarios()
        total_scenarios = len(all_scenarios)
        all_rows = []
        
        num_batches = (total_scenarios + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_scenarios)
            current_batch = all_scenarios[start_idx:end_idx]
            
            print(f"\n🚀 Lote {i+1}/{num_batches} (Escenarios {start_idx+1} a {end_idx})...")
            
            try:
                batch_results = Parallel(n_jobs=safe_jobs, backend='loky', timeout=99999)(
                    delayed(self._run_scenario_wrapper)(args) 
                    for args in tqdm(current_batch, desc=f"  Lote {i+1}", leave=False)
                )
                
                for res in batch_results:
                    all_rows.extend(res)
                
                self._save_checkpoint(all_rows, excel_filename)
                
                del batch_results, current_batch
                clear_all_sessions()
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error en lote {i+1}: {e}")
                self._save_checkpoint(all_rows, f"RECOVERY_{excel_filename}")
                raise e

        print("\n" + "="*80)
        print(f"✅ COMPLETADO - {len(all_rows)} filas generadas")
        print("="*80)
        
        return pd.DataFrame(all_rows)

    def _run_scenario_wrapper(self, args):
        """Wrapper para ejecución paralela."""
        (arima_cfg, dist, var, seed, n_test_steps, 
         n_train, n_calib, n_boot) = args
        
        pipeline = Pipeline140SinSesgos_ARIMA(n_boot=n_boot, seed=seed, verbose=False)
        pipeline.N_TEST_STEPS = n_test_steps
        pipeline.N_TRAIN = n_train
        pipeline.N_CALIBRATION = n_calib
        
        return pipeline._run_single_scenario(arima_cfg, dist, var, seed)

    def _save_checkpoint(self, rows, filename):
        """Guarda progreso en Excel."""
        if not rows: 
            return
        
        df_temp = pd.DataFrame(rows)
        
        ordered_cols = [
            'Paso', 'proces_simulacion', 'Distribución', 'Varianza error', 'Valor_Observado',
            'AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 'EnCQR-LSTM',
            'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap'
        ]
        final_cols = [c for c in ordered_cols if c in df_temp.columns]
        remaining_cols = [c for c in df_temp.columns if c not in final_cols]
        final_cols.extend(remaining_cols)
        
        df_temp = df_temp[final_cols]
        df_temp.to_excel(filename, index=False)


class Pipeline140SinSesgos_SETAR:
    """
    Pipeline para procesos SETAR con la misma filosofía que ARMA/ARIMA:
    1. Optimización con distribución verdadera
    2. Congelamiento de parámetros
    3. NO re-estimación en ventana rodante
    """
    
    N_TEST_STEPS = 12
    N_CALIBRATION = 40
    N_TRAIN = 200

    # Configuraciones SETAR basadas en los 7 casos especificados
    SETAR_CONFIGS = [
        {
            'nombre': 'SETAR-1',
            'phi_regime1': [0.6],
            'phi_regime2': [-0.5],
            'threshold': 0.0,
            'delay': 1,
            'description': 'SETAR(2;1,1) - AR(1) con d=1'
        },
        {
            'nombre': 'SETAR-2',
            'phi_regime1': [0.7],
            'phi_regime2': [-0.7],
            'threshold': 0.0,
            'delay': 2,
            'description': 'SETAR(2;1,1) - AR(1) con d=2'
        },
        {
            'nombre': 'SETAR-3',
            'phi_regime1': [0.5, -0.2],
            'phi_regime2': [-0.3, 0.1],
            'threshold': 0.5,
            'delay': 1,
            'description': 'SETAR(2;2,2) - AR(2) con d=1'
        },
        {
            'nombre': 'SETAR-4',
            'phi_regime1': [0.8, -0.15],
            'phi_regime2': [-0.6, 0.2],
            'threshold': 1.0,
            'delay': 2,
            'description': 'SETAR(2;2,2) - AR(2) con d=2'
        },
        {
            'nombre': 'SETAR-5',
            'phi_regime1': [0.4, -0.1, 0.05],
            'phi_regime2': [-0.3, 0.1, -0.05],
            'threshold': 0.0,
            'delay': 1,
            'description': 'SETAR(2;3,3) - AR(3) con d=1'
        },
        {
            'nombre': 'SETAR-6',
            'phi_regime1': [0.5, -0.3, 0.1],
            'phi_regime2': [-0.4, 0.2, -0.05],
            'threshold': 0.5,
            'delay': 2,
            'description': 'SETAR(2;3,3) - AR(3) con d=2'
        },
        {
            'nombre': 'SETAR-7',
            'phi_regime1': [0.3, 0.1],
            'phi_regime2': [-0.2, -0.1],
            'threshold': 0.8,
            'delay': 3,
            'description': 'SETAR(2;2,2) - AR(2) con d=3'
        }
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        
    def _setup_models(self, seed: int):
        """Inicializa los 9 modelos con seed específica."""
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(
                block_length='auto', n_boot=self.n_boot,
                random_state=seed, verbose=False,
                hyperparam_ranges={'block_length': [2, 50]}, optimize=True
            ),
            'Sieve Bootstrap': SieveBootstrapModel(
                order='auto', n_boot=self.n_boot,
                random_state=seed, verbose=False,
                hyperparam_ranges={'order': [1, 20]}, optimize=True
            ),
            'LSPM': LSPM(random_state=seed, verbose=False),
            'LSPMW': LSPMW(rho=0.95, random_state=seed, verbose=False),
            'DeepAR': DeepARModel(
                hidden_size=15, n_lags=5, num_layers=1, dropout=0.1,
                lr=0.01, batch_size=32, epochs=30, num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            ),
            'AREPD': AREPD(
                n_lags=5, rho=0.9, alpha=0.1, poly_degree=2,
                random_state=seed, verbose=False, optimize=True
            ),
            'MCPS': MondrianCPSModel(
                n_lags=10, n_bins=8, test_size=0.25,
                random_state=seed, verbose=False, optimize=True
            ),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(
                n_lags=15, n_pred_bins=8, n_vol_bins=4, volatility_window=20,
                test_size=0.25, random_state=seed, verbose=False, optimize=True
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=20, B=3, units=32, n_layers=2, lr=0.005,
                batch_size=16, epochs=20, num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            )
        }
    
    def _generate_true_distribution(self, simulator: SETARSimulation, 
                                    series_history: np.ndarray,
                                    errors_history: np.ndarray,
                                    n_samples: int = 5000) -> np.ndarray:
        """
        Genera la distribución verdadera del siguiente paso usando el simulador SETAR.
        Para SETAR, el régimen se determina automáticamente según el umbral.
        """
        return simulator.get_true_next_step_samples(
            series_history=series_history,
            errors_history=errors_history,
            n_samples=n_samples
        )
    
    def _optimize_models_con_distribucion_verdadera(self, models: dict, 
                                                     train_calib_series: np.ndarray,
                                                     train_calib_errors: np.ndarray,
                                                     simulator: SETARSimulation):
        """
        PASO 1: Optimiza hiperparámetros usando la distribución verdadera.
        """
        true_distribution = self._generate_true_distribution(
            simulator=simulator,
            series_history=train_calib_series,
            errors_history=train_calib_errors,
            n_samples=5000
        )
        
        if self.verbose:
            print(f"  Distribución verdadera: μ={np.mean(true_distribution):.4f}, "
                  f"σ={np.std(true_distribution):.4f}")
            # Mostrar información del régimen
            regime_props = simulator.get_regime_proportions()
            print(f"  Proporciones de régimen: {regime_props}")
        
        for name, model in models.items():
            try:
                if hasattr(model, 'optimize_hyperparameters'):
                    df_tc = pd.DataFrame({'valor': train_calib_series})
                    model.optimize_hyperparameters(df_tc, true_distribution)
                    
                    if hasattr(model, 'optimize'):
                        model.optimize = False
                        
                    if self.verbose and hasattr(model, 'best_params'):
                        print(f"    {name}: {model.best_params}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"    Error optimizando {name}: {e}")

    def _freeze_all_models(self, models: dict, train_calib_series: np.ndarray):
        """
        PASO 2: CRÍTICO - Congela TODOS los modelos después de optimización.
        """
        if self.verbose:
            print("\n  CONGELANDO MODELOS:")
        
        for name, model in models.items():
            try:
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(train_calib_series)
                    
                    if self.verbose:
                        frozen_attrs = []
                        if hasattr(model, '_frozen_block_length'):
                            frozen_attrs.append(f"block_length={model._frozen_block_length}")
                        if hasattr(model, '_frozen_order'):
                            frozen_attrs.append(f"order={model._frozen_order}")
                        if hasattr(model, '_frozen_mean'):
                            frozen_attrs.append(f"mean={model._frozen_mean:.4f}")
                        if hasattr(model, '_frozen_std'):
                            frozen_attrs.append(f"std={model._frozen_std:.4f}")
                        if hasattr(model, '_is_frozen') and model._is_frozen:
                            frozen_attrs.append("✓ frozen")
                        if hasattr(model, '_fitted_artifacts') and model._fitted_artifacts:
                            frozen_attrs.append("✓ artifacts")
                        if hasattr(model, '_trained_ensemble') and model._trained_ensemble:
                            frozen_attrs.append("✓ ensemble")
                        
                        if frozen_attrs:
                            print(f"    {name}: {', '.join(frozen_attrs)}")
                            
            except Exception as e:
                if self.verbose:
                    print(f"    Error congelando {name}: {e}")

    def _run_single_scenario(self, setar_config: dict, distribution: str, 
                            variance: float, scenario_seed: int) -> list:
        """
        Ejecuta un escenario completo SETAR:
        1. Optimiza hiperparámetros (usa distribución verdadera)
        2. Congela TODOS los parámetros
        3. Predice sin re-estimar
        """
        try:
            total_needed = self.N_TRAIN + self.N_CALIBRATION + self.N_TEST_STEPS
            
            # Crear simulador SETAR
            simulator = SETARSimulation(
                model_type=setar_config['nombre'],
                phi_regime1=setar_config['phi_regime1'],
                phi_regime2=setar_config['phi_regime2'],
                threshold=setar_config['threshold'],
                delay=setar_config['delay'],
                noise_dist=distribution,
                sigma=np.sqrt(variance),
                seed=scenario_seed,
                verbose=False
            )
            
            # Simular serie completa con errores
            full_series, full_errors = simulator.simulate(n=total_needed, burn_in=50)
            
            # Dividir datos
            calib_end = self.N_TRAIN + self.N_CALIBRATION
            train_calib_series = full_series[:calib_end]
            train_calib_errors = full_errors[:calib_end]
            
            # Inicializar modelos
            models = self._setup_models(scenario_seed)
            
            # PASO 1: Optimizar con distribución verdadera
            self._optimize_models_con_distribucion_verdadera(
                models=models,
                train_calib_series=train_calib_series,
                train_calib_errors=train_calib_errors,
                simulator=simulator
            )
            
            # PASO 2: CONGELAR TODOS LOS MODELOS
            self._freeze_all_models(models, train_calib_series)
            
            # PASO 3: Predicción rodante SIN re-estimación
            results_rows = []
            
            for step in range(self.N_TEST_STEPS):
                current_idx = calib_end + step
                history_series = full_series[:current_idx]
                true_value = full_series[current_idx]
                
                step_result = {
                    'Paso': step + 1,
                    'proces_simulacion': setar_config['nombre'],
                    'Descripción': setar_config['description'],
                    'Distribución': distribution,
                    'Varianza error': variance,
                    'Valor_Observado': true_value
                }
                
                for name, model in models.items():
                    try:
                        if isinstance(model, (CircularBlockBootstrapModel, SieveBootstrapModel)):
                            pred_samples = model.fit_predict(history_series)
                        else:
                            df_hist = pd.DataFrame({'valor': history_series})
                            pred_samples = model.fit_predict(df_hist)
                        
                        pred_samples = np.asarray(pred_samples).flatten()
                        crps_val = crps(pred_samples, true_value)
                        step_result[name] = crps_val
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"    {name} error en paso {step+1}: {e}")
                        step_result[name] = np.nan
                
                results_rows.append(step_result)
            
            # Promedio del escenario
            avg_row = {
                'Paso': 'Promedio',
                'proces_simulacion': setar_config['nombre'],
                'Descripción': setar_config['description'],
                'Distribución': distribution,
                'Varianza error': variance,
                'Valor_Observado': np.nan
            }
            
            model_names = list(models.keys())
            for model_name in model_names:
                vals = [r[model_name] for r in results_rows if not pd.isna(r.get(model_name))]
                avg_row[model_name] = np.mean(vals) if vals else np.nan
            
            results_rows.append(avg_row)
            
            del models, simulator
            clear_all_sessions()
            
            return results_rows
            
        except Exception as e:
            if self.verbose:
                print(f"Error en escenario: {e}")
            return []
    
    def generate_all_scenarios(self) -> list:
        """Genera lista de argumentos para ejecución paralela."""
        scenarios = []
        scenario_id = 0
        
        for setar_cfg in self.SETAR_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((
                        setar_cfg.copy(),
                        dist,
                        var,
                        self.seed + scenario_id,
                        self.N_TEST_STEPS,
                        self.N_TRAIN,
                        self.N_CALIBRATION,
                        self.n_boot
                    ))
                    scenario_id += 1
        
        return scenarios
    
    def run_all(self, excel_filename: str = "resultados_140_SETAR.xlsx", 
                batch_size: int = 10):
        """
        Ejecuta los escenarios SETAR con el flujo correcto garantizado.
        """
        print("="*80)
        print("EVALUACIÓN 140 ESCENARIOS SETAR - VERSIÓN CORREGIDA SIN SESGOS")
        print("="*80)
        
        cpu_count = os.cpu_count() or 4
        safe_jobs = min(6, max(1, int(cpu_count * 0.75)))
        
        print(f"⚡ Usando {safe_jobs} núcleos en paralelo")
        print(f"⚡ Tamaño del lote: {batch_size}")
        print(f"\n✓ FLUJO CORRECTO:")
        print("  1. Optimización con distribución verdadera")
        print("  2. Congelamiento de TODOS los parámetros")
        print("  3. Predicción rodante SIN re-estimación")
        print("\n📊 MODELOS SETAR:")
        for cfg in self.SETAR_CONFIGS:
            print(f"  • {cfg['nombre']}: {cfg['description']}")
        print("="*80 + "\n")
        
        all_scenarios = self.generate_all_scenarios()
        total_scenarios = len(all_scenarios)
        all_rows = []
        
        num_batches = (total_scenarios + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_scenarios)
            current_batch = all_scenarios[start_idx:end_idx]
            
            print(f"\n🚀 Lote {i+1}/{num_batches} (Escenarios {start_idx+1} a {end_idx})...")
            
            try:
                batch_results = Parallel(n_jobs=safe_jobs, backend='loky', timeout=99999)(
                    delayed(self._run_scenario_wrapper)(args) 
                    for args in tqdm(current_batch, desc=f"  Lote {i+1}", leave=False)
                )
                
                for res in batch_results:
                    all_rows.extend(res)
                
                self._save_checkpoint(all_rows, excel_filename)
                
                del batch_results, current_batch
                clear_all_sessions()
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error en lote {i+1}: {e}")
                self._save_checkpoint(all_rows, f"RECOVERY_{excel_filename}")
                raise e

        print("\n" + "="*80)
        print(f"✅ COMPLETADO - {len(all_rows)} filas generadas")
        print("="*80)
        
        return pd.DataFrame(all_rows)

    def _run_scenario_wrapper(self, args):
        """Wrapper para ejecución paralela."""
        (setar_cfg, dist, var, seed, n_test_steps, 
         n_train, n_calib, n_boot) = args
        
        pipeline = Pipeline140SinSesgos_SETAR(n_boot=n_boot, seed=seed, verbose=False)
        pipeline.N_TEST_STEPS = n_test_steps
        pipeline.N_TRAIN = n_train
        pipeline.N_CALIBRATION = n_calib
        
        return pipeline._run_single_scenario(setar_cfg, dist, var, seed)

    def _save_checkpoint(self, rows, filename):
        """Guarda progreso en Excel."""
        if not rows: 
            return
        
        df_temp = pd.DataFrame(rows)
        
        ordered_cols = [
            'Paso', 'proces_simulacion', 'Descripción', 'Distribución', 
            'Varianza error', 'Valor_Observado',
            'AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 'EnCQR-LSTM',
            'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap'
        ]
        final_cols = [c for c in ordered_cols if c in df_temp.columns]
        remaining_cols = [c for c in df_temp.columns if c not in final_cols]
        final_cols.extend(remaining_cols)
        
        df_temp = df_temp[final_cols]
        df_temp.to_excel(filename, index=False)

# Agregar a pipeline.py

class TransformadorDiferenciacionIntegracion:
    """
    Maneja la diferenciación e integración de series temporales.
    Mantiene el último valor observado para poder revertir la transformación.
    """
    def __init__(self, d: int = 1, verbose: bool = False):
        """
        Args:
            d: Orden de diferenciación (usualmente 1 para ARIMA)
            verbose: Mostrar información de debug
        """
        self.d = d
        self.verbose = verbose
        self._last_value = None
        self._diff_history = []
    
    def diferenciar_serie(self, serie: np.ndarray) -> np.ndarray:
        """
        Aplica diferenciación de orden d a la serie.
        Guarda el último valor para poder integrar después.
        
        Args:
            serie: Serie temporal original
            
        Returns:
            Serie diferenciada
        """
        if self.d == 0:
            return serie
        
        serie_diff = serie.copy()
        for _ in range(self.d):
            self._last_value = serie_diff[-1]
            serie_diff = np.diff(serie_diff)
        
        self._diff_history.append(self._last_value)
        
        if self.verbose:
            print(f"  Diferenciación d={self.d}: {len(serie)} → {len(serie_diff)} puntos")
            print(f"  Último valor guardado: {self._last_value:.4f}")
        
        return serie_diff
    
    def integrar_predicciones(self, predicciones: np.ndarray, 
                             ultimo_valor_observado: float = None) -> np.ndarray:
        """
        Revierte la diferenciación en las predicciones.
        
        Args:
            predicciones: Predicciones en espacio diferenciado
            ultimo_valor_observado: Último valor de la serie original (opcional)
            
        Returns:
            Predicciones en espacio original
        """
        if self.d == 0:
            return predicciones
        
        # Usar el último valor proporcionado o el guardado
        base_value = ultimo_valor_observado if ultimo_valor_observado is not None else self._last_value
        
        if base_value is None:
            raise ValueError("No hay valor base para integrar. "
                           "Debe llamarse diferenciar_serie() primero o proporcionar ultimo_valor_observado.")
        
        # Integración: Y_t = Y_{t-1} + ΔY_t
        predicciones_integradas = base_value + predicciones
        
        if self.verbose:
            print(f"  Integración: {len(predicciones)} predicciones")
            print(f"  Valor base: {base_value:.4f}")
            print(f"  Rango integrado: [{predicciones_integradas.min():.4f}, "
                  f"{predicciones_integradas.max():.4f}]")
        
        return predicciones_integradas


def _predict_with_model_diferenciacion(name: str, model, history_df: pd.DataFrame,
                                       history_series: np.ndarray,
                                       transformador: TransformadorDiferenciacionIntegracion = None,
                                       verbose: bool = False) -> Dict:
    """
    Wrapper para predicción de modelos con manejo opcional de diferenciación.
    
    Args:
        name: Nombre del modelo
        model: Instancia del modelo
        history_df: DataFrame con historia completa
        history_series: Array numpy con historia completa
        transformador: Transformador de diferenciación (None si no se usa)
        verbose: Mostrar información
        
    Returns:
        Dict con 'samples' (predicciones) y 'error' (si hubo error)
    """
    try:
        # PASO 1: Preparar datos según se use diferenciación o no
        if transformador is not None:
            # CON DIFERENCIACIÓN
            serie_diff = transformador.diferenciar_serie(history_series)
            df_diff = pd.DataFrame({'valor': serie_diff})
            ultimo_valor_original = history_series[-1]
            
            if verbose:
                print(f"  {name}: Prediciendo en espacio diferenciado "
                      f"(n={len(serie_diff)}, último={ultimo_valor_original:.4f})")
            
            # Modelos que trabajan con DataFrame
            if name in ['LSPM', 'LSPMW', 'AREPD', 'DeepAR', 'MCPS', 'AV-MCPS', 'EnCQR-LSTM']:
                samples_diff = model.fit_predict(df_diff)
            # Modelos de bootstrap que trabajan con arrays
            elif name in ['Block Bootstrapping', 'Sieve Bootstrap']:
                samples_diff = model.fit_predict(serie_diff)
            else:
                samples_diff = model.fit_predict(df_diff)
            
            # PASO 2: Integrar predicciones
            samples = transformador.integrar_predicciones(
                samples_diff, 
                ultimo_valor_observado=ultimo_valor_original
            )
            
        else:
            # SIN DIFERENCIACIÓN (comportamiento original)
            if name in ['LSPM', 'LSPMW', 'AREPD', 'DeepAR', 'MCPS', 'AV-MCPS', 'EnCQR-LSTM']:
                samples = model.fit_predict(history_df)
            elif name in ['Block Bootstrapping', 'Sieve Bootstrap']:
                samples = model.fit_predict(history_series)
            else:
                samples = model.fit_predict(history_df)
        
        # Validación de salida
        samples = np.asarray(samples).flatten()
        
        if len(samples) == 0 or np.all(np.isnan(samples)):
            return {'samples': np.array([]), 'error': f'{name}: Predicciones vacías/NaN'}
        
        return {'samples': samples, 'error': None}
        
    except Exception as e:
        if verbose:
            print(f"  Error en {name}: {str(e)}")
        return {'samples': np.array([]), 'error': f'{name}: {str(e)}'}


class Pipeline140ConDiferenciacion_ARIMA:
    """
    Pipeline ARIMA con OPCIÓN de diferenciación adicional.
    
    Este pipeline permite evaluar el impacto de diferenciar ANTES de aplicar
    los modelos de predicción conformal, comparando:
    
    A) SIN diferenciación adicional: Los modelos reciben Y_t directamente
    B) CON diferenciación adicional: Los modelos reciben ΔY_t, predicciones se integran
    
    Filosofía:
    1. Optimización con distribución verdadera (en espacio correspondiente)
    2. Congelamiento de parámetros
    3. Predicción rodante sin re-estimación
    """
    
    N_TEST_STEPS = 12
    N_CALIBRATION = 40
    N_TRAIN = 200
    
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
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False,
                 usar_diferenciacion: bool = False):
        """
        Args:
            n_boot: Número de muestras bootstrap
            seed: Semilla aleatoria
            verbose: Mostrar información detallada
            usar_diferenciacion: Si True, diferencia ANTES de aplicar modelos
        """
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.usar_diferenciacion = usar_diferenciacion
        self.rng = np.random.default_rng(seed)
    
    def _setup_models(self, seed: int):
        """Inicializa los 9 modelos."""
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(
                block_length='auto', n_boot=self.n_boot,
                random_state=seed, verbose=False,
                hyperparam_ranges={'block_length': [2, 50]}, optimize=True
            ),
            'Sieve Bootstrap': SieveBootstrapModel(
                order='auto', n_boot=self.n_boot,
                random_state=seed, verbose=False,
                hyperparam_ranges={'order': [1, 20]}, optimize=True
            ),
            'LSPM': LSPM(random_state=seed, verbose=False),
            'LSPMW': LSPMW(rho=0.95, random_state=seed, verbose=False),
            'DeepAR': DeepARModel(
                hidden_size=15, n_lags=5, num_layers=1, dropout=0.1,
                lr=0.01, batch_size=32, epochs=30, num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            ),
            'AREPD': AREPD(
                n_lags=5, rho=0.9, alpha=0.1, poly_degree=2,
                random_state=seed, verbose=False, optimize=True
            ),
            'MCPS': MondrianCPSModel(
                n_lags=10, n_bins=8, test_size=0.25,
                random_state=seed, verbose=False, optimize=True
            ),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(
                n_lags=15, n_pred_bins=8, n_vol_bins=4, volatility_window=20,
                test_size=0.25, random_state=seed, verbose=False, optimize=True
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=20, B=3, units=32, n_layers=2, lr=0.005,
                batch_size=16, epochs=20, num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            )
        }
    
    def _generate_true_distribution(self, simulator: ARIMASimulation,
                                    series_history: np.ndarray,
                                    errors_history: np.ndarray,
                                    transformador: TransformadorDiferenciacionIntegracion = None,
                                    n_samples: int = 5000) -> np.ndarray:
        """
        Genera distribución verdadera del siguiente paso.
        
        Si usar_diferenciacion=True:
            - La distribución teórica se calcula en espacio diferenciado
            - Esto es correcto porque ARIMA(p,1,q) modela ΔY_t
        """
        # La distribución teórica SIEMPRE se calcula en el espacio original
        # porque get_true_next_step_samples() maneja la integración internamente
        true_samples = simulator.get_true_next_step_samples(
            series_history=series_history,
            errors_history=errors_history,
            n_samples=n_samples
        )
        
        # Si usamos diferenciación, convertimos a espacio diferenciado
        if transformador is not None:
            ultimo_valor = series_history[-1]
            true_samples_diff = true_samples - ultimo_valor
            return true_samples_diff
        
        return true_samples
    
    def _optimize_models_con_distribucion_verdadera(self, models: dict,
                                                     train_calib_series: np.ndarray,
                                                     train_calib_errors: np.ndarray,
                                                     simulator: ARIMASimulation,
                                                     transformador: TransformadorDiferenciacionIntegracion = None):
        """
        PASO 1: Optimización de hiperparámetros.
        """
        # Generar distribución teórica (en espacio adecuado)
        true_distribution = self._generate_true_distribution(
            simulator=simulator,
            series_history=train_calib_series,
            errors_history=train_calib_errors,
            transformador=transformador,
            n_samples=5000
        )
        
        if self.verbose:
            espacio = "diferenciado" if transformador is not None else "original"
            print(f"  Distribución verdadera ({espacio}): "
                  f"μ={np.mean(true_distribution):.4f}, "
                  f"σ={np.std(true_distribution):.4f}")
        
        # Preparar datos para optimización
        if transformador is not None:
            serie_diff = transformador.diferenciar_serie(train_calib_series)
            df_tc = pd.DataFrame({'valor': serie_diff})
        else:
            df_tc = pd.DataFrame({'valor': train_calib_series})
        
        for name, model in models.items():
            try:
                if hasattr(model, 'optimize_hyperparameters'):
                    model.optimize_hyperparameters(df_tc, true_distribution)
                    
                    if hasattr(model, 'optimize'):
                        model.optimize = False
                    
                    if self.verbose and hasattr(model, 'best_params'):
                        print(f"    {name}: {model.best_params}")
            except Exception as e:
                if self.verbose:
                    print(f"    Error optimizando {name}: {e}")
    
    def _freeze_all_models(self, models: dict, train_calib_series: np.ndarray,
                          transformador: TransformadorDiferenciacionIntegracion = None):
        """
        PASO 2: Congelamiento de modelos.
        """
        if self.verbose:
            espacio = "diferenciado" if transformador is not None else "original"
            print(f"\n  CONGELANDO MODELOS (espacio {espacio}):")
        
        # Preparar datos para freeze
        if transformador is not None:
            serie_diff = transformador.diferenciar_serie(train_calib_series)
            freeze_data = serie_diff
        else:
            freeze_data = train_calib_series
        
        for name, model in models.items():
            try:
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(freeze_data)
                    
                    if self.verbose:
                        frozen_attrs = []
                        if hasattr(model, '_frozen_block_length'):
                            frozen_attrs.append(f"block_length={model._frozen_block_length}")
                        if hasattr(model, '_frozen_order'):
                            frozen_attrs.append(f"order={model._frozen_order}")
                        if hasattr(model, '_frozen_mean'):
                            frozen_attrs.append(f"mean={model._frozen_mean:.4f}")
                        if hasattr(model, '_frozen_std'):
                            frozen_attrs.append(f"std={model._frozen_std:.4f}")
                        if hasattr(model, '_is_frozen') and model._is_frozen:
                            frozen_attrs.append("✓ frozen")
                        
                        if frozen_attrs:
                            print(f"    {name}: {', '.join(frozen_attrs)}")
            except Exception as e:
                if self.verbose:
                    print(f"    Error congelando {name}: {e}")
    
    def _run_single_scenario(self, arima_config: dict, distribution: str,
                            variance: float, scenario_seed: int) -> list:
        """
        Ejecuta un escenario completo.
        """
        try:
            total_needed = self.N_TRAIN + self.N_CALIBRATION + self.N_TEST_STEPS
            
            # Crear simulador
            simulator = ARIMASimulation(
                model_type=arima_config['nombre'],
                phi=arima_config['phi'],
                theta=arima_config['theta'],
                noise_dist=distribution,
                sigma=np.sqrt(variance),
                seed=scenario_seed,
                verbose=False
            )
            
            # Simular serie
            full_series, full_errors = simulator.simulate(n=total_needed, burn_in=50)
            
            # Dividir datos
            calib_end = self.N_TRAIN + self.N_CALIBRATION
            train_calib_series = full_series[:calib_end]
            train_calib_errors = full_errors[:calib_end]
            
            # Crear transformador si se usa diferenciación
            transformador_optim = TransformadorDiferenciacionIntegracion(
                d=1, verbose=False
            ) if self.usar_diferenciacion else None
            
            # Inicializar modelos
            models = self._setup_models(scenario_seed)
            
            # PASO 1: Optimizar
            self._optimize_models_con_distribucion_verdadera(
                models=models,
                train_calib_series=train_calib_series,
                train_calib_errors=train_calib_errors,
                simulator=simulator,
                transformador=transformador_optim
            )
            
            # PASO 2: Congelar
            self._freeze_all_models(
                models=models,
                train_calib_series=train_calib_series,
                transformador=transformador_optim
            )
            
            # PASO 3: Predicción rodante
            results_rows = []
            
            for step in range(self.N_TEST_STEPS):
                current_idx = calib_end + step
                history_series = full_series[:current_idx]
                true_value = full_series[current_idx]
                
                # Transformador fresco para este paso
                step_transformador = TransformadorDiferenciacionIntegracion(
                    d=1, verbose=False
                ) if self.usar_diferenciacion else None
                
                step_result = {
                    'Paso': step + 1,
                    'proces_simulacion': arima_config['nombre'],
                    'Distribución': distribution,
                    'Varianza error': variance,
                    'Valor_Observado': true_value,
                    'Con_Diferenciacion': 'Sí' if self.usar_diferenciacion else 'No'
                }
                
                for name, model in models.items():
                    try:
                        result = _predict_with_model_diferenciacion(
                            name=name,
                            model=model,
                            history_df=pd.DataFrame({'valor': history_series}),
                            history_series=history_series,
                            transformador=step_transformador,
                            verbose=False
                        )
                        
                        if result['samples'].size > 0 and result['error'] is None:
                            crps_val = crps(result['samples'], true_value)
                            step_result[name] = crps_val
                        else:
                            step_result[name] = np.nan
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"    {name} error en paso {step+1}: {e}")
                        step_result[name] = np.nan
                
                results_rows.append(step_result)
            
            # Promedio del escenario
            avg_row = {
                'Paso': 'Promedio',
                'proces_simulacion': arima_config['nombre'],
                'Distribución': distribution,
                'Varianza error': variance,
                'Valor_Observado': np.nan,
                'Con_Diferenciacion': 'Sí' if self.usar_diferenciacion else 'No'
            }
            
            model_names = list(models.keys())
            for model_name in model_names:
                vals = [r[model_name] for r in results_rows 
                       if not pd.isna(r.get(model_name))]
                avg_row[model_name] = np.mean(vals) if vals else np.nan
            
            results_rows.append(avg_row)
            
            del models, simulator
            clear_all_sessions()
            
            return results_rows
            
        except Exception as e:
            if self.verbose:
                print(f"Error en escenario: {e}")
            return []
    
    def generate_all_scenarios(self) -> list:
        """Genera lista de escenarios."""
        scenarios = []
        scenario_id = 0
        
        for arima_cfg in self.ARIMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((
                        arima_cfg.copy(),
                        dist,
                        var,
                        self.seed + scenario_id,
                        self.N_TEST_STEPS,
                        self.N_TRAIN,
                        self.N_CALIBRATION,
                        self.n_boot,
                        self.usar_diferenciacion
                    ))
                    scenario_id += 1
        
        return scenarios
    
    def run_all(self, excel_filename: str, batch_size: int = 10):
        """
        Ejecuta todos los escenarios.
        """
        diferenciacion_str = "CON" if self.usar_diferenciacion else "SIN"
        print("="*80)
        print(f"EVALUACIÓN 140 ESCENARIOS ARIMA - {diferenciacion_str} DIFERENCIACIÓN ADICIONAL")
        print("="*80)
        
        cpu_count = os.cpu_count() or 4
        safe_jobs = min(6, max(1, int(cpu_count * 0.75)))
        
        print(f"⚡ Usando {safe_jobs} núcleos en paralelo")
        print(f"⚡ Tamaño del lote: {batch_size}")
        print(f"⚡ Diferenciación adicional: {diferenciacion_str}")
        print(f"\n✓ FLUJO CORRECTO:")
        print("  1. Optimización con distribución verdadera")
        print("  2. Congelamiento de TODOS los parámetros")
        print("  3. Predicción rodante SIN re-estimación")
        if self.usar_diferenciacion:
            print("\n📊 DIFERENCIACIÓN:")
            print("  • Los modelos trabajan con ΔY_t (incrementos)")
            print("  • Las predicciones se integran de vuelta a Y_t")
        print("="*80 + "\n")
        
        all_scenarios = self.generate_all_scenarios()
        total_scenarios = len(all_scenarios)
        all_rows = []
        
        num_batches = (total_scenarios + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_scenarios)
            current_batch = all_scenarios[start_idx:end_idx]
            
            print(f"\n🚀 Lote {i+1}/{num_batches} (Escenarios {start_idx+1} a {end_idx})...")
            
            try:
                batch_results = Parallel(n_jobs=safe_jobs, backend='loky', timeout=99999)(
                    delayed(self._run_scenario_wrapper)(args)
                    for args in tqdm(current_batch, desc=f"  Lote {i+1}", leave=False)
                )
                
                for res in batch_results:
                    all_rows.extend(res)
                
                self._save_checkpoint(all_rows, excel_filename)
                
                del batch_results, current_batch
                clear_all_sessions()
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error en lote {i+1}: {e}")
                self._save_checkpoint(all_rows, f"RECOVERY_{excel_filename}")
                raise e
        
        print("\n" + "="*80)
        print(f"✅ COMPLETADO - {len(all_rows)} filas generadas")
        print("="*80)
        
        return pd.DataFrame(all_rows)
    
    def _run_scenario_wrapper(self, args):
        """Wrapper para paralelización."""
        (arima_cfg, dist, var, seed, n_test_steps,
         n_train, n_calib, n_boot, usar_diff) = args
        
        pipeline = Pipeline140ConDiferenciacion_ARIMA(
            n_boot=n_boot, seed=seed, verbose=False,
            usar_diferenciacion=usar_diff
        )
        pipeline.N_TEST_STEPS = n_test_steps
        pipeline.N_TRAIN = n_train
        pipeline.N_CALIBRATION = n_calib
        
        return pipeline._run_single_scenario(arima_cfg, dist, var, seed)
    
    def _save_checkpoint(self, rows, filename):
        """Guarda progreso."""
        if not rows:
            return
        
        df_temp = pd.DataFrame(rows)
        
        ordered_cols = [
            'Paso', 'proces_simulacion', 'Distribución', 'Varianza error',
            'Con_Diferenciacion', 'Valor_Observado',
            'AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 'EnCQR-LSTM',
            'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap'
        ]
        final_cols = [c for c in ordered_cols if c in df_temp.columns]
        remaining_cols = [c for c in df_temp.columns if c not in final_cols]
        final_cols.extend(remaining_cols)
        
        df_temp = df_temp[final_cols]
        df_temp.to_excel(filename, index=False)

# ============================================================================
# CLASES PARA AGREGAR AL FINAL DE pipeline.py
# ============================================================================
# Evalúa ARIMA(p,d,q) con d=1,...,10 en DOS MODALIDADES SIMULTÁNEAS:
# A) Proceso SIN diferenciación adicional (modelos reciben Y_t directamente)
# B) Proceso CON diferenciación adicional (modelos reciben ΔY_t, se integra después)
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
    Pipeline que evalúa procesos ARIMA(p,d,q) con d=1,...,10 en DOS MODALIDADES:
    
    MODALIDAD A (Sin diferenciación adicional):
    - Los modelos reciben Y_t directamente
    - Optimización y predicción en espacio original
    
    MODALIDAD B (Con diferenciación adicional):
    - Los modelos reciben ΔY_t (serie diferenciada)
    - Optimización en espacio diferenciado
    - Predicciones se integran de vuelta
    
    Para cada (d, ARMA_config, distribución, varianza):
    - Ejecuta AMBAS modalidades
    - Compara cuál funciona mejor
    
    Total: 10 (d) × 7 (ARMA) × 5 (dist) × 4 (var) × 2 (modalidades) = 2,800 filas
    """
    
    N_TEST_STEPS = 12
    N_CALIBRATION = 40
    N_TRAIN = 200
    
    # Configuraciones base ARMA(p,q)
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
    D_VALUES = list(range(1, 11))  # d = 1, 2, ..., 10
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
    
    def _setup_models(self, seed: int):
        """Inicializa los 9 modelos."""
        return {
            'Block Bootstrapping': CircularBlockBootstrapModel(
                block_length='auto', n_boot=self.n_boot,
                random_state=seed, verbose=False,
                hyperparam_ranges={'block_length': [2, 50]}, optimize=True
            ),
            'Sieve Bootstrap': SieveBootstrapModel(
                order='auto', n_boot=self.n_boot,
                random_state=seed, verbose=False,
                hyperparam_ranges={'order': [1, 20]}, optimize=True
            ),
            'LSPM': LSPM(random_state=seed, verbose=False),
            'LSPMW': LSPMW(rho=0.95, random_state=seed, verbose=False),
            'DeepAR': DeepARModel(
                hidden_size=15, n_lags=5, num_layers=1, dropout=0.1,
                lr=0.01, batch_size=32, epochs=30, num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            ),
            'AREPD': AREPD(
                n_lags=5, rho=0.9, alpha=0.1, poly_degree=2,
                random_state=seed, verbose=False, optimize=True
            ),
            'MCPS': MondrianCPSModel(
                n_lags=10, n_bins=8, test_size=0.25,
                random_state=seed, verbose=False, optimize=True
            ),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(
                n_lags=15, n_pred_bins=8, n_vol_bins=4, volatility_window=20,
                test_size=0.25, random_state=seed, verbose=False, optimize=True
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=20, B=3, units=32, n_layers=2, lr=0.005,
                batch_size=16, epochs=20, num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            )
        }
    
    def _generate_true_distribution(self, simulator: ARIMAMultiDSimulation,
                                    series_history: np.ndarray,
                                    errors_history: np.ndarray,
                                    transformador: TransformadorDiferenciacionIntegracion = None,
                                    n_samples: int = 5000) -> np.ndarray:
        """
        Genera distribución verdadera del siguiente paso.
        
        Si transformador es None: distribución en espacio original Y_t
        Si transformador existe: distribución en espacio diferenciado ΔY_t
        """
        # Distribución teórica siempre se calcula en espacio original
        true_samples = simulator.get_true_next_step_samples(
            series_history=series_history,
            errors_history=errors_history,
            n_samples=n_samples
        )
        
        # Si usamos diferenciación, convertir a espacio diferenciado
        if transformador is not None:
            ultimo_valor = series_history[-1]
            true_samples_diff = true_samples - ultimo_valor
            return true_samples_diff
        
        return true_samples
    
    def _optimize_models_con_distribucion_verdadera(self, models: dict,
                                                     train_calib_series: np.ndarray,
                                                     train_calib_errors: np.ndarray,
                                                     simulator: ARIMAMultiDSimulation,
                                                     transformador: TransformadorDiferenciacionIntegracion = None):
        """PASO 1: Optimización de hiperparámetros."""
        true_distribution = self._generate_true_distribution(
            simulator=simulator,
            series_history=train_calib_series,
            errors_history=train_calib_errors,
            transformador=transformador,
            n_samples=5000
        )
        
        if self.verbose:
            espacio = "diferenciado" if transformador is not None else "original"
            print(f"  Distribución verdadera ({espacio}): "
                  f"μ={np.mean(true_distribution):.4f}, "
                  f"σ={np.std(true_distribution):.4f}")
        
        # Preparar datos
        if transformador is not None:
            serie_diff = transformador.diferenciar_serie(train_calib_series)
            df_tc = pd.DataFrame({'valor': serie_diff})
        else:
            df_tc = pd.DataFrame({'valor': train_calib_series})
        
        for name, model in models.items():
            try:
                if hasattr(model, 'optimize_hyperparameters'):
                    model.optimize_hyperparameters(df_tc, true_distribution)
                    if hasattr(model, 'optimize'):
                        model.optimize = False
            except Exception as e:
                if self.verbose:
                    print(f"    Error optimizando {name}: {e}")
    
    def _freeze_all_models(self, models: dict, train_calib_series: np.ndarray,
                          transformador: TransformadorDiferenciacionIntegracion = None):
        """PASO 2: Congelamiento de modelos."""
        if transformador is not None:
            serie_diff = transformador.diferenciar_serie(train_calib_series)
            freeze_data = serie_diff
        else:
            freeze_data = train_calib_series
        
        for name, model in models.items():
            try:
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(freeze_data)
            except Exception as e:
                if self.verbose:
                    print(f"    Error congelando {name}: {e}")
    
    def _run_single_scenario_doble_modalidad(self, arma_config: dict, d_value: int,
                                             distribution: str, variance: float,
                                             scenario_seed: int) -> list:
        """
        Ejecuta un escenario completo en AMBAS modalidades.
        
        Returns:
            Lista con 2 conjuntos de resultados:
            - 1 conjunto para modalidad SIN diferenciación
            - 1 conjunto para modalidad CON diferenciación
        """
        all_results = []
        
        # Generar datos UNA SOLA VEZ
        try:
            total_needed = self.N_TRAIN + self.N_CALIBRATION + self.N_TEST_STEPS
            
            simulator = ARIMAMultiDSimulation(
                phi=arma_config['phi'],
                theta=arma_config['theta'],
                d=d_value,
                noise_dist=distribution,
                sigma=np.sqrt(variance),
                seed=scenario_seed,
                verbose=False
            )
            
            full_series, full_errors = simulator.simulate(n=total_needed, burn_in=50)
            calib_end = self.N_TRAIN + self.N_CALIBRATION
            train_calib_series = full_series[:calib_end]
            train_calib_errors = full_errors[:calib_end]
            
        except Exception as e:
            if self.verbose:
                print(f"Error generando datos: {e}")
            return []
        
        # ============================================================
        # MODALIDAD A: SIN DIFERENCIACIÓN ADICIONAL
        # ============================================================
        results_A = self._run_single_modalidad(
            arma_config=arma_config,
            d_value=d_value,
            distribution=distribution,
            variance=variance,
            scenario_seed=scenario_seed,
            full_series=full_series,
            full_errors=full_errors,
            train_calib_series=train_calib_series,
            train_calib_errors=train_calib_errors,
            simulator=simulator,
            usar_diferenciacion=False
        )
        all_results.extend(results_A)
        
        # ============================================================
        # MODALIDAD B: CON DIFERENCIACIÓN ADICIONAL
        # ============================================================
        results_B = self._run_single_modalidad(
            arma_config=arma_config,
            d_value=d_value,
            distribution=distribution,
            variance=variance,
            scenario_seed=scenario_seed + 100000,  # Seed diferente para independencia
            full_series=full_series,
            full_errors=full_errors,
            train_calib_series=train_calib_series,
            train_calib_errors=train_calib_errors,
            simulator=simulator,
            usar_diferenciacion=True
        )
        all_results.extend(results_B)
        
        del simulator
        clear_all_sessions()
        
        return all_results
    
    def _run_single_modalidad(self, arma_config: dict, d_value: int,
                             distribution: str, variance: float, scenario_seed: int,
                             full_series: np.ndarray, full_errors: np.ndarray,
                             train_calib_series: np.ndarray, train_calib_errors: np.ndarray,
                             simulator: ARIMAMultiDSimulation,
                             usar_diferenciacion: bool) -> list:
        """Ejecuta una modalidad específica."""
        try:
            calib_end = self.N_TRAIN + self.N_CALIBRATION
            
            # Crear transformador si es necesario
            transformador_optim = TransformadorDiferenciacionIntegracion(
                d=1, verbose=False
            ) if usar_diferenciacion else None
            
            # Inicializar modelos
            models = self._setup_models(scenario_seed)
            
            # PASO 1: Optimizar
            self._optimize_models_con_distribucion_verdadera(
                models=models,
                train_calib_series=train_calib_series,
                train_calib_errors=train_calib_errors,
                simulator=simulator,
                transformador=transformador_optim
            )
            
            # PASO 2: Congelar
            self._freeze_all_models(
                models=models,
                train_calib_series=train_calib_series,
                transformador=transformador_optim
            )
            
            # PASO 3: Predicción rodante
            results_rows = []
            p = len(arma_config['phi'])
            q = len(arma_config['theta'])
            model_name = f"ARIMA({p},{d_value},{q})"
            modalidad_str = "CON_DIFF" if usar_diferenciacion else "SIN_DIFF"
            
            for step in range(self.N_TEST_STEPS):
                current_idx = calib_end + step
                history_series = full_series[:current_idx]
                true_value = full_series[current_idx]
                
                step_transformador = TransformadorDiferenciacionIntegracion(
                    d=1, verbose=False
                ) if usar_diferenciacion else None
                
                step_result = {
                    'Paso': step + 1,
                    'Proceso': model_name,
                    'p': p,
                    'd': d_value,
                    'q': q,
                    'ARMA_base': arma_config['nombre'],
                    'Distribución': distribution,
                    'Varianza': variance,
                    'Modalidad': modalidad_str,
                    'Valor_Observado': true_value
                }
                
                for name, model in models.items():
                    try:
                        result = _predict_with_model_diferenciacion(
                            name=name,
                            model=model,
                            history_df=pd.DataFrame({'valor': history_series}),
                            history_series=history_series,
                            transformador=step_transformador,
                            verbose=False
                        )
                        
                        if result['samples'].size > 0 and result['error'] is None:
                            crps_val = crps(result['samples'], true_value)
                            step_result[name] = crps_val
                        else:
                            step_result[name] = np.nan
                    except Exception as e:
                        if self.verbose:
                            print(f"    {name} error: {e}")
                        step_result[name] = np.nan
                
                results_rows.append(step_result)
            
            # Promedio
            avg_row = {
                'Paso': 'Promedio',
                'Proceso': model_name,
                'p': p,
                'd': d_value,
                'q': q,
                'ARMA_base': arma_config['nombre'],
                'Distribución': distribution,
                'Varianza': variance,
                'Modalidad': modalidad_str,
                'Valor_Observado': np.nan
            }
            
            model_names = list(models.keys())
            for model_name_iter in model_names:
                vals = [r[model_name_iter] for r in results_rows 
                       if not pd.isna(r.get(model_name_iter))]
                avg_row[model_name_iter] = np.mean(vals) if vals else np.nan
            
            results_rows.append(avg_row)
            
            del models
            return results_rows
            
        except Exception as e:
            if self.verbose:
                print(f"Error en modalidad: {e}")
            return []
    
    def generate_all_scenarios(self) -> list:
        """
        Genera lista de escenarios.
        Cada escenario se ejecutará en AMBAS modalidades.
        """
        scenarios = []
        scenario_id = 0
        
        for d_val in self.D_VALUES:
            for arma_cfg in self.ARMA_CONFIGS:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        scenarios.append((
                            arma_cfg.copy(),
                            d_val,
                            dist,
                            var,
                            self.seed + scenario_id
                        ))
                        scenario_id += 1
        
        return scenarios
    
    def run_all(self, excel_filename: str = "resultados_ARIMA_d1_a_d10_DOBLE_MODALIDAD.xlsx",
                batch_size: int = 20):
        """Ejecuta todos los escenarios en ambas modalidades."""
        print("="*80)
        print("EVALUACIÓN ARIMA d=1,...,10 - DOBLE MODALIDAD (SIN_DIFF + CON_DIFF)")
        print("="*80)
        
        cpu_count = os.cpu_count() or 4
        safe_jobs = min(6, max(1, int(cpu_count * 0.75)))
        
        total_base_scenarios = len(self.D_VALUES) * len(self.ARMA_CONFIGS) * len(self.DISTRIBUTIONS) * len(self.VARIANCES)
        total_rows_expected = total_base_scenarios * 2 * (self.N_TEST_STEPS + 1)  # ×2 modalidades
        
        print(f"⚡ Usando {safe_jobs} núcleos en paralelo")
        print(f"⚡ Tamaño del lote: {batch_size}")
        print(f"\n📊 CONFIGURACIÓN:")
        print(f"  • Valores de d: {self.D_VALUES}")
        print(f"  • Configuraciones ARMA: {len(self.ARMA_CONFIGS)}")
        print(f"  • Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"  • Varianzas: {len(self.VARIANCES)}")
        print(f"  • Modalidades: 2 (SIN_DIFF + CON_DIFF)")
        print(f"  • ESCENARIOS BASE: {total_base_scenarios}")
        print(f"  • FILAS TOTALES ESPERADAS: ~{total_rows_expected}")
        print("="*80 + "\n")
        
        all_scenarios = self.generate_all_scenarios()
        total_scenarios = len(all_scenarios)
        all_rows = []
        
        num_batches = (total_scenarios + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_scenarios)
            current_batch = all_scenarios[start_idx:end_idx]
            
            print(f"\n🚀 Lote {i+1}/{num_batches} (Escenarios base {start_idx+1} a {end_idx})...")
            
            try:
                batch_results = Parallel(n_jobs=safe_jobs, backend='loky', timeout=99999)(
                    delayed(self._run_scenario_wrapper)(args)
                    for args in tqdm(current_batch, desc=f"  Lote {i+1}", leave=False)
                )
                
                for res in batch_results:
                    all_rows.extend(res)
                
                self._save_checkpoint(all_rows, excel_filename)
                
                del batch_results, current_batch
                clear_all_sessions()
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error en lote {i+1}: {e}")
                self._save_checkpoint(all_rows, f"RECOVERY_{excel_filename}")
                raise e
        
        print("\n" + "="*80)
        print(f"✅ COMPLETADO - {len(all_rows)} filas generadas")
        print("="*80)
        
        return pd.DataFrame(all_rows)
    
    def _run_scenario_wrapper(self, args):
        """Wrapper para paralelización."""
        arma_cfg, d_val, dist, var, seed = args
        
        pipeline = PipelineARIMA_MultiD_DobleModalidad(
            n_boot=self.n_boot, seed=seed, verbose=False
        )
        pipeline.N_TEST_STEPS = self.N_TEST_STEPS
        pipeline.N_TRAIN = self.N_TRAIN
        pipeline.N_CALIBRATION = self.N_CALIBRATION
        
        return pipeline._run_single_scenario_doble_modalidad(
            arma_cfg, d_val, dist, var, seed
        )
    
    def _save_checkpoint(self, rows, filename):
        """Guarda progreso en Excel."""
        if not rows:
            return
        
        df_temp = pd.DataFrame(rows)
        
        ordered_cols = [
            'Paso', 'Proceso', 'p', 'd', 'q', 'ARMA_base',
            'Distribución', 'Varianza', 'Modalidad', 'Valor_Observado',
            'AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 'EnCQR-LSTM',
            'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap'
        ]
        final_cols = [c for c in ordered_cols if c in df_temp.columns]
        remaining_cols = [c for c in df_temp.columns if c not in final_cols]
        final_cols.extend(remaining_cols)
        
        df_temp = df_temp[final_cols]
        df_temp.to_excel(filename, index=False)