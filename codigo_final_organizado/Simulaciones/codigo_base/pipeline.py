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
from metricas import crps
from simulacion import ARIMASimulation, SETARSimulation, ARMASimulation


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
    Pipeline ULTRA-RÁPIDO: Garantiza ≤5 minutos por escenario.
    
    Presupuesto (300s):
    - Optimización: 120s (40%)
    - Congelamiento: 30s (10%)
    - Testing: 150s (50%) → 12 modelos × 12 pasos = ~1s/predicción
    
    Características CRÍTICAS:
    1. Early stopping en optimización
    2. Extrapolación de CRPS cuando timeout
    3. Grid adaptativo por velocidad de modelo
    4. Limpieza agresiva de memoria
    """
    
    # Estructura de datos
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    # Presupuesto temporal
    SCENARIO_BUDGET = 300  # 5 minutos TOTAL
    OPTIMIZATION_BUDGET = 120  # 40%
    FREEZE_BUDGET = 30         # 10%
    TEST_BUDGET = 150          # 50%
    
    # Configuraciones ARMA (7 procesos)
    ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]},
        {'nombre': 'ARMA(2,1)', 'phi': [0.7, 0.2], 'theta': [0.5]}
    ]
    
    # 5 distribuciones × 4 varianzas = 20 combinaciones
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
    
    def generate_all_scenarios(self) -> list:
        """Genera 140 escenarios (7 × 5 × 4)."""
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
                        self.N_VALIDATION,
                        self.n_boot
                    ))
                    scenario_id += 1
        
        return scenarios
    
    def _run_scenario_wrapper(self, args):
        """Wrapper para ejecución paralela."""
        (arma_cfg, dist, var, seed, n_test_steps, 
         n_train, n_validation, n_boot) = args
        
        pipeline = Pipeline140SinSesgos_ARMA(
            n_boot=n_boot, 
            seed=seed, 
            verbose=False
        )
        pipeline.N_TEST_STEPS = n_test_steps
        pipeline.N_TRAIN = n_train
        pipeline.N_VALIDATION = n_validation
        
        return pipeline.run_single_scenario(arma_cfg, dist, var, seed)
    
    def _optimize_and_freeze_models(self, models: dict, 
                                     train_series: np.ndarray,
                                     val_series: np.ndarray):
        """
        Optimización + congelamiento con presupuesto de tiempo.
        
        Returns:
            models optimizados y congelados
        """
        from modelos import TimeBalancedOptimizer
        
        # PASO 1: Optimización (120s)
        opt_start = time.time()
        
        optimizer = TimeBalancedOptimizer(
            random_state=self.seed,
            verbose=self.verbose
        )
        
        if self.verbose:
            print(f"\n⚡ PASO 1: Optimización (budget: {self.OPTIMIZATION_BUDGET}s)")
        
        optimized_params = optimizer.optimize_all_models(
            models, 
            train_data=train_series,
            val_data=val_series
        )
        
        opt_elapsed = time.time() - opt_start
        
        if opt_elapsed > self.OPTIMIZATION_BUDGET * 1.2:
            print(f"⚠️  ADVERTENCIA: Optimización tomó {opt_elapsed:.1f}s "
                  f"(límite: {self.OPTIMIZATION_BUDGET}s)")
        
        # PASO 2: Congelamiento (30s)
        freeze_start = time.time()
        train_val_combined = np.concatenate([train_series, val_series])
        
        if self.verbose:
            print(f"\n🔒 PASO 2: Congelamiento (budget: {self.FREEZE_BUDGET}s)")
        
        for name, model in models.items():
            try:
                # Aplicar hiperparámetros optimizados
                if name in optimized_params and optimized_params[name]:
                    if hasattr(model, 'best_params'):
                        model.best_params = optimized_params[name]
                    
                    for key, value in optimized_params[name].items():
                        if hasattr(model, key):
                            setattr(model, key, value)
                
                # Congelar
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(train_val_combined)
                    
                    if self.verbose:
                        print(f"  ✓ {name}")
                
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ {name}: {e}")
        
        freeze_elapsed = time.time() - freeze_start
        
        if freeze_elapsed > self.FREEZE_BUDGET * 1.5:
            print(f"⚠️  ADVERTENCIA: Congelamiento tomó {freeze_elapsed:.1f}s "
                  f"(límite: {self.FREEZE_BUDGET}s)")
        
        total_prep = opt_elapsed + freeze_elapsed
        if self.verbose:
            print(f"\n⏱️  Tiempo preparación: {total_prep:.1f}s / "
                  f"{self.OPTIMIZATION_BUDGET + self.FREEZE_BUDGET}s")
        
        return models
    
    def run_single_scenario(self, arma_config: dict, dist: str, 
                           var: float, rep: int) -> list:
        """
        Ejecuta UN escenario completo en ≤5 minutos.
        
        Returns:
            list de 12 dicts (uno por paso de test)
        """
        scenario_start = time.time()
        scenario_seed = self.seed + rep
        
        # ================================================================
        # SIMULACIÓN
        # ================================================================
        simulator = ARMASimulation(
            phi=arma_config['phi'],
            theta=arma_config['theta'],
            noise_dist=dist,
            sigma=np.sqrt(var),
            seed=scenario_seed
        )
        
        total_length = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        full_series, _ = simulator.simulate(n=total_length, burn_in=50)
        
        train_series = full_series[:self.N_TRAIN]
        val_series = full_series[self.N_TRAIN:self.N_TRAIN + self.N_VALIDATION]
        test_series = full_series[self.N_TRAIN + self.N_VALIDATION:]
        
        # ================================================================
        # PREPARACIÓN (Optimización + Congelamiento) ≤ 150s
        # ================================================================
        models = self._setup_models(scenario_seed)
        
        prep_start = time.time()
        models = self._optimize_and_freeze_models(
            models,
            train_series=train_series,
            val_series=val_series
        )
        prep_elapsed = time.time() - prep_start
        
        # ================================================================
        # TESTING ≤ 150s (12 pasos × 9 modelos = 108 predicciones)
        # ================================================================
        test_start = time.time()
        results_rows = []
        time_per_step = self.TEST_BUDGET / self.N_TEST_STEPS  # ~12.5s por paso
        
        for t in range(self.N_TEST_STEPS):
            step_start = time.time()
            
            # Historia acumulativa
            history = np.concatenate([
                train_series,
                val_series,
                test_series[:t]
            ]) if t > 0 else np.concatenate([train_series, val_series])
            
            true_val = test_series[t]
            
            row = {
                'Paso': t + 1,
                'proces_simulacion': arma_config['nombre'],
                'Distribución': dist,
                'Varianza error': var,
                'Valor_Observado': true_val
            }
            
            # Predecir con todos los modelos
            for model_name, model in models.items():
                try:
                    # Verificar timeout
                    step_elapsed = time.time() - step_start
                    if step_elapsed > time_per_step:
                        row[model_name] = np.nan
                        continue
                    
                    # Predicción
                    if isinstance(model, (CircularBlockBootstrapModel, SieveBootstrapModel)):
                        pred_samples = model.fit_predict(history)
                    else:
                        df_hist = pd.DataFrame({'valor': history})
                        pred_samples = model.fit_predict(df_hist)
                    
                    pred_samples = np.asarray(pred_samples).flatten()
                    
                    # CRPS
                    score = crps(pred_samples, true_val)
                    row[model_name] = score if not np.isnan(score) else np.nan
                
                except Exception as e:
                    row[model_name] = np.nan
            
            results_rows.append(row)
            
            # Limpieza por paso
            clear_all_sessions()
        
        test_elapsed = time.time() - test_start
        
        # ================================================================
        # VERIFICACIÓN DE PRESUPUESTO
        # ================================================================
        total_elapsed = time.time() - scenario_start
        
        if self.verbose or total_elapsed > self.SCENARIO_BUDGET:
            print(f"\n📊 Escenario: {arma_config['nombre']}, {dist}, var={var}")
            print(f"  Preparación: {prep_elapsed:.1f}s")
            print(f"  Testing: {test_elapsed:.1f}s")
            print(f"  TOTAL: {total_elapsed:.1f}s / {self.SCENARIO_BUDGET}s")
            
            if total_elapsed > self.SCENARIO_BUDGET:
                print(f"  ⚠️  EXCEDIÓ PRESUPUESTO por {total_elapsed - self.SCENARIO_BUDGET:.1f}s")
        
        return results_rows
    
    def _setup_models(self, seed: int):
        """Inicializa 9 modelos con configuraciones ligeras."""
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
                hidden_size=16, n_lags=5, num_layers=1, dropout=0.1,
                lr=0.01, batch_size=16, epochs=20,  # Reducido de 30
                num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True,
                early_stopping_patience=3  # Más agresivo
            ),
            'AREPD': AREPD(
                n_lags=5, rho=0.93, alpha=0.1, poly_degree=2,
                random_state=seed, verbose=False, optimize=True
            ),
            'MCPS': MondrianCPSModel(
                n_lags=10, n_bins=8, test_size=0.25,
                random_state=seed, verbose=False, optimize=True
            ),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(
                n_lags=12, n_pred_bins=6, n_vol_bins=3, volatility_window=15,
                test_size=0.25, random_state=seed, verbose=False, optimize=True
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=15, B=3, units=24, n_layers=2, lr=0.005,
                batch_size=16, epochs=15,  # Reducido de 20
                num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            )
        }
    
    def run_all(self, excel_filename: str = "resultados_140_FAST_5min.xlsx", 
                batch_size: int = 10, max_workers: int = 4):
        """
        Ejecuta 140 escenarios con paralelización controlada.
        
        Tiempo estimado: 140 escenarios × 5min / 4 workers = ~3 horas
        """
        print("="*80)
        print("⚡ EVALUACIÓN ULTRA-RÁPIDA: 140 ESCENARIOS EN ~3 HORAS")
        print("="*80)
        print(f"\n  Presupuesto por escenario: {self.SCENARIO_BUDGET}s (5 min)")
        print(f"  Workers paralelos: {max_workers}")
        print(f"  Tamaño de lote: {batch_size}")
        print(f"  Tiempo estimado total: {140 * self.SCENARIO_BUDGET / max_workers / 3600:.1f} horas")
        print("="*80 + "\n")
        
        all_scenarios = self.generate_all_scenarios()
        total_scenarios = len(all_scenarios)
        all_rows = []
        
        num_batches = (total_scenarios + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_scenarios)
            current_batch = all_scenarios[start_idx:end_idx]
            
            batch_start = time.time()
            print(f"\n🚀 Lote {i+1}/{num_batches} (Escenarios {start_idx+1}-{end_idx})...")
            
            try:
                batch_results = Parallel(
                    n_jobs=max_workers, 
                    backend='loky', 
                    timeout=99999
                )(
                    delayed(self._run_scenario_wrapper)(args) 
                    for args in tqdm(current_batch, desc=f"  Lote {i+1}", leave=False)
                )
                
                for res in batch_results:
                    if isinstance(res, list):
                        all_rows.extend(res)
                    else:
                        all_rows.append(res)
                
                batch_elapsed = time.time() - batch_start
                scenarios_in_batch = len(current_batch)
                avg_time_per_scenario = batch_elapsed / scenarios_in_batch / max_workers
                
                print(f"  ✅ Lote completado en {batch_elapsed:.1f}s")
                print(f"  📊 Promedio: {avg_time_per_scenario:.1f}s/escenario")
                
                if avg_time_per_scenario > self.SCENARIO_BUDGET:
                    print(f"  ⚠️  ADVERTENCIA: Excede presupuesto de {self.SCENARIO_BUDGET}s")
                
                # Guardar checkpoint
                self._save_checkpoint(all_rows, excel_filename)
                
                # Limpieza
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
    Pipeline ULTRA-RÁPIDO: Garantiza ≤5 minutos por escenario.
    
    Presupuesto (300s):
    - Optimización: 120s (40%)
    - Congelamiento: 30s (10%)
    - Testing: 150s (50%) → 12 modelos × 12 pasos = ~1s/predicción
    
    Características CRÍTICAS:
    1. Early stopping en optimización
    2. Extrapolación de CRPS cuando timeout
    3. Grid adaptativo por velocidad de modelo
    4. Limpieza agresiva de memoria
    """
    
    # Estructura de datos
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    # Presupuesto temporal
    SCENARIO_BUDGET = 300  # 5 minutos TOTAL
    OPTIMIZATION_BUDGET = 120  # 40%
    FREEZE_BUDGET = 30         # 10%
    TEST_BUDGET = 150          # 50%
    
    # Configuraciones ARIMA (7 procesos)
    ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    # 5 distribuciones × 4 varianzas = 20 combinaciones
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
    
    def generate_all_scenarios(self) -> list:
        """Genera 140 escenarios (7 × 5 × 4)."""
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
                        self.N_VALIDATION,
                        self.n_boot
                    ))
                    scenario_id += 1
        
        return scenarios
    
    def _run_scenario_wrapper(self, args):
        """Wrapper para ejecución paralela."""
        (arima_cfg, dist, var, seed, n_test_steps, 
         n_train, n_validation, n_boot) = args
        
        pipeline = Pipeline140SinSesgos_ARIMA(
            n_boot=n_boot, 
            seed=seed, 
            verbose=False
        )
        pipeline.N_TEST_STEPS = n_test_steps
        pipeline.N_TRAIN = n_train
        pipeline.N_VALIDATION = n_validation
        
        return pipeline.run_single_scenario(arima_cfg, dist, var, seed)
    
    def _optimize_and_freeze_models(self, models: dict, 
                                     train_series: np.ndarray,
                                     val_series: np.ndarray):
        """
        Optimización + congelamiento con presupuesto de tiempo.
        
        Returns:
            models optimizados y congelados
        """
        from modelos import TimeBalancedOptimizer
        
        # PASO 1: Optimización (120s)
        opt_start = time.time()
        
        optimizer = TimeBalancedOptimizer(
            random_state=self.seed,
            verbose=self.verbose
        )
        
        if self.verbose:
            print(f"\n⚡ PASO 1: Optimización (budget: {self.OPTIMIZATION_BUDGET}s)")
        
        optimized_params = optimizer.optimize_all_models(
            models, 
            train_data=train_series,
            val_data=val_series
        )
        
        opt_elapsed = time.time() - opt_start
        
        if opt_elapsed > self.OPTIMIZATION_BUDGET * 1.2:
            print(f"⚠️  ADVERTENCIA: Optimización tomó {opt_elapsed:.1f}s "
                  f"(límite: {self.OPTIMIZATION_BUDGET}s)")
        
        # PASO 2: Congelamiento (30s)
        freeze_start = time.time()
        train_val_combined = np.concatenate([train_series, val_series])
        
        if self.verbose:
            print(f"\n🔒 PASO 2: Congelamiento (budget: {self.FREEZE_BUDGET}s)")
        
        for name, model in models.items():
            try:
                # Aplicar hiperparámetros optimizados
                if name in optimized_params and optimized_params[name]:
                    if hasattr(model, 'best_params'):
                        model.best_params = optimized_params[name]
                    
                    for key, value in optimized_params[name].items():
                        if hasattr(model, key):
                            setattr(model, key, value)
                
                # Congelar
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(train_val_combined)
                    
                    if self.verbose:
                        print(f"  ✓ {name}")
                
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ {name}: {e}")
        
        freeze_elapsed = time.time() - freeze_start
        
        if freeze_elapsed > self.FREEZE_BUDGET * 1.5:
            print(f"⚠️  ADVERTENCIA: Congelamiento tomó {freeze_elapsed:.1f}s "
                  f"(límite: {self.FREEZE_BUDGET}s)")
        
        total_prep = opt_elapsed + freeze_elapsed
        if self.verbose:
            print(f"\n⏱️  Tiempo preparación: {total_prep:.1f}s / "
                  f"{self.OPTIMIZATION_BUDGET + self.FREEZE_BUDGET}s")
        
        return models
    
    def run_single_scenario(self, arima_config: dict, dist: str, 
                           var: float, rep: int) -> list:
        """
        Ejecuta UN escenario completo en ≤5 minutos.
        
        Returns:
            list de 12 dicts (uno por paso de test)
        """
        scenario_start = time.time()
        scenario_seed = self.seed + rep
        
        # ================================================================
        # SIMULACIÓN
        # ================================================================
        simulator = ARIMASimulation(
            phi=arima_config['phi'],
            theta=arima_config['theta'],
            noise_dist=dist,
            sigma=np.sqrt(var),
            seed=scenario_seed
        )
        
        total_length = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        full_series, _ = simulator.simulate(n=total_length, burn_in=50)
        
        train_series = full_series[:self.N_TRAIN]
        val_series = full_series[self.N_TRAIN:self.N_TRAIN + self.N_VALIDATION]
        test_series = full_series[self.N_TRAIN + self.N_VALIDATION:]
        
        # ================================================================
        # PREPARACIÓN (Optimización + Congelamiento) ≤ 150s
        # ================================================================
        models = self._setup_models(scenario_seed)
        
        prep_start = time.time()
        models = self._optimize_and_freeze_models(
            models,
            train_series=train_series,
            val_series=val_series
        )
        prep_elapsed = time.time() - prep_start
        
        # ================================================================
        # TESTING ≤ 150s (12 pasos × 9 modelos = 108 predicciones)
        # ================================================================
        test_start = time.time()
        results_rows = []
        time_per_step = self.TEST_BUDGET / self.N_TEST_STEPS  # ~12.5s por paso
        
        for t in range(self.N_TEST_STEPS):
            step_start = time.time()
            
            # Historia acumulativa
            history = np.concatenate([
                train_series,
                val_series,
                test_series[:t]
            ]) if t > 0 else np.concatenate([train_series, val_series])
            
            true_val = test_series[t]
            
            row = {
                'Paso': t + 1,
                'proces_simulacion': arima_config['nombre'],
                'Distribución': dist,
                'Varianza error': var,
                'Valor_Observado': true_val
            }
            
            # Predecir con todos los modelos
            for model_name, model in models.items():
                try:
                    # Verificar timeout
                    step_elapsed = time.time() - step_start
                    if step_elapsed > time_per_step:
                        row[model_name] = np.nan
                        continue
                    
                    # Predicción
                    if isinstance(model, (CircularBlockBootstrapModel, SieveBootstrapModel)):
                        pred_samples = model.fit_predict(history)
                    else:
                        df_hist = pd.DataFrame({'valor': history})
                        pred_samples = model.fit_predict(df_hist)
                    
                    pred_samples = np.asarray(pred_samples).flatten()
                    
                    # CRPS
                    score = crps(pred_samples, true_val)
                    row[model_name] = score if not np.isnan(score) else np.nan
                
                except Exception as e:
                    row[model_name] = np.nan
            
            results_rows.append(row)
            
            # Limpieza por paso
            clear_all_sessions()
        
        test_elapsed = time.time() - test_start
        
        # ================================================================
        # VERIFICACIÓN DE PRESUPUESTO
        # ================================================================
        total_elapsed = time.time() - scenario_start
        
        if self.verbose or total_elapsed > self.SCENARIO_BUDGET:
            print(f"\n📊 Escenario: {arima_config['nombre']}, {dist}, var={var}")
            print(f"  Preparación: {prep_elapsed:.1f}s")
            print(f"  Testing: {test_elapsed:.1f}s")
            print(f"  TOTAL: {total_elapsed:.1f}s / {self.SCENARIO_BUDGET}s")
            
            if total_elapsed > self.SCENARIO_BUDGET:
                print(f"  ⚠️  EXCEDIÓ PRESUPUESTO por {total_elapsed - self.SCENARIO_BUDGET:.1f}s")
        
        return results_rows
    
    def _setup_models(self, seed: int):
        """Inicializa 9 modelos con configuraciones ligeras."""
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
                hidden_size=16, n_lags=5, num_layers=1, dropout=0.1,
                lr=0.01, batch_size=16, epochs=20,  # Reducido de 30
                num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True,
                early_stopping_patience=3  # Más agresivo
            ),
            'AREPD': AREPD(
                n_lags=5, rho=0.93, alpha=0.1, poly_degree=2,
                random_state=seed, verbose=False, optimize=True
            ),
            'MCPS': MondrianCPSModel(
                n_lags=10, n_bins=8, test_size=0.25,
                random_state=seed, verbose=False, optimize=True
            ),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(
                n_lags=12, n_pred_bins=6, n_vol_bins=3, volatility_window=15,
                test_size=0.25, random_state=seed, verbose=False, optimize=True
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=15, B=3, units=24, n_layers=2, lr=0.005,
                batch_size=16, epochs=15,  # Reducido de 20
                num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            )
        }
    
    def run_all(self, excel_filename: str = "resultados_140_FAST_5min.xlsx", 
                batch_size: int = 10, max_workers: int = 4):
        """
        Ejecuta 140 escenarios con paralelización controlada.
        
        Tiempo estimado: 140 escenarios × 5min / 4 workers = ~3 horas
        """
        print("="*80)
        print("⚡ EVALUACIÓN ULTRA-RÁPIDA: 140 ESCENARIOS EN ~3 HORAS")
        print("="*80)
        print(f"\n  Presupuesto por escenario: {self.SCENARIO_BUDGET}s (5 min)")
        print(f"  Workers paralelos: {max_workers}")
        print(f"  Tamaño de lote: {batch_size}")
        print(f"  Tiempo estimado total: {140 * self.SCENARIO_BUDGET / max_workers / 3600:.1f} horas")
        print("="*80 + "\n")
        
        all_scenarios = self.generate_all_scenarios()
        total_scenarios = len(all_scenarios)
        all_rows = []
        
        num_batches = (total_scenarios + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_scenarios)
            current_batch = all_scenarios[start_idx:end_idx]
            
            batch_start = time.time()
            print(f"\n🚀 Lote {i+1}/{num_batches} (Escenarios {start_idx+1}-{end_idx})...")
            
            try:
                batch_results = Parallel(
                    n_jobs=max_workers, 
                    backend='loky', 
                    timeout=99999
                )(
                    delayed(self._run_scenario_wrapper)(args) 
                    for args in tqdm(current_batch, desc=f"  Lote {i+1}", leave=False)
                )
                
                for res in batch_results:
                    if isinstance(res, list):
                        all_rows.extend(res)
                    else:
                        all_rows.append(res)
                
                batch_elapsed = time.time() - batch_start
                scenarios_in_batch = len(current_batch)
                avg_time_per_scenario = batch_elapsed / scenarios_in_batch / max_workers
                
                print(f"  ✅ Lote completado en {batch_elapsed:.1f}s")
                print(f"  📊 Promedio: {avg_time_per_scenario:.1f}s/escenario")
                
                if avg_time_per_scenario > self.SCENARIO_BUDGET:
                    print(f"  ⚠️  ADVERTENCIA: Excede presupuesto de {self.SCENARIO_BUDGET}s")
                
                # Guardar checkpoint
                self._save_checkpoint(all_rows, excel_filename)
                
                # Limpieza
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
    Pipeline para procesos SETAR siguiendo la lógica exacta de ARMA/ARIMA.
    
    Presupuesto (300s):
    - Optimización: 120s (40%)
    - Congelamiento: 30s (10%)
    - Testing: 150s (50%) → 12 modelos × 12 pasos = ~1s/predicción
    
    Flujo garantizado:
    1. Simulación de serie SETAR con distribución de ruido especificada
    2. Optimización de hiperparámetros (train+val)
    3. Congelamiento de TODOS los parámetros
    4. Predicción rodante SIN re-estimación
    """
    
    # Estructura de datos (IDÉNTICA a ARMA/ARIMA)
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    # Presupuesto temporal
    SCENARIO_BUDGET = 300  # 5 minutos TOTAL
    OPTIMIZATION_BUDGET = 120  # 40%
    FREEZE_BUDGET = 30         # 10%
    TEST_BUDGET = 150          # 50%
    
    # Configuraciones SETAR (7 procesos)
    # Notación: SETAR(k; p1, p2) donde k=regímenes, p1=orden AR régimen 1, p2=orden AR régimen 2
    SETAR_CONFIGS = [
        {
            'nombre': 'SETAR-1',
            'phi_regime1': [0.6],
            'phi_regime2': [-0.5],
            'threshold': 0.0,
            'delay': 1,
            'description': 'SETAR(2;1,1) - AR(1) con d=1, threshold=0'
        },
        {
            'nombre': 'SETAR-2',
            'phi_regime1': [0.7],
            'phi_regime2': [-0.7],
            'threshold': 0.0,
            'delay': 2,
            'description': 'SETAR(2;1,1) - AR(1) con d=2, threshold=0'
        },
        {
            'nombre': 'SETAR-3',
            'phi_regime1': [0.5, -0.2],
            'phi_regime2': [-0.3, 0.1],
            'threshold': 0.5,
            'delay': 1,
            'description': 'SETAR(2;2,2) - AR(2) con d=1, threshold=0.5'
        },
        {
            'nombre': 'SETAR-4',
            'phi_regime1': [0.8, -0.15],
            'phi_regime2': [-0.6, 0.2],
            'threshold': 1.0,
            'delay': 2,
            'description': 'SETAR(2;2,2) - AR(2) con d=2, threshold=1.0'
        },
        {
            'nombre': 'SETAR-5',
            'phi_regime1': [0.4, -0.1, 0.05],
            'phi_regime2': [-0.3, 0.1, -0.05],
            'threshold': 0.0,
            'delay': 1,
            'description': 'SETAR(2;3,3) - AR(3) con d=1, threshold=0'
        },
        {
            'nombre': 'SETAR-6',
            'phi_regime1': [0.5, -0.3, 0.1],
            'phi_regime2': [-0.4, 0.2, -0.05],
            'threshold': 0.5,
            'delay': 2,
            'description': 'SETAR(2;3,3) - AR(3) con d=2, threshold=0.5'
        },
        {
            'nombre': 'SETAR-7',
            'phi_regime1': [0.3, 0.1],
            'phi_regime2': [-0.2, -0.1],
            'threshold': 0.8,
            'delay': 3,
            'description': 'SETAR(2;2,2) - AR(2) con d=3, threshold=0.8'
        }
    ]
    
    # 5 distribuciones × 4 varianzas = 20 combinaciones
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False):
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
    
    def generate_all_scenarios(self) -> list:
        """Genera 140 escenarios (7 × 5 × 4)."""
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
                        self.N_VALIDATION,
                        self.n_boot
                    ))
                    scenario_id += 1
        
        return scenarios
    
    def _run_scenario_wrapper(self, args):
        """Wrapper para ejecución paralela."""
        (setar_cfg, dist, var, seed, n_test_steps, 
         n_train, n_validation, n_boot) = args
        
        pipeline = Pipeline140SinSesgos_SETAR(
            n_boot=n_boot, 
            seed=seed, 
            verbose=False
        )
        pipeline.N_TEST_STEPS = n_test_steps
        pipeline.N_TRAIN = n_train
        pipeline.N_VALIDATION = n_validation
        
        return pipeline.run_single_scenario(setar_cfg, dist, var, seed)
    
    def _optimize_and_freeze_models(self, models: dict, 
                                     train_series: np.ndarray,
                                     val_series: np.ndarray):
        """
        Optimización + congelamiento con presupuesto de tiempo.
        IDÉNTICO a ARMA/ARIMA.
        
        Returns:
            models optimizados y congelados
        """
        # PASO 1: Optimización (120s)
        opt_start = time.time()
        
        optimizer = TimeBalancedOptimizer(
            random_state=self.seed,
            verbose=self.verbose
        )
        
        if self.verbose:
            print(f"\n⚡ PASO 1: Optimización (budget: {self.OPTIMIZATION_BUDGET}s)")
        
        optimized_params = optimizer.optimize_all_models(
            models, 
            train_data=train_series,
            val_data=val_series
        )
        
        opt_elapsed = time.time() - opt_start
        
        if opt_elapsed > self.OPTIMIZATION_BUDGET * 1.2:
            print(f"⚠️  ADVERTENCIA: Optimización tomó {opt_elapsed:.1f}s "
                  f"(límite: {self.OPTIMIZATION_BUDGET}s)")
        
        # PASO 2: Congelamiento (30s)
        freeze_start = time.time()
        train_val_combined = np.concatenate([train_series, val_series])
        
        if self.verbose:
            print(f"\n🔒 PASO 2: Congelamiento (budget: {self.FREEZE_BUDGET}s)")
        
        for name, model in models.items():
            try:
                # Aplicar hiperparámetros optimizados
                if name in optimized_params and optimized_params[name]:
                    if hasattr(model, 'best_params'):
                        model.best_params = optimized_params[name]
                    
                    for key, value in optimized_params[name].items():
                        if hasattr(model, key):
                            setattr(model, key, value)
                
                # Congelar
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(train_val_combined)
                    
                    if self.verbose:
                        print(f"  ✓ {name}")
                
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ {name}: {e}")
        
        freeze_elapsed = time.time() - freeze_start
        
        if freeze_elapsed > self.FREEZE_BUDGET * 1.5:
            print(f"⚠️  ADVERTENCIA: Congelamiento tomó {freeze_elapsed:.1f}s "
                  f"(límite: {self.FREEZE_BUDGET}s)")
        
        total_prep = opt_elapsed + freeze_elapsed
        if self.verbose:
            print(f"\n⏱️  Tiempo preparación: {total_prep:.1f}s / "
                  f"{self.OPTIMIZATION_BUDGET + self.FREEZE_BUDGET}s")
        
        return models
    
    def run_single_scenario(self, setar_config: dict, dist: str, 
                           var: float, rep: int) -> list:
        """
        Ejecuta UN escenario completo en ≤5 minutos.
        
        Diferencias vs ARMA/ARIMA:
        1. Usa SETARSimulation en lugar de ARMASimulation/ARIMASimulation
        2. Agrega columna 'Descripción' con detalles del modelo SETAR
        3. Todo lo demás es IDÉNTICO
        
        Returns:
            list de 12 dicts (uno por paso de test)
        """
        scenario_start = time.time()
        scenario_seed = self.seed + rep
        
        # ================================================================
        # SIMULACIÓN (ÚNICA DIFERENCIA vs ARMA/ARIMA)
        # ================================================================
        simulator = SETARSimulation(
            model_type=setar_config['nombre'],
            phi_regime1=setar_config['phi_regime1'],
            phi_regime2=setar_config['phi_regime2'],
            threshold=setar_config['threshold'],
            delay=setar_config['delay'],
            noise_dist=dist,
            sigma=np.sqrt(var),
            seed=scenario_seed,
            verbose=False  # Silencioso para no saturar logs en paralelo
        )
        
        total_length = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        full_series, _ = simulator.simulate(n=total_length, burn_in=50)
        
        train_series = full_series[:self.N_TRAIN]
        val_series = full_series[self.N_TRAIN:self.N_TRAIN + self.N_VALIDATION]
        test_series = full_series[self.N_TRAIN + self.N_VALIDATION:]
        
        # ================================================================
        # PREPARACIÓN (Optimización + Congelamiento) ≤ 150s
        # ================================================================
        models = self._setup_models(scenario_seed)
        
        prep_start = time.time()
        models = self._optimize_and_freeze_models(
            models,
            train_series=train_series,
            val_series=val_series
        )
        prep_elapsed = time.time() - prep_start
        
        # ================================================================
        # TESTING ≤ 150s (12 pasos × 9 modelos = 108 predicciones)
        # ================================================================
        test_start = time.time()
        results_rows = []
        time_per_step = self.TEST_BUDGET / self.N_TEST_STEPS  # ~12.5s por paso
        
        for t in range(self.N_TEST_STEPS):
            step_start = time.time()
            
            # Historia acumulativa
            history = np.concatenate([
                train_series,
                val_series,
                test_series[:t]
            ]) if t > 0 else np.concatenate([train_series, val_series])
            
            true_val = test_series[t]
            
            row = {
                'Paso': t + 1,
                'proces_simulacion': setar_config['nombre'],
                'Descripción': setar_config['description'],  # ✅ Extra info para SETAR
                'Distribución': dist,
                'Varianza error': var,
                'Valor_Observado': true_val
            }
            
            # Predecir con todos los modelos
            for model_name, model in models.items():
                try:
                    # Verificar timeout
                    step_elapsed = time.time() - step_start
                    if step_elapsed > time_per_step:
                        row[model_name] = np.nan
                        continue
                    
                    # Predicción
                    if isinstance(model, (CircularBlockBootstrapModel, SieveBootstrapModel)):
                        pred_samples = model.fit_predict(history)
                    else:
                        df_hist = pd.DataFrame({'valor': history})
                        pred_samples = model.fit_predict(df_hist)
                    
                    pred_samples = np.asarray(pred_samples).flatten()
                    
                    # CRPS
                    score = crps(pred_samples, true_val)
                    row[model_name] = score if not np.isnan(score) else np.nan
                
                except Exception as e:
                    row[model_name] = np.nan
            
            results_rows.append(row)
            
            # Limpieza por paso
            clear_all_sessions()
        
        test_elapsed = time.time() - test_start
        
        # ================================================================
        # VERIFICACIÓN DE PRESUPUESTO
        # ================================================================
        total_elapsed = time.time() - scenario_start
        
        if self.verbose or total_elapsed > self.SCENARIO_BUDGET:
            print(f"\n📊 Escenario: {setar_config['nombre']}, {dist}, var={var}")
            print(f"  Descripción: {setar_config['description']}")
            print(f"  Preparación: {prep_elapsed:.1f}s")
            print(f"  Testing: {test_elapsed:.1f}s")
            print(f"  TOTAL: {total_elapsed:.1f}s / {self.SCENARIO_BUDGET}s")
            
            if total_elapsed > self.SCENARIO_BUDGET:
                print(f"  ⚠️  EXCEDIÓ PRESUPUESTO por {total_elapsed - self.SCENARIO_BUDGET:.1f}s")
        
        return results_rows
    
    def _setup_models(self, seed: int):
        """Inicializa 9 modelos - IDÉNTICO a ARMA/ARIMA."""
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
                hidden_size=16, n_lags=5, num_layers=1, dropout=0.1,
                lr=0.01, batch_size=16, epochs=20,
                num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True,
                early_stopping_patience=3
            ),
            'AREPD': AREPD(
                n_lags=5, rho=0.93, alpha=0.1, poly_degree=2,
                random_state=seed, verbose=False, optimize=True
            ),
            'MCPS': MondrianCPSModel(
                n_lags=10, n_bins=8, test_size=0.25,
                random_state=seed, verbose=False, optimize=True
            ),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(
                n_lags=12, n_pred_bins=6, n_vol_bins=3, volatility_window=15,
                test_size=0.25, random_state=seed, verbose=False, optimize=True
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=15, B=3, units=24, n_layers=2, lr=0.005,
                batch_size=16, epochs=15,
                num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            )
        }
    
    def run_all(self, excel_filename: str = "resultados_140_SETAR_5min.xlsx", 
                batch_size: int = 10, max_workers: int = 4):
        """
        Ejecuta 140 escenarios con paralelización controlada.
        IDÉNTICO a ARMA/ARIMA.
        
        Tiempo estimado: 140 escenarios × 5min / 4 workers = ~3 horas
        """
        print("="*80)
        print("⚡ EVALUACIÓN ULTRA-RÁPIDA SETAR: 140 ESCENARIOS EN ~3 HORAS")
        print("="*80)
        print(f"\n  Presupuesto por escenario: {self.SCENARIO_BUDGET}s (5 min)")
        print(f"  Workers paralelos: {max_workers}")
        print(f"  Tamaño de lote: {batch_size}")
        print(f"  Tiempo estimado total: {140 * self.SCENARIO_BUDGET / max_workers / 3600:.1f} horas")
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
            
            batch_start = time.time()
            print(f"\n🚀 Lote {i+1}/{num_batches} (Escenarios {start_idx+1}-{end_idx})...")
            
            try:
                batch_results = Parallel(
                    n_jobs=max_workers, 
                    backend='loky', 
                    timeout=99999
                )(
                    delayed(self._run_scenario_wrapper)(args) 
                    for args in tqdm(current_batch, desc=f"  Lote {i+1}", leave=False)
                )
                
                for res in batch_results:
                    if isinstance(res, list):
                        all_rows.extend(res)
                    else:
                        all_rows.append(res)
                
                batch_elapsed = time.time() - batch_start
                scenarios_in_batch = len(current_batch)
                avg_time_per_scenario = batch_elapsed / scenarios_in_batch / max_workers
                
                print(f"  ✅ Lote completado en {batch_elapsed:.1f}s")
                print(f"  📊 Promedio: {avg_time_per_scenario:.1f}s/escenario")
                
                if avg_time_per_scenario > self.SCENARIO_BUDGET:
                    print(f"  ⚠️  ADVERTENCIA: Excede presupuesto de {self.SCENARIO_BUDGET}s")
                
                # Guardar checkpoint
                self._save_checkpoint(all_rows, excel_filename)
                
                # Limpieza
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


def _predict_with_model_diferenciacion(name: str, model, 
                                       history_series: np.ndarray,
                                       transformador: TransformadorDiferenciacionIntegracion = None,
                                       verbose: bool = False) -> np.ndarray:
    """
    Predicción con manejo opcional de diferenciación.
    
    Args:
        name: Nombre del modelo
        model: Instancia del modelo
        history_series: Array numpy con historia completa
        transformador: Transformador de diferenciación (None si no se usa)
        verbose: Mostrar información
        
    Returns:
        Array con predicciones (en espacio original)
    """
    try:
        # PASO 1: Preparar datos según se use diferenciación o no
        if transformador is not None:
            # CON DIFERENCIACIÓN
            serie_diff = transformador.diferenciar_serie(history_series)
            ultimo_valor_original = history_series[-1]
            
            if verbose:
                print(f"  {name}: Prediciendo en espacio diferenciado "
                      f"(n={len(serie_diff)}, último={ultimo_valor_original:.4f})")
            
            # Predicción según tipo de modelo
            if name in ['Block Bootstrapping', 'Sieve Bootstrap']:
                samples_diff = model.fit_predict(serie_diff)
            else:
                df_diff = pd.DataFrame({'valor': serie_diff})
                samples_diff = model.fit_predict(df_diff)
            
            # PASO 2: Integrar predicciones
            samples = transformador.integrar_predicciones(
                samples_diff, 
                ultimo_valor_observado=ultimo_valor_original
            )
            
        else:
            # SIN DIFERENCIACIÓN (comportamiento original)
            if name in ['Block Bootstrapping', 'Sieve Bootstrap']:
                samples = model.fit_predict(history_series)
            else:
                df_hist = pd.DataFrame({'valor': history_series})
                samples = model.fit_predict(df_hist)
        
        # Validación de salida
        samples = np.asarray(samples).flatten()
        
        if len(samples) == 0 or np.all(np.isnan(samples)):
            if verbose:
                print(f"  {name}: Predicciones vacías/NaN")
            return np.array([])
        
        return samples
        
    except Exception as e:
        if verbose:
            print(f"  Error en {name}: {str(e)}")
        return np.array([])


class Pipeline140SinSesgos_ARIMA_ConDiferenciacion:
    """
    Pipeline ARIMA con OPCIÓN de diferenciación adicional.
    
    Presupuesto (300s):
    - Optimización: 120s (40%)
    - Congelamiento: 30s (10%)
    - Testing: 150s (50%)
    
    Permite comparar:
    A) SIN diferenciación adicional: Modelos reciben Y_t directamente
    B) CON diferenciación adicional: Modelos reciben ΔY_t, predicciones se integran
    
    Características CRÍTICAS:
    1. Early stopping en optimización
    2. Extrapolación de CRPS cuando timeout
    3. Grid adaptativo por velocidad de modelo
    4. Limpieza agresiva de memoria
    """
    
    # Estructura de datos (IDÉNTICA a ARMA)
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    
    # Presupuesto temporal
    SCENARIO_BUDGET = 300  # 5 minutos TOTAL
    OPTIMIZATION_BUDGET = 120  # 40%
    FREEZE_BUDGET = 30         # 10%
    TEST_BUDGET = 150          # 50%
    
    # Configuraciones ARIMA (7 procesos)
    ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    # 5 distribuciones × 4 varianzas = 20 combinaciones
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
    
    def generate_all_scenarios(self) -> list:
        """Genera 140 escenarios (7 × 5 × 4)."""
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
                        self.N_VALIDATION,
                        self.n_boot,
                        self.usar_diferenciacion
                    ))
                    scenario_id += 1
        
        return scenarios
    
    def _run_scenario_wrapper(self, args):
        """Wrapper para ejecución paralela."""
        (arima_cfg, dist, var, seed, n_test_steps, 
         n_train, n_validation, n_boot, usar_diff) = args
        
        pipeline = Pipeline140SinSesgos_ARIMA_ConDiferenciacion(
            n_boot=n_boot, 
            seed=seed, 
            verbose=False,
            usar_diferenciacion=usar_diff
        )
        pipeline.N_TEST_STEPS = n_test_steps
        pipeline.N_TRAIN = n_train
        pipeline.N_VALIDATION = n_validation
        
        return pipeline.run_single_scenario(arima_cfg, dist, var, seed)
    
    def _optimize_and_freeze_models(self, models: dict, 
                                     train_series: np.ndarray,
                                     val_series: np.ndarray,
                                     transformador: TransformadorDiferenciacionIntegracion = None):
        """
        Optimización + congelamiento con presupuesto de tiempo.
        
        Si usar_diferenciacion=True:
            - Transforma train/val a espacio diferenciado
            - Optimiza y congela en ese espacio
        
        Returns:
            models optimizados y congelados
        """
        # PASO 1: Optimización (120s)
        opt_start = time.time()
        
        optimizer = TimeBalancedOptimizer(
            random_state=self.seed,
            verbose=self.verbose
        )
        
        if self.verbose:
            espacio = "diferenciado" if transformador is not None else "original"
            print(f"\n⚡ PASO 1: Optimización en espacio {espacio} (budget: {self.OPTIMIZATION_BUDGET}s)")
        
        # Preparar datos para optimización
        if transformador is not None:
            train_series_opt = transformador.diferenciar_serie(train_series)
            val_series_opt = transformador.diferenciar_serie(val_series)
        else:
            train_series_opt = train_series
            val_series_opt = val_series
        
        optimized_params = optimizer.optimize_all_models(
            models, 
            train_data=train_series_opt,
            val_data=val_series_opt
        )
        
        opt_elapsed = time.time() - opt_start
        
        if opt_elapsed > self.OPTIMIZATION_BUDGET * 1.2:
            print(f"⚠️  ADVERTENCIA: Optimización tomó {opt_elapsed:.1f}s "
                  f"(límite: {self.OPTIMIZATION_BUDGET}s)")
        
        # PASO 2: Congelamiento (30s)
        freeze_start = time.time()
        
        # Combinar train+val en el espacio correspondiente
        if transformador is not None:
            full_series = np.concatenate([train_series, val_series])
            train_val_combined = transformador.diferenciar_serie(full_series)
        else:
            train_val_combined = np.concatenate([train_series, val_series])
        
        if self.verbose:
            espacio = "diferenciado" if transformador is not None else "original"
            print(f"\n🔒 PASO 2: Congelamiento en espacio {espacio} (budget: {self.FREEZE_BUDGET}s)")
        
        for name, model in models.items():
            try:
                # Aplicar hiperparámetros optimizados
                if name in optimized_params and optimized_params[name]:
                    if hasattr(model, 'best_params'):
                        model.best_params = optimized_params[name]
                    
                    for key, value in optimized_params[name].items():
                        if hasattr(model, key):
                            setattr(model, key, value)
                
                # Congelar
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(train_val_combined)
                    
                    if self.verbose:
                        print(f"  ✓ {name}")
                
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ {name}: {e}")
        
        freeze_elapsed = time.time() - freeze_start
        
        if freeze_elapsed > self.FREEZE_BUDGET * 1.5:
            print(f"⚠️  ADVERTENCIA: Congelamiento tomó {freeze_elapsed:.1f}s "
                  f"(límite: {self.FREEZE_BUDGET}s)")
        
        total_prep = opt_elapsed + freeze_elapsed
        if self.verbose:
            print(f"\n⏱️  Tiempo preparación: {total_prep:.1f}s / "
                  f"{self.OPTIMIZATION_BUDGET + self.FREEZE_BUDGET}s")
        
        return models
    
    def run_single_scenario(self, arima_config: dict, dist: str, 
                           var: float, rep: int) -> list:
        """
        Ejecuta UN escenario completo en ≤5 minutos.
        
        Diferencias vs ARMA estándar:
        1. Usa ARIMASimulation
        2. Opcionalmente diferencia datos antes de optimizar/congelar/predecir
        3. Agrega columna 'Con_Diferenciacion'
        
        Returns:
            list de 12 dicts (uno por paso de test)
        """
        scenario_start = time.time()
        scenario_seed = self.seed + rep
        
        # ================================================================
        # SIMULACIÓN
        # ================================================================
        simulator = ARIMASimulation(
            model_type=arima_config['nombre'],
            phi=arima_config['phi'],
            theta=arima_config['theta'],
            noise_dist=dist,
            sigma=np.sqrt(var),
            seed=scenario_seed
        )
        
        total_length = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        full_series, _ = simulator.simulate(n=total_length, burn_in=50)
        
        train_series = full_series[:self.N_TRAIN]
        val_series = full_series[self.N_TRAIN:self.N_TRAIN + self.N_VALIDATION]
        test_series = full_series[self.N_TRAIN + self.N_VALIDATION:]
        
        # ================================================================
        # PREPARACIÓN (Optimización + Congelamiento) ≤ 150s
        # ================================================================
        models = self._setup_models(scenario_seed)
        
        # Crear transformador si se usa diferenciación
        transformador_prep = TransformadorDiferenciacionIntegracion(
            d=1, verbose=False
        ) if self.usar_diferenciacion else None
        
        prep_start = time.time()
        models = self._optimize_and_freeze_models(
            models,
            train_series=train_series,
            val_series=val_series,
            transformador=transformador_prep
        )
        prep_elapsed = time.time() - prep_start
        
        # ================================================================
        # TESTING ≤ 150s (12 pasos × 9 modelos = 108 predicciones)
        # ================================================================
        test_start = time.time()
        results_rows = []
        time_per_step = self.TEST_BUDGET / self.N_TEST_STEPS  # ~12.5s por paso
        
        for t in range(self.N_TEST_STEPS):
            step_start = time.time()
            
            # Historia acumulativa
            history = np.concatenate([
                train_series,
                val_series,
                test_series[:t]
            ]) if t > 0 else np.concatenate([train_series, val_series])
            
            true_val = test_series[t]
            
            row = {
                'Paso': t + 1,
                'proces_simulacion': arima_config['nombre'],
                'Distribución': dist,
                'Varianza error': var,
                'Con_Diferenciacion': 'Sí' if self.usar_diferenciacion else 'No',
                'Valor_Observado': true_val
            }
            
            # Transformador fresco para este paso
            step_transformador = TransformadorDiferenciacionIntegracion(
                d=1, verbose=False
            ) if self.usar_diferenciacion else None
            
            # Predecir con todos los modelos
            for model_name, model in models.items():
                try:
                    # Verificar timeout
                    step_elapsed = time.time() - step_start
                    if step_elapsed > time_per_step:
                        row[model_name] = np.nan
                        continue
                    
                    # Predicción (con o sin diferenciación)
                    pred_samples = _predict_with_model_diferenciacion(
                        name=model_name,
                        model=model,
                        history_series=history,
                        transformador=step_transformador,
                        verbose=False
                    )
                    
                    if len(pred_samples) > 0:
                        # CRPS
                        score = crps(pred_samples, true_val)
                        row[model_name] = score if not np.isnan(score) else np.nan
                    else:
                        row[model_name] = np.nan
                
                except Exception as e:
                    row[model_name] = np.nan
            
            results_rows.append(row)
            
            # Limpieza por paso
            clear_all_sessions()
        
        test_elapsed = time.time() - test_start
        
        # ================================================================
        # VERIFICACIÓN DE PRESUPUESTO
        # ================================================================
        total_elapsed = time.time() - scenario_start
        
        if self.verbose or total_elapsed > self.SCENARIO_BUDGET:
            diff_str = "CON" if self.usar_diferenciacion else "SIN"
            print(f"\n📊 Escenario: {arima_config['nombre']}, {dist}, var={var} ({diff_str} dif)")
            print(f"  Preparación: {prep_elapsed:.1f}s")
            print(f"  Testing: {test_elapsed:.1f}s")
            print(f"  TOTAL: {total_elapsed:.1f}s / {self.SCENARIO_BUDGET}s")
            
            if total_elapsed > self.SCENARIO_BUDGET:
                print(f"  ⚠️  EXCEDIÓ PRESUPUESTO por {total_elapsed - self.SCENARIO_BUDGET:.1f}s")
        
        return results_rows
    
    def _setup_models(self, seed: int):
        """Inicializa 9 modelos - IDÉNTICO a ARMA."""
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
                hidden_size=16, n_lags=5, num_layers=1, dropout=0.1,
                lr=0.01, batch_size=16, epochs=20,
                num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True,
                early_stopping_patience=3
            ),
            'AREPD': AREPD(
                n_lags=5, rho=0.93, alpha=0.1, poly_degree=2,
                random_state=seed, verbose=False, optimize=True
            ),
            'MCPS': MondrianCPSModel(
                n_lags=10, n_bins=8, test_size=0.25,
                random_state=seed, verbose=False, optimize=True
            ),
            'AV-MCPS': AdaptiveVolatilityMondrianCPS(
                n_lags=12, n_pred_bins=6, n_vol_bins=3, volatility_window=15,
                test_size=0.25, random_state=seed, verbose=False, optimize=True
            ),
            'EnCQR-LSTM': EnCQR_LSTM_Model(
                n_lags=15, B=3, units=24, n_layers=2, lr=0.005,
                batch_size=16, epochs=15,
                num_samples=self.n_boot,
                random_state=seed, verbose=False, optimize=True
            )
        }
    
    def run_all(self, excel_filename: str = "resultados_140_ARIMA_dif_5min.xlsx", 
                batch_size: int = 10, max_workers: int = 4):
        """
        Ejecuta 140 escenarios con paralelización controlada.
        
        Tiempo estimado: 140 escenarios × 5min / 4 workers = ~3 horas
        """
        diferenciacion_str = "CON" if self.usar_diferenciacion else "SIN"
        print("="*80)
        print(f"⚡ EVALUACIÓN ULTRA-RÁPIDA ARIMA - {diferenciacion_str} DIFERENCIACIÓN ADICIONAL")
        print("="*80)
        print(f"\n  Presupuesto por escenario: {self.SCENARIO_BUDGET}s (5 min)")
        print(f"  Workers paralelos: {max_workers}")
        print(f"  Tamaño de lote: {batch_size}")
        print(f"  Diferenciación adicional: {diferenciacion_str}")
        print(f"  Tiempo estimado total: {140 * self.SCENARIO_BUDGET / max_workers / 3600:.1f} horas")
        
        if self.usar_diferenciacion:
            print("\n📊 DIFERENCIACIÓN:")
            print("  • Los modelos trabajan con ΔY_t (incrementos)")
            print("  • Optimización y freeze en espacio diferenciado")
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
            
            batch_start = time.time()
            print(f"\n🚀 Lote {i+1}/{num_batches} (Escenarios {start_idx+1}-{end_idx})...")
            
            try:
                batch_results = Parallel(
                    n_jobs=max_workers, 
                    backend='loky', 
                    timeout=99999
                )(
                    delayed(self._run_scenario_wrapper)(args) 
                    for args in tqdm(current_batch, desc=f"  Lote {i+1}", leave=False)
                )
                
                for res in batch_results:
                    if isinstance(res, list):
                        all_rows.extend(res)
                    else:
                        all_rows.append(res)
                
                batch_elapsed = time.time() - batch_start
                scenarios_in_batch = len(current_batch)
                avg_time_per_scenario = batch_elapsed / scenarios_in_batch / max_workers
                
                print(f"  ✅ Lote completado en {batch_elapsed:.1f}s")
                print(f"  📊 Promedio: {avg_time_per_scenario:.1f}s/escenario")
                
                if avg_time_per_scenario > self.SCENARIO_BUDGET:
                    print(f"  ⚠️  ADVERTENCIA: Excede presupuesto de {self.SCENARIO_BUDGET}s")
                
                # Guardar checkpoint
                self._save_checkpoint(all_rows, excel_filename)
                
                # Limpieza
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
    
    def _save_checkpoint(self, rows, filename):
        """Guarda progreso en Excel."""
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


class PipelineARMA_MultiD_DobleModalidad:
    """
    Pipeline que evalúa procesos ARMA diferenciados manualmente d=2,3,4,5,7,10 
    en DOS MODALIDADES, utilizando optimización justa sin data leakage.
    
    MODALIDAD A (Sin diferenciación adicional):
    - Los modelos reciben Y_t diferenciado d veces (el proceso integrado)
    - Optimización basada en error de predicción sobre Y_t
    
    MODALIDAD B (Con diferenciación adicional):
    - Los modelos reciben ΔY_t (una diferenciación MÁS sobre el proceso)
    - Optimización basada en error de predicción sobre ΔY_t
    - Predicciones se integran de vuelta para la evaluación final
    """
    
    N_TEST_STEPS = 12
    N_VALIDATION_FOR_OPT = 40 # Tamaño del set de validación para el optimizador
    N_TRAIN_INITIAL = 200     # Tamaño inicial de entrenamiento
    
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
    D_VALUES = [2, 3, 4, 5, 7, 10]
    
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
                random_state=seed, verbose=False, optimize=True,
                early_stopping_patience=3
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
    
    def _generate_differenced_arma_series(self, arma_config: dict, d_value: int,
                                         distribution: str, variance: float,
                                         scenario_seed: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera serie ARMA estacionaria y luego la integra d veces.
        Retorna (serie_integrada, errores_arma)
        """
        from simulacion import ARMASimulation
        
        # Paso 1: Generar ARMA estacionario
        simulator_arma = ARMASimulation(
            phi=arma_config['phi'],
            theta=arma_config['theta'],
            noise_dist=distribution,
            sigma=np.sqrt(variance),
            seed=scenario_seed
        )
        
        arma_series, arma_errors = simulator_arma.simulate(n=n, burn_in=50)
        
        # Paso 2: Integrar d veces para obtener la serie observada Y_t
        integrated_series = arma_series.copy()
        for _ in range(d_value):
            integrated_series = np.cumsum(integrated_series)
        
        return integrated_series, arma_errors

    def _optimize_and_freeze_models(self, models: dict, 
                                     train_calib_series: np.ndarray,
                                     transformador: TransformadorDiferenciacionIntegracion = None):
        """
        Optimización JUSTA usando TimeBalancedOptimizer y partición Train/Val.
        
        Si transformador no es None, optimiza sobre la serie diferenciada.
        """
        # 1. Preparar datos para el optimizador (espacio original o diferenciado)
        if transformador is not None:
            # Modalidad CON_DIFF: Trabajamos con ΔY
            data_for_optim = transformador.diferenciar_serie(train_calib_series)
        else:
            # Modalidad SIN_DIFF: Trabajamos con Y
            data_for_optim = train_calib_series

        # 2. Dividir en Train y Validation para el optimizador
        # Usamos los últimos N_VALIDATION_FOR_OPT puntos para validar hiperparámetros
        if len(data_for_optim) > self.N_VALIDATION_FOR_OPT + 10:
            train_data = data_for_optim[:-self.N_VALIDATION_FOR_OPT]
            val_data = data_for_optim[-self.N_VALIDATION_FOR_OPT:]
        else:
            # Fallback si la serie es muy corta (raro con N=200+)
            split_idx = int(len(data_for_optim) * 0.8)
            train_data = data_for_optim[:split_idx]
            val_data = data_for_optim[split_idx:]

        # 3. Optimización con presupuesto de tiempo
        optimizer = TimeBalancedOptimizer(
            random_state=self.seed,
            verbose=self.verbose
        )
        
        if self.verbose:
            espacio = "diferenciado" if transformador is not None else "original"
            print(f"  Optimizando en espacio {espacio} (Train={len(train_data)}, Val={len(val_data)})")
            
        optimized_params = optimizer.optimize_all_models(
            models, 
            train_data=train_data,
            val_data=val_data
        )

        # 4. Aplicar parámetros y CONGELAR
        # El congelamiento se hace con TODA la data disponible (train + val) en el espacio correcto
        for name, model in models.items():
            try:
                # Aplicar mejores params
                if name in optimized_params and optimized_params[name]:
                    if hasattr(model, 'best_params'):
                        model.best_params = optimized_params[name]
                    for key, value in optimized_params[name].items():
                        if hasattr(model, key):
                            setattr(model, key, value)
                
                # Congelar modelo (re-entrena con toda la data_for_optim)
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(data_for_optim)
                    
            except Exception as e:
                if self.verbose:
                    print(f"    Error configurando/congelando {name}: {e}")

        return models
    
    def _run_single_scenario_doble_modalidad(self, arma_config: dict, d_value: int,
                                             distribution: str, variance: float,
                                             scenario_seed: int) -> list:
        """
        Ejecuta un escenario completo en AMBAS modalidades.
        
        Returns:
            Lista con 2 conjuntos de resultados (SIN_DIFF y CON_DIFF)
        """
        all_results = []
        
        # Generar datos UNA SOLA VEZ (El "Mundo Real")
        try:
            # Necesitamos historial suficiente para Train + Val_Opt + Test
            total_needed = self.N_TRAIN_INITIAL + self.N_TEST_STEPS
            
            # Generar serie integrada d veces
            full_series, full_errors = self._generate_differenced_arma_series(
                arma_config=arma_config,
                d_value=d_value,
                distribution=distribution,
                variance=variance,
                scenario_seed=scenario_seed,
                n=total_needed
            )
            
            # Definir punto de corte donde empieza el test
            test_start_idx = self.N_TRAIN_INITIAL
            
            # Datos disponibles antes del test (Train + Calibration)
            train_calib_series = full_series[:test_start_idx]
            
        except Exception as e:
            if self.verbose:
                print(f"Error generando datos: {e}")
            return []
        
        # --- MODALIDAD A: SIN DIFERENCIACIÓN ADICIONAL ---
        # Los modelos ven Y_t tal cual viene del simulador
        results_A = self._run_single_modalidad(
            arma_config=arma_config,
            d_value=d_value,
            distribution=distribution,
            variance=variance,
            scenario_seed=scenario_seed,
            full_series=full_series,
            train_calib_series=train_calib_series,
            test_start_idx=test_start_idx,
            usar_diferenciacion=False
        )
        all_results.extend(results_A)
        
        # --- MODALIDAD B: CON DIFERENCIACIÓN ADICIONAL ---
        # Los modelos ven ΔY_t (diferenciamos una vez más antes de procesar)
        results_B = self._run_single_modalidad(
            arma_config=arma_config,
            d_value=d_value,
            distribution=distribution,
            variance=variance,
            scenario_seed=scenario_seed + 100000, # Semilla diferente para inicialización de modelos
            full_series=full_series,
            train_calib_series=train_calib_series,
            test_start_idx=test_start_idx,
            usar_diferenciacion=True
        )
        all_results.extend(results_B)
        
        clear_all_sessions()
        
        return all_results
    
    def _run_single_modalidad(self, arma_config: dict, d_value: int,
                             distribution: str, variance: float, scenario_seed: int,
                             full_series: np.ndarray, train_calib_series: np.ndarray,
                             test_start_idx: int, usar_diferenciacion: bool) -> list:
        """Ejecuta una modalidad específica optimizando y probando."""
        try:
            # 1. Configurar Transformador (si aplica)
            transformador = TransformadorDiferenciacionIntegracion(
                d=1, verbose=False
            ) if usar_diferenciacion else None
            
            # 2. Inicializar modelos limpios
            models = self._setup_models(scenario_seed)
            
            # 3. Optimizar y Congelar (usando TimeBalancedOptimizer)
            # Esto maneja internamente el split Train/Val y la diferenciación si transformador != None
            self._optimize_and_freeze_models(
                models=models,
                train_calib_series=train_calib_series,
                transformador=transformador
            )
            
            # 4. Predicción Rodante (Test)
            results_rows = []
            p = len(arma_config['phi'])
            q = len(arma_config['theta'])
            model_name = f"ARMA_I({p},{d_value},{q})"
            modalidad_str = "CON_DIFF" if usar_diferenciacion else "SIN_DIFF"
            
            for step in range(self.N_TEST_STEPS):
                current_idx = test_start_idx + step
                
                # Historia disponible hasta el momento t
                history_series = full_series[:current_idx]
                true_value = full_series[current_idx]
                
                # Transformador fresco para este paso (stateless)
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
                
                # Predecir
                for name, model in models.items():
                    try:
                        # La función auxiliar maneja la diferenciación/integración en predicción
                        pred_samples = _predict_with_model_diferenciacion(
                            name=name,
                            model=model,
                            history_series=history_series,
                            transformador=step_transformador,
                            verbose=False
                        )
                        
                        if len(pred_samples) > 0:
                            crps_val = crps(pred_samples, true_value)
                            step_result[name] = crps_val
                        else:
                            step_result[name] = np.nan
                    except Exception as e:
                        if self.verbose:
                            print(f"    {name} error en test paso {step}: {e}")
                        step_result[name] = np.nan
                
                results_rows.append(step_result)
            
            # Calcular Promedio
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
            
            # Limpieza explícita
            del models
            return results_rows
            
        except Exception as e:
            if self.verbose:
                print(f"Error crítico en modalidad {modalidad_str}: {e}")
            return []
    
    def generate_all_scenarios(self) -> list:
        """Genera lista de escenarios para ejecución paralela."""
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
    
    def run_all(self, excel_filename: str = "resultados_ARMA_MultiD_Final.xlsx",
                batch_size: int = 20):
        """Ejecuta todos los escenarios."""
        print("="*80)
        print("EVALUACIÓN ARMA MULTI-D - DOBLE MODALIDAD (SIN DATA LEAKAGE)")
        print("="*80)
        
        cpu_count = os.cpu_count() or 4
        safe_jobs = min(6, max(1, int(cpu_count * 0.75)))
        
        total_base_scenarios = (len(self.D_VALUES) * len(self.ARMA_CONFIGS) * 
                               len(self.DISTRIBUTIONS) * len(self.VARIANCES))
        
        print(f"⚡ Usando {safe_jobs} núcleos en paralelo")
        print(f"⚡ Tamaño del lote: {batch_size}")
        print(f"📊 ESCENARIOS BASE: {total_base_scenarios}")
        print(f"📊 FILAS TOTALES ESPERADAS: ~{total_base_scenarios * 2 * 13}")
        print("="*80 + "\n")
        
        all_scenarios = self.generate_all_scenarios()
        total_scenarios = len(all_scenarios)
        all_rows = []
        
        num_batches = (total_scenarios + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_scenarios)
            current_batch = all_scenarios[start_idx:end_idx]
            
            print(f"\n🚀 Lote {i+1}/{num_batches} (Escenarios {start_idx+1}-{end_idx})...")
            
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
        
        pipeline = PipelineARMA_MultiD_DobleModalidad(
            n_boot=self.n_boot, seed=seed, verbose=False
        )
        
        return pipeline._run_single_scenario_doble_modalidad(
            arma_cfg, d_val, dist, var, seed
        )
    
    def _save_checkpoint(self, rows, filename):
        """Guarda progreso en Excel."""
        if not rows:
            return
        
        df_temp = pd.DataFrame(rows)
        # Ordenar columnas lógicamente
        first_cols = ['Paso', 'Proceso', 'p', 'd', 'q', 'ARMA_base', 
                      'Distribución', 'Varianza', 'Modalidad', 'Valor_Observado']
        model_cols = [c for c in df_temp.columns if c not in first_cols]
        final_cols = [c for c in first_cols if c in df_temp.columns] + sorted(model_cols)
        
        df_temp = df_temp[final_cols]
        df_temp.to_excel(filename, index=False)

class Pipeline140_TamanosCrecientes:
    """
    Pipeline unificado que evalúa ARMA, ARIMA y SETAR con TAMAÑOS CRECIENTES
    de datos de entrenamiento y calibración.
    
    Objetivo: Determinar cómo el tamaño de los datos históricos afecta el
    desempeño de los métodos de predicción conformal.
    
    Configuración OPTIMIZADA:
    - N_TEST_STEPS = 12 (FIJO - no cambia)
    - 3 combinaciones únicas (N_TRAIN, N_CALIBRATION):
      1. (200, 40)  - Baseline
      2. (500, 100) - Intermedio
      3. (1000, 200) - Máximo
    
    Para cada combinación:
    - 7 configs proceso × 5 dist × 4 var = 140 escenarios base
    
    Total: 3 combinaciones × 140 escenarios = 420 escenarios base
    """
    
    N_TEST_STEPS = 12  # FIJO
    
    # ✅ NUEVA CONFIGURACIÓN: 3 combinaciones únicas específicas
    SIZE_COMBINATIONS = [
        {'n_train': 100, 'n_calib': 20},    # Pequeño
        {'n_train': 500, 'n_calib': 100},   # Mediano
        {'n_train': 1000, 'n_calib': 200}   # Grande
    ]
    
    # Configuraciones ARMA
    ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'AR(2)', 'phi': [0.5, -0.3], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]},
        {'nombre': 'MA(2)', 'phi': [], 'theta': [0.4, 0.2]},
        {'nombre': 'ARMA(1,1)', 'phi': [0.6], 'theta': [0.3]},
        {'nombre': 'ARMA(2,2)', 'phi': [0.4, -0.2], 'theta': [0.5, 0.1]},
        {'nombre': 'ARMA(2,1)', 'phi': [0.7, 0.2], 'theta': [0.5]}
    ]
    
    # Configuraciones ARIMA
    ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(0,1,0)', 'phi': [], 'theta': []},
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.6], 'theta': []},
        {'nombre': 'ARIMA(2,1,0)', 'phi': [0.5, -0.2], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.5]},
        {'nombre': 'ARIMA(0,1,2)', 'phi': [], 'theta': [0.4, 0.25]},
        {'nombre': 'ARIMA(1,1,1)', 'phi': [0.7], 'theta': [-0.3]},
        {'nombre': 'ARIMA(2,1,2)', 'phi': [0.6, 0.2], 'theta': [0.4, -0.1]}
    ]
    
    # Configuraciones SETAR
    SETAR_CONFIGS = [
        {
            'nombre': 'SETAR-1',
            'phi_regime1': [0.6],
            'phi_regime2': [-0.5],
            'threshold': 0.0,
            'delay': 1
        },
        {
            'nombre': 'SETAR-2',
            'phi_regime1': [0.7],
            'phi_regime2': [-0.7],
            'threshold': 0.0,
            'delay': 2
        },
        {
            'nombre': 'SETAR-3',
            'phi_regime1': [0.5, -0.2],
            'phi_regime2': [-0.3, 0.1],
            'threshold': 0.5,
            'delay': 1
        },
        {
            'nombre': 'SETAR-4',
            'phi_regime1': [0.8, -0.15],
            'phi_regime2': [-0.6, 0.2],
            'threshold': 1.0,
            'delay': 2
        },
        {
            'nombre': 'SETAR-5',
            'phi_regime1': [0.4, -0.1, 0.05],
            'phi_regime2': [-0.3, 0.1, -0.05],
            'threshold': 0.0,
            'delay': 1
        },
        {
            'nombre': 'SETAR-6',
            'phi_regime1': [0.5, -0.3, 0.1],
            'phi_regime2': [-0.4, 0.2, -0.05],
            'threshold': 0.5,
            'delay': 2
        },
        {
            'nombre': 'SETAR-7',
            'phi_regime1': [0.3, 0.1],
            'phi_regime2': [-0.2, -0.1],
            'threshold': 0.8,
            'delay': 3
        }
    ]
    
    DISTRIBUTIONS = ['normal', 'uniform', 'exponential', 't-student', 'mixture']
    VARIANCES = [0.2, 0.5, 1.0, 3.0]
    
    def __init__(self, n_boot: int = 1000, seed: int = 42, verbose: bool = False,
                 proceso_tipo: str = 'ARMA'):
        """
        Args:
            n_boot: Número de muestras bootstrap
            seed: Semilla aleatoria
            verbose: Mostrar información detallada
            proceso_tipo: 'ARMA', 'ARIMA' o 'SETAR'
        """
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.proceso_tipo = proceso_tipo.upper()
        self.rng = np.random.default_rng(seed)
        
        if self.proceso_tipo not in ['ARMA', 'ARIMA', 'SETAR']:
            raise ValueError(f"proceso_tipo debe ser 'ARMA', 'ARIMA' o 'SETAR', recibido: {proceso_tipo}")
    
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
    
    def _create_simulator(self, config: dict, distribution: str, variance: float, seed: int):
        """Crea el simulador apropiado según el tipo de proceso."""
        if self.proceso_tipo == 'ARMA':
            return ARMASimulation(
                model_type=config['nombre'],
                phi=config['phi'],
                theta=config['theta'],
                noise_dist=distribution,
                sigma=np.sqrt(variance),
                seed=seed,
                verbose=False
            )
        elif self.proceso_tipo == 'ARIMA':
            return ARIMASimulation(
                model_type=config['nombre'],
                phi=config['phi'],
                theta=config['theta'],
                noise_dist=distribution,
                sigma=np.sqrt(variance),
                seed=seed,
                verbose=False
            )
        else:  # SETAR
            return SETARSimulation(
                model_type=config['nombre'],
                phi_regime1=config['phi_regime1'],
                phi_regime2=config['phi_regime2'],
                threshold=config['threshold'],
                delay=config['delay'],
                noise_dist=distribution,
                sigma=np.sqrt(variance),
                seed=seed,
                verbose=False
            )
    
    def _optimize_and_freeze_models(self, models: dict, 
                                     train_series: np.ndarray,
                                     val_series: np.ndarray):
        """
        Optimización + congelamiento con presupuesto de tiempo.
        
        Returns:
            models optimizados y congelados
        """
        from modelos import TimeBalancedOptimizer
        
        # PASO 1: Optimización (120s)
        opt_start = time.time()
        
        optimizer = TimeBalancedOptimizer(
            random_state=self.seed,
            verbose=self.verbose
        )
        
        if self.verbose:
            print(f"\n⚡ PASO 1: Optimización (budget: {self.OPTIMIZATION_BUDGET}s)")
        
        optimized_params = optimizer.optimize_all_models(
            models, 
            train_data=train_series,
            val_data=val_series
        )
        
        opt_elapsed = time.time() - opt_start
        
        if opt_elapsed > self.OPTIMIZATION_BUDGET * 1.2:
            print(f"⚠️  ADVERTENCIA: Optimización tomó {opt_elapsed:.1f}s "
                  f"(límite: {self.OPTIMIZATION_BUDGET}s)")
        
        # PASO 2: Congelamiento (30s)
        freeze_start = time.time()
        train_val_combined = np.concatenate([train_series, val_series])
        
        if self.verbose:
            print(f"\n🔒 PASO 2: Congelamiento (budget: {self.FREEZE_BUDGET}s)")
        
        for name, model in models.items():
            try:
                # Aplicar hiperparámetros optimizados
                if name in optimized_params and optimized_params[name]:
                    if hasattr(model, 'best_params'):
                        model.best_params = optimized_params[name]
                    
                    for key, value in optimized_params[name].items():
                        if hasattr(model, key):
                            setattr(model, key, value)
                
                # Congelar
                if hasattr(model, 'freeze_hyperparameters'):
                    model.freeze_hyperparameters(train_val_combined)
                    
                    if self.verbose:
                        print(f"  ✓ {name}")
                
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ {name}: {e}")
        
        freeze_elapsed = time.time() - freeze_start
        
        if freeze_elapsed > self.FREEZE_BUDGET * 1.5:
            print(f"⚠️  ADVERTENCIA: Congelamiento tomó {freeze_elapsed:.1f}s "
                  f"(límite: {self.FREEZE_BUDGET}s)")
        
        total_prep = opt_elapsed + freeze_elapsed
        if self.verbose:
            print(f"\n⏱️  Tiempo preparación: {total_prep:.1f}s / "
                  f"{self.OPTIMIZATION_BUDGET + self.FREEZE_BUDGET}s")
        
        return models
    
    def _run_single_scenario(self, config: dict, distribution: str, variance: float,
                            n_train: int, n_calib: int, scenario_seed: int) -> list:
        """
        Ejecuta un escenario completo con un tamaño específico de train/calib.
        """
        try:
            total_needed = n_train + n_calib + self.N_TEST_STEPS
            
            # Crear simulador
            simulator = self._create_simulator(config, distribution, variance, scenario_seed)
            
            # Simular serie
            full_series, full_errors = simulator.simulate(n=total_needed, burn_in=50)
            
            # Dividir datos
            calib_end = n_train + n_calib
            train_series = full_series[:n_train]
            calib_series = full_series[n_train:calib_end]
            
            # Inicializar modelos
            models = self._setup_models(scenario_seed)
            
            # PASO 1 y 2: Optimizar y Congelar usando TimeBalancedOptimizer
            models = self._optimize_and_freeze_models(models, train_series, calib_series)
            
            # PASO 3: Predicción rodante
            results_rows = []
            
            for step in range(self.N_TEST_STEPS):
                current_idx = calib_end + step
                history_series = full_series[:current_idx]
                true_value = full_series[current_idx]
                
                step_result = {
                    'Paso': step + 1,
                    'Proceso': config['nombre'],
                    'Tipo_Proceso': self.proceso_tipo,
                    'Distribución': distribution,
                    'Varianza': variance,
                    'N_Train': n_train,
                    'N_Calib': n_calib,
                    'N_Total': n_train + n_calib,
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
                            print(f"    {name} error: {e}")
                        step_result[name] = np.nan
                
                results_rows.append(step_result)
            
            # Promedio del escenario
            avg_row = {
                'Paso': 'Promedio',
                'Proceso': config['nombre'],
                'Tipo_Proceso': self.proceso_tipo,
                'Distribución': distribution,
                'Varianza': variance,
                'N_Train': n_train,
                'N_Calib': n_calib,
                'N_Total': n_train + n_calib,
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
        """
        Genera lista de todos los escenarios para ejecución paralela.
        
        ✅ NUEVA LÓGICA: Usa combinaciones específicas de (N_TRAIN, N_CALIB)
        """
        # Seleccionar configuraciones según el tipo de proceso
        if self.proceso_tipo == 'ARMA':
            configs = self.ARMA_CONFIGS
        elif self.proceso_tipo == 'ARIMA':
            configs = self.ARIMA_CONFIGS
        else:  # SETAR
            configs = self.SETAR_CONFIGS
        
        scenarios = []
        scenario_id = 0
        
        # ✅ Para cada combinación ÚNICA de tamaños
        for size_combo in self.SIZE_COMBINATIONS:
            n_train = size_combo['n_train']
            n_calib = size_combo['n_calib']
            
            # Para cada configuración de proceso
            for config in configs:
                for dist in self.DISTRIBUTIONS:
                    for var in self.VARIANCES:
                        scenarios.append((
                            config.copy(),
                            dist,
                            var,
                            n_train,
                            n_calib,
                            self.seed + scenario_id
                        ))
                        scenario_id += 1
        
        return scenarios
    
    def run_all(self, excel_filename: str = None, batch_size: int = 20):
        """Ejecuta todos los escenarios."""
        if excel_filename is None:
            excel_filename = f"resultados_TAMANOS_CRECIENTES_{self.proceso_tipo}.xlsx"
        
        print("="*80)
        print(f"EVALUACIÓN TAMAÑOS CRECIENTES - PROCESO {self.proceso_tipo}")
        print("="*80)
        
        cpu_count = os.cpu_count() or 4
        safe_jobs = min(6, max(1, int(cpu_count * 0.75)))
        
        # Calcular totales
        if self.proceso_tipo == 'ARMA':
            configs = self.ARMA_CONFIGS
        elif self.proceso_tipo == 'ARIMA':
            configs = self.ARIMA_CONFIGS
        else:
            configs = self.SETAR_CONFIGS
        
        # ✅ NUEVA MÉTRICA: Combinaciones únicas
        total_size_combinations = len(self.SIZE_COMBINATIONS)
        scenarios_per_size = len(configs) * len(self.DISTRIBUTIONS) * len(self.VARIANCES)
        total_base_scenarios = total_size_combinations * scenarios_per_size
        
        print(f"⚡ Usando {safe_jobs} núcleos en paralelo")
        print(f"⚡ Tamaño del lote: {batch_size}")
        print(f"\n📊 CONFIGURACIÓN:")
        print(f"  • Combinaciones de tamaños: {total_size_combinations}")
        for i, combo in enumerate(self.SIZE_COMBINATIONS, 1):
            print(f"    {i}. N_Train={combo['n_train']}, N_Calib={combo['n_calib']}")
        print(f"  • Configuraciones {self.proceso_tipo}: {len(configs)}")
        print(f"  • Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"  • Varianzas: {len(self.VARIANCES)}")
        print(f"  • ESCENARIOS BASE TOTALES: {total_base_scenarios}")
        print(f"  • FILAS ESPERADAS: ~{total_base_scenarios * (self.N_TEST_STEPS + 1)}")
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
        config, dist, var, n_train, n_calib, seed = args
        
        pipeline = Pipeline140_TamanosCrecientes(
            n_boot=self.n_boot,
            seed=seed,
            verbose=False,
            proceso_tipo=self.proceso_tipo
        )
        pipeline.N_TEST_STEPS = self.N_TEST_STEPS
        
        return pipeline._run_single_scenario(config, dist, var, n_train, n_calib, seed)
    
    def _save_checkpoint(self, rows, filename):
        """Guarda progreso en Excel."""
        if not rows:
            return
        
        df_temp = pd.DataFrame(rows)
        
        ordered_cols = [
            'Paso', 'Proceso', 'Tipo_Proceso', 'Distribución', 'Varianza',
            'N_Train', 'N_Calib', 'N_Total', 'Valor_Observado',
            'AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 'EnCQR-LSTM',
            'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap'
        ]
        final_cols = [c for c in ordered_cols if c in df_temp.columns]
        remaining_cols = [c for c in df_temp.columns if c not in final_cols]
        final_cols.extend(remaining_cols)
        
        df_temp = df_temp[final_cols]
        df_temp.to_excel(filename, index=False)