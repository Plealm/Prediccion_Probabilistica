import numpy as np
import pandas as pd
import warnings
import gc
import os
from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

from simulacion import ARMASimulation
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


def _run_scenario_wrapper(args):
    """
    Función auxiliar para ejecutar un escenario en un proceso separado.
    MODIFICADA: Ahora incluye parámetros del simulador para generar distribución verdadera.
    """
    (arma_cfg, dist, var, seed, n_test_steps, 
     n_train, n_calib, n_boot) = args
    
    pipeline = Pipeline140SinSesgos(n_boot=n_boot, seed=seed)
    pipeline.N_TEST_STEPS = n_test_steps
    pipeline.N_TRAIN = n_train
    pipeline.N_CALIBRATION = n_calib
    
    return pipeline._run_single_scenario(arma_cfg, dist, var, seed)


class Pipeline140SinSesgos:
    """
    Pipeline optimizado con optimización usando DISTRIBUCIÓN VERDADERA del simulador.
    """
    
    N_TEST_STEPS = 10
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
        NUEVO MÉTODO: Genera la distribución verdadera del siguiente paso
        usando el simulador ARMA con los parámetros reales.
        
        Args:
            simulator: Instancia de ARMASimulation con los parámetros verdaderos
            series_history: Historia de la serie X_1, ..., X_n
            errors_history: Historia de los errores ε_1, ..., ε_n
            n_samples: Número de muestras a generar
            
        Returns:
            Array con n_samples muestras de la distribución verdadera de X_{n+1}
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
        NUEVO MÉTODO: Optimiza hiperparámetros usando la distribución verdadera.
        
        Esta es la implementación "limpia" que usa el conocimiento del DGP
        para generar la referencia perfecta.
        """
        # Generar distribución verdadera para el punto justo después de train+calib
        true_distribution = self._generate_true_distribution(
            simulator=simulator,
            series_history=train_calib_series,
            errors_history=train_calib_errors,
            n_samples=5000
        )
        
        if self.verbose:
            print(f"  Distribución verdadera generada: media={np.mean(true_distribution):.4f}, "
                  f"std={np.std(true_distribution):.4f}")
        
        # Optimizar cada modelo contra la distribución verdadera
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

    def _freeze_bootstrap_models(self, models: dict, train_calib_series: np.ndarray):
        """Congela hiperparámetros de modelos bootstrap."""
        for name, model in models.items():
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_calib_series)

    def _run_single_scenario(self, arma_config: dict, distribution: str, 
                            variance: float, scenario_seed: int) -> list:
        """
        Ejecuta un escenario completo.
        MODIFICADO: Usa distribución verdadera para optimización.
        """
        try:
            total_needed = self.N_TRAIN + self.N_CALIBRATION + self.N_TEST_STEPS
            
            # Crear simulador (lo necesitamos para generar distribución verdadera)
            simulator = ARMASimulation(
                model_type=arma_config['nombre'],
                phi=arma_config['phi'],
                theta=arma_config['theta'],
                noise_dist=distribution,
                sigma=np.sqrt(variance),
                seed=scenario_seed,
                verbose=False
            )
            
            # Simular serie completa CON errores (necesarios para distribución verdadera)
            full_series, full_errors = simulator.simulate(n=total_needed, burn_in=50)
            
            # Dividir datos
            calib_end = self.N_TRAIN + self.N_CALIBRATION
            train_calib_series = full_series[:calib_end]
            train_calib_errors = full_errors[:calib_end]
            
            # Inicializar modelos
            models = self._setup_models(scenario_seed)
            
            # ========================================================
            # CAMBIO PRINCIPAL: Optimización con distribución verdadera
            # ========================================================
            self._optimize_models_con_distribucion_verdadera(
                models=models,
                train_calib_series=train_calib_series,
                train_calib_errors=train_calib_errors,
                simulator=simulator
            )
            
            # Congelar modelos bootstrap
            self._freeze_bootstrap_models(models, train_calib_series)
            
            results_rows = []
            
            # Ventana de test (10 pasos)
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
                        if isinstance(model, (CircularBlockBootstrapModel, SieveBootstrapModel)):
                            pred_samples = model.fit_predict(history_series)
                        else:
                            df_hist = pd.DataFrame({'valor': history_series})
                            pred_samples = model.fit_predict(df_hist)
                        
                        pred_samples = np.asarray(pred_samples).flatten()
                        crps_val = crps(pred_samples, true_value)
                        step_result[name] = crps_val
                        
                    except Exception as e:
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
    
    def run_all(self, excel_filename: str = "resultados_140_DIST_VERDADERA.xlsx", 
                batch_size: int = 10):
        """
        Ejecuta los escenarios en lotes con optimización basada en distribución verdadera.
        """
        print("="*80)
        print("EVALUACIÓN 140 ESCENARIOS - DISTRIBUCIÓN VERDADERA PARA OPTIMIZACIÓN")
        print("="*80)
        
        cpu_count = os.cpu_count() or 4
        safe_jobs = min(6, max(1, int(cpu_count * 0.75)))
        
        print(f"⚡ Usando {safe_jobs} núcleos en paralelo")
        print(f"⚡ Tamaño del lote: {batch_size}")
        print(f"⚡ Método de optimización: Distribución Verdadera del Simulador")
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
                    delayed(_run_scenario_wrapper)(args) 
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