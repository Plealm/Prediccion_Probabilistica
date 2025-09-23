import numpy as np
import pandas as pd
from IPython.display import display
from typing import Dict, List, Any
import concurrent.futures
import os
from tqdm import tqdm

# Importar las clases de los modelos y el simulador
from arma_simulation import ARMASimulation
from models import (
    EnhancedBootstrappingModel, LSPM, LSPMW, DeepARModel, 
    SieveBootstrap, AREPD, MondrianCPSModel, EnCQR_LSTM_Model
)
from metrics import ecrps

# ==============================================================================
# FUNCIÓN AUXILIAR PARA PROCESAR UN MODELO EN UN PROCESO SEPARADO
# ==============================================================================
# <-- CAMBIO: La firma de la función ahora acepta 'model_name' explícitamente.
def _process_model_wrapper(model_name: str, model_info: Dict, sim_config_dict: Dict, 
                           history_df: pd.DataFrame, history_series: np.ndarray, 
                           history_errors: np.ndarray, 
                           random_state: int, verbose: bool) -> Dict:
    """
    Wrapper que encapsula la lógica para procesar un único modelo.
    Esta función es serializable y será ejecutada en un proceso separado.
    """
    # <-- CAMBIO: Ya no se extrae 'model_name' del diccionario 'model_info'.
    ModelClass = model_info['class']
    model_init_args = model_info['init_args']

    local_rng = np.random.default_rng(random_state)
    local_simulator_instance = ARMASimulation(**sim_config_dict, seed=random_state)

    reference_noise_for_opt = local_simulator_instance.get_true_next_step_samples(history_series, history_errors, 5000)
    theoretical_samples = local_simulator_instance.get_true_next_step_samples(history_series, history_errors, 20000)

    if ModelClass == EnhancedBootstrappingModel:
        model_instance = ModelClass(sim_config_dict, random_state=random_state, verbose=verbose, **model_init_args)
    else:
        model_instance = ModelClass(random_state=random_state, verbose=verbose, **model_init_args)

    if verbose: print(f"--- Iniciando procesamiento para: {model_name} ---")

    try:
        if hasattr(model_instance, 'optimize_hyperparameters'):
            model_instance.optimize_hyperparameters(history_df, reference_noise_for_opt)
        elif hasattr(model_instance, 'grid_search'):
            model_instance.grid_search(history_series)
        
        if model_name == 'Block Bootstrapping':
            prediction_output = model_instance.fit_predict(history_series) 
            samples = np.array(prediction_output[0]).flatten()
        else:
            prediction_output = model_instance.fit_predict(history_df)
            if isinstance(prediction_output, list) and prediction_output and isinstance(prediction_output[0], dict):
                values = [d['value'] for d in prediction_output]
                probs = np.array([d['probability'] for d in prediction_output])
                if np.sum(probs) > 1e-9: probs /= np.sum(probs)
                else: probs = np.ones(len(values)) / len(values)
                samples = local_rng.choice(values, size=5000, p=probs, replace=True)
            else:
                samples = np.array(prediction_output).flatten()

        ecrps_value = ecrps(samples, theoretical_samples)
        
        if verbose: print(f"--- Finalizado procesamiento para: {model_name} ---")
        return {'name': model_name, 'samples': samples, 'ecrps_value': ecrps_value}

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        if verbose: print(f"¡ERROR en modelo '{model_name}': {e}\n{error_trace}")
        return {'name': model_name, 'samples': np.array([]), 'ecrps_value': np.nan}


class Pipeline:
    """
    Orquesta un backtesting con ventana rodante para múltiples modelos.
    (Versión PARALELIZADA para máxima velocidad).
    """
    N_TEST_STEPS = 10

    def __init__(self, model_type='ARMA(1,1)', phi=[0.7], theta=[0.3], sigma=1.2, noise_dist='t-student', n_samples=250, seed=42, verbose=False):
        self.config = {
            'model_type': model_type, 'phi': phi, 'theta': theta, 'sigma': sigma,
            'noise_dist': noise_dist, 'n_samples': n_samples, 'seed': seed,
            'verbose': verbose
        }
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        self.simulator = None
        self.full_series, self.full_errors, self.burn_in_len = None, None, None
        self.rolling_ecrps: List[Dict] = []
        self.initial_train_len = None

    def _get_model_definitions(self) -> Dict:
        """Define las clases de modelos y sus parámetros iniciales."""
        p_order = len(self.config['phi']) if self.config['phi'] else 2

        return { 
            'Block Bootstrapping': {'class': EnhancedBootstrappingModel, 'init_args': {}},
            'Sieve Bootstrap': {'class': SieveBootstrap, 'init_args': {'p_order': p_order}},
            'LSPM': {'class': LSPM, 'init_args': {}},
            'LSPMW': {'class': LSPMW, 'init_args': {}},
            'AREPD': {'class': AREPD, 'init_args': {}},
            'DeepAR': {'class': DeepARModel, 'init_args': {'epochs': 20, 'num_samples': 5000}}, 
            'EnCQR-LSTM': {'class': EnCQR_LSTM_Model, 'init_args': {'epochs': 8, 'B': 2, 'units': 32, 'n_layers': 1, 'num_samples': 5000}},
            'MCPS': {'class': MondrianCPSModel, 'init_args': {}}
        }
    
    def execute(self):
        """Ejecuta el pipeline de forma robusta y paralela."""
        
        sim_config_for_init = {k: v for k, v in self.config.items() if k not in ['n_samples', 'verbose']}
        self.simulator = ARMASimulation(**sim_config_for_init)
        
        full_series_with_burn_in, _ = self.simulator.simulate(n=self.config['n_samples'], burn_in=50, return_just_series=True)
        self.burn_in_len = 50 
        
        self.full_series, self.full_errors = self.simulator.simulate(n=self.config['n_samples'], burn_in=50, return_just_series=False)
        self.initial_train_len = len(self.full_series) - self.N_TEST_STEPS
        
        all_model_definitions = self._get_model_definitions()
        
        sim_config_dict_for_children = {
            'model_type': self.config['model_type'], 'phi': self.config['phi'], 
            'theta': self.config['theta'], 'sigma': self.config['sigma'],
            'noise_dist': self.config['noise_dist']
        }

        if self.verbose: print(f"\n--- 2. Iniciando Backtesting Paralelo Robusto para {self.N_TEST_STEPS} pasos ---")
        
        for t in tqdm(range(self.N_TEST_STEPS), desc="Progreso del Backtesting"):
            step_t = self.initial_train_len + t
            if self.verbose: print(f"\n{'='*40}\n >> Paso {t+1}/{self.N_TEST_STEPS} (Prediciendo para t={step_t}) \n{'='*40}")
            
            history_series = self.full_series[:step_t]
            history_errors = self.full_errors[:step_t]
            df_history = pd.DataFrame({'valor': history_series})
            
            step_ecrps = {'Paso': step_t}
            
            max_p_workers = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_p_workers) as executor:
                futures = {
                    executor.submit(
                        _process_model_wrapper, 
                        name,                                    # <-- CAMBIO: Se pasa 'name' como primer argumento.
                        model_def,
                        sim_config_dict_for_children,
                        df_history, history_series, history_errors,
                        self.config['seed'] + t,
                        self.verbose
                    ): name for name, model_def in all_model_definitions.items()
                }
                
                for future in concurrent.futures.as_completed(futures):
                    model_name_completed = futures[future]
                    try:
                        result = future.result()
                        step_ecrps[model_name_completed] = result['ecrps_value']
                    except Exception as exc:
                        print(f"¡ERROR fatal procesando '{model_name_completed}' en el Pipeline: {exc}")
                        step_ecrps[model_name_completed] = np.nan
            
            self.rolling_ecrps.append(step_ecrps)
            
        self._display_results(full_series_with_burn_in)
        
    def _display_results(self, full_series_with_burn_in: np.ndarray):
        print(f"\n{'='*70}\nResultados Finales del Backtesting\n{'='*70}")
        
        print("\n[RESUMEN] Las gráficas de series y densidades han sido deshabilitadas.")
        print(f"Serie simulada con {len(full_series_with_burn_in)} puntos (incluyendo {self.burn_in_len} de burn-in).")
        print(f"Evaluación realizada sobre los últimos {self.N_TEST_STEPS} pasos.")
        
        print("\n[SALIDA] Tabla de ECRPS por Paso con Mejor Modelo")
        if not self.rolling_ecrps:
            print("No se generaron resultados de ECRPS.")
            return

        ecrps_df = pd.DataFrame(self.rolling_ecrps).set_index('Paso')
        
        ecrps_df_clean = ecrps_df.dropna(axis=1, how='all')

        if not ecrps_df_clean.empty:
            ecrps_df['Mejor Modelo'] = ecrps_df_clean.idxmin(axis=1)
            averages = ecrps_df_clean.drop(columns='Mejor Modelo', errors='ignore').mean(numeric_only=True)
            
            if not averages.empty:
                best_overall_model = averages.idxmin()
            else:
                best_overall_model = "N/A"

            ecrps_df.loc['Promedio'] = averages
            ecrps_df.loc['Promedio', 'Mejor Modelo'] = best_overall_model
        else:
            print("No hay modelos válidos con resultados de ECRPS para mostrar.")

            display(ecrps_df)