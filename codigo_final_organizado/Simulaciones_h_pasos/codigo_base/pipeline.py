import numpy as np
import pandas as pd
import time
import gc
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from modelos import (CircularBlockBootstrapModel, SieveBootstrapModel, LSPM, 
                     DeepARModel, MondrianCPSModel, TimeBalancedOptimizer)
from metricas import crps
from simulacion import ARMASimulation

def clear_all_sessions():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except: pass

class PipelineARMA_100Trayectorias:
    """
    Pipeline ARMA que genera 100 trayectorias estocásticas para predicción 
    multi-paso (h-steps ahead) mediante muestreo recursivo.
    """
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    N_TRAJECTORIES = 100  # Número de trayectorias solicitado

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

    def _setup_models(self, seed):
        return {
            'LSPM': LSPM(random_state=seed),
            'DeepAR': DeepARModel(
                hidden_size=20, n_lags=10, epochs=25, num_samples=self.n_boot,
                random_state=seed, early_stopping_patience=4
            ),
            'Sieve Bootstrap': SieveBootstrapModel(n_boot=self.n_boot, random_state=seed),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed)
        }

    def run_single_scenario(self, arma_config: dict, dist: str, var: float, rep: int) -> list:
        scenario_seed = self.seed + rep
        
        # 1. Simulación
        simulator = ARMASimulation(
            phi=arma_config['phi'], theta=arma_config['theta'],
            noise_dist=dist, sigma=np.sqrt(var), seed=scenario_seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        full_series, _ = simulator.simulate(n=total_len, burn_in=50)
        
        train_series = full_series[:self.N_TRAIN]
        val_series = full_series[self.N_TRAIN:self.N_TRAIN + self.N_VALIDATION]
        test_series = full_series[self.N_TRAIN + self.N_VALIDATION:]
        
        # 2. Optimización y Freeze
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        
        # Optimizar con Train y Val
        best_params = optimizer.optimize_all_models(models, train_series, val_series)
        train_val_combined = np.concatenate([train_series, val_series])
        
        for name, model in models.items():
            if name in best_params and best_params[name]:
                for k, v in best_params[name].items():
                    if hasattr(model, k): setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_combined)

        # 3. Generación de 100 Trayectorias Estocásticas
        # Diccionario para guardar: {modelo: matriz(100 trayectorias x 12 pasos)}
        model_forecasts = {name: np.zeros((self.N_TRAJECTORIES, self.N_TEST_STEPS)) for name in models.keys()}
        
        for name, model in models.items():
            for i in range(self.N_TRAJECTORIES):
                # Cada trayectoria empieza con el historial real (Train + Val)
                current_history = train_val_combined.copy()
                
                for h in range(self.N_TEST_STEPS):
                    # Predicción del siguiente paso basada en lo que el modelo cree que pasó antes
                    if name == 'Sieve Bootstrap':
                        pred_dist = model.fit_predict(current_history)
                    else:
                        pred_dist = model.fit_predict(pd.DataFrame({'valor': current_history}))
                    
                    # Seleccionar aleatoriamente un valor de la distribución (equivale a elegir un cuantil aleatorio)
                    sampled_val = self.rng.choice(np.asarray(pred_dist).flatten())
                    
                    # Guardar en la trayectoria y actualizar historial para el paso h+1
                    model_forecasts[name][i, h] = sampled_val
                    current_history = np.append(current_history, sampled_val)
            
            clear_all_sessions()

        # 4. Cálculo de Métricas (CRPS por cada paso h)
        results_rows = []
        for h in range(self.N_TEST_STEPS):
            row = {
                'Paso_H': h + 1,
                'proces_simulacion': arma_config['nombre'],
                'Distribución': dist,
                'Varianza': var,
                'Valor_Real': test_series[h]
            }
            
            for name in models.keys():
                # La distribución para el paso H son los 100 valores de las 100 trayectorias en ese h
                dist_h = model_forecasts[name][:, h]
                row[name] = crps(dist_h, test_series[h])
            
            results_rows.append(row)
            
        return results_rows

    def run_all(self, filename="resultados_trayectorias_ARMA.xlsx"):
        scenarios = []
        scenario_id = 0
        for arma_cfg in self.ARMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((arma_cfg.copy(), dist, var, scenario_id))
                    scenario_id += 1
        
        all_results = []
        print(f"🚀 Iniciando evaluación de {len(scenarios)} escenarios con {self.N_TRAJECTORIES} trayectorias cada uno...")
        
        # Procesamiento en paralelo
        results = Parallel(n_jobs=4, backend='loky')(
            delayed(self.run_single_scenario)(*s) for s in tqdm(scenarios)
        )
        
        for r in results: all_results.extend(r)
        
        df = pd.DataFrame(all_results)
        df.to_excel(filename, index=False)
        print(f"✅ Proceso completado. Resultados guardados en {filename}")
        return df
