import numpy as np
import pandas as pd
import gc
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from modelos import (LSPM, DeepARModel, SieveBootstrapModel, 
                     MondrianCPSModel, TimeBalancedOptimizer)
from metricas import ecrps
from simulacion import ARMASimulation
warnings.filterwarnings("ignore")


def clear_all_sessions():
    """Limpia memoria y cache GPU si está disponible."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
    except: 
        pass


class PipelineARMA_100Trayectorias:
    """
    Pipeline ARMA que genera 100 trayectorias estocásticas para predicción 
    multi-paso (h-steps ahead) mediante muestreo recursivo.
    
    Compara distribuciones de modelos (100 trayectorias) contra la distribución 
    teórica del proceso (1000 trayectorias) usando ECRPS.
    """
    N_TEST_STEPS = 12
    N_VALIDATION = 40
    N_TRAIN = 200
    N_TRAJECTORIES_MODEL = 100    # Trayectorias por modelo
    N_TRAJECTORIES_TRUE = 1000    # Trayectorias para la "Verdad Teórica"

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
        """Inicializa los modelos con configuración estándar."""
        return {
            'LSPM': LSPM(random_state=seed),
            'DeepAR': DeepARModel(
                hidden_size=20, n_lags=10, epochs=25, 
                num_samples=self.n_boot,
                random_state=seed, early_stopping_patience=4
            ),
            'Sieve Bootstrap': SieveBootstrapModel(
                n_boot=self.n_boot, random_state=seed
            ),
            'MCPS': MondrianCPSModel(n_lags=10, random_state=seed)
        }

    def _generate_true_distribution(self, simulator, history, steps):
        """
        Genera la 'Verdad Teórica' proyectando el proceso real 
        N_TRAJECTORIES_TRUE veces desde el presente hacia el futuro.
        
        Args:
            simulator: Instancia de ARMASimulation con parámetros reales
            history: Serie histórica observada
            steps: Número de pasos a proyectar
            
        Returns:
            np.ndarray: Matriz (N_TRAJECTORIES_TRUE, steps) con trayectorias teóricas
        """
        true_forecasts = np.zeros((self.N_TRAJECTORIES_TRUE, steps))
        
        p = len(simulator.phi)
        q = len(simulator.theta)
        max_lag = max(p, q, 1)
        
        for i in range(self.N_TRAJECTORIES_TRUE):
            # Crear una nueva instancia del simulador con seed diferente para cada trayectoria
            temp_simulator = ARMASimulation(
                phi=simulator.phi.tolist() if hasattr(simulator.phi, 'tolist') else simulator.phi,
                theta=simulator.theta.tolist() if hasattr(simulator.theta, 'tolist') else simulator.theta,
                noise_dist=simulator.noise_dist,
                sigma=simulator.sigma,
                seed=simulator.seed + i if simulator.seed else None
            )
            
            # Generar trayectoria completa con condiciones iniciales del historial
            # Concatenar historial + nueva proyección
            full_length = len(history) + steps
            full_series, full_errors = temp_simulator.simulate(n=full_length, burn_in=50)
            
            # Extraer solo los últimos 'steps' pasos (la proyección futura)
            true_forecasts[i, :] = full_series[-steps:]
            
        return true_forecasts

    def run_single_scenario(self, arma_config: dict, dist: str, var: float, rep: int) -> tuple:
        """
        Ejecuta un escenario completo: simula datos, optimiza modelos, genera trayectorias
        y calcula ECRPS contra distribución teórica.
        
        Args:
            arma_config: Configuración ARMA (phi, theta, nombre)
            dist: Distribución del ruido
            var: Varianza del ruido
            rep: ID del escenario para semilla
            
        Returns:
            tuple: (results_rows, predictions_dict, scenario_name)
                - results_rows: Lista de diccionarios con resultados por paso
                - predictions_dict: Dict con distribuciones por paso para graficar
                - scenario_name: Nombre identificador del escenario
        """
        scenario_seed = self.seed + rep
        scenario_name = f"{arma_config['nombre']}_{dist}_var{var}"
        
        if self.verbose:
            print(f"\n🔄 Procesando: {scenario_name}")
        
        # 1. SIMULACIÓN: Generar serie temporal completa
        simulator = ARMASimulation(
            phi=arma_config['phi'], theta=arma_config['theta'],
            noise_dist=dist, sigma=np.sqrt(var), seed=scenario_seed
        )
        total_len = self.N_TRAIN + self.N_VALIDATION + self.N_TEST_STEPS
        full_series, _ = simulator.simulate(n=total_len, burn_in=50)
        
        train_series = full_series[:self.N_TRAIN]
        val_series = full_series[self.N_TRAIN:self.N_TRAIN + self.N_VALIDATION]
        train_val_combined = np.concatenate([train_series, val_series])
        
        # 2. DISTRIBUCIÓN TEÓRICA: Generar 1000 trayectorias del proceso real
        true_dist_paths = self._generate_true_distribution(
            simulator, train_val_combined, self.N_TEST_STEPS
        )
        
        # 3. OPTIMIZACIÓN: Entrenar y optimizar modelos
        models = self._setup_models(scenario_seed)
        optimizer = TimeBalancedOptimizer(random_state=scenario_seed, verbose=self.verbose)
        
        best_params = optimizer.optimize_all_models(models, train_series, val_series)
        
        # Aplicar mejores hiperparámetros y congelar
        for name, model in models.items():
            if name in best_params and best_params[name]:
                for k, v in best_params[name].items():
                    if hasattr(model, k): 
                        setattr(model, k, v)
            if hasattr(model, 'freeze_hyperparameters'):
                model.freeze_hyperparameters(train_val_combined)

        # 4. GENERACIÓN DE TRAYECTORIAS: 100 por modelo mediante aleatorización recursiva
        model_forecasts = {
            name: np.zeros((self.N_TRAJECTORIES_MODEL, self.N_TEST_STEPS)) 
            for name in models.keys()
        }
        
        for name, model in models.items():
            if self.verbose:
                print(f"  Generando {self.N_TRAJECTORIES_MODEL} trayectorias para {name}...")
            
            for i in range(self.N_TRAJECTORIES_MODEL):
                # Cada trayectoria empieza con el historial real (Train + Val)
                current_history = train_val_combined.copy()
                
                for h in range(self.N_TEST_STEPS):
                    # Predicción del siguiente paso basada en historial actual
                    if name == 'Sieve Bootstrap':
                        pred_dist = model.fit_predict(current_history)
                    else:
                        pred_dist = model.fit_predict(pd.DataFrame({'valor': current_history}))
                    
                    # ALEATORIZACIÓN: Seleccionar valor aleatorio de la distribución predictiva
                    sampled_val = self.rng.choice(np.asarray(pred_dist).flatten())
                    
                    # Guardar en trayectoria y actualizar historial para el paso h+1
                    model_forecasts[name][i, h] = sampled_val
                    current_history = np.append(current_history, sampled_val)
            
            clear_all_sessions()

        # 5. PREPARAR DATOS PARA GRÁFICOS
        predictions_dict = {}
        for h in range(self.N_TEST_STEPS):
            predictions_dict[h] = {
                'true_distribution': true_dist_paths[:, h],  # 1000 muestras teóricas
                'model_predictions': {
                    name: model_forecasts[name][:, h]  # 100 muestras por modelo
                    for name in models.keys()
                }
            }

        # 6. CÁLCULO DE MÉTRICAS: ECRPS por cada paso h
        results_rows = []
        for h in range(self.N_TEST_STEPS):
            row = {
                'Paso_H': h + 1,
                'Proceso': arma_config['nombre'],
                'Distribución': dist,
                'Varianza': var
            }
            
            # La distribución teórica para el paso H (1000 valores)
            true_samples_h = true_dist_paths[:, h]
            
            for name in models.keys():
                # La distribución del modelo para el paso H (100 valores)
                model_samples_h = model_forecasts[name][:, h]
                
                # ECRPS: Compara distribución modelo vs distribución teórica
                row[name] = ecrps(model_samples_h, true_samples_h)
            
            results_rows.append(row)
        
        return results_rows, predictions_dict, scenario_name

    def run_all(self, excel_filename="resultados_trayectorias_ARMA.xlsx", 
                n_jobs=4):
        """
        Ejecuta todos los escenarios en paralelo y guarda resultados en Excel.
        
        Args:
            excel_filename: Nombre del archivo Excel de salida
            n_jobs: Número de procesos paralelos
            
        Returns:
            pd.DataFrame: DataFrame con todos los resultados
        """
        # Generar lista de escenarios
        scenarios = []
        scenario_id = 0
        for arma_cfg in self.ARMA_CONFIGS:
            for dist in self.DISTRIBUTIONS:
                for var in self.VARIANCES:
                    scenarios.append((arma_cfg.copy(), dist, var, scenario_id))
                    scenario_id += 1
        
        print("="*80)
        print(f"🚀 INICIANDO EVALUACIÓN DE {len(scenarios)} ESCENARIOS")
        print("="*80)
        print(f"📊 Configuración:")
        print(f"   - Trayectorias por modelo: {self.N_TRAJECTORIES_MODEL}")
        print(f"   - Trayectorias teóricas: {self.N_TRAJECTORIES_TRUE}")
        print(f"   - Pasos de predicción: {self.N_TEST_STEPS}")
        print(f"   - Procesos ARMA: {len(self.ARMA_CONFIGS)}")
        print(f"   - Distribuciones: {len(self.DISTRIBUTIONS)}")
        print(f"   - Varianzas: {len(self.VARIANCES)}")
        print(f"   - Procesos paralelos: {n_jobs}")
        print()
        
        # Procesamiento en paralelo
        all_results = []
        all_predictions = {}
        
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(self.run_single_scenario)(*s) for s in tqdm(scenarios, desc="Escenarios")
        )
        
        # Consolidar resultados
        for idx, (rows, preds, name) in enumerate(results):
            all_results.extend(rows)
            all_predictions[name] = preds
        
        # Guardar en Excel
        df = pd.DataFrame(all_results)
        df.to_excel(excel_filename, index=False)
        
        print()
        print("="*80)
        print("✅ PROCESO COMPLETADO")
        print("="*80)
        print(f"📁 Resultados guardados en: {excel_filename}")
        print(f"📊 Total de filas: {len(df)}")
        print(f"📈 Columnas: {list(df.columns)}")
        print()
        
        # Almacenar predicciones para uso posterior en gráficos
        self._predictions_cache = all_predictions
        
        return df

    def get_predictions_dict(self):
        """
        Retorna el diccionario de predicciones del último run_all().
        Útil para generar gráficos después de la ejecución.
        """
        if hasattr(self, '_predictions_cache'):
            return self._predictions_cache
        else:
            raise ValueError("No hay predicciones disponibles. Ejecuta run_all() primero.")