# pipeline.py
import numpy as np
import pandas as pd
from IPython.display import display
from typing import Dict, List

# Asegúrate de que tus importaciones sean correctas
from simulacion import ARMASimulation
from plot import PlotManager
from modelos import EnhancedBootstrappingModel, LSPM, LSPMW, AREPD, DeepARModel, SieveBootstrap
from metrica import ecrps

class Pipeline:
    """
    Orquesta un backtesting con ventana rodante para múltiples modelos,
    incluyendo un paso de optimización de hiperparámetros en cada paso.
    """
    N_TEST_STEPS = 10

    def __init__(self, model_type='ARMA(1,1)', phi=[0.7], theta=[0.3], sigma=1.2, noise_dist='t-student', n_samples=250, seed=42, verbose=False):
        self.config = {
            'model_type': model_type, 'phi': phi, 'theta': theta, 'sigma': sigma,
            'noise_dist': noise_dist, 'n_samples': n_samples, 'seed': seed
        }
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        
        self.simulator, self.full_series, self.full_errors, self.burn_in_len = None, None, None, None
        self.rolling_ecrps: List[Dict] = []

    def _setup_models(self) -> Dict:
        """Inicializa todos los modelos a ser evaluados."""
        seed = self.config['seed']
        p_order = len(self.config['phi']) if self.config['phi'] else 2
        
        return { 
            'Block Bootstrapping': EnhancedBootstrappingModel(self.simulator, random_state=seed, verbose=self.verbose),
            'Sieve Bootstrap': SieveBootstrap(p_order=p_order, random_state=seed, verbose=self.verbose),
            'LSPM': LSPM(random_state=seed, verbose=self.verbose),
            'LSPMW': LSPMW(random_state=seed, verbose=self.verbose),
            'AREPD': AREPD(random_state=seed, verbose=self.verbose),
            'DeepAR': DeepARModel(random_state=seed, verbose=self.verbose, epochs=20) 
        }

    def execute(self):
        """Ejecuta el pipeline completo."""
        
        # --- 1. SIMULACIÓN INICIAL ---
        if self.verbose: print("--- 1. Ejecutando Simulación Inicial ---")
        sim_config = {k: v for k, v in self.config.items() if k != 'n_samples'}
        self.simulator = ARMASimulation(**sim_config)
        self.full_series, self.full_errors = self.simulator.simulate(n=self.config['n_samples'], burn_in=50)
        self.burn_in_len = 50 

        initial_train_len = len(self.full_series) - self.N_TEST_STEPS
        
        # --- PREPARACIÓN DE MODELOS Y COLORES ---
        models = self._setup_models()
        model_keys = list(models.keys())
        colors = PlotManager._STYLE['default_colors']
        color_map = {name: colors[i % len(colors)] for i, name in enumerate(model_keys)}
        color_map['Teórica'] = '#000000' # Negro para la distribución teórica

        # --- 2. BUCLE DE VENTANA RODANTE ---
        if self.verbose: print(f"\n--- 2. Iniciando Backtesting de Ventana Rodante para {self.N_TEST_STEPS} pasos ---")
        
        for t in range(self.N_TEST_STEPS):
            step_t = initial_train_len + t
            print(f"\n{'='*40}\n >> Paso {t+1}/{self.N_TEST_STEPS} (Prediciendo para t={step_t}) \n{'='*40}")
            
            history_series = self.full_series[:step_t]
            history_errors = self.full_errors[:step_t]
            df_history = pd.DataFrame({'valor': history_series})

            # --- 2.1 [MODIFICADO] RE-OPTIMIZACIÓN DE HIPERPARÁMETROS EN CADA PASO ---
            if self.verbose: print(f"\n--- Optimizando Hiperparámetros en {len(history_series)} puntos ---")
            
            reference_noise_for_opt = self.simulator.get_true_next_step_samples(history_series, history_errors, 5000)

            for name, model in models.items():
                if hasattr(model, 'optimize_hyperparameters'):
                    model.optimize_hyperparameters(df_history, reference_noise_for_opt)
                elif hasattr(model, 'grid_search'):
                    temp_model_for_gs = EnhancedBootstrappingModel(self.simulator, random_state=self.config['seed'], verbose=self.verbose)
                    temp_model_for_gs.arma_simulator.series = history_series
                    temp_model_for_gs.grid_search()
                    model.n_lags = temp_model_for_gs.n_lags 
                    if self.verbose: print(f"-> {name} n_lags actualizado a: {model.n_lags}")
                else:
                     if hasattr(model, 'n_lags'):
                         model.n_lags = len(self.config['phi']) if self.config['phi'] else 2

            # --- 2.2 PREDICCIÓN Y EVALUACIÓN DEL PASO ACTUAL ---
            theoretical_samples = self.simulator.get_true_next_step_samples(history_series, history_errors, 20000)
            
            step_ecrps = {'Paso': step_t}
            step_distributions = {'Teórica': theoretical_samples}
            
            for name, model in models.items():
                fit_predict_data = df_history
                if name == 'Block Bootstrapping':
                    # Este modelo requiere una lógica especial para adaptarse a la ventana rodante
                    bb_model_step = EnhancedBootstrappingModel(self.simulator, random_state=self.config['seed'])
                    bb_model_step.n_lags = model.n_lags # Usar lags optimizados
                    bb_model_step.arma_simulator.series = history_series
                    prediction_output = bb_model_step.fit_predict(history_series)
                    samples = np.array(prediction_output[0]).flatten() # Tomar solo la primera predicción
                else:
                    prediction_output = model.fit_predict(fit_predict_data)
                    if isinstance(prediction_output, list) and prediction_output and isinstance(prediction_output[0], dict):
                        values = [d['value'] for d in prediction_output]
                        probs = np.array([d['probability'] for d in prediction_output])
                        if np.sum(probs) > 1e-9: probs /= np.sum(probs)
                        else: probs = np.ones(len(values)) / len(values)
                        samples = self.rng.choice(values, size=5000, p=probs, replace=True)
                    else: # Para modelos como DeepAR que devuelven un array
                        samples = np.array(prediction_output).flatten()
                
                step_distributions[name] = samples
                step_ecrps[name] = ecrps(samples, theoretical_samples)

            self.rolling_ecrps.append(step_ecrps)
            
            # --- 2.3 [MODIFICADO] GENERAR GRÁFICO DE DENSIDADES PARA EL PASO ACTUAL ---
            metrics_for_plot = {name: val for name, val in step_ecrps.items() if name != 'Paso'}
            plot_title = f"Comparación de Densidades Predictivas en el Paso t={step_t}"
            PlotManager.plot_density_comparison(step_distributions, metrics_for_plot, plot_title, color_map)

        # --- 3. MOSTRAR RESULTADOS FINALES ---
        self._display_results()

    def _display_results(self):
        """Formatea y muestra todas las salidas finales solicitadas."""
        print(f"\n{'='*70}\nResultados Finales del Backtesting\n{'='*70}")
        
        print("\n[SALIDA 1/2] Gráfico de la Serie Temporal con Divisiones")
        series_with_burn_in, _ = self.simulator.simulate(n=self.config['n_samples'], burn_in=self.burn_in_len, return_just_series=True)
        PlotManager.plot_series_split(series_with_burn_in, self.burn_in_len, self.N_TEST_STEPS)
        
        print("\n[SALIDA 2/2] Tabla de ECRPS por Paso de la Ventana Rodante (Plana)")
        ecrps_df = pd.DataFrame(self.rolling_ecrps).set_index('Paso')
        
        # Añadir fila de promedios para un resumen rápido
        ecrps_df.loc['Promedio'] = ecrps_df.mean()
        
        # [MODIFICADO] Mostrar el DataFrame sin estilo
        display(ecrps_df)