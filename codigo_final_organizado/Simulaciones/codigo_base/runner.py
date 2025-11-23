import warnings
import time
import os
from tqdm import tqdm
warnings.filterwarnings("ignore")

# Configuración de hilos para evitar conflictos con joblib
n_threads = str(os.cpu_count())
os.environ["OMP_NUM_THREADS"] = n_threads
os.environ["OPENBLAS_NUM_THREADS"] = n_threads
os.environ["MKL_NUM_THREADS"] = n_threads
os.environ["NUMEXPR_NUM_THREADS"] = n_threads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pipeline import Pipeline140SinSesgos
import pandas as pd
import numpy as np


def run_analysis(df_final):
    """Función común para análisis exhaustivo de resultados."""
    print("\n" + "="*80)
    print("ANÁLISIS EXHAUSTIVO DE RESULTADOS")
    print("="*80)
    
    model_cols = ['AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 
                  'EnCQR-LSTM', 'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap']
    
    # Filtrar columnas que realmente existen en el DF
    model_cols = [c for c in model_cols if c in df_final.columns]
    
    if 'Paso' in df_final.columns:
        df_steps = df_final[df_final['Paso'] != 'Promedio'].copy()
    else:
        df_steps = df_final.copy()
    
    if len(df_steps) == 0:
        print("⚠️ No hay datos suficientes para el análisis.")
        return

    # 1. RANKING GLOBAL
    print("\n🏆 1. RANKING GLOBAL (Media CRPS)")
    print("-" * 80)
    
    means = {}
    for model in model_cols:
        val = df_steps[model].mean()
        means[model] = val
    
    sorted_models = sorted(means.keys(), key=lambda x: means[x])
    
    print(f"{'Rank':<6} {'Modelo':<25} {'CRPS Medio':<15}")
    print("-" * 60)
    for i, m in enumerate(sorted_models):
        print(f"{i+1:<6} {m:<25} {means[m]:.6f}")

    # 2. MEJOR POR ESCENARIO
    print("\n🎯 2. VICTORIAS (Mejor modelo por paso)")
    print("-" * 80)
    wins = {m: 0 for m in model_cols}
    total = 0
    
    for _, row in df_steps.iterrows():
        scores = {m: row[m] for m in model_cols if not pd.isna(row[m])}
        if scores:
            winner = min(scores, key=scores.get)
            wins[winner] += 1
            total += 1
            
    for m in sorted(wins, key=wins.get, reverse=True):
        if total > 0:
            pct = (wins[m] / total) * 100
            print(f"  {m:<25}: {wins[m]:3d} victorias ({pct:.1f}%)")

    print("\n" + "="*80)
    print("FIN DEL ANÁLISIS")
    print("="*80)


def main_full_140():
    """Ejecución completa de 140 escenarios con gestión de memoria."""
    start_time = time.time()
    
    print("="*80)
    print("INICIANDO SIMULACIÓN DE 140 ESCENARIOS")
    print("="*80)
    
    # Crear pipeline
    pipeline = Pipeline140SinSesgos(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    # Ejecutar por lotes
    # batch_size=10 significa que procesará 10 escenarios en paralelo, 
    # limpiará RAM, guardará Excel y seguirá con los siguientes 10.
    df_final = pipeline.run_all(
        excel_filename="resultados_140_FINAL_FIXED.xlsx",
        batch_size=10 
    )
    
    # Análisis final
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_two_scenarios():
    """Ejecución con SOLO DOS ESCENARIOS (para verificación rápida)."""
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON SOLO 2 ESCENARIOS")
    print("="*80)
    
    pipeline = Pipeline140SinSesgos(n_boot=1000, seed=42, verbose=False)
    
    # --- MONKEY PATCHING PARA PRUEBA RÁPIDA ---
    # Sobreescribimos el método que genera escenarios para que solo devuelva 2
    # IMPORTANTE: Debe devolver la tupla completa de 8 elementos que espera el wrapper paralelo
    
    def generate_two_scenarios(self):
        scenarios = []
        
        # Escenario 1: AR(1) normal
        arma_cfg1 = {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []}
        scenarios.append((
            arma_cfg1, 'normal', 1.0, self.seed + 0,
            self.N_TEST_STEPS, self.N_TRAIN, self.N_CALIBRATION, self.n_boot
        ))
        
        # Escenario 2: MA(1) t-student
        arma_cfg2 = {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]}
        scenarios.append((
            arma_cfg2, 't-student', 1.0, self.seed + 1,
            self.N_TEST_STEPS, self.N_TRAIN, self.N_CALIBRATION, self.n_boot
        ))
        
        return scenarios
    
    # Inyectar el método en la instancia
    pipeline.generate_all_scenarios = generate_two_scenarios.__get__(pipeline)
    
    # Ejecutar
    # NOTA: Eliminado batch_size=2 porque run_all ya no lo acepta
    df_final = pipeline.run_all(
        excel_filename="resultados_2_ESCENARIOS.xlsx"
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final
