import warnings
import time
import os
from tqdm import tqdm
warnings.filterwarnings("ignore")

n_threads = str(os.cpu_count())
os.environ["OMP_NUM_THREADS"] = n_threads
os.environ["OPENBLAS_NUM_THREADS"] = n_threads
os.environ["MKL_NUM_THREADS"] = n_threads
os.environ["NUMEXPR_NUM_THREADS"] = n_threads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pipeline import PipelineARMA_100Trayectorias
import pandas as pd
import numpy as np


def run_analysis(df_final):
    """Función común para análisis exhaustivo de resultados."""
    print("\n" + "="*80)
    print("ANÁLISIS EXHAUSTIVO DE RESULTADOS - 100 TRAYECTORIAS")
    print("="*80)
    
    # Columnas de modelos disponibles
    model_cols = ['LSPM', 'DeepAR', 'Sieve Bootstrap', 'MCPS']
    model_cols = [c for c in model_cols if c in df_final.columns]
    
    if len(df_final) == 0:
        print("⚠️ No hay datos suficientes para el análisis.")
        return

    # 1. RANKING GLOBAL POR MODELO
    print("\n🏆 1. RANKING GLOBAL (Media CRPS)")
    print("-" * 80)
    
    means = {}
    for model in model_cols:
        val = df_final[model].mean()
        means[model] = val
    
    sorted_models = sorted(means.keys(), key=lambda x: means[x])
    
    print(f"{'Rank':<6} {'Modelo':<25} {'CRPS Medio':<15}")
    print("-" * 60)
    for i, m in enumerate(sorted_models):
        print(f"{i+1:<6} {m:<25} {means[m]:.6f}")

    # 2. VICTORIAS POR PASO
    print("\n🎯 2. VICTORIAS (Mejor modelo por cada paso-escenario)")
    print("-" * 80)
    wins = {m: 0 for m in model_cols}
    total = 0
    
    for _, row in df_final.iterrows():
        scores = {m: row[m] for m in model_cols if not pd.isna(row[m])}
        if scores:
            winner = min(scores, key=scores.get)
            wins[winner] += 1
            total += 1
            
    for m in sorted(wins, key=wins.get, reverse=True):
        if total > 0:
            pct = (wins[m] / total) * 100
            print(f"  {m:<25}: {wins[m]:4d} victorias ({pct:.1f}%)")

    # 3. ANÁLISIS POR HORIZONTE DE PREDICCIÓN
    print("\n📈 3. RENDIMIENTO POR HORIZONTE (Paso H)")
    print("-" * 80)
    
    if 'Paso_H' in df_final.columns:
        for h in sorted(df_final['Paso_H'].unique()):
            df_h = df_final[df_final['Paso_H'] == h]
            print(f"\n  Paso H={h}:")
            for model in model_cols:
                mean_crps = df_h[model].mean()
                print(f"    {model:<20}: {mean_crps:.6f}")

    # 4. ANÁLISIS POR TIPO DE PROCESO
    print("\n🔄 4. RENDIMIENTO POR PROCESO ARMA")
    print("-" * 80)
    
    if 'proces_simulacion' in df_final.columns:
        for proceso in sorted(df_final['proces_simulacion'].unique()):
            df_proc = df_final[df_final['proces_simulacion'] == proceso]
            print(f"\n  {proceso}:")
            for model in model_cols:
                mean_crps = df_proc[model].mean()
                print(f"    {model:<20}: {mean_crps:.6f}")

    # 5. ANÁLISIS POR DISTRIBUCIÓN
    print("\n📊 5. RENDIMIENTO POR DISTRIBUCIÓN")
    print("-" * 80)
    
    if 'Distribución' in df_final.columns:
        for dist in sorted(df_final['Distribución'].unique()):
            df_dist = df_final[df_final['Distribución'] == dist]
            print(f"\n  {dist}:")
            for model in model_cols:
                mean_crps = df_dist[model].mean()
                print(f"    {model:<20}: {mean_crps:.6f}")

    print("\n" + "="*80)
    print("FIN DEL ANÁLISIS")
    print("="*80)


def main_full_140():
    """Ejecución completa de 140 escenarios con 100 trayectorias cada uno."""
    start_time = time.time()
    
    print("="*80)
    print("INICIANDO SIMULACIÓN DE 140 ESCENARIOS CON 100 TRAYECTORIAS")
    print("="*80)
    
    pipeline = PipelineARMA_100Trayectorias(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    df_final = pipeline.run_all(
        filename="resultados_140_trayectorias_FINAL.xlsx"
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_one_scenario():
    """
    Ejecuta solo UN escenario para pruebas rápidas.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON SOLO 1 ESCENARIO (100 TRAYECTORIAS)")
    print("="*80)
    
    # Crear pipeline
    pipeline = PipelineARMA_100Trayectorias(n_boot=1000, seed=42, verbose=True)
    
    # Configurar solo 1 escenario: AR(1) con ruido normal y varianza 1.0
    pipeline.ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    print(f"\n📊 Configuración del escenario:")
    print(f"   - Proceso: AR(1) con φ=0.9")
    print(f"   - Distribución: Normal")
    print(f"   - Varianza: 1.0")
    print(f"   - Trayectorias: {pipeline.N_TRAJECTORIES}")
    print(f"   - Pasos adelante: {pipeline.N_TEST_STEPS}")
    print()
    
    # Ejecutar
    df_final = pipeline.run_all(
        filename="resultados_1_escenario_trayectorias.xlsx"
    )
    
    # Análisis simplificado
    print("\n" + "="*60)
    print("RESULTADOS DEL ESCENARIO ÚNICO")
    print("="*60)
    
    model_cols = ['LSPM', 'DeepAR', 'Sieve Bootstrap', 'MCPS']
    model_cols = [c for c in model_cols if c in df_final.columns]
    
    print(f"\n{'Paso_H':<8} {'Valor_Real':<12} ", end="")
    for m in model_cols:
        print(f"{m:<15}", end="")
    print()
    print("-" * 80)
    
    for _, row in df_final.iterrows():
        print(f"{row['Paso_H']:<8} {row['Valor_Real']:<12.4f} ", end="")
        for m in model_cols:
            print(f"{row[m]:<15.6f}", end="")
        print()
    
    # Resumen
    print("\n📊 CRPS PROMEDIO POR MODELO:")
    print("-" * 40)
    for m in model_cols:
        mean_crps = df_final[m].mean()
        print(f"  {m:<20}: {mean_crps:.6f}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final


