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

from pipeline import Pipeline140SinSesgos_ARMA, Pipeline140SinSesgos_ARIMA, Pipeline140SinSesgos_SETAR, Pipeline140ConDiferenciacion_ARIMA
import pandas as pd
import numpy as np


def run_analysis(df_final):
    """Función común para análisis exhaustivo de resultados."""
    print("\n" + "="*80)
    print("ANÁLISIS EXHAUSTIVO DE RESULTADOS")
    print("="*80)
    
    model_cols = ['AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 
                  'EnCQR-LSTM', 'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap']
    
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
    
    pipeline = Pipeline140SinSesgos_ARMA(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_140_FINAL_FIXED.xlsx",
        batch_size=10 
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_two_scenarios():
    """
    FIX: Ejecuta solo 2 escenarios DENTRO del wrapper de paralelización.
    No hace monkey patching problemático.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON SOLO 2 ESCENARIOS")
    print("="*80)
    
    # Crear pipeline con configuración especial
    pipeline = Pipeline140SinSesgos_ARMA(n_boot=1000, seed=42, verbose=True)
    
    # Configurar solo 2 escenarios manualmente
    pipeline.ARMA_CONFIGS = [
        {'nombre': 'AR(1)', 'phi': [0.9], 'theta': []},
        {'nombre': 'MA(1)', 'phi': [], 'theta': [0.7]}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    # Ahora generate_all_scenarios() solo generará 2 escenarios
    df_final = pipeline.run_all(
        excel_filename="resultados_2_ESCENARIOS.xlsx",
        batch_size=2
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final

def main_full_140_ARIMA():
    """
    Ejecución completa de 140 escenarios ARIMA con gestión de memoria.
    """
    start_time = time.time()
    
    print("="*80)
    print("INICIANDO SIMULACIÓN DE 140 ESCENARIOS ARIMA")
    print("="*80)
    
    pipeline = Pipeline140SinSesgos_ARIMA(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_140_ARIMA_FINAL.xlsx",
        batch_size=10 
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_two_scenarios_ARIMA():
    """
    Ejecuta solo 2 escenarios ARIMA para pruebas rápidas.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON SOLO 2 ESCENARIOS ARIMA")
    print("="*80)
    
    # Crear pipeline con configuración especial
    pipeline = Pipeline140SinSesgos_ARIMA(n_boot=1000, seed=42, verbose=True)
    
    # Configurar solo 2 escenarios manualmente
    pipeline.ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.7], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.6]}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    # Ahora generate_all_scenarios() solo generará 2 escenarios
    df_final = pipeline.run_all(
        excel_filename="resultados_2_ESCENARIOS_ARIMA.xlsx",
        batch_size=2
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final

def main_full_140_SETAR():
    """
    Ejecución completa de 140 escenarios SETAR con gestión de memoria.
    """
    start_time = time.time()
    
    print("="*80)
    print("INICIANDO SIMULACIÓN DE 140 ESCENARIOS SETAR")
    print("="*80)
    
    pipeline = Pipeline140SinSesgos_SETAR(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_140_SETAR_FINAL.xlsx",
        batch_size=10 
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_two_scenarios_SETAR():
    """
    Ejecuta solo 2 escenarios SETAR para pruebas rápidas.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON SOLO 2 ESCENARIOS SETAR")
    print("="*80)
    
    # Crear pipeline con configuración especial
    pipeline = Pipeline140SinSesgos_SETAR(n_boot=1000, seed=42, verbose=True)
    
    # Configurar solo 2 escenarios manualmente
    pipeline.SETAR_CONFIGS = [
        {
            'nombre': 'SETAR-1',
            'phi_regime1': [0.6],
            'phi_regime2': [-0.5],
            'threshold': 0.0,
            'delay': 1,
            'description': 'SETAR(2;1,1) - AR(1) con d=1'
        },
        {
            'nombre': 'SETAR-3',
            'phi_regime1': [0.5, -0.2],
            'phi_regime2': [-0.3, 0.1],
            'threshold': 0.5,
            'delay': 1,
            'description': 'SETAR(2;2,2) - AR(2) con d=1'
        }
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    # Ahora generate_all_scenarios() solo generará 2 escenarios
    df_final = pipeline.run_all(
        excel_filename="resultados_2_ESCENARIOS_SETAR.xlsx",
        batch_size=2
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final

def main_two_scenarios_diferenciado():
    """
    Ejecuta 2 escenarios ARIMA CON diferenciación adicional para pruebas rápidas.
    """
    start_time = time.time()
    
    print("="*80)
    print("EVALUACIÓN CON 2 ESCENARIOS ARIMA - CON DIFERENCIACIÓN ADICIONAL")
    print("="*80)
    
    from pipeline import Pipeline140ConDiferenciacion_ARIMA
    
    # Crear pipeline CON diferenciación
    pipeline = Pipeline140ConDiferenciacion_ARIMA(
        n_boot=1000, 
        seed=42, 
        verbose=True,
        usar_diferenciacion=True  # ✅ ACTIVAR DIFERENCIACIÓN
    )
    
    # Configurar solo 2 escenarios
    pipeline.ARIMA_CONFIGS = [
        {'nombre': 'ARIMA(1,1,0)', 'phi': [0.7], 'theta': []},
        {'nombre': 'ARIMA(0,1,1)', 'phi': [], 'theta': [0.6]}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    df_final = pipeline.run_all(
        excel_filename="resultados_2_ESCENARIOS_ARIMA_CON_DIFF.xlsx",
        batch_size=2
    )
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final


def main_full_140_diferenciado():
    """
    Ejecución completa de 140 escenarios ARIMA CON diferenciación adicional.
    
    Este pipeline aplica diferenciación ANTES de pasar los datos a los modelos,
    permitiendo comparar si trabajar en espacio de incrementos (ΔY_t) mejora
    el desempeño de los métodos de predicción conformal.
    """
    start_time = time.time()
    
    print("="*80)
    print("INICIANDO SIMULACIÓN DE 140 ESCENARIOS ARIMA - CON DIFERENCIACIÓN")
    print("="*80)
    
    from pipeline import Pipeline140ConDiferenciacion_ARIMA
    
    # Crear pipeline CON diferenciación
    pipeline = Pipeline140ConDiferenciacion_ARIMA(
        n_boot=1000,
        seed=42,
        verbose=False,
        usar_diferenciacion=True  # ✅ ACTIVAR DIFERENCIACIÓN
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_140_ARIMA_CON_DIFERENCIACION.xlsx",
        batch_size=10
    )
    
    print("\n" + "="*80)
    print("ANÁLISIS DE RESULTADOS - CON DIFERENCIACIÓN")
    print("="*80)
    
    run_analysis(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final