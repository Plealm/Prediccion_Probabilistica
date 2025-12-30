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

from pipeline import Pipeline140SinSesgos_ARMA, Pipeline140SinSesgos_ARIMA, Pipeline140SinSesgos_SETAR, Pipeline140_TamanosCrecientes, Pipeline240_ProporcionesVariables, PipelineARIMA_MultiD_SieveOnly
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
    
    from pipeline import Pipeline140SinSesgos_ARIMA_ConDiferenciacion
    
    # Crear pipeline CON diferenciación
    pipeline = Pipeline140SinSesgos_ARIMA_ConDiferenciacion(
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
    
    from pipeline import Pipeline140SinSesgos_ARIMA_ConDiferenciacion
    
    # Crear pipeline CON diferenciación
    pipeline = Pipeline140SinSesgos_ARIMA_ConDiferenciacion(
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


# ============================================================================
# CÓDIGO PARA AGREGAR AL FINAL DE runner.py
# ============================================================================

def analisis_completo_doble_modalidad(df_final):
    """
    Análisis exhaustivo para resultados con doble modalidad (SIN_DIFF vs CON_DIFF).
    
    Compara:
    1. Desempeño por cada valor de d
    2. SIN_DIFF vs CON_DIFF: ¿cuál funciona mejor?
    3. Tendencias según d aumenta
    4. Mejor d por modelo y modalidad
    """
    print("\n" + "="*80)
    print("ANÁLISIS EXHAUSTIVO - DOBLE MODALIDAD (SIN_DIFF vs CON_DIFF)")
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
    
    # Asegurar tipos correctos
    if 'd' in df_steps.columns:
        df_steps['d'] = pd.to_numeric(df_steps['d'], errors='coerce')
    
    d_values = sorted(df_steps['d'].unique())
    modalidades = sorted(df_steps['Modalidad'].unique()) if 'Modalidad' in df_steps.columns else []
    
    # =================================================================
    # 1. COMPARACIÓN GLOBAL: SIN_DIFF vs CON_DIFF
    # =================================================================
    print("\n" + "="*80)
    print("🔍 1. COMPARACIÓN GLOBAL POR MODALIDAD")
    print("="*80)
    
    for modalidad in modalidades:
        df_mod = df_steps[df_steps['Modalidad'] == modalidad]
        
        if len(df_mod) == 0:
            continue
        
        print(f"\n{'='*60}")
        print(f"MODALIDAD: {modalidad}")
        print(f"{'='*60}")
        
        means = {}
        for model in model_cols:
            if model in df_mod.columns:
                val = df_mod[model].mean()
                if not pd.isna(val):
                    means[model] = val
        
        if not means:
            print("  (Sin datos válidos)")
            continue
        
        sorted_models = sorted(means.keys(), key=lambda x: means[x])
        
        print(f"{'Rank':<6} {'Modelo':<25} {'CRPS Medio':<15}")
        print("-" * 60)
        for i, m in enumerate(sorted_models):
            print(f"{i+1:<6} {m:<25} {means[m]:.6f}")
    
    # =================================================================
    # 2. RANKING POR CADA d Y MODALIDAD
    # =================================================================
    print("\n" + "="*80)
    print("📊 2. RANKING POR CADA VALOR DE d (AMBAS MODALIDADES)")
    print("="*80)
    
    for d_val in d_values:
        df_d = df_steps[df_steps['d'] == d_val]
        
        if len(df_d) == 0:
            continue
        
        print(f"\n{'='*70}")
        print(f"d = {d_val}")
        print(f"{'='*70}")
        
        for modalidad in modalidades:
            df_d_mod = df_d[df_d['Modalidad'] == modalidad]
            
            if len(df_d_mod) == 0:
                continue
            
            print(f"\n  --- {modalidad} ---")
            
            means = {}
            for model in model_cols:
                if model in df_d_mod.columns:
                    val = df_d_mod[model].mean()
                    if not pd.isna(val):
                        means[model] = val
            
            if not means:
                print("    (Sin datos válidos)")
                continue
            
            sorted_models = sorted(means.keys(), key=lambda x: means[x])
            
            print(f"  {'Rank':<6} {'Modelo':<25} {'CRPS':<12}")
            print("  " + "-" * 50)
            for i, m in enumerate(sorted_models[:5]):  # Top 5
                print(f"  {i+1:<6} {m:<25} {means[m]:.6f}")
    
    # =================================================================
    # 3. VICTORIAS POR MODELO EN CADA MODALIDAD
    # =================================================================
    print("\n" + "="*80)
    print("🎯 3. VICTORIAS POR MODALIDAD (Mejor modelo por paso)")
    print("="*80)
    
    for modalidad in modalidades:
        df_mod = df_steps[df_steps['Modalidad'] == modalidad]
        
        if len(df_mod) == 0:
            continue
        
        print(f"\n{'='*60}")
        print(f"MODALIDAD: {modalidad}")
        print(f"{'='*60}")
        
        wins = {m: 0 for m in model_cols}
        total = 0
        
        for _, row in df_mod.iterrows():
            scores = {m: row[m] for m in model_cols if not pd.isna(row[m])}
            if scores:
                winner = min(scores, key=scores.get)
                wins[winner] += 1
                total += 1
        
        if total == 0:
            print("  (Sin datos válidos)")
            continue
        
        for m in sorted(wins, key=wins.get, reverse=True):
            if wins[m] > 0:
                pct = (wins[m] / total) * 100
                print(f"  {m:<25}: {wins[m]:4d} victorias ({pct:.1f}%)")
    
    # =================================================================
    # 4. TENDENCIAS: Desempeño según d (por modalidad)
    # =================================================================
    print("\n" + "="*80)
    print("📈 4. TENDENCIAS: Desempeño según d aumenta")
    print("="*80)
    
    for modalidad in modalidades:
        print(f"\n{'='*70}")
        print(f"MODALIDAD: {modalidad}")
        print(f"{'='*70}")
        
        print(f"\n{'Modelo':<25} ", end="")
        for d_val in d_values:
            print(f"d={d_val:<3}", end="  ")
        print()
        print("-" * (25 + 7 * len(d_values)))
        
        for model in model_cols:
            print(f"{model:<25} ", end="")
            for d_val in d_values:
                df_d_mod = df_steps[(df_steps['d'] == d_val) & 
                                    (df_steps['Modalidad'] == modalidad)]
                if model in df_d_mod.columns:
                    val = df_d_mod[model].mean()
                    if not pd.isna(val):
                        print(f"{val:.4f}", end="  ")
                    else:
                        print("  ---  ", end="  ")
                else:
                    print("  ---  ", end="  ")
            print()
    
    # =================================================================
    # 5. COMPARACIÓN DIRECTA: SIN_DIFF vs CON_DIFF por modelo
    # =================================================================
    print("\n" + "="*80)
    print("⚖️  5. COMPARACIÓN DIRECTA: SIN_DIFF vs CON_DIFF")
    print("="*80)
    
    if len(modalidades) == 2:
        mod_sin = [m for m in modalidades if 'SIN' in m][0]
        mod_con = [m for m in modalidades if 'CON' in m][0]
        
        print(f"\n{'Modelo':<25} {mod_sin:<12} {mod_con:<12} {'Diferencia':<12} {'Mejor':<10}")
        print("-" * 75)
        
        for model in model_cols:
            df_sin = df_steps[df_steps['Modalidad'] == mod_sin]
            df_con = df_steps[df_steps['Modalidad'] == mod_con]
            
            if model in df_sin.columns and model in df_con.columns:
                val_sin = df_sin[model].mean()
                val_con = df_con[model].mean()
                
                if not pd.isna(val_sin) and not pd.isna(val_con):
                    diff = val_con - val_sin
                    mejor = mod_sin if val_sin < val_con else mod_con
                    
                    print(f"{model:<25} {val_sin:.6f}   {val_con:.6f}   "
                          f"{diff:+.6f}   {mejor:<10}")
    
    # =================================================================
    # 6. MEJOR d POR MODELO Y MODALIDAD
    # =================================================================
    print("\n" + "="*80)
    print("🎲 6. MEJOR VALOR DE d PARA CADA MODELO Y MODALIDAD")
    print("="*80)
    
    for modalidad in modalidades:
        print(f"\n{'='*60}")
        print(f"MODALIDAD: {modalidad}")
        print(f"{'='*60}")
        
        print(f"\n{'Modelo':<25} {'Mejor d':<10} {'CRPS en ese d':<15}")
        print("-" * 60)
        
        for model in model_cols:
            best_d = None
            best_crps = float('inf')
            
            for d_val in d_values:
                df_d_mod = df_steps[(df_steps['d'] == d_val) & 
                                    (df_steps['Modalidad'] == modalidad)]
                if model in df_d_mod.columns:
                    val = df_d_mod[model].mean()
                    if not pd.isna(val) and val < best_crps:
                        best_crps = val
                        best_d = d_val
            
            if best_d is not None:
                print(f"{model:<25} {best_d:<10} {best_crps:.6f}")
    
    # =================================================================
    # 7. RESUMEN EJECUTIVO
    # =================================================================
    print("\n" + "="*80)
    print("📋 7. RESUMEN EJECUTIVO")
    print("="*80)
    
    # Mejor modalidad global
    if len(modalidades) == 2:
        crps_sin = df_steps[df_steps['Modalidad'] == mod_sin][model_cols].mean().mean()
        crps_con = df_steps[df_steps['Modalidad'] == mod_con][model_cols].mean().mean()
        
        print(f"\n✓ MEJOR MODALIDAD GLOBAL:")
        print(f"  • {mod_sin}: CRPS promedio = {crps_sin:.6f}")
        print(f"  • {mod_con}: CRPS promedio = {crps_con:.6f}")
        
        if crps_sin < crps_con:
            print(f"  → GANADOR: {mod_sin} (diferencia: {crps_con - crps_sin:.6f})")
        else:
            print(f"  → GANADOR: {mod_con} (diferencia: {crps_sin - crps_con:.6f})")
    
    # Mejor modelo global
    global_means = {}
    for model in model_cols:
        if model in df_steps.columns:
            val = df_steps[model].mean()
            if not pd.isna(val):
                global_means[model] = val
    
    if global_means:
        best_model = min(global_means, key=global_means.get)
        print(f"\n✓ MEJOR MODELO GLOBAL:")
        print(f"  → {best_model}: CRPS = {global_means[best_model]:.6f}")
    
    print("\n" + "="*80)
    print("FIN DEL ANÁLISIS")
    print("="*80)


def main_full_2800():
    """
    Ejecución completa: 2,800 filas (1,400 escenarios × 2 modalidades).
    - d = 1, 2, ..., 10
    - 7 configuraciones ARMA
    - 5 distribuciones
    - 4 varianzas
    - 2 modalidades (SIN_DIFF + CON_DIFF)
    """
    start_time = time.time()
    
    print("="*80)
    print("INICIANDO SIMULACIÓN COMPLETA: 2,800 FILAS")
    print("="*80)
    
    from pipeline import PipelineARIMA_MultiD_DobleModalidad
    
    pipeline = PipelineARIMA_MultiD_DobleModalidad(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_ARIMA_d1_a_d10_DOBLE_MODALIDAD_COMPLETO.xlsx",
        batch_size=20
    )
    
    # Análisis exhaustivo
    analisis_completo_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_test_reducido_doble():
    """
    Test reducido: 2 valores de d, 2 ARMA, 1 distribución, 1 varianza, 2 modalidades.
    Total: 2 × 2 × 1 × 1 × 2 = 8 filas base
    Con 12 pasos + 1 promedio = 104 filas totales
    """
    start_time = time.time()
    
    print("="*80)
    print("TEST REDUCIDO: DOBLE MODALIDAD (d=1,2)")
    print("="*80)
    
    from pipeline import PipelineARIMA_MultiD_DobleModalidad
    
    pipeline = PipelineARIMA_MultiD_DobleModalidad(
        n_boot=1000, seed=42, verbose=True
    )
    
    # Configuración reducida
    pipeline.D_VALUES = [1, 2]
    pipeline.ARMA_CONFIGS = [
        {'nombre': 'RW', 'phi': [], 'theta': []},
        {'nombre': 'AR(1)', 'phi': [0.6], 'theta': []}
    ]
    pipeline.DISTRIBUTIONS = ['normal']
    pipeline.VARIANCES = [1.0]
    
    df_final = pipeline.run_all(
        excel_filename="resultados_TEST_DOBLE_MODALIDAD.xlsx",
        batch_size=4
    )
    
    analisis_completo_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final


def main_rango_d_doble_modalidad(d_min=1, d_max=5):
    """
    Rango personalizado de d con ambas modalidades.
    
    Args:
        d_min: Valor mínimo de d (default: 1)
        d_max: Valor máximo de d (default: 5)
    
    Ejemplo: main_rango_d_doble_modalidad(d_min=1, d_max=5)
    """
    start_time = time.time()
    
    print("="*80)
    print(f"SIMULACIÓN ARIMA d={d_min},...,{d_max} - DOBLE MODALIDAD")
    print("="*80)
    
    from pipeline import PipelineARIMA_MultiD_DobleModalidad
    
    pipeline = PipelineARIMA_MultiD_DobleModalidad(
        n_boot=1000, seed=42, verbose=False
    )
    
    # Configurar rango de d
    pipeline.D_VALUES = list(range(d_min, d_max + 1))
    
    total_base_scenarios = (len(pipeline.D_VALUES) * len(pipeline.ARMA_CONFIGS) * 
                           len(pipeline.DISTRIBUTIONS) * len(pipeline.VARIANCES))
    
    print(f"📊 Escenarios base: {total_base_scenarios}")
    print(f"   • Valores de d: {pipeline.D_VALUES}")
    print(f"   • Modalidades: 2 (SIN_DIFF + CON_DIFF)")
    print(f"   • Filas esperadas: ~{total_base_scenarios * 2 * 13}")
    print("="*80 + "\n")
    
    df_final = pipeline.run_all(
        excel_filename=f"resultados_ARIMA_d{d_min}_a_d{d_max}_DOBLE_MOD.xlsx",
        batch_size=20
    )
    
    analisis_completo_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def comparar_d_especificos_doble(d_lista=[1, 3, 5, 10]):
    """
    Compara valores específicos de d en ambas modalidades.
    
    Args:
        d_lista: Lista de valores de d a comparar (default: [1, 3, 5, 10])
    
    Ejemplo: comparar_d_especificos_doble([1, 5, 10])
    """
    start_time = time.time()
    
    print("="*80)
    print(f"COMPARACIÓN d={d_lista} - DOBLE MODALIDAD")
    print("="*80)
    
    from pipeline import PipelineARIMA_MultiD_DobleModalidad
    
    pipeline = PipelineARIMA_MultiD_DobleModalidad(
        n_boot=1000, seed=42, verbose=False
    )
    
    # Configurar valores específicos de d
    pipeline.D_VALUES = d_lista
    
    total_base_scenarios = (len(pipeline.D_VALUES) * len(pipeline.ARMA_CONFIGS) * 
                           len(pipeline.DISTRIBUTIONS) * len(pipeline.VARIANCES))
    
    print(f"📊 Escenarios base: {total_base_scenarios}")
    print(f"   • Modalidades: 2 (SIN_DIFF + CON_DIFF)")
    print("="*80 + "\n")
    
    filename = f"resultados_ARIMA_d_{'_'.join(map(str, d_lista))}_DOBLE_MOD.xlsx"
    df_final = pipeline.run_all(excel_filename=filename, batch_size=20)
    
    analisis_completo_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def analisis_tamanos_crecientes(df_final):
    """
    Análisis especializado para resultados con tamaños crecientes de datos.
    
    Evalúa:
    1. Impacto de N_Train en el desempeño
    2. Impacto de N_Calib en el desempeño
    3. Mejor combinación (N_Train, N_Calib) por modelo
    4. Rendimiento marginal de agregar más datos
    5. Punto de saturación (cuando más datos no ayudan)
    """
    print("\n" + "="*80)
    print("ANÁLISIS EXHAUSTIVO - TAMAÑOS CRECIENTES DE DATOS")
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
    
    # Asegurar tipos numéricos
    for col in ['N_Train', 'N_Calib', 'N_Total']:
        if col in df_steps.columns:
            df_steps[col] = pd.to_numeric(df_steps[col], errors='coerce')
    
    train_sizes = sorted(df_steps['N_Train'].unique())
    calib_sizes = sorted(df_steps['N_Calib'].unique())
    
    # =================================================================
    # 1. RANKING GLOBAL POR TAMAÑO TOTAL
    # =================================================================
    print("\n" + "="*80)
    print("📊 1. RANKING GLOBAL POR TAMAÑO TOTAL DE DATOS")
    print("="*80)
    
    for n_total in sorted(df_steps['N_Total'].unique()):
        df_size = df_steps[df_steps['N_Total'] == n_total]
        
        if len(df_size) == 0:
            continue
        
        print(f"\n{'='*70}")
        print(f"N_Total = {n_total}")
        print(f"{'='*70}")
        
        means = {}
        for model in model_cols:
            if model in df_size.columns:
                val = df_size[model].mean()
                if not pd.isna(val):
                    means[model] = val
        
        if not means:
            print("  (Sin datos válidos)")
            continue
        
        sorted_models = sorted(means.keys(), key=lambda x: means[x])
        
        print(f"{'Rank':<6} {'Modelo':<25} {'CRPS Medio':<15}")
        print("-" * 60)
        for i, m in enumerate(sorted_models[:5]):  # Top 5
            print(f"{i+1:<6} {m:<25} {means[m]:.6f}")
    
    # =================================================================
    # 2. IMPACTO DE N_TRAIN (fijando N_CALIB)
    # =================================================================
    print("\n" + "="*80)
    print("📈 2. IMPACTO DE N_TRAIN EN EL DESEMPEÑO")
    print("="*80)
    
    for n_calib in calib_sizes:
        df_calib = df_steps[df_steps['N_Calib'] == n_calib]
        
        if len(df_calib) == 0:
            continue
        
        print(f"\n{'='*70}")
        print(f"N_Calib = {n_calib} (fijo)")
        print(f"{'='*70}")
        
        print(f"\n{'Modelo':<25} ", end="")
        for n_train in train_sizes:
            print(f"N_Train={n_train:<5}", end="  ")
        print()
        print("-" * (25 + 13 * len(train_sizes)))
        
        for model in model_cols:
            print(f"{model:<25} ", end="")
            for n_train in train_sizes:
                df_subset = df_calib[df_calib['N_Train'] == n_train]
                if model in df_subset.columns:
                    val = df_subset[model].mean()
                    if not pd.isna(val):
                        print(f"{val:.5f}", end="  ")
                    else:
                        print("  ----  ", end="  ")
                else:
                    print("  ----  ", end="  ")
            print()
    
    # =================================================================
    # 3. IMPACTO DE N_CALIB (fijando N_TRAIN)
    # =================================================================
    print("\n" + "="*80)
    print("📈 3. IMPACTO DE N_CALIB EN EL DESEMPEÑO")
    print("="*80)
    
    for n_train in train_sizes:
        df_train = df_steps[df_steps['N_Train'] == n_train]
        
        if len(df_train) == 0:
            continue
        
        print(f"\n{'='*70}")
        print(f"N_Train = {n_train} (fijo)")
        print(f"{'='*70}")
        
        print(f"\n{'Modelo':<25} ", end="")
        for n_calib in calib_sizes:
            print(f"N_Calib={n_calib:<4}", end="  ")
        print()
        print("-" * (25 + 12 * len(calib_sizes)))
        
        for model in model_cols:
            print(f"{model:<25} ", end="")
            for n_calib in calib_sizes:
                df_subset = df_train[df_train['N_Calib'] == n_calib]
                if model in df_subset.columns:
                    val = df_subset[model].mean()
                    if not pd.isna(val):
                        print(f"{val:.5f}", end="  ")
                    else:
                        print(" ---- ", end="  ")
                else:
                    print(" ---- ", end="  ")
            print()
    
    # =================================================================
    # 4. MEJOR COMBINACIÓN POR MODELO
    # =================================================================
    print("\n" + "="*80)
    print("🎯 4. MEJOR COMBINACIÓN (N_TRAIN, N_CALIB) POR MODELO")
    print("="*80)
    
    print(f"\n{'Modelo':<25} {'Mejor N_Train':<15} {'Mejor N_Calib':<15} {'CRPS':<12}")
    print("-" * 75)
    
    for model in model_cols:
        best_crps = float('inf')
        best_train = None
        best_calib = None
        
        for n_train in train_sizes:
            for n_calib in calib_sizes:
                df_subset = df_steps[(df_steps['N_Train'] == n_train) & 
                                     (df_steps['N_Calib'] == n_calib)]
                if model in df_subset.columns:
                    val = df_subset[model].mean()
                    if not pd.isna(val) and val < best_crps:
                        best_crps = val
                        best_train = n_train
                        best_calib = n_calib
        
        if best_train is not None:
            print(f"{model:<25} {best_train:<15} {best_calib:<15} {best_crps:.6f}")

    # =================================================================
    # 5. MEJORA MARGINAL AL AGREGAR DATOS
    # =================================================================
    print("\n" + "="*80)
    print("💹 5. MEJORA MARGINAL AL INCREMENTAR DATOS")
    print("="*80)
    
    print("\n--- Mejora al incrementar N_Train (N_Calib fijo) ---")
    
    for n_calib in calib_sizes:
        df_calib = df_steps[df_steps['N_Calib'] == n_calib]
        
        if len(df_calib) == 0:
            continue
        
        print(f"\nN_Calib = {n_calib}:")
        print(f"{'Modelo':<25} ", end="")
        
        # Calcular mejoras entre tamaños consecutivos
        for i in range(len(train_sizes) - 1):
            n_small = train_sizes[i]
            n_large = train_sizes[i+1]
            print(f"{n_small}→{n_large:<5}", end="  ")
        print()
        print("-" * 60)
        
        for model in model_cols:
            print(f"{model:<25} ", end="")
            
            for i in range(len(train_sizes) - 1):
                n_small = train_sizes[i]
                n_large = train_sizes[i+1]
                
                df_small = df_calib[df_calib['N_Train'] == n_small]
                df_large = df_calib[df_calib['N_Train'] == n_large]
                
                if model in df_small.columns and model in df_large.columns:
                    val_small = df_small[model].mean()
                    val_large = df_large[model].mean()
                    
                    if not pd.isna(val_small) and not pd.isna(val_large):
                        mejora = val_small - val_large  # Positivo = mejora
                        mejora_pct = (mejora / val_small) * 100 if val_small != 0 else 0
                        print(f"{mejora_pct:+5.1f}%", end="  ")
                    else:
                        print("  --- ", end="  ")
                else:
                    print("  --- ", end="  ")
            print()

    # =================================================================
    # 6. PUNTO DE SATURACIÓN
    # =================================================================
    print("\n" + "="*80)
    print("🎚️ 6. ANÁLISIS DE SATURACIÓN (¿Cuándo más datos no ayudan?)")
    print("="*80)
    
    print("\nCriterio: Mejora < 2% al incrementar datos = SATURADO\n")
    
    for model in model_cols:
        print(f"\n{model}:")
        
        # Buscar saturación en N_Train
        saturado_train = False
        for n_calib in calib_sizes:
            if saturado_train:
                break
            
            for i in range(len(train_sizes) - 1):
                n_small = train_sizes[i]
                n_large = train_sizes[i+1]
                
                df_small = df_steps[(df_steps['N_Train'] == n_small) & 
                                   (df_steps['N_Calib'] == n_calib)]
                df_large = df_steps[(df_steps['N_Train'] == n_large) & 
                                   (df_steps['N_Calib'] == n_calib)]
                
                if model in df_small.columns and model in df_large.columns:
                    val_small = df_small[model].mean()
                    val_large = df_large[model].mean()
                    
                    if not pd.isna(val_small) and not pd.isna(val_large):
                        mejora_pct = ((val_small - val_large) / val_small) * 100 if val_small != 0 else 0
                        
                        if mejora_pct < 2.0:
                            print(f"  ✓ Saturado en N_Train ≥ {n_small} (N_Calib={n_calib})")
                            saturado_train = True
                            break

    # =================================================================
    # 7. RESUMEN EJECUTIVO
    # =================================================================
    print("\n" + "="*80)
    print("📋 7. RESUMEN EJECUTIVO")
    print("="*80)
    
    # Mejor modelo global
    global_means = {}
    for model in model_cols:
        if model in df_steps.columns:
            val = df_steps[model].mean()
            if not pd.isna(val):
                global_means[model] = val
    
    if global_means:
        best_model = min(global_means, key=global_means.get)
        worst_model = max(global_means, key=global_means.get)
        
        print(f"\n✅ MEJOR MODELO GLOBAL:")
        print(f"   → {best_model}: CRPS = {global_means[best_model]:.6f}")
        
        print(f"\n❌ PEOR MODELO GLOBAL:")
        print(f"   → {worst_model}: CRPS = {global_means[worst_model]:.6f}")
    
    # Mejor tamaño total
    size_means = df_steps.groupby('N_Total')[model_cols].mean().mean(axis=1)
    best_size = size_means.idxmin()
    
    print(f"\n🎯 MEJOR TAMAÑO TOTAL DE DATOS:")
    print(f"   → N_Total = {best_size}: CRPS promedio = {size_means[best_size]:.6f}")
    
    print("\n" + "="*80)
    print("FIN DEL ANÁLISIS")
    print("="*80)


# =================================================================
# FUNCIONES RUNNER PRINCIPALES
# =================================================================

def analisis_tamanos_crecientes(df_final):
    """
    Análisis especializado para resultados con tamaños crecientes de datos.
    """
    print("\n" + "="*80)
    print("ANÁLISIS EXHAUSTIVO - TAMAÑOS CRECIENTES DE DATOS")
    print("="*80)
    
    model_cols = ['AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 
                  'EnCQR-LSTM', 'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap']
    model_cols = [c for c in model_cols if c in df_final.columns]
    
    if len(df_final) == 0:
        print("⚠️ No hay datos suficientes para el análisis.")
        return
    
    df_steps = df_final.copy()
    
    # Asegurar tipos numéricos
    for col in ['N_Train', 'N_Calib', 'N_Total']:
        if col in df_steps.columns:
            df_steps[col] = pd.to_numeric(df_steps[col], errors='coerce')
    
    train_sizes = sorted(df_steps['N_Train'].dropna().unique())
    calib_sizes = sorted(df_steps['N_Calib'].dropna().unique())
    
    # 1. RANKING GLOBAL POR TAMAÑO TOTAL
    print("\n📊 1. RANKING GLOBAL POR TAMAÑO TOTAL DE DATOS")
    for n_total in sorted(df_steps['N_Total'].dropna().unique()):
        df_size = df_steps[df_steps['N_Total'] == n_total]
        print(f"\n--- N_Total = {n_total} ---")
        means = df_size[model_cols].mean().sort_values()
        for i, (model, val) in enumerate(means.head(5).items()):
            print(f" {i+1}. {model:<20} {val:.6f}")

    # 2. IMPACTO DE N_TRAIN (fijando N_CALIB)
    print("\n📈 2. IMPACTO DE N_TRAIN EN EL DESEMPEÑO")
    for n_calib in calib_sizes:
        print(f"\nN_Calib = {n_calib} (fijo):")
        pivot = df_steps[df_steps['N_Calib'] == n_calib].groupby('N_Train')[model_cols].mean()
        print(pivot.T)

    # 3. IMPACTO DE N_CALIB (fijando N_TRAIN)
    print("\n📈 3. IMPACTO DE N_CALIB EN EL DESEMPEÑO")
    for n_train in train_sizes:
        print(f"\nN_Train = {n_train} (fijo):")
        pivot = df_steps[df_steps['N_Train'] == n_train].groupby('N_Calib')[model_cols].mean()
        print(pivot.T)

    # 4. MEJOR COMBINACIÓN POR MODELO
    print("\n🎯 4. MEJOR COMBINACIÓN (N_TRAIN, N_CALIB) POR MODELO")
    for model in model_cols:
        idx = df_steps.groupby(['N_Train', 'N_Calib'])[model].mean().idxmin()
        val = df_steps.groupby(['N_Train', 'N_Calib'])[model].mean().min()
        print(f" {model:<20}: N_Train={idx[0]}, N_Calib={idx[1]} | CRPS={val:.6f}")

    # 5. RESUMEN EJECUTIVO
    print("\n📋 5. RESUMEN EJECUTIVO")
    global_means = df_steps[model_cols].mean()
    print(f" ✅ MEJOR MODELO GLOBAL: {global_means.idxmin()} ({global_means.min():.6f})")
    
    size_means = df_steps.groupby('N_Total')[model_cols].mean().mean(axis=1)
    print(f" 🎯 MEJOR TAMAÑO TOTAL: N_Total={size_means.idxmin()} ({size_means.min():.6f})")


# =================================================================
# FUNCIONES RUNNER
# =================================================================

def main_tamanos_crecientes_ARMA(train_sizes=None, calib_sizes=None):
    start_time = time.time()
    print("\n" + "="*50 + "\nPROCESO ARMA\n" + "="*50)
    
    pipeline = Pipeline140_TamanosCrecientes(proceso_tipo='ARMA')
    if train_sizes: pipeline.TRAIN_SIZES = train_sizes
    if calib_sizes: pipeline.CALIB_SIZES = calib_sizes
    
    df = pipeline.run_all(excel_filename="resultados_TAMANOS_ARMA.xlsx", batch_size=20)
    analisis_tamanos_crecientes(df)
    print(f"⏱ Tiempo: {time.time()-start_time:.1f}s")
    return df

def main_tamanos_crecientes_ARIMA(train_sizes=None, calib_sizes=None):
    start_time = time.time()
    print("\n" + "="*50 + "\nPROCESO ARIMA\n" + "="*50)
    
    pipeline = Pipeline140_TamanosCrecientes(proceso_tipo='ARIMA')
    if train_sizes: pipeline.TRAIN_SIZES = train_sizes
    if calib_sizes: pipeline.CALIB_SIZES = calib_sizes
    
    df = pipeline.run_all(excel_filename="resultados_TAMANOS_ARIMA.xlsx", batch_size=20)
    analisis_tamanos_crecientes(df)
    print(f"⏱ Tiempo: {time.time()-start_time:.1f}s")
    return df

def main_tamanos_crecientes_SETAR(train_sizes=None, calib_sizes=None):
    start_time = time.time()
    print("\n" + "="*50 + "\nPROCESO SETAR\n" + "="*50)
    
    pipeline = Pipeline140_TamanosCrecientes(proceso_tipo='SETAR')
    if train_sizes: pipeline.TRAIN_SIZES = train_sizes
    if calib_sizes: pipeline.CALIB_SIZES = calib_sizes
    
    df = pipeline.run_all(excel_filename="resultados_TAMANOS_SETAR.xlsx", batch_size=20)
    analisis_tamanos_crecientes(df)
    print(f"⏱ Tiempo: {time.time()-start_time:.1f}s")
    return df

def main_test_tamanos_reducido():
    print("🚀 INICIANDO TEST REDUCIDO")
    test_train = [100, 200]
    test_calib = [20, 40]
    results = {}

    for tipo in ['ARMA', 'ARIMA', 'SETAR']:
        pipeline = Pipeline140_TamanosCrecientes(n_boot=100, proceso_tipo=tipo)
        pipeline.TRAIN_SIZES = test_train
        pipeline.CALIB_SIZES = test_calib
        
        # Limitar configuraciones para el test
        pipeline.CONFIGS[tipo] = pipeline.CONFIGS[tipo][:1] 
        pipeline.DISTRIBUTIONS = ['normal']
        pipeline.VARIANCES = [1.0]
        
        print(f"Ejecutando {tipo}...")
        results[tipo] = pipeline.run_all(excel_filename=f"test_{tipo}.xlsx", batch_size=2)
    
    return results

def main_comparacion_todos_procesos(train_sizes=None, calib_sizes=None):
    """
    Ejecuta el estudio completo para los 3 tipos de procesos.
    """
    results = {}
    results['ARMA'] = main_tamanos_crecientes_ARMA(train_sizes, calib_sizes)
    results['ARIMA'] = main_tamanos_crecientes_ARIMA(train_sizes, calib_sizes)
    results['SETAR'] = main_tamanos_crecientes_SETAR(train_sizes, calib_sizes)
    
    print("\n" + "="*80)
    print("📊 COMPARATIVA FINAL GLOBAL")
    print("="*80)
    for tipo, df in results.items():
        media = df[['AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 
                    'EnCQR-LSTM', 'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap']].mean().mean()
        print(f" {tipo:<10} | CRPS Promedio Global: {media:.6f}")
    
    return results


# =================================================================
# ANÁLISIS DE PROPORCIONES VARIABLES (N=240)
# =================================================================

def main_proporciones_240_ARMA():
    """
    Ejecuta análisis de proporciones variables (N=240) para procesos ARMA.
    
    Evalúa cómo la proporción de datos de calibración (10%-50%) afecta
    el desempeño cuando el tamaño total de datos es fijo (240).
    """
    start_time = time.time()
    
    print("="*80)
    print("ANÁLISIS PROPORCIONES VARIABLES (N=240) - PROCESOS ARMA")
    print("="*80)
    
    pipeline = Pipeline240_ProporcionesVariables(
        n_boot=1000,
        seed=42,
        verbose=False,
        proceso_tipo='ARMA'
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_PROPORCIONES_240_ARMA.xlsx",
        batch_size=20
    )
    
    analisis_proporciones_240(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_proporciones_240_ARIMA():
    """
    Ejecuta análisis de proporciones variables (N=240) para procesos ARIMA.
    """
    start_time = time.time()
    
    print("="*80)
    print("ANÁLISIS PROPORCIONES VARIABLES (N=240) - PROCESOS ARIMA")
    print("="*80)
    
    pipeline = Pipeline240_ProporcionesVariables(
        n_boot=1000,
        seed=42,
        verbose=False,
        proceso_tipo='ARIMA'
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_PROPORCIONES_240_ARIMA.xlsx",
        batch_size=20
    )
    
    analisis_proporciones_240(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_proporciones_240_SETAR():
    """
    Ejecuta análisis de proporciones variables (N=240) para procesos SETAR.
    """
    start_time = time.time()
    
    print("="*80)
    print("ANÁLISIS PROPORCIONES VARIABLES (N=240) - PROCESOS SETAR")
    print("="*80)
    
    pipeline = Pipeline240_ProporcionesVariables(
        n_boot=1000,
        seed=42,
        verbose=False,
        proceso_tipo='SETAR'
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_PROPORCIONES_240_SETAR.xlsx",
        batch_size=20
    )
    
    analisis_proporciones_240(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_test_proporciones_240_reducido():
    """
    Test reducido: solo 2 proporciones (10% y 30%) para cada proceso.
    Útil para verificar que todo funciona antes de ejecutar el experimento completo.
    """
    start_time = time.time()
    print("="*80)
    print("TEST REDUCIDO - PROPORCIONES 240 (TODOS LOS PROCESOS)")
    print("="*80)
    
    # Definir proporciones de prueba
    test_proportions = [
        {'prop_tag': '10%', 'n_train': 216, 'n_calib': 24, 'prop_val': 0.10},
        {'prop_tag': '30%', 'n_train': 168, 'n_calib': 72, 'prop_val': 0.30}
    ]
    
    results = {}
    
    # ARMA
    print("\n" + "="*60)
    print("EJECUTANDO ARMA...")
    print("="*60)
    pipeline_arma = Pipeline240_ProporcionesVariables(
        n_boot=1000, seed=42, verbose=False, proceso_tipo='ARMA'
    )
    pipeline_arma.SIZE_COMBINATIONS = test_proportions
    pipeline_arma.CONFIGS['ARMA'] = pipeline_arma.CONFIGS['ARMA'][:2]  # Solo 2 configs
    pipeline_arma.DISTRIBUTIONS = ['normal']
    pipeline_arma.VARIANCES = [1.0]
    
    results['ARMA'] = pipeline_arma.run_all(
        excel_filename="resultados_TEST_PROP240_ARMA.xlsx",
        batch_size=4
    )
    
    # ARIMA
    print("\n" + "="*60)
    print("EJECUTANDO ARIMA...")
    print("="*60)
    pipeline_arima = Pipeline240_ProporcionesVariables(
        n_boot=1000, seed=42, verbose=False, proceso_tipo='ARIMA'
    )
    pipeline_arima.SIZE_COMBINATIONS = test_proportions
    pipeline_arima.CONFIGS['ARIMA'] = pipeline_arima.CONFIGS['ARIMA'][:2]  # Solo 2 configs
    pipeline_arima.DISTRIBUTIONS = ['normal']
    pipeline_arima.VARIANCES = [1.0]
    
    results['ARIMA'] = pipeline_arima.run_all(
        excel_filename="resultados_TEST_PROP240_ARIMA.xlsx",
        batch_size=4
    )
    
    # SETAR
    print("\n" + "="*60)
    print("EJECUTANDO SETAR...")
    print("="*60)
    pipeline_setar = Pipeline240_ProporcionesVariables(
        n_boot=1000, seed=42, verbose=False, proceso_tipo='SETAR'
    )
    pipeline_setar.SIZE_COMBINATIONS = test_proportions
    pipeline_setar.CONFIGS['SETAR'] = pipeline_setar.CONFIGS['SETAR'][:2]  # Solo 2 configs
    pipeline_setar.DISTRIBUTIONS = ['normal']
    pipeline_setar.VARIANCES = [1.0]
    
    results['SETAR'] = pipeline_setar.run_all(
        excel_filename="resultados_TEST_PROP240_SETAR.xlsx",
        batch_size=4
    )
    
    # Análisis conjunto
    print("\n" + "="*80)
    print("ANÁLISIS COMPARATIVO DE LOS TRES PROCESOS")
    print("="*80)
    
    for proceso_tipo, df in results.items():
        print(f"\n{'='*60}")
        print(f"PROCESO: {proceso_tipo}")
        print(f"{'='*60}")
        analisis_proporciones_240(df)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total test: {elapsed:.1f}s")
    
    return results


def main_comparacion_proporciones_240_todos():
    """
    Ejecuta el experimento completo para los tres tipos de procesos
    y genera un análisis comparativo final.
    
    Total de escenarios: 3 procesos × 5 proporciones × 7 configs × 5 dist × 4 var = 2100 escenarios
    """
    start_time = time.time()
    
    print("="*80)
    print("EXPERIMENTO COMPLETO - PROPORCIONES 240 (TODOS LOS PROCESOS)")
    print("="*80)
    print("\nEste experimento ejecutará:")
    print("  • 3 tipos de procesos (ARMA, ARIMA, SETAR)")
    print("  • 5 proporciones de calibración (10%, 20%, 30%, 40%, 50%)")
    print("  • 7 configuraciones por proceso")
    print("  • 5 distribuciones de ruido")
    print("  • 4 niveles de varianza")
    print("  • TOTAL: ~2100 escenarios base × 13 pasos = ~27,300 filas\n")
    
    input("Presiona ENTER para continuar o Ctrl+C para cancelar...")
    
    results = {}
    
    # ARMA
    print("\n" + "="*80)
    print("1/3: EJECUTANDO ARMA...")
    print("="*80)
    results['ARMA'] = main_proporciones_240_ARMA()
    
    # ARIMA
    print("\n" + "="*80)
    print("2/3: EJECUTANDO ARIMA...")
    print("="*80)
    results['ARIMA'] = main_proporciones_240_ARIMA()
    
    # SETAR
    print("\n" + "="*80)
    print("3/3: EJECUTANDO SETAR...")
    print("="*80)
    results['SETAR'] = main_proporciones_240_SETAR()
    
    # Análisis comparativo final
    print("\n" + "="*80)
    print("ANÁLISIS COMPARATIVO FINAL - TODOS LOS PROCESOS")
    print("="*80)
    
    # Combinar todos los resultados
    df_combined = pd.concat([
        results['ARMA'].assign(Tipo_Proceso='ARMA'),
        results['ARIMA'].assign(Tipo_Proceso='ARIMA'),
        results['SETAR'].assign(Tipo_Proceso='SETAR')
    ], ignore_index=True)
    
    # Guardar resultados combinados
    df_combined.to_excel("resultados_PROPORCIONES_240_TODOS.xlsx", index=False)
    print("\n✅ Resultados combinados guardados en: resultados_PROPORCIONES_240_TODOS.xlsx")
    
    # Análisis individual por proceso
    for proceso_tipo, df in results.items():
        print(f"\n{'='*60}")
        print(f"PROCESO: {proceso_tipo}")
        print(f"{'='*60}")
        analisis_proporciones_240(df)
    
    # Análisis comparativo entre procesos
    print("\n" + "="*80)
    print("COMPARACIÓN ENTRE PROCESOS")
    print("="*80)
    analisis_comparativo_entre_procesos_240(df_combined)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total del experimento: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return results, df_combined


def analisis_comparativo_entre_procesos_240(df_combined: pd.DataFrame):
    """
    Análisis comparativo entre diferentes tipos de procesos (ARMA, ARIMA, SETAR).
    
    Args:
        df_combined: DataFrame combinado con resultados de todos los procesos
    """
    print("\n" + "="*60)
    print("ANÁLISIS COMPARATIVO ENTRE TIPOS DE PROCESOS")
    print("="*60)
    
    # Filtrar solo filas de promedio
    df_avg = df_combined[df_combined['Paso'] == 'Promedio'].copy()
    
    if df_avg.empty:
        print("⚠️  No hay datos de promedio para analizar")
        return
    
    # Columnas de modelos
    model_cols = ['AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 
                  'EnCQR-LSTM', 'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap']
    model_cols = [col for col in model_cols if col in df_avg.columns]
    
    print(f"\n📊 Resumen General:")
    print(f"  • Total de escenarios: {len(df_avg)}")
    print(f"  • Tipos de procesos: {sorted(df_avg['Tipo_Proceso'].unique())}")
    print(f"  • Proporciones: {sorted(df_avg['Prop_Calib'].unique())}")
    
    # Desempeño promedio por tipo de proceso
    print("\n" + "="*60)
    print("DESEMPEÑO PROMEDIO POR TIPO DE PROCESO")
    print("="*60)
    
    for proceso in sorted(df_avg['Tipo_Proceso'].unique()):
        df_proc = df_avg[df_avg['Tipo_Proceso'] == proceso]
        print(f"\n🔹 {proceso}")
        print("   " + "-"*50)
        
        for model in model_cols:
            if model in df_proc.columns:
                vals = df_proc[model].dropna()
                if len(vals) > 0:
                    print(f"   {model:25s}: {vals.mean():.4f} ± {vals.std():.4f}")
    
    # Mejor modelo por tipo de proceso
    print("\n" + "="*60)
    print("MEJOR MODELO POR TIPO DE PROCESO")
    print("="*60)
    
    for proceso in sorted(df_avg['Tipo_Proceso'].unique()):
        df_proc = df_avg[df_avg['Tipo_Proceso'] == proceso]
        print(f"\n🏆 {proceso}")
        
        model_means = {}
        for model in model_cols:
            if model in df_proc.columns:
                vals = df_proc[model].dropna()
                if len(vals) > 0:
                    model_means[model] = vals.mean()
        
        if model_means:
            ranked = sorted(model_means.items(), key=lambda x: x[1])
            for i, (model, score) in enumerate(ranked[:5], 1):
                print(f"   {i}. {model:25s}: {score:.4f}")
    
    # Análisis por proporción y tipo de proceso
    print("\n" + "="*60)
    print("MEJOR PROPORCIÓN POR MODELO Y TIPO DE PROCESO")
    print("="*60)
    
    for proceso in sorted(df_avg['Tipo_Proceso'].unique()):
        print(f"\n📈 {proceso}")
        df_proc = df_avg[df_avg['Tipo_Proceso'] == proceso]
        
        for model in model_cols:
            if model in df_proc.columns:
                best_prop = None
                best_score = float('inf')
                
                for prop in df_proc['Prop_Calib'].unique():
                    df_prop = df_proc[df_proc['Prop_Calib'] == prop]
                    vals = df_prop[model].dropna()
                    if len(vals) > 0:
                        mean_score = vals.mean()
                        if mean_score < best_score:
                            best_score = mean_score
                            best_prop = prop
                
                if best_prop:
                    print(f"   {model:25s}: {best_prop} (CRPS={best_score:.4f})")
    
    # Consistencia entre procesos
    print("\n" + "="*60)
    print("CONSISTENCIA DE MODELOS ENTRE PROCESOS")
    print("="*60)
    print("\n(Modelos que mantienen buen desempeño en todos los tipos de proceso)")
    
    for model in model_cols:
        if model in df_avg.columns:
            ranks_by_process = {}
            
            for proceso in df_avg['Tipo_Proceso'].unique():
                df_proc = df_avg[df_avg['Tipo_Proceso'] == proceso]
                
                model_means = {}
                for m in model_cols:
                    if m in df_proc.columns:
                        vals = df_proc[m].dropna()
                        if len(vals) > 0:
                            model_means[m] = vals.mean()
                
                if model_means and model in model_means:
                    ranked = sorted(model_means.items(), key=lambda x: x[1])
                    rank = next((i+1 for i, (m, _) in enumerate(ranked) if m == model), None)
                    if rank:
                        ranks_by_process[proceso] = rank
            
            if len(ranks_by_process) == len(df_avg['Tipo_Proceso'].unique()):
                avg_rank = sum(ranks_by_process.values()) / len(ranks_by_process)
                ranks_str = ", ".join([f"{k}={v}" for k, v in ranks_by_process.items()])
                print(f"   {model:25s}: Promedio={avg_rank:.1f} ({ranks_str})")


def analisis_proporciones_240(df: pd.DataFrame):
    """
    Análisis específico para resultados de proporciones variables con N=240.
    
    Args:
        df: DataFrame con resultados del pipeline
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE PROPORCIONES VARIABLES (N=240)")
    print("="*60)
    
    # Filtrar solo filas de promedio
    df_avg = df[df['Paso'] == 'Promedio'].copy()
    
    if df_avg.empty:
        print("⚠️  No hay datos de promedio para analizar")
        return
    
    # Columnas de modelos
    model_cols = ['AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 
                  'EnCQR-LSTM', 'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap']
    model_cols = [col for col in model_cols if col in df_avg.columns]
    
    print(f"\n📊 Resumen General:")
    print(f"  • Total de filas: {len(df)}")
    print(f"  • Escenarios únicos: {len(df_avg)}")
    print(f"  • Proporciones evaluadas: {sorted(df_avg['Prop_Calib'].unique())}")
    print(f"  • Procesos: {sorted(df_avg['Proceso'].unique())}")
    print(f"  • Distribuciones: {sorted(df_avg['Distribución'].unique())}")
    print(f"  • Varianzas: {sorted(df_avg['Varianza'].unique())}")
    
    # Análisis por proporción
    print("\n" + "="*60)
    print("DESEMPEÑO PROMEDIO POR PROPORCIÓN DE CALIBRACIÓN")
    print("="*60)
    
    for prop in sorted(df_avg['Prop_Calib'].unique()):
        df_prop = df_avg[df_avg['Prop_Calib'] == prop]
        print(f"\n📈 Proporción Calibración: {prop}")
        print(f"   (N_Train={df_prop['N_Train'].iloc[0]}, N_Calib={df_prop['N_Calib'].iloc[0]})")
        print("   " + "-"*50)
        
        for model in model_cols:
            if model in df_prop.columns:
                vals = df_prop[model].dropna()
                if len(vals) > 0:
                    print(f"   {model:25s}: {vals.mean():.4f} ± {vals.std():.4f}")
    
    # Ranking por proporción
    print("\n" + "="*60)
    print("RANKING DE MODELOS POR PROPORCIÓN")
    print("="*60)
    
    for prop in sorted(df_avg['Prop_Calib'].unique()):
        df_prop = df_avg[df_avg['Prop_Calib'] == prop]
        print(f"\n🏆 Proporción: {prop}")
        
        model_means = {}
        for model in model_cols:
            if model in df_prop.columns:
                vals = df_prop[model].dropna()
                if len(vals) > 0:
                    model_means[model] = vals.mean()
        
        if model_means:
            ranked = sorted(model_means.items(), key=lambda x: x[1])
            for i, (model, score) in enumerate(ranked[:5], 1):
                print(f"   {i}. {model:25s}: {score:.4f}")
    
    # Análisis de tendencias
    print("\n" + "="*60)
    print("TENDENCIAS: EFECTO DE AUMENTAR PROPORCIÓN DE CALIBRACIÓN")
    print("="*60)
    
    props_sorted = sorted(df_avg['Prop_Calib'].unique(), 
                         key=lambda x: float(x.strip('%')))
    
    for model in model_cols:
        if model in df_avg.columns:
            scores_by_prop = []
            for prop in props_sorted:
                df_prop = df_avg[df_avg['Prop_Calib'] == prop]
                vals = df_prop[model].dropna()
                if len(vals) > 0:
                    scores_by_prop.append(vals.mean())
            
            if len(scores_by_prop) >= 2:
                trend = "📈 MEJORA" if scores_by_prop[-1] < scores_by_prop[0] else "📉 EMPEORA"
                change_pct = ((scores_by_prop[-1] - scores_by_prop[0]) / scores_by_prop[0]) * 100
                print(f"   {model:25s}: {trend} ({change_pct:+.1f}%)")
    
    # Mejor proporción por modelo
    print("\n" + "="*60)
    print("MEJOR PROPORCIÓN POR MODELO")
    print("="*60)
    
    for model in model_cols:
        if model in df_avg.columns:
            best_prop = None
            best_score = float('inf')
            
            for prop in df_avg['Prop_Calib'].unique():
                df_prop = df_avg[df_avg['Prop_Calib'] == prop]
                vals = df_prop[model].dropna()
                if len(vals) > 0:
                    mean_score = vals.mean()
                    if mean_score < best_score:
                        best_score = mean_score
                        best_prop = prop
            
            if best_prop:
                print(f"   {model:25s}: {best_prop} (CRPS={best_score:.4f})")


def analisis_comparativo_proporciones_240(df_combined: pd.DataFrame):
    """
    Análisis comparativo entre los tres tipos de procesos.
    
    Args:
        df_combined: DataFrame con resultados de ARMA, ARIMA y SETAR combinados
    """
    print("\n" + "="*60)
    print("ANÁLISIS COMPARATIVO ENTRE PROCESOS")
    print("="*60)
    
    df_avg = df_combined[df_combined['Paso'] == 'Promedio'].copy()
    
    model_cols = ['AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 
                  'EnCQR-LSTM', 'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap']
    model_cols = [col for col in model_cols if col in df_avg.columns]
    
    # Desempeño por tipo de proceso
    print("\n📊 DESEMPEÑO PROMEDIO POR TIPO DE PROCESO:")
    print("-"*60)
    
    for proceso in sorted(df_avg['Tipo_Proceso'].unique()):
        df_proc = df_avg[df_avg['Tipo_Proceso'] == proceso]
        print(f"\n{proceso}:")
        
        for model in model_cols:
            if model in df_proc.columns:
                vals = df_proc[model].dropna()
                if len(vals) > 0:
                    print(f"  {model:25s}: {vals.mean():.4f} ± {vals.std():.4f}")
    
    # Mejor proceso por modelo
    print("\n" + "="*60)
    print("MEJOR TIPO DE PROCESO POR MODELO")
    print("="*60)
    
    for model in model_cols:
        if model in df_avg.columns:
            best_proceso = None
            best_score = float('inf')
            
            for proceso in df_avg['Tipo_Proceso'].unique():
                df_proc = df_avg[df_avg['Tipo_Proceso'] == proceso]
                vals = df_proc[model].dropna()
                if len(vals) > 0:
                    mean_score = vals.mean()
                    if mean_score < best_score:
                        best_score = mean_score
                        best_proceso = proceso
            
            if best_proceso:
                print(f"  {model:25s}: {best_proceso} (CRPS={best_score:.4f})")
    
    print("\n" + "="*60)


# =================================================================
# Sieve multiple D ARIMA
# =================================================================

def diebold_mariano_test_modificado(errors1, errors2, h=1):
    """
    Test Diebold-Mariano modificado con corrección Harvey-Leybourne-Newbold (1997)
    
    Parameters:
    -----------
    errors1, errors2 : array-like
        Errores de pronóstico (ECRPS) de los dos modelos
    h : int
        Horizonte de pronóstico (forecast horizon)
    
    Returns:
    --------
    hln_dm_stat : float
        Estadístico DM corregido (HLN-DM)
    p_value : float
        P-valor usando distribución t-Student con T-1 grados de libertad
    dm_stat : float
        Estadístico DM original (sin corrección)
    """
    from scipy import stats
    
    # Calcular diferencial de pérdida
    d = errors1 - errors2
    d_bar = np.mean(d)
    T = len(d)
    
    # Calcular autocovarianzas
    def gamma_d(k):
        if k == 0:
            return np.var(d, ddof=1)
        else:
            return np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
    
    # Estimar la varianza de largo plazo usando Newey-West
    # Para h-step-ahead forecasts, incluimos hasta h-1 lags
    gamma_0 = gamma_d(0)
    gamma_sum = gamma_0
    
    if h > 1:
        for k in range(1, h):
            gamma_k = gamma_d(k)
            gamma_sum += 2 * gamma_k
    
    var_d = gamma_sum / T
    
    if var_d <= 0:
        return 0, 1.0, 0
    
    # Estadístico DM original
    dm_stat = d_bar / np.sqrt(var_d)
    
    # Corrección Harvey-Leybourne-Newbold (1997)
    correction_factor = np.sqrt((T + 1 - 2*h + h*(h-1)) / T)
    hln_dm_stat = correction_factor * dm_stat
    
    # P-valor usando t-Student con T-1 grados de libertad
    df = T - 1
    p_value = 2 * (1 - stats.t.cdf(abs(hln_dm_stat), df))
    
    return hln_dm_stat, p_value, dm_stat


def test_diebold_mariano_por_d(df_final):
    """
    Realiza test Diebold-Mariano modificado (HLN) para cada valor de d.
    Compara SIN_DIFF vs CON_DIFF.
    
    Retorna DataFrame con resultados del test estadístico.
    """
    print("\n" + "="*100)
    print("TEST DIEBOLD-MARIANO MODIFICADO (HLN): SIN_DIFF vs CON_DIFF")
    print("="*100)
    
    # Verificar que existe la columna Sieve Bootstrap
    if 'Sieve Bootstrap' not in df_final.columns:
        print("⚠️ No se encontró la columna 'Sieve Bootstrap'")
        return None
    
    # Filtrar datos válidos
    datos = df_final[['d', 'Modalidad', 'Sieve Bootstrap']].copy()
    datos.rename(columns={'Sieve Bootstrap': 'ECRPS'}, inplace=True)
    datos = datos.dropna()
    
    # Obtener valores únicos de d
    valores_d = sorted(datos['d'].unique())
    
    # Lista para almacenar resultados
    resultados = []
    
    # Iterar sobre cada valor de d
    for d_val in valores_d:
        # Filtrar datos para el d actual
        datos_d = datos[datos['d'] == d_val].copy()
        
        # Separar por modalidad
        sin_diff = datos_d[datos_d['Modalidad'] == 'SIN_DIFF']['ECRPS'].values
        con_diff = datos_d[datos_d['Modalidad'] == 'CON_DIFF']['ECRPS'].values
        
        # Verificar que ambas modalidades tengan datos
        if len(sin_diff) == 0 or len(con_diff) == 0:
            print(f"⚠️ d={d_val} no tiene datos para ambas modalidades")
            continue
        
        # Verificar que tengan la misma longitud
        if len(sin_diff) != len(con_diff):
            print(f"⚠️ d={d_val} tiene diferente número de observaciones")
            min_len = min(len(sin_diff), len(con_diff))
            sin_diff = sin_diff[:min_len]
            con_diff = con_diff[:min_len]
        
        # Calcular estadísticas descriptivas
        ecrps_sin_diff_mean = np.mean(sin_diff)
        ecrps_con_diff_mean = np.mean(con_diff)
        diferencia = ecrps_sin_diff_mean - ecrps_con_diff_mean
        
        # Realizar test Diebold-Mariano modificado
        hln_dm_stat, p_value, dm_stat = diebold_mariano_test_modificado(sin_diff, con_diff, h=1)
        
        # Determinar significancia
        if p_value < 0.01:
            significativo = "***"
        elif p_value < 0.05:
            significativo = "**"
        elif p_value < 0.10:
            significativo = "*"
        else:
            significativo = "No"
        
        # Determinar mejor modalidad
        if diferencia < 0:
            mejor = "SIN_DIFF"
        elif diferencia > 0:
            mejor = "CON_DIFF"
        else:
            mejor = "Empate"
        
        # Agregar a resultados
        resultados.append({
            'd': int(d_val),
            'N_obs': len(sin_diff),
            'ECRPS_SIN_DIFF': ecrps_sin_diff_mean,
            'ECRPS_CON_DIFF': ecrps_con_diff_mean,
            'Diferencia': diferencia,
            'DM_stat': dm_stat,
            'HLN-DM_stat': hln_dm_stat,
            'p_valor': p_value,
            'Significativo': significativo,
            'Mejor': mejor
        })
    
    # Crear DataFrame con resultados
    resultados_df = pd.DataFrame(resultados)
    
    if len(resultados_df) == 0:
        print("⚠️ No se pudieron calcular tests estadísticos")
        return None
    
    # Formatear para mejor visualización
    resultados_df['ECRPS_SIN_DIFF'] = resultados_df['ECRPS_SIN_DIFF'].round(6)
    resultados_df['ECRPS_CON_DIFF'] = resultados_df['ECRPS_CON_DIFF'].round(6)
    resultados_df['Diferencia'] = resultados_df['Diferencia'].round(6)
    resultados_df['DM_stat'] = resultados_df['DM_stat'].round(4)
    resultados_df['HLN-DM_stat'] = resultados_df['HLN-DM_stat'].round(4)
    resultados_df['p_valor'] = resultados_df['p_valor'].round(4)
    
    # Mostrar resultados
    print("\nH0: No hay diferencia significativa entre modalidades")
    print("H1: Hay diferencia significativa entre modalidades")
    print("\nSignificancia: *** p<0.01, ** p<0.05, * p<0.10, No = no significativo")
    print("\n")
    print(resultados_df.to_string(index=False))
    
    # Resumen de resultados
    print("\n" + "="*100)
    print("RESUMEN DEL TEST ESTADÍSTICO")
    print("="*100)
    
    n_significativos = len(resultados_df[resultados_df['Significativo'] != 'No'])
    n_total = len(resultados_df)
    n_sin_diff_mejor = len(resultados_df[resultados_df['Mejor'] == 'SIN_DIFF'])
    n_con_diff_mejor = len(resultados_df[resultados_df['Mejor'] == 'CON_DIFF'])
    
    print(f"\n✓ Total de comparaciones: {n_total}")
    print(f"✓ Diferencias significativas: {n_significativos} ({100*n_significativos/n_total:.1f}%)")
    print(f"✓ No significativas: {n_total - n_significativos} ({100*(n_total-n_significativos)/n_total:.1f}%)")
    print(f"\n✓ SIN_DIFF mejor: {n_sin_diff_mejor} casos ({100*n_sin_diff_mejor/n_total:.1f}%)")
    print(f"✓ CON_DIFF mejor: {n_con_diff_mejor} casos ({100*n_con_diff_mejor/n_total:.1f}%)")
    
    # Calcular estadística global
    sin_diff_global = datos[datos['Modalidad'] == 'SIN_DIFF']['ECRPS'].mean()
    con_diff_global = datos[datos['Modalidad'] == 'CON_DIFF']['ECRPS'].mean()
    
    print(f"\n✓ ECRPS promedio global:")
    print(f"  • SIN_DIFF: {sin_diff_global:.6f}")
    print(f"  • CON_DIFF: {con_diff_global:.6f}")
    print(f"  • Diferencia: {sin_diff_global - con_diff_global:+.6f}")
    
    if sin_diff_global < con_diff_global:
        mejora = ((con_diff_global - sin_diff_global) / con_diff_global) * 100
        print(f"  • SIN_DIFF es {mejora:.2f}% mejor globalmente")
    else:
        mejora = ((sin_diff_global - con_diff_global) / sin_diff_global) * 100
        print(f"  • CON_DIFF es {mejora:.2f}% mejor globalmente")
    
    return resultados_df


def analisis_sieve_doble_modalidad(df_final):
    """
    Análisis especializado para resultados de Sieve Bootstrap con doble modalidad.
    
    Incluye:
    1. Test Diebold-Mariano modificado (HLN) por cada d
    2. Desempeño por valor de d
    3. SIN_DIFF vs CON_DIFF
    4. Tendencias según d aumenta
    5. Mejor d por modalidad
    """
    print("\n" + "="*80)
    print("ANÁLISIS SIEVE BOOTSTRAP - DOBLE MODALIDAD")
    print("="*80)
    
    if 'Sieve Bootstrap' not in df_final.columns:
        print("⚠️ No se encontró columna 'Sieve Bootstrap'")
        return
    
    if 'Paso' in df_final.columns:
        df_steps = df_final[df_final['Paso'] != 'Promedio'].copy()
    else:
        df_steps = df_final.copy()
    
    if len(df_steps) == 0:
        print("⚠️ No hay datos suficientes para el análisis.")
        return
    
    # Asegurar tipos correctos
    if 'd' in df_steps.columns:
        df_steps['d'] = pd.to_numeric(df_steps['d'], errors='coerce')
    
    d_values = sorted(df_steps['d'].unique())
    modalidades = sorted(df_steps['Modalidad'].unique()) if 'Modalidad' in df_steps.columns else []
    
    # =================================================================
    # 0. TEST DIEBOLD-MARIANO MODIFICADO (NUEVO)
    # =================================================================
    print("\n" + "="*80)
    print("🔬 0. TEST ESTADÍSTICO DIEBOLD-MARIANO MODIFICADO")
    print("="*80)
    
    dm_results = test_diebold_mariano_por_d(df_final)
    
    if dm_results is not None:
        # Guardar resultados del test en Excel
        excel_filename = "resultados_DM_test_sieve.xlsx"
        dm_results.to_excel(excel_filename, index=False)
        print(f"\n✅ Resultados del test guardados en: {excel_filename}")
    
    # =================================================================
    # 1. COMPARACIÓN GLOBAL: SIN_DIFF vs CON_DIFF
    # =================================================================
    print("\n" + "="*80)
    print("🔍 1. COMPARACIÓN GLOBAL POR MODALIDAD")
    print("="*80)
    
    for modalidad in modalidades:
        df_mod = df_steps[df_steps['Modalidad'] == modalidad]
        
        if len(df_mod) == 0:
            continue
        
        val = df_mod['Sieve Bootstrap'].mean()
        n_valid = df_mod['Sieve Bootstrap'].notna().sum()
        
        print(f"\n{modalidad}:")
        print(f"  CRPS Promedio: {val:.6f}")
        print(f"  Observaciones válidas: {n_valid}/{len(df_mod)}")
    
    # =================================================================
    # 2. DESEMPEÑO POR CADA d Y MODALIDAD
    # =================================================================
    print("\n" + "="*80)
    print("📊 2. DESEMPEÑO POR VALOR DE d")
    print("="*80)
    
    print(f"\n{'d':<6} {'SIN_DIFF':<15} {'CON_DIFF':<15} {'Mejor':<12}")
    print("-" * 55)
    
    for d_val in d_values:
        df_d = df_steps[df_steps['d'] == d_val]
        
        sin_diff = df_d[df_d['Modalidad'] == 'SIN_DIFF']['Sieve Bootstrap'].mean()
        con_diff = df_d[df_d['Modalidad'] == 'CON_DIFF']['Sieve Bootstrap'].mean()
        
        if not pd.isna(sin_diff) and not pd.isna(con_diff):
            mejor = "SIN_DIFF" if sin_diff < con_diff else "CON_DIFF"
            print(f"{d_val:<6} {sin_diff:.6f}      {con_diff:.6f}      {mejor:<12}")
    
    # =================================================================
    # 3. TENDENCIAS POR DISTRIBUCIÓN Y VARIANZA
    # =================================================================
    print("\n" + "="*80)
    print("📈 3. TENDENCIAS POR DISTRIBUCIÓN (Promedio por d)")
    print("="*80)
    
    distribuciones = sorted(df_steps['Distribución'].unique())
    
    for dist in distribuciones:
        print(f"\n--- {dist} ---")
        df_dist = df_steps[df_steps['Distribución'] == dist]
        
        print(f"{'d':<6} {'SIN_DIFF':<12} {'CON_DIFF':<12}")
        print("-" * 35)
        
        for d_val in d_values:
            df_d_dist = df_dist[df_dist['d'] == d_val]
            
            sin_val = df_d_dist[df_d_dist['Modalidad'] == 'SIN_DIFF']['Sieve Bootstrap'].mean()
            con_val = df_d_dist[df_d_dist['Modalidad'] == 'CON_DIFF']['Sieve Bootstrap'].mean()
            
            sin_str = f"{sin_val:.6f}" if not pd.isna(sin_val) else "---"
            con_str = f"{con_val:.6f}" if not pd.isna(con_val) else "---"
            
            print(f"{d_val:<6} {sin_str:<12} {con_str:<12}")
    
    # =================================================================
    # 4. MEJOR d POR MODALIDAD
    # =================================================================
    print("\n" + "="*80)
    print("🎲 4. MEJOR VALOR DE d POR MODALIDAD")
    print("="*80)
    
    for modalidad in modalidades:
        df_mod = df_steps[df_steps['Modalidad'] == modalidad]
        
        best_d = None
        best_crps = float('inf')
        
        for d_val in d_values:
            df_d_mod = df_mod[df_mod['d'] == d_val]
            val = df_d_mod['Sieve Bootstrap'].mean()
            
            if not pd.isna(val) and val < best_crps:
                best_crps = val
                best_d = d_val
        
        print(f"\n{modalidad}:")
        print(f"  Mejor d: {best_d}")
        print(f"  CRPS en d={best_d}: {best_crps:.6f}")
    
    # =================================================================
    # 5. COMPARACIÓN DIRECTA: SIN_DIFF vs CON_DIFF
    # =================================================================
    print("\n" + "="*80)
    print("⚖️  5. COMPARACIÓN DIRECTA GLOBAL")
    print("="*80)
    
    if len(modalidades) == 2:
        mod_sin = [m for m in modalidades if 'SIN' in m][0]
        mod_con = [m for m in modalidades if 'CON' in m][0]
        
        df_sin = df_steps[df_steps['Modalidad'] == mod_sin]
        df_con = df_steps[df_steps['Modalidad'] == mod_con]
        
        val_sin = df_sin['Sieve Bootstrap'].mean()
        val_con = df_con['Sieve Bootstrap'].mean()
        
        if not pd.isna(val_sin) and not pd.isna(val_con):
            diff = val_con - val_sin
            mejor = mod_sin if val_sin < val_con else mod_con
            pct_diff = abs(diff / val_sin) * 100
            
            print(f"\n{mod_sin}: {val_sin:.6f}")
            print(f"{mod_con}: {val_con:.6f}")
            print(f"\nDiferencia: {diff:+.6f} ({pct_diff:.2f}%)")
            print(f"GANADOR: {mejor}")
    
    # =================================================================
    # 6. RESUMEN POR PROCESO ARMA
    # =================================================================
    print("\n" + "="*80)
    print("🔄 6. RESUMEN POR PROCESO ARMA BASE")
    print("="*80)
    
    arma_bases = sorted(df_steps['ARMA_base'].unique())
    
    print(f"\n{'ARMA Base':<15} {'SIN_DIFF':<12} {'CON_DIFF':<12} {'Mejor':<12}")
    print("-" * 55)
    
    for arma in arma_bases:
        df_arma = df_steps[df_steps['ARMA_base'] == arma]
        
        sin_val = df_arma[df_arma['Modalidad'] == 'SIN_DIFF']['Sieve Bootstrap'].mean()
        con_val = df_arma[df_arma['Modalidad'] == 'CON_DIFF']['Sieve Bootstrap'].mean()
        
        if not pd.isna(sin_val) and not pd.isna(con_val):
            mejor = "SIN_DIFF" if sin_val < con_val else "CON_DIFF"
            print(f"{arma:<15} {sin_val:.6f}   {con_val:.6f}   {mejor:<12}")
    
    # =================================================================
    # 7. RESUMEN EJECUTIVO
    # =================================================================
    print("\n" + "="*80)
    print("📋 7. RESUMEN EJECUTIVO")
    print("="*80)
    
    # Mejor modalidad global
    if len(modalidades) == 2:
        crps_sin = df_steps[df_steps['Modalidad'] == mod_sin]['Sieve Bootstrap'].mean()
        crps_con = df_steps[df_steps['Modalidad'] == mod_con]['Sieve Bootstrap'].mean()
        
        print(f"\n✓ MEJOR MODALIDAD GLOBAL:")
        print(f"  • {mod_sin}: CRPS = {crps_sin:.6f}")
        print(f"  • {mod_con}: CRPS = {crps_con:.6f}")
        
        if crps_sin < crps_con:
            mejora = ((crps_con - crps_sin) / crps_sin) * 100
            print(f"  → GANADOR: {mod_sin} (mejora de {mejora:.2f}%)")
        else:
            mejora = ((crps_sin - crps_con) / crps_con) * 100
            print(f"  → GANADOR: {mod_con} (mejora de {mejora:.2f}%)")
    
    # Mejor d global
    best_d_global = None
    best_crps_global = float('inf')
    
    for d_val in d_values:
        df_d = df_steps[df_steps['d'] == d_val]
        val = df_d['Sieve Bootstrap'].mean()
        
        if not pd.isna(val) and val < best_crps_global:
            best_crps_global = val
            best_d_global = d_val
    
    if best_d_global is not None:
        print(f"\n✓ MEJOR VALOR DE d GLOBAL:")
        print(f"  → d = {best_d_global}: CRPS = {best_crps_global:.6f}")
    
    print("\n" + "="*80)
    print("FIN DEL ANÁLISIS")
    print("="*80)


# =============================================================================
# RUNNERS ESPECIALIZADOS
# =============================================================================

def main_sieve_full():
    """
    Ejecución completa SOLO con Sieve Bootstrap.
    Incluye análisis estadístico completo con test Diebold-Mariano.
    
    Total esperado: 23,580 filas
    - 7 valores de d
    - 7 configuraciones ARMA
    - 5 distribuciones
    - 4 varianzas
    - 2 modalidades (SIN_DIFF + CON_DIFF)
    - 12 pasos de predicción
    = 7 × 7 × 5 × 4 × 2 × 12 = 23,520 filas
    """
    start_time = time.time()
    
    print("="*80)
    print("SIMULACIÓN SIEVE BOOTSTRAP - CONFIGURACIÓN COMPLETA")
    print("Ejecutando análisis exhaustivo con test estadístico Diebold-Mariano")
    print("="*80)
    
    pipeline = PipelineARIMA_MultiD_SieveOnly(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    df_final = pipeline.run_all(
        excel_filename="resultados_SIEVE_d1_a_d10_COMPLETO.xlsx",
        batch_size=30,
        n_jobs=4
    )
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETO DE RESULTADOS")
    print("="*80)
    
    # Análisis descriptivo especializado
    analisis_sieve_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/60:.2f} minutos)")
    
    return df_final


def main_sieve_test_reducido():
    """
    Test reducido SOLO con Sieve Bootstrap.
    Configuración mínima para validación rápida.
    Incluye test estadístico Diebold-Mariano.
    """
    start_time = time.time()
    
    print("="*80)
    print("TEST REDUCIDO: SIEVE BOOTSTRAP (d=1,2,3)")
    print("="*80)
    
    pipeline = PipelineARIMA_MultiD_SieveOnly(
        n_boot=1000, 
        seed=42, 
        verbose=True
    )
    
    # Configuración reducida
    pipeline.D_VALUES = [1, 2, 3]
    pipeline.ARMA_CONFIGS = [
        {'nombre': 'RW', 'phi': [], 'theta': []},
        {'nombre': 'AR(1)', 'phi': [0.6], 'theta': []}
    ]
    pipeline.DISTRIBUTIONS = ['normal', 'uniform']
    pipeline.VARIANCES = [0.5, 1.0]
    
    df_final = pipeline.run_all(
        excel_filename="resultados_TEST_SIEVE.xlsx",
        batch_size=8,
        n_jobs=2
    )
    
    print("\n" + "="*80)
    print("ANÁLISIS DE RESULTADOS DEL TEST")
    print("="*80)
    
    analisis_sieve_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final


def main_sieve_solo_d1():
    """
    Configuración especial: SOLO d=1 con Sieve Bootstrap.
    Útil para comparación directa con Pipeline140SinSesgos_ARIMA_ConDiferenciacion.
    Incluye test estadístico.
    """
    start_time = time.time()
    
    print("="*80)
    print("SIEVE BOOTSTRAP: SOLO d=1 (VALIDACIÓN)")
    print("="*80)
    
    pipeline = PipelineARIMA_MultiD_SieveOnly(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    # Solo d=1
    pipeline.D_VALUES = [1]
    
    df_final = pipeline.run_all(
        excel_filename="resultados_SIEVE_d1_VALIDACION.xlsx",
        batch_size=20,
        n_jobs=4
    )
    
    print("\n" + "="*80)
    print("ANÁLISIS ESPECIALIZADO PARA d=1")
    print("="*80)
    
    analisis_sieve_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final


def main_sieve_alto_orden():
    """
    Configuración especial: SOLO órdenes altos (d=5,7,10).
    Para investigar comportamiento en integración alta.
    Incluye test estadístico Diebold-Mariano.
    """
    start_time = time.time()
    
    print("="*80)
    print("SIEVE BOOTSTRAP: ÓRDENES ALTOS (d=5,7,10)")
    print("="*80)
    
    pipeline = PipelineARIMA_MultiD_SieveOnly(
        n_boot=1000,
        seed=42,
        verbose=False
    )
    
    # Solo órdenes altos
    pipeline.D_VALUES = [5, 7, 10]
    
    df_final = pipeline.run_all(
        excel_filename="resultados_SIEVE_ALTO_ORDEN.xlsx",
        batch_size=25,
        n_jobs=4
    )
    
    print("\n" + "="*80)
    print("ANÁLISIS ESPECIALIZADO PARA ÓRDENES ALTOS")
    print("="*80)
    
    analisis_sieve_doble_modalidad(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return df_final