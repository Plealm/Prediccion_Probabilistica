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

from pipeline import Pipeline140SinSesgos_ARMA, Pipeline140SinSesgos_ARIMA, Pipeline140SinSesgos_SETAR, Pipeline140_TamanosCrecientes 
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

def main_tamanos_crecientes_ARMA(train_sizes=None, calib_sizes=None):
    """
    Ejecuta análisis de tamaños crecientes para procesos ARMA.
    Args:
        train_sizes: Lista de tamaños de entrenamiento (default: [100, 200, 300, 500, 1000])
        calib_sizes: Lista de tamaños de calibración (default: [20, 40, 60, 100, 200])
    """
    start_time = time.time()
    
    print("="*80)
    print("ANÁLISIS TAMAÑOS CRECIENTES - PROCESOS ARMA")
    print("="*80)
    
    pipeline = Pipeline140_TamanosCrecientes(
        n_boot=1000,
        seed=42,
        verbose=False,
        proceso_tipo='ARMA'
    )
    
    # Configurar tamaños personalizados si se proporcionan
    if train_sizes is not None:
        pipeline.TRAIN_SIZES = train_sizes
    if calib_sizes is not None:
        pipeline.CALIB_SIZES = calib_sizes
    
    df_final = pipeline.run_all(
        excel_filename="resultados_TAMANOS_CRECIENTES_ARMA.xlsx",
        batch_size=20
    )
    
    analisis_tamanos_crecientes(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_tamanos_crecientes_ARIMA(train_sizes=None, calib_sizes=None):
    """
    Ejecuta análisis de tamaños crecientes para procesos ARIMA.
    """
    start_time = time.time()
    print("="*80)
    print("ANÁLISIS TAMAÑOS CRECIENTES - PROCESOS ARIMA")
    print("="*80)
    
    pipeline = Pipeline140_TamanosCrecientes(
        n_boot=1000,
        seed=42,
        verbose=False,
        proceso_tipo='ARIMA'
    )
    
    if train_sizes is not None:
        pipeline.TRAIN_SIZES = train_sizes
    if calib_sizes is not None:
        pipeline.CALIB_SIZES = calib_sizes
    
    df_final = pipeline.run_all(
        excel_filename="resultados_TAMANOS_CRECIENTES_ARIMA.xlsx",
        batch_size=20
    )
    
    analisis_tamanos_crecientes(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_tamanos_crecientes_SETAR(train_sizes=None, calib_sizes=None):
    """
    Ejecuta análisis de tamaños crecientes para procesos SETAR.
    """
    start_time = time.time()
    print("="*80)
    print("ANÁLISIS TAMAÑOS CRECIENTES - PROCESOS SETAR")
    print("="*80)
    
    pipeline = Pipeline140_TamanosCrecientes(
        n_boot=1000,
        seed=42,
        verbose=False,
        proceso_tipo='SETAR'
    )
    
    if train_sizes is not None:
        pipeline.TRAIN_SIZES = train_sizes
    if calib_sizes is not None:
        pipeline.CALIB_SIZES = calib_sizes
    
    df_final = pipeline.run_all(
        excel_filename="resultados_TAMANOS_CRECIENTES_SETAR.xlsx",
        batch_size=20
    )
    
    analisis_tamanos_crecientes(df_final)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return df_final


def main_test_tamanos_reducido():
    """
    Test reducido: solo 2 tamaños de train y 2 de calib para cada proceso.
    """
    start_time = time.time()
    print("="*80)
    print("TEST REDUCIDO - TAMAÑOS CRECIENTES (TODOS LOS PROCESOS)")
    print("="*80)
    
    test_train_sizes = [100, 200]
    test_calib_sizes = [20, 40]
    
    results = {}
    
    # ARMA
    print("\n" + "="*60)
    print("EJECUTANDO ARMA...")
    print("="*60)
    pipeline_arma = Pipeline140_TamanosCrecientes(
        n_boot=1000, seed=42, verbose=False, proceso_tipo='ARMA'
    )
    pipeline_arma.TRAIN_SIZES = test_train_sizes
    pipeline_arma.CALIB_SIZES = test_calib_sizes
    pipeline_arma.ARMA_CONFIGS = pipeline_arma.ARMA_CONFIGS[:2]  # Solo 2 configs
    pipeline_arma.DISTRIBUTIONS = ['normal']
    pipeline_arma.VARIANCES = [1.0]
    
    results['ARMA'] = pipeline_arma.run_all(
        excel_filename="resultados_TEST_TAMANOS_ARMA.xlsx",
        batch_size=4
    )
    
    # ARIMA
    print("\n" + "="*60)
    print("EJECUTANDO ARIMA...")
    print("="*60)
    pipeline_arima = Pipeline140_TamanosCrecientes(
        n_boot=1000, seed=42, verbose=False, proceso_tipo='ARIMA'
    )
    pipeline_arima.TRAIN_SIZES = test_train_sizes
    pipeline_arima.CALIB_SIZES = test_calib_sizes
    pipeline_arima.ARIMA_CONFIGS = pipeline_arima.ARIMA_CONFIGS[:2]
    pipeline_arima.DISTRIBUTIONS = ['normal']
    pipeline_arima.VARIANCES = [1.0]
    
    results['ARIMA'] = pipeline_arima.run_all(
        excel_filename="resultados_TEST_TAMANOS_ARIMA.xlsx",
        batch_size=4
    )
    
    # SETAR
    print("\n" + "="*60)
    print("EJECUTANDO SETAR...")
    print("="*60)
    pipeline_setar = Pipeline140_TamanosCrecientes(
        n_boot=1000, seed=42, verbose=False, proceso_tipo='SETAR'
    )
    pipeline_setar.TRAIN_SIZES = test_train_sizes
    pipeline_setar.CALIB_SIZES = test_calib_sizes
    pipeline_setar.SETAR_CONFIGS = pipeline_setar.SETAR_CONFIGS[:2]
    pipeline_setar.DISTRIBUTIONS = ['normal']
    pipeline_setar.VARIANCES = [1.0]
    
    results['SETAR'] = pipeline_setar.run_all(
        excel_filename="resultados_TEST_TAMANOS_SETAR.xlsx",
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
        analisis_tamanos_crecientes(df)
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s")
    
    return results


def main_comparacion_todos_procesos(train_sizes=None, calib_sizes=None):
    """
    Ejecuta análisis completo para los 3 tipos de procesos y genera comparación.
    Args:
        train_sizes: Lista de tamaños de entrenamiento
        calib_sizes: Lista de tamaños de calibración
    """
    start_time = time.time()
    
    print("="*80)
    print("ANÁLISIS COMPARATIVO COMPLETO - ARMA vs ARIMA vs SETAR")
    print("="*80)
    
    results = {}
    
    # ARMA
    results['ARMA'] = main_tamanos_crecientes_ARMA(train_sizes, calib_sizes)
    
    # ARIMA
    results['ARIMA'] = main_tamanos_crecientes_ARIMA(train_sizes, calib_sizes)
    
    # SETAR
    results['SETAR'] = main_tamanos_crecientes_SETAR(train_sizes, calib_sizes)
    
    # Comparación final
    print("\n" + "="*80)
    print("📊 COMPARACIÓN FINAL: ARMA vs ARIMA vs SETAR")
    print("="*80)
    
    model_cols = ['AREPD', 'AV-MCPS', 'Block Bootstrapping', 'DeepAR', 
                  'EnCQR-LSTM', 'LSPM', 'LSPMW', 'MCPS', 'Sieve Bootstrap']
    
    for proceso_tipo, df in results.items():
        df_steps = df[df['Paso'] != 'Promedio'] if 'Paso' in df.columns else df
        
        print(f"\n{proceso_tipo}:")
        for model in model_cols:
            if model in df_steps.columns:
                val = df_steps[model].mean()
                if not pd.isna(val):
                    print(f"  {model:<25}: {val:.6f}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱  Tiempo total: {elapsed:.1f}s ({elapsed/3600:.2f} horas)")
    
    return results