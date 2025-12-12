
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import seaborn as sns
import pandas as pd
import os

def visualize_predictions(pipeline, series_index=0, output_prefix="dataset", output_folder="Resultados"):
    """
    Genera visualizaciones verticales detalladas guardadas en la carpeta especificada.
    
    Args:
        pipeline: Instancia de PipelineElectricity o PipelineTraffic.
        series_index: Índice de la serie a evaluar.
        output_prefix: Prefijo para los archivos (ej. 'traffic' o 'electricity').
        output_folder: Carpeta donde se guardarán las imágenes y el Excel.
    """
    print(f"\n🎨 Generando visualizaciones detalladas para la serie {series_index}...")

    # 1. Crear carpeta de Resultados Dinámica
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"  📂 Carpeta '{output_folder}' creada.")
    
    # 2. Ejecutar Pipeline
    df_results, predictions_dict = pipeline.run_evaluation(
        series_index=series_index,
        save_predictions=True
    )
    
    if predictions_dict is None:
        print("❌ Error: No se obtuvieron predicciones.")
        return

    # 3. Guardar Excel en carpeta Resultados
    excel_name = os.path.join(output_folder, f"{output_prefix}_resultados.xlsx")
    df_results.to_excel(excel_name, index=False)
    print(f"  💾 Resultados guardados en: {excel_name}")

    # -------------------------------------------------------------------------
    # 4. IDENTIFICAR RANKING DE MODELOS
    # -------------------------------------------------------------------------
    model_names = [c for c in df_results.columns if c not in ['Paso', 'Valor_Observado', 'timestamp']]
    
    mean_scores = df_results[model_names].mean().sort_values()
    sorted_models = mean_scores.index.tolist()
    
    top_3_best = sorted_models[:3]
    top_3_worst = sorted_models[-3:]
    
    top_2_best = sorted_models[:2]
    absolute_worst = sorted_models[-1]
    
    print(f"  🏆 Mejores 3: {', '.join(top_3_best)}")
    print(f"  💀 Peores 3:  {', '.join(top_3_worst)}")

    steps = sorted(predictions_dict.keys())

    # -------------------------------------------------------------------------
    # 5. FUNCIÓN GENERADORA DE IMÁGENES VERTICALES
    # -------------------------------------------------------------------------
    def generate_vertical_plot(models_to_plot, filename_suffix, palette=None):
        n_steps = len(steps)
        fig, axes = plt.subplots(nrows=n_steps, ncols=1, figsize=(10, 4 * n_steps), constrained_layout=True)
        
        if n_steps == 1: axes = [axes]

        for i, t in enumerate(steps):
            ax = axes[i]
            data = predictions_dict[t]
            timestamp = data['timestamp']
            true_val = data['true_value']
            step_num = t + 1
            
            row_res = df_results.loc[df_results['Paso'] == step_num].iloc[0]

            plot_df_list = []
            crps_text = "📊 CRPS:\n"
            
            for m in models_to_plot:
                preds = data['predictions'].get(m, [])
                preds = preds[np.isfinite(preds)] 
                
                if len(preds) > 0:
                    plot_df_list.append(pd.DataFrame({'Valor': preds, 'Modelo': m}))
                    val_crps = row_res[m]
                    name_display = (m[:12] + '..') if len(m) > 12 else m
                    crps_text += f"{name_display:<14}: {val_crps:.3f}\n"
                else:
                    crps_text += f"{m:<14}: NaN\n"
            
            date_str = timestamp.strftime('%Y-%m-%d %H:%M')
            ax.set_title(f"{date_str} | Paso {step_num}/{n_steps}", 
                         fontsize=14, fontweight='bold', loc='center', color='#333333')
            
            if plot_df_list:
                full_df = pd.concat(plot_df_list)
                sns.kdeplot(
                    data=full_df, x='Valor', hue='Modelo',
                    fill=True, alpha=0.25, linewidth=2.5,
                    ax=ax, palette=palette, common_norm=False,
                    legend=True
                )
                ax.axvline(x=true_val, color='black', linestyle='--', linewidth=2.5, label='Valor Real')
                
                all_vals = full_df['Valor'].values
                min_x = min(np.min(all_vals), true_val)
                max_x = max(np.max(all_vals), true_val)
                margin = (max_x - min_x) * 0.25
                ax.set_xlim(min_x - margin, max_x + margin)
            
            ax.set_ylabel("Densidad")
            ax.set_xlabel("Valor (Traffic)")
            ax.grid(True, alpha=0.3, linestyle=':')

            at = AnchoredText(
                crps_text.strip(),
                prop=dict(size=10, family='monospace'), 
                frameon=True, loc='upper right'
            )
            at.patch.set_boxstyle("round,pad=0.5,rounding_size=0.2")
            at.patch.set_facecolor("#f8f9fa")
            at.patch.set_edgecolor("gray")
            at.patch.set_alpha(0.95)
            ax.add_artist(at)

            ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9, shadow=True)

        fname = os.path.join(output_folder, f"{output_prefix}_{filename_suffix}.png")
        plt.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Guardado: {fname}")

    # -------------------------------------------------------------------------
    # 6. GENERACIÓN DE IMÁGENES
    # -------------------------------------------------------------------------
    print("\n  📸 Generando imágenes individuales...")
    for model in model_names:
        generate_vertical_plot([model], f"modelo_{model}")

    print("\n  📸 Generando comparativas...")
    
    # Top 3 Mejores
    palette_best = {top_3_best[0]: '#2ca02c', top_3_best[1]: '#1f77b4', top_3_best[2]: '#9467bd'} if len(top_3_best) >= 3 else None
    generate_vertical_plot(top_3_best, "COMPARACION_TOP3_MEJORES", palette=palette_best)
    
    # Top 3 Peores
    palette_worst = {top_3_worst[-1]: '#800000', top_3_worst[-2]: '#d62728', top_3_worst[-3]: '#ff7f0e'} if len(top_3_worst) >= 3 else None
    generate_vertical_plot(top_3_worst, "COMPARACION_TOP3_PEORES", palette=palette_worst)
    
    # Top 2 vs Peor
    palette_vs = {top_2_best[0]: '#2ca02c', top_2_best[1]: '#1f77b4', absolute_worst: '#d62728'}
    generate_vertical_plot(top_2_best + [absolute_worst], "COMPARACION_TOP2_VS_PEOR", palette=palette_vs)
    
    print(f"\n✨ Proceso finalizado. Revisa la carpeta '{output_folder}'.")