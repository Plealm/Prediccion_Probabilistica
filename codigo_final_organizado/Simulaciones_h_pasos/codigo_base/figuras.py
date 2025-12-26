import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy.stats import gaussian_kde
from matplotlib.offsetbox import AnchoredText
import warnings
warnings.filterwarnings("ignore")


class PlotManager:
    """Generador de gráficos para análisis de trayectorias estocásticas."""
    
    _STYLE = {
        'figsize': (14, 6),
        'grid_style': {'alpha': 0.3, 'linestyle': ':'},
        'default_colors': {
            'LSPM': '#1f77b4',
            'DeepAR': '#ff7f0e',
            'Sieve Bootstrap': '#2ca02c',
            'MCPS': '#d62728',
            'Teórica': '#9467bd'
        },
        'font_scale': 1.2,
        'context': 'notebook'
    }

    @classmethod
    def _base_plot(cls, title: str, xlabel: str, ylabel: str, figsize=None):
        """Configuración base para todos los gráficos."""
        plt.figure(figsize=figsize or cls._STYLE['figsize'])
        sns.set_style("whitegrid")
        sns.set_context(cls._STYLE['context'], font_scale=cls._STYLE['font_scale'])
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, **cls._STYLE['grid_style'])

    @classmethod
    def plot_scenario_densities(cls, scenario_name: str, predictions_dict: dict,
                                df_results: pd.DataFrame, model_names: List[str],
                                save_path: str = None):
        """
        Genera gráfico vertical con densidades paso a paso.
        Similar a plot_type_a_vertical_stack: compara distribuciones de modelos
        contra la distribución teórica.
        
        Args:
            scenario_name: Nombre identificador del escenario
            predictions_dict: Dict con estructura {step_idx: {'true_distribution': array,
                                                               'model_predictions': dict}}
            df_results: DataFrame con ECRPS por paso
            model_names: Lista de nombres de modelos a graficar
            save_path: Ruta para guardar el gráfico
        """
        steps = sorted(predictions_dict.keys())
        n_steps = len(steps)
        
        # Altura dinámica: 2.5 pulgadas por paso
        fig, axes = plt.subplots(nrows=n_steps, ncols=1, 
                                figsize=(12, 2.5 * n_steps), 
                                constrained_layout=True)
        
        if n_steps == 1: 
            axes = [axes]
        
        fig.suptitle(f"Densidades Predictivas vs Teórica - {scenario_name}", 
                    fontsize=20, fontweight='bold', y=1.002)

        for i, step_idx in enumerate(steps):
            ax = axes[i]
            step_num = step_idx + 1
            data = predictions_dict[step_idx]
            
            true_dist = data['true_distribution']  # 1000 muestras teóricas
            model_preds = data['model_predictions']  # Dict con 100 muestras por modelo
            
            # Filtrar resultados para este paso
            step_results = df_results[df_results['Paso_H'] == step_num]
            if len(step_results) == 0:
                continue
            step_results = step_results.iloc[0]
            
            # Determinar límites X comunes para todas las distribuciones
            vals_step = list(true_dist)
            for model_name in model_names:
                if model_name in model_preds:
                    vals_step.extend(model_preds[model_name])
            
            if len(vals_step) > 1:
                mn, mx = np.min(vals_step), np.max(vals_step)
                margin = (mx - mn) * 0.2
                ax.set_xlim(mn - margin, mx + margin)

            # 1. DISTRIBUCIÓN TEÓRICA (más prominente)
            true_dist_clean = true_dist[np.isfinite(true_dist)]
            if len(true_dist_clean) > 1 and np.var(true_dist_clean) > 0:
                try:
                    sns.kdeplot(true_dist_clean, fill=True, 
                               color=cls._STYLE['default_colors']['Teórica'], 
                               alpha=0.3, ax=ax, linewidth=3, 
                               label='Teórica (1000 traj.)')
                except:
                    ax.hist(true_dist_clean, bins=50, density=True, 
                           color=cls._STYLE['default_colors']['Teórica'], 
                           alpha=0.3, label='Teórica')

            # 2. DISTRIBUCIONES DE MODELOS
            legend_labels = []
            for model_name in model_names:
                if model_name not in model_preds:
                    continue
                    
                samples = np.array(model_preds[model_name])
                samples = samples[np.isfinite(samples)]
                
                if len(samples) < 2:
                    continue
                
                color = cls._STYLE['default_colors'].get(model_name, 'gray')
                ecrps_val = step_results[model_name]
                ecrps_txt = f"{ecrps_val:.4f}" if pd.notna(ecrps_val) else "NaN"
                
                # Graficar KDE o histograma
                if np.var(samples) > 0:
                    try:
                        sns.kdeplot(samples, fill=True, color=color, 
                                   alpha=0.15, ax=ax, linewidth=2)
                    except:
                        ax.hist(samples, bins=30, density=True, 
                               color=color, alpha=0.15)
                
                legend_labels.append(f"{model_name}: {ecrps_txt}")

            # Media de la distribución teórica
            true_mean = np.mean(true_dist_clean)
            ax.axvline(true_mean, color='black', linestyle='--', 
                      linewidth=2, alpha=0.7)
            
            # Títulos y etiquetas
            ax.set_title(f"Paso {step_num} | Media Teórica: {true_mean:.2f}", 
                        fontsize=12, fontweight='bold', loc='right')
            ax.set_ylabel("Densidad", fontsize=10)
            ax.set_xlabel("Valor", fontsize=10)
            ax.grid(True, alpha=0.3, linestyle=':')
            
            # Recuadro con ECRPS (superior izquierdo)
            full_txt = "ECRPS:\n" + "\n".join(legend_labels)
            at = AnchoredText(full_txt, prop=dict(size=9), 
                            frameon=True, loc='upper left')
            at.patch.set_boxstyle("round,pad=0.3")
            at.patch.set_alpha(0.85)
            ax.add_artist(at)
            
            # Leyenda de la teórica (superior derecho)
            ax.legend(loc='upper right', fontsize=9, framealpha=0.85)

        # Guardar figura
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Gráfico guardado: {save_path}")
        plt.close()

    @classmethod
    def plot_density_comparison(cls, distributions: Dict[str, np.ndarray],
                               true_distribution: Optional[np.ndarray] = None,
                               metric_values: Optional[Dict[str, float]] = None,
                               title: str = "Comparación de Distribuciones",
                               save_path: str = None):
        """
        Compara densidades predictivas con KDE en un solo gráfico.
        Útil para comparar un paso específico.
        
        Args:
            distributions: Dict {model_name: array de muestras}
            true_distribution: Array con distribución teórica (opcional)
            metric_values: Dict {model_name: ecrps_value}
            title: Título del gráfico
            save_path: Ruta para guardar
        """
        cls._base_plot(title, "Valor", "Densidad")
        
        metric_values = metric_values or {}
        
        x_min, x_max = np.inf, -np.inf
        
        # Graficar distribución teórica si está disponible
        if true_distribution is not None:
            true_clean = true_distribution[np.isfinite(true_distribution)]
            if len(true_clean) > 10 and np.var(true_clean) > 0:
                kde = gaussian_kde(true_clean)
                x = np.linspace(true_clean.min(), true_clean.max(), 500)
                y = kde(x)
                plt.plot(x, y, label='Teórica', 
                        color=cls._STYLE['default_colors']['Teórica'], 
                        linewidth=3, linestyle='--')
                plt.fill_between(x, y, alpha=0.2, 
                               color=cls._STYLE['default_colors']['Teórica'])
                x_min = min(x_min, true_clean.min())
                x_max = max(x_max, true_clean.max())
        
        # Graficar distribuciones de modelos
        for name, samples in distributions.items():
            samples = samples[np.isfinite(samples)]
            if len(samples) < 10:
                continue
                
            kde = gaussian_kde(samples)
            x = np.linspace(samples.min(), samples.max(), 500)
            y = kde(x)
            
            color = cls._STYLE['default_colors'].get(name, '#333333')
            label = f"{name}"
            if name in metric_values:
                label += f" (ECRPS: {metric_values[name]:.4f})"
            
            plt.plot(x, y, label=label, color=color, linewidth=2.5)
            plt.fill_between(x, y, alpha=0.15, color=color)
            
            x_min = min(x_min, samples.min())
            x_max = max(x_max, samples.max())
        
        plt.xlim(x_min * 0.95 if x_min > 0 else x_min * 1.05, 
                 x_max * 1.05 if x_max > 0 else x_max * 0.95)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @classmethod
    def plot_results_summary(cls, df_results: pd.DataFrame,
                            model_names: List[str],
                            step_name: str = "Todos los pasos",
                            save_path: str = None):
        """
        Boxplot + tabla resumen de ECRPS.
        
        Args:
            df_results: DataFrame con resultados
            model_names: Lista de modelos
            step_name: Nombre descriptivo del análisis
            save_path: Ruta para guardar
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                       gridspec_kw={'width_ratios': [2.5, 1]})
        
        # Preparar datos para boxplot
        if 'Paso_H' in df_results.columns:
            id_col = 'Paso_H'
        else:
            df_results = df_results.copy()
            df_results['_idx'] = range(len(df_results))
            id_col = '_idx'
        
        df_plot = df_results.melt(id_vars=[id_col], value_vars=model_names,
                                  var_name='Modelo', value_name='ECRPS')
        df_plot['Modelo'] = pd.Categorical(df_plot['Modelo'], 
                                           categories=model_names, ordered=True)
        
        # Boxplot
        colors = [cls._STYLE['default_colors'].get(m, '#333333') for m in model_names]
        sns.boxplot(data=df_plot, x='Modelo', y='ECRPS', ax=ax1, palette=colors)
        ax1.set_title(f"Distribución ECRPS - {step_name}", fontsize=14, fontweight='bold')
        ax1.set_ylabel("ECRPS", fontsize=12)
        ax1.tick_params(axis='x', rotation=30)
        ax1.grid(True, alpha=0.3)
        
        # Tabla resumen
        summary = df_results[model_names].agg(['mean', 'std', 'min', 'max']).round(4).T
        summary = summary.sort_values('mean')
        summary['rank'] = range(1, len(summary) + 1)
        
        ax2.axis('off')
        table = ax2.table(cellText=summary.values,
                         colLabels=['Media', 'Std', 'Mín', 'Máx', 'Rank'],
                         rowLabels=summary.index,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Colorear celdas por ranking
        for i, model in enumerate(summary.index):
            color = cls._STYLE['default_colors'].get(model, '#ffffff')
            table[(i+1, -1)].set_facecolor(color)
            table[(i+1, -1)].set_alpha(0.3)
        
        ax2.set_title("Ranking (menor=mejor)", fontsize=12, pad=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return summary

    @classmethod
    def plot_ecrps_evolution(cls, results_by_step: Dict[int, pd.DataFrame],
                            model_names: List[str],
                            save_path: str = None):
        """
        Evolución del ECRPS por horizonte de predicción.
        
        Args:
            results_by_step: Dict {step: DataFrame con media de ECRPS por modelo}
            model_names: Lista de modelos
            save_path: Ruta para guardar
        """
        cls._base_plot("Evolución ECRPS por Horizonte", 
                      "Paso de Predicción", "ECRPS (media)")
        
        x = sorted(results_by_step.keys())
        for model in model_names:
            y = [results_by_step[step][model].values[0] for step in x]
            color = cls._STYLE['default_colors'].get(model, '#333333')
            plt.plot(x, y, marker='o', label=model, linewidth=2.5, 
                    color=color, markersize=8)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(x)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @classmethod
    def plot_ranking_heatmap(cls, df_results: pd.DataFrame,
                            model_names: List[str],
                            title: str = "Ranking por Paso",
                            save_path: str = None):
        """
        Heatmap de ranking por paso.
        
        Args:
            df_results: DataFrame con ECRPS por modelo
            model_names: Lista de modelos
            title: Título del gráfico
            save_path: Ruta para guardar
        """
        rank_df = df_results[model_names].rank(axis=1, ascending=True)
        mean_rank = rank_df.mean().sort_values()
        ordered = mean_rank.index.tolist()
        
        plt.figure(figsize=(10, max(6, len(rank_df) * 0.3)))
        sns.heatmap(rank_df[ordered], annot=True, cmap="RdYlGn_r", 
                   fmt=".0f", cbar_kws={'label': 'Ranking'},
                   linewidths=0.5, linecolor='white')
        
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.ylabel("Paso")
        plt.xlabel("Modelo")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @classmethod
    def plot_variance_sensitivity(cls, df_results: pd.DataFrame,
                                  model_names: List[str],
                                  save_path: str = None):
        """
        Análisis de sensibilidad a la varianza.
        
        Args:
            df_results: DataFrame con columna 'Varianza'
            model_names: Lista de modelos
            save_path: Ruta para guardar
        """
        if 'Varianza' not in df_results.columns:
            print("⚠️ No hay columna 'Varianza' en los resultados")
            return
        
        cls._base_plot("Sensibilidad a la Varianza", 
                      "Varianza del Ruido", "ECRPS (media)")
        
        variances = sorted(df_results['Varianza'].unique())
        
        for model in model_names:
            means = [df_results[df_results['Varianza']==v][model].mean() 
                    for v in variances]
            color = cls._STYLE['default_colors'].get(model, '#333333')
            plt.plot(variances, means, marker='o', label=model, 
                    linewidth=2.5, color=color, markersize=8)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(variances)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()