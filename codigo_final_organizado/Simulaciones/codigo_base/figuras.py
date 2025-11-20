# figuras.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

class PlotManager:
    """Clase para generar todos los gráficos del análisis de forma consistente"""
    
    _STYLE = {
        'figsize': (14, 6),
        'grid_style': {'alpha': 0.3, 'linestyle': ':'},
        'default_colors': [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
        ],
        'font_scale': 1.3,
        'context': 'notebook'
    }

    @classmethod
    def _base_plot(cls, title: str, xlabel: str, ylabel: str, figsize=None):
        """Configura figura base con estilo consistente"""
        plt.figure(figsize=figsize or cls._STYLE['figsize'])
        sns.set_style("whitegrid")
        sns.set_context(cls._STYLE['context'], font_scale=cls._STYLE['font_scale'])
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, **cls._STYLE['grid_style'])

    @classmethod
    def plot_series_split(cls, series: np.ndarray, burn_in_len: int, train_len: int, test_len: int,
                          save_path: str = None):
        """Grafica serie temporal con divisiones burn-in / train / test"""
        cls._base_plot("Serie Temporal Generada - División de Datos", "Tiempo", "Valor")
        
        t = np.arange(len(series))
        plt.plot(t[:burn_in_len], series[:burn_in_len], label='Burn-in', color='gray', alpha=0.6)
        plt.plot(t[burn_in_len:burn_in_len + train_len], 
                 series[burn_in_len:burn_in_len + train_len], 
                 label='Entrenamiento', color='blue')
        plt.plot(t[burn_in_len + train_len:burn_in_len + train_len + test_len],
                 series[burn_in_len + train_len:burn_in_len + train_len + test_len],
                 label='Test (5 pasos)', color='red', linewidth=2)
        
        plt.axvline(burn_in_len, color='black', linestyle='--', alpha=0.7)
        plt.axvline(burn_in_len + train_len, color='black', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @classmethod
    def plot_density_comparison(cls, distributions: Dict[str, np.ndarray],
                               true_values: Optional[List[float]] = None,
                               metric_values: Optional[Dict[str, float]] = None,
                               title: str = "Comparación de Distribuciones Predictivas (Paso 1)",
                               colors: Optional[Dict[str, str]] = None,
                               save_path: str = None):
        """Compara densidades predictivas con KDE + valores reales"""
        cls._base_plot(title, "Valor", "Densidad")
        
        colors = colors or {}
        default_colors = cls._STYLE['default_colors']
        color_idx = 0
        
        x_min, x_max = np.inf, -np.inf
        for name, samples in distributions.items():
            if len(samples) < 10:
                continue
            kde = gaussian_kde(samples)
            x = np.linspace(samples.min(), samples.max(), 500)
            y = kde(x)
            color = colors.get(name, default_colors[color_idx % len(default_colors)])
            plt.plot(x, y, label=f"{name} (ECRPS: {metric_values.get(name, np.nan):.4f})", 
                    color=color, linewidth=2.5)
            plt.fill_between(x, y, alpha=0.15, color=color)
            x_min, x_max = min(x_min, samples.min()), max(x_max, samples.max())
            color_idx += 1
        
        # Líneas verticales para valores reales
        if true_values:
            for i, val in enumerate(true_values):
                plt.axvline(val, color='black', linestyle='-', linewidth=2,
                           label='Real' if i == 0 else None)
        
        plt.xlim(x_min * 0.95, x_max * 1.05)
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
        """Boxplot + tabla resumen de ECRPS por modelo"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [3, 1]})
        
        # Boxplot
        df_plot = df_results.melt(id_vars=['scenario'], var_name='Modelo', value_name='ECRPS')
        df_plot = df_plot[df_plot['Modelo'].isin(model_names)]
        df_plot['Modelo'] = pd.Categorical(df_plot['Modelo'], categories=model_names, ordered=True)
        
        sns.boxplot(data=df_plot, x='Modelo', y='ECRPS', ax=ax1, palette='tab10')
        ax1.set_title(f"Distribución de ECRPS - {step_name}", fontsize=16, fontweight='bold')
        ax1.set_ylabel("ECRPS", fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        
        # Tabla resumen
        summary = df_results[model_names].agg(['mean', 'std', 'min', 'max']).round(4)
        summary = summary.T
        summary = summary.sort_values('mean')
        summary['rank'] = np.arange(1, len(summary) + 1)
        
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(cellText=summary.values,
                         colLabels=['Media', 'Std', 'Mín', 'Máx', 'Rank'],
                         rowLabels=summary.index,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        ax2.set_title("Ranking ECRPS (menor = mejor)", fontsize=14, pad=20)
        
        plt.suptitle(f"Resumen de Resultados - {step_name}", fontsize=18, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return summary

    @classmethod
    def plot_ecrps_evolution(cls, results_by_step: Dict[int, pd.DataFrame],
                            model_names: List[str],
                            save_path: str = None):
        """Evolución del ECRPS a lo largo de los 5 pasos de predicción"""
        cls._base_plot("Evolución del ECRPS por Horizonte de Predicción", 
                      "Horizonte (paso)", "ECRPS (media)")
        
        means = {}
        stds = {}
        for step in range(1, 6):
            if step in results_by_step:
                df = results_by_step[step][model_names]
                means[step] = df.mean()
                stds[step] = df.std()
        
        x = sorted(means.keys())
        for model in model_names:
            y = [means[step][model] for step in x]
            e = [stds[step][model] for step in x]
            color = cls._STYLE['default_colors'][model_names.index(model) % len(cls._STYLE['default_colors'])]
            plt.plot(x, y, marker='o', label=model, linewidth=2.5, color=color)
            plt.fill_between(x, np.array(y)-np.array(e), np.array(y)+np.array(e), alpha=0.2, color=color)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(x)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @classmethod
    def plot_ranking_heatmap(cls, df_results: pd.DataFrame,
                            model_names: List[str],
                            title: str = "Ranking por Escenario",
                            save_path: str = None):
        """Heatmap de ranking por escenario"""
        rank_df = df_results[model_names].rank(axis=1, ascending=True)
        mean_rank = rank_df.mean().sort_values()
        ordered_models = mean_rank.index.tolist()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(rank_df[ordered_models], annot=True, cmap="YlGnBu_r", fmt=".0f",
                   cbar_kws={'label': 'Ranking (1 = mejor)'})
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel("Escenario")
        plt.xlabel("Modelo")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()