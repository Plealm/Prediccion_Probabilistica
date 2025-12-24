# figuras.py (CORREGIDO)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")


class PlotManager:
    """Generador de gráficos consistente."""
    
    _STYLE = {
        'figsize': (14, 6),
        'grid_style': {'alpha': 0.3, 'linestyle': ':'},
        'default_colors': [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ],
        'font_scale': 1.3,
        'context': 'notebook'
    }

    @classmethod
    def _base_plot(cls, title: str, xlabel: str, ylabel: str, figsize=None):
        plt.figure(figsize=figsize or cls._STYLE['figsize'])
        sns.set_style("whitegrid")
        sns.set_context(cls._STYLE['context'], font_scale=cls._STYLE['font_scale'])
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, **cls._STYLE['grid_style'])

    @classmethod
    def plot_density_comparison(cls, distributions: Dict[str, np.ndarray],
                               true_values: Optional[List[float]] = None,
                               metric_values: Optional[Dict[str, float]] = None,
                               title: str = "Comparación de Distribuciones",
                               colors: Optional[Dict[str, str]] = None,
                               save_path: str = None):
        """Compara densidades predictivas con KDE."""
        cls._base_plot(title, "Valor", "Densidad")
        
        colors = colors or {}
        default_colors = cls._STYLE['default_colors']
        color_idx = 0
        metric_values = metric_values or {}
        
        x_min, x_max = np.inf, -np.inf
        for name, samples in distributions.items():
            if len(samples) < 10:
                continue
            kde = gaussian_kde(samples)
            x = np.linspace(samples.min(), samples.max(), 500)
            y = kde(x)
            color = colors.get(name, default_colors[color_idx % len(default_colors)])
            label = f"{name}"
            if name in metric_values:
                label += f" (ECRPS: {metric_values[name]:.4f})"
            plt.plot(x, y, label=label, color=color, linewidth=2.5)
            plt.fill_between(x, y, alpha=0.15, color=color)
            x_min, x_max = min(x_min, samples.min()), max(x_max, samples.max())
            color_idx += 1
        
        if true_values:
            for i, val in enumerate(true_values):
                plt.axvline(val, color='black', linestyle='-', linewidth=2,
                           label='Real' if i == 0 else None)
        
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
        """Boxplot + tabla resumen de ECRPS."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                       gridspec_kw={'width_ratios': [2.5, 1]})
        
        # CORRECCIÓN: usar 'Paso' como id_var si existe, sino crear índice
        if 'Paso' in df_results.columns:
            id_col = 'Paso'
        else:
            df_results = df_results.copy()
            df_results['_idx'] = range(len(df_results))
            id_col = '_idx'
        
        df_plot = df_results.melt(id_vars=[id_col], value_vars=model_names,
                                  var_name='Modelo', value_name='ECRPS')
        df_plot['Modelo'] = pd.Categorical(df_plot['Modelo'], 
                                           categories=model_names, ordered=True)
        
        sns.boxplot(data=df_plot, x='Modelo', y='ECRPS', ax=ax1, palette='tab10')
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
        """Evolución del ECRPS por horizonte."""
        cls._base_plot("Evolución ECRPS por Horizonte", "Paso", "ECRPS (media)")
        
        x = sorted(results_by_step.keys())
        for i, model in enumerate(model_names):
            y = [results_by_step[step][model].values[0] for step in x]
            color = cls._STYLE['default_colors'][i % len(cls._STYLE['default_colors'])]
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
        """Heatmap de ranking."""
        rank_df = df_results[model_names].rank(axis=1, ascending=True)
        mean_rank = rank_df.mean().sort_values()
        ordered = mean_rank.index.tolist()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(rank_df[ordered], annot=True, cmap="YlGnBu_r", 
                   fmt=".0f", cbar_kws={'label': 'Ranking'})
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.ylabel("Paso")
        plt.xlabel("Modelo")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()