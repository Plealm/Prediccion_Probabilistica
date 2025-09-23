# plot.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict

class PlotManager:
    """Clase de utilidad para generar los gráficos del análisis."""
    # Estilos actualizados según lo solicitado
    _STYLE = {'figsize': (14, 6), 'grid_style': {'alpha': 0.3, 'linestyle': ':'},
              'default_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#000000']}

    @classmethod
    def _base_plot(cls, title, xlabel, ylabel, figsize=None):
        """Crea la base para un gráfico estándar."""
        fig_size = figsize if figsize else cls._STYLE['figsize']
        plt.figure(figsize=fig_size)
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(**cls._STYLE['grid_style'])
        plt.gca().spines[['top', 'right']].set_visible(False)

    @classmethod
    def plot_series_split(cls, series: np.ndarray, burn_in_len: int, test_len: int):
        """Grafica la serie temporal mostrando las divisiones de burn-in, train y test."""
        cls._base_plot("Serie Temporal Simulada con Divisiones", "Tiempo", "Valor")
        
        train_len = len(series) - burn_in_len - test_len
        
        plt.plot(series, label='Serie Completa', color=cls._STYLE['default_colors'][0])
        
        plt.axvspan(0, burn_in_len, color='gray', alpha=0.3, label=f'Burn-in ({burn_in_len} puntos)')
        plt.axvspan(burn_in_len, burn_in_len + train_len, color='green', alpha=0.2, label=f'Entrenamiento Inicial ({train_len} puntos)')
        plt.axvspan(burn_in_len + train_len, len(series), color='red', alpha=0.2, label=f'Test (Ventana Rodante) ({test_len} puntos)')
        
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_density_comparison(cls, distributions: Dict[str, np.ndarray], metric_values: Dict[str, float], title: str, colors: Dict[str, str] = None):
        """
        [NUEVA FUNCIÓN]
        Compara las densidades de las distribuciones predictivas para un único paso de tiempo.
        """
        cls._base_plot(title, "Valor", "Densidad")
        
        if colors is None:
            colors = {name: cls._STYLE['default_colors'][i % len(cls._STYLE['default_colors'])] for i, name in enumerate(distributions.keys())}

        # Dibuja cada distribución
        for name, data in distributions.items():
            color = colors.get(name, '#333333') 
            
            # Estilo diferencial para la distribución teórica (real)
            linestyle = '-' if name == 'Teórica' else '--'
            linewidth = 3.0 if name == 'Teórica' else 2.0

            clean_data = data[np.isfinite(data)]
            if len(clean_data) > 1 and np.std(clean_data) > 1e-9:
                sns.kdeplot(clean_data, color=color, label=name, linestyle=linestyle, linewidth=linewidth, warn_singular=False)
            else: # Maneja el caso de predicciones puntuales
                point_prediction = np.mean(clean_data)
                plt.axvline(point_prediction, color=color, linestyle=linestyle, linewidth=linewidth, label=f'{name} (Puntual)')

        # Ordena las métricas y las muestra en una caja de texto
        sorted_metrics = sorted(metric_values.items(), key=lambda x: x[1])
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in sorted_metrics])
        
        plt.text(0.98, 0.98, f'ECRPS vs Teórica:\n{metrics_text}', transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        
        plt.legend(loc='upper left', frameon=True)
        plt.tight_layout()
        plt.show()