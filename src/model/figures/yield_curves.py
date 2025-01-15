import matplotlib.pyplot as plt
import numpy as np
from .base import BaseFigure

class YieldCurveFigures(BaseFigure):
    def plot_yield_curve_vs_fitted(self, ax_short, ax_10y):
        common_index = self.atsm.yield_curve["short"].index.intersection(self.atsm.yield_curve_fitted.index)
        if self.model == "ff":
            common_index = common_index.intersection(self.atsm.macro_data["long"]["rf_LR"].index)
            
        # Plot short rate
        ax_short.plot(self.atsm.yield_curve["short"]\
                     .loc[common_index, str(self.atsm.start_period)], 
                     label='Yield Curve at start_period',
                     color='royalblue')
        ax_short.plot(self.atsm.yield_curve_fitted\
                     .loc[common_index, str(self.atsm.start_period)],  
                     label='Yield Curve Fitted at start_period',
                     color='lightskyblue',
                     marker='o',
                     markersize=5,
                     linestyle='dashed')
        if self.model == "ff":
            ax_short.plot(self.atsm.macro_data["long"]["rf_LR"]\
                         .loc[common_index], 
                         label='Long-run RF')
        self.set_title(ax_short, 'Short Rate')
        ax_short.legend()

        # Plot 10-year rate
        ax_10y.plot(self.atsm.yield_curve["short"]\
                   .loc[common_index, str(10*self.atsm.freq_factor)], 
                   label='Yield Curve at 10*freq_factor',
                   color='royalblue')
        ax_10y.plot(self.atsm.yield_curve_fitted\
                   .loc[common_index, str(10*self.atsm.freq_factor)],  
                   label='Yield Curve Fitted at 10*freq_factor',
                   color='lightskyblue',
                   marker='x',
                   markersize=5,
                   linestyle='dashed')
        if self.model == "ff":
            ax_10y.plot(self.atsm.macro_data["long"]["rf_LR"]\
                       .loc[common_index], 
                       label='Long-run RF',
                       linestyle='-.')
        self.set_title(ax_10y, '10 years')
        ax_10y.legend()

        return ax_short, ax_10y

    def plot_yield_curve_vs_fitted_standalone(self):
        fig, (ax_short, ax_10y) = plt.subplots(1, 2, figsize=(20, 6))
        self.plot_yield_curve_vs_fitted(ax_short, ax_10y)
        self.remove_frame(ax_short)
        self.remove_frame(ax_10y)
        self.add_grid(ax_short)
        self.add_grid(ax_10y)
        self.set_legend(ax_short)
        self.set_legend(ax_10y)
        plt.tight_layout()
        return fig

    def plot_full_yield_curve(self, ax):
        num_columns = self.atsm.yield_curve_fitted.shape[1]
        colors = plt.cm.Blues(np.linspace(0.2, 1, num_columns))
        for i, column in enumerate(self.atsm.yield_curve_fitted.columns):
            ax.plot(self.atsm.yield_curve_fitted[column], color=colors[i])
        self.set_title(ax, 'Fitted Yield Curve')
        ax.set_ylabel('Yield')
        ax.set_xlabel('')
        self.create_colorbar(ax, num_columns, label='Maturity')
        return ax

    def plot_full_yield_curve_standalone(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_full_yield_curve(ax)
        self.remove_frame(ax)
        self.add_grid(ax)
        plt.tight_layout()
        return fig

    def plot_yield_curve_cycle(self, ax):
        if self.model != "ff":
            return ax
            
        num_columns = self.atsm.yield_curve_cycle.shape[1]
        colors = plt.cm.Blues(np.linspace(0.2, 1, num_columns))
        for i, column in enumerate(self.atsm.yield_curve_cycle.columns):
            ax.plot(self.atsm.yield_curve_cycle[column], color=colors[i])
        self.set_title(ax, 'Yield Curve Cycle')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        self.create_colorbar(ax, num_columns, label='Maturity')
        return ax

    def plot_yield_curve_cycle_standalone(self):
        if self.model != "ff":
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_yield_curve_cycle(ax)
        self.remove_frame(ax)
        self.add_grid(ax)
        plt.tight_layout()
        return fig 