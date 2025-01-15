import matplotlib.pyplot as plt
import numpy as np
from .base import BaseFigure

class ReturnsFigures(BaseFigure):
    def plot_E(self, ax):
        num_columns = self.atsm.E.shape[1]
        colors = plt.cm.Blues(np.linspace(0.2, 1, num_columns))
        for i, column in enumerate(self.atsm.E.columns):
            ax.plot(self.atsm.E[column], color=colors[i])
        self.set_title(ax, 'E')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        self.create_colorbar(ax, num_columns)
        return ax

    def plot_E_standalone(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_E(ax)
        self.remove_frame(ax)
        self.add_grid(ax)
        plt.tight_layout()
        return fig

    def plot_excess_returns(self, ax):
        num_columns = self.atsm.xr.shape[1]
        colors = plt.cm.Blues(np.linspace(0.2, 1, num_columns))
        for i, column in enumerate(self.atsm.xr.columns):
            ax.plot(self.atsm.xr[column], color=colors[i])
        self.set_title(ax, 'Excess Returns (xr)')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        self.create_colorbar(ax, num_columns)
        return ax

    def plot_excess_returns_standalone(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_excess_returns(ax)
        self.remove_frame(ax)
        self.add_grid(ax)
        plt.tight_layout()
        return fig 