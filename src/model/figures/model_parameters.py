import matplotlib.pyplot as plt
import numpy as np
from .base import BaseFigure

class ModelParametersFigures(BaseFigure):
    def plot_A(self, ax):
        ax.plot(self.atsm.A)
        self.set_title(ax, 'A')
        return ax

    def plot_A_standalone(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_A(ax)
        self.remove_frame(ax)
        self.add_grid(ax)
        plt.tight_layout()
        return fig

    def plot_B(self, ax):
        num_columns = self.atsm.B.T.shape[1]
        blues = plt.cm.Blues(np.linspace(0.3, 1, num_columns))
        for i, col in enumerate(range(num_columns)):
            ax.plot(self.atsm.B.T[:, col], color=blues[i], label=f'$B_{{{col+1}}}$')
        self.set_title(ax, 'B')
        ax.legend([str(col + 1) for col in range(num_columns)])
        return ax

    def plot_B_standalone(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_B(ax)
        self.remove_frame(ax)
        self.add_grid(ax)
        self.set_legend(ax)
        plt.tight_layout()
        return fig

    def plot_X(self, ax):
        num_columns = self.atsm.X.shape[1]
        blues = plt.cm.Blues(np.linspace(1, 0.3, num_columns))
        for i, column in enumerate(self.atsm.X.columns):
            ax.plot(self.atsm.X[column], color=blues[i], label=f'$X_{{{i+1},t}}$')
        self.set_title(ax, 'X')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        legend = ax.legend(loc='upper right')
        plt.setp(legend.get_texts(), fontsize=self.labelsize)
        ax.tick_params(labelsize=self.labelsize)
        return ax

    def plot_X_standalone(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_X(ax)
        self.remove_frame(ax)
        self.add_grid(ax)
        plt.tight_layout()
        return fig

    def plot_v(self, ax):
        num_columns = self.atsm.v.shape[1]
        blues = plt.cm.Blues(np.linspace(1, 0.3, num_columns))
        for i, column in enumerate(self.atsm.v.columns):
            ax.plot(self.atsm.v[column], color=blues[i], label=f'$v_{{{i+1},t}}$')
        self.set_title(ax, 'v')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        legend = ax.legend(loc='upper right')
        plt.setp(legend.get_texts(), fontsize=self.labelsize)
        ax.tick_params(labelsize=self.labelsize)
        return ax

    def plot_v_standalone(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_v(ax)
        self.remove_frame(ax)
        self.add_grid(ax)
        plt.tight_layout()
        return fig 