import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import rc

# Set up LaTeX font for all text
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

class Figures:
    def __init__(self, atsm, model):
        self.atsm = atsm
        self.model = model
        self.selected_columns = [0, 
                                 self.atsm.freq_factor - 1, 
                                 5 * self.atsm.freq_factor - 1, 
                                 10 * self.atsm.freq_factor - 1]
        self.total_plots = 10  # A, B, X, v, E, xr, Short Rate, 10 Years, Full Yield Curve, Yield Curve Cycle
        self.rows = 5
        self.cols = 2
        self.fontsize = 24  # Default fontsize for titles
        self.labelsize = 22  # Default fontsize for legend labels
        self.grid_linewidth = 0.75# Default linewidth for grid

    def create_figure(self):
        return plt.figure(figsize=(20, 6 * self.rows), dpi=40)

    def remove_frame(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def set_title(self, ax, title):
        ax.set_title(title, fontsize=self.fontsize)

    def add_grid(self, ax):
        ax.grid(True, linewidth=self.grid_linewidth)

    def set_legend(self, ax):
        if ax.get_legend():
            ax.legend(fontsize=self.labelsize)

    @property
    def results(self):
        fig = self.create_figure()
        
        # A plot
        ax_a = fig.add_subplot(self.rows, self.cols, 1)
        self.A(ax_a)
        self.remove_frame(ax_a)
        self.add_grid(ax_a)
        
        # B plot
        ax_b = fig.add_subplot(self.rows, self.cols, 2)
        self.B(ax_b)
        self.remove_frame(ax_b)
        self.add_grid(ax_b)
        self.set_legend(ax_b)
        
        # X plot
        ax_x = fig.add_subplot(self.rows, self.cols, 3)
        self.X(ax_x)
        self.remove_frame(ax_x)
        self.add_grid(ax_x)
        
        # v plot
        ax_v = fig.add_subplot(self.rows, self.cols, 4)
        self.v(ax_v)
        self.remove_frame(ax_v)
        self.add_grid(ax_v)
        
        # xr plot
        ax_xr = fig.add_subplot(self.rows, self.cols, 5)
        self.xr(ax_xr)
        self.remove_frame(ax_xr)
        self.add_grid(ax_xr)
            
        # E plot
        ax_e = fig.add_subplot(self.rows, self.cols, 6)
        self.E(ax_e)
        self.remove_frame(ax_e)
        self.add_grid(ax_e)
        
        # Yield curve plots
        ax_short = fig.add_subplot(self.rows, self.cols, 7)
        ax_10y = fig.add_subplot(self.rows, self.cols, 8)
        self.yield_curve_vs_fitted(ax_short, ax_10y)
        self.remove_frame(ax_short)
        self.remove_frame(ax_10y)
        self.add_grid(ax_short)
        self.add_grid(ax_10y)
        self.set_legend(ax_short)
        self.set_legend(ax_10y)
        
        # Full Yield Curve plot
        ax_full = fig.add_subplot(self.rows, self.cols, 9)
        self.full_yield_curve(ax_full)
        self.remove_frame(ax_full)
        self.add_grid(ax_full)
        
        # Yield curve cycle plot
        if self.model == "ff":
            ax_cycle = fig.add_subplot(self.rows, self.cols, 10)
            self.yield_curve_cycle(ax_cycle)
            self.remove_frame(ax_cycle)
            self.add_grid(ax_cycle)
        
        plt.tight_layout()
        return fig

    def A(self, ax):
        ax.plot(self.atsm.A)
        self.set_title(ax, 'A')
        return ax

    def B(self, ax):
        num_columns = self.atsm.B.T.shape[1]
        blues = plt.cm.Blues(np.linspace(0.3, 1, num_columns))
        for i, col in enumerate(range(num_columns)):
            ax.plot(self.atsm.B.T[:, col], color=blues[i], label=f'$B_{{{col+1}}}$')
        self.set_title(ax, 'B')
        ax.legend([str(col + 1) for col in range(num_columns)])
        return ax
    


    def X(self, ax):
        num_columns = self.atsm.X.shape[1]
        blues = plt.cm.Blues(np.linspace(1, 0.3, num_columns))
        for i, column in enumerate(self.atsm.X.columns):
            ax.plot(self.atsm.X[column], color=blues[i], label=f'$X_{{{i+1},t}}$')
        self.set_title(ax, 'X')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        legend = ax.legend(loc='upper right')  # Ensure the legend is placed in the upper right location
        plt.setp(legend.get_texts(), fontsize=self.labelsize)  # Update the labelsize of the legend
        ax.tick_params(labelsize=self.labelsize)
        return ax

    def v(self, ax):
        num_columns = self.atsm.v.shape[1]
        blues = plt.cm.Blues(np.linspace(1, 0.3, num_columns))
        for i, column in enumerate(self.atsm.v.columns):
            ax.plot(self.atsm.v[column], color=blues[i], label=f'$v_{{{i+1},t}}$')
        self.set_title(ax, 'v')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        legend = ax.legend(loc='upper right')  # Ensure the legend is placed in the upper right location
        plt.setp(legend.get_texts(), fontsize=self.labelsize)  # Update the labelsize of the legend
        ax.tick_params(labelsize=self.labelsize)
        return ax
    
    def E(self, ax):
        num_columns = self.atsm.E.shape[1]
        colors = plt.cm.Blues(np.linspace(0.2, 1, num_columns))
        for i, column in enumerate(self.atsm.E.columns):
            ax.plot(self.atsm.E[column], color=colors[i])
        self.set_title(ax, 'E')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        
        # Create color bar as legend inside the graph at the top center
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=num_columns-1))
        sm.set_array([])
        cax = ax.inset_axes([0.3, 0.90, 0.4, 0.03])  # [left, bottom, width, height]        
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_ticks([0, num_columns-1])
        cbar.set_ticklabels(['1', str(num_columns)])
        cbar.set_label('Column', labelpad=-10, fontsize=self.labelsize)
        cbar.ax.tick_params(labelsize=self.labelsize)
        return ax

    def xr(self, ax):
        num_columns = self.atsm.xr.shape[1]
        colors = plt.cm.Blues(np.linspace(0.2, 1, num_columns))
        for i, column in enumerate(self.atsm.xr.columns):
            ax.plot(self.atsm.xr[column], color=colors[i])
        self.set_title(ax, 'Excess Returns (xr)')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        
        # Create color bar as legend inside the graph at the top center
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=num_columns-1))
        sm.set_array([])
        cax = ax.inset_axes([0.3, 0.90, 0.4, 0.03])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_ticks([0, num_columns-1])
        cbar.set_ticklabels(['1', str(num_columns)])
        cbar.set_label('Column', labelpad=-10, fontsize=self.labelsize)
        cbar.ax.tick_params(labelsize=self.labelsize)
        return ax

    def yield_curve_vs_fitted(self, ax_short, ax_10y):
        common_index = self.atsm.yield_curve["short"].index.intersection(self.atsm.yield_curve_fitted.index)
        if self.model == "ff":
            common_index = common_index.intersection(self.atsm.macro_data["long"]["rf_LR"].index)
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

    def full_yield_curve(self, ax):
        num_columns = self.atsm.yield_curve_fitted.shape[1]
        colors = plt.cm.Blues(np.linspace(0.2, 1, num_columns))
        for i, column in enumerate(self.atsm.yield_curve_fitted.columns):
            ax.plot(self.atsm.yield_curve_fitted[column], color=colors[i])
        self.set_title(ax, 'Fitted Yield Curve')
        ax.set_ylabel('Yield')
        ax.set_xlabel('')
        
        # Create color bar as legend inside the graph at the top center
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=num_columns-1))
        sm.set_array([])
        cax = ax.inset_axes([0.3, 0.90, 0.4, 0.03])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_ticks([0, num_columns-1])
        cbar.set_ticklabels(['1', str(num_columns)])
        cbar.set_label('Maturity', labelpad=-10, fontsize=self.labelsize)
        cbar.ax.tick_params(labelsize=self.labelsize)
        return ax

    def yield_curve_cycle(self, ax):
        num_columns = self.atsm.yield_curve_cycle.shape[1]
        colors = plt.cm.Blues(np.linspace(0.2, 1, num_columns))
        for i, column in enumerate(self.atsm.yield_curve_cycle.columns):
            ax.plot(self.atsm.yield_curve_cycle[column], color=colors[i])
        self.set_title(ax, 'Yield Curve Cycle')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        
        # Create color bar as legend inside the graph at the top center
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=num_columns-1))
        sm.set_array([])
        cax = ax.inset_axes([0.3, 0.90, 0.4, 0.03])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_ticks([0, num_columns-1])
        cbar.set_ticklabels(['1', str(num_columns)])
        cbar.set_label('Maturity', labelpad=-10, fontsize=self.labelsize)
        return ax