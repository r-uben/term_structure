import matplotlib.pyplot as plt
from matplotlib import rc

# Set up LaTeX font for all text
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

class BaseFigure:
    def __init__(self, atsm, model):
        self.atsm = atsm
        self.model = model
        self.selected_columns = [0, 
                               self.atsm.freq_factor - 1, 
                               5 * self.atsm.freq_factor - 1, 
                               10 * self.atsm.freq_factor - 1]
        self.fontsize = 24  # Default fontsize for titles
        self.labelsize = 22  # Default fontsize for legend labels
        self.grid_linewidth = 0.75  # Default linewidth for grid

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

    def create_colorbar(self, ax, num_columns, label='Column'):
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=num_columns-1))
        sm.set_array([])
        cax = ax.inset_axes([0.3, 0.90, 0.4, 0.03])
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_ticks([0, num_columns-1])
        cbar.set_ticklabels(['1', str(num_columns)])
        cbar.set_label(label, labelpad=-10, fontsize=self.labelsize)
        cbar.ax.tick_params(labelsize=self.labelsize)
        return cbar 