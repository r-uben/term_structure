import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl

# Set up LaTeX font for all text
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

# Set up dark theme style
plt.style.use('dark_background')
mpl.rcParams.update({
    'axes.facecolor': '#1C1C1C',  # Darker background
    'figure.facecolor': '#1C1C1C',
    'axes.edgecolor': '#FFFFFF',   # White edges
    'grid.color': '#404040',       # Lighter grid
    'grid.alpha': 0.3,             # Semi-transparent grid
    'text.color': '#FFFFFF',       # White text
    'axes.labelcolor': '#FFFFFF',  # White labels
    'xtick.color': '#FFFFFF',      # White ticks
    'ytick.color': '#FFFFFF',
    'axes.prop_cycle': plt.cycler(color=['#00B4D8',   # Light blue
                                        '#FF9F1C',     # Light orange
                                        '#90BE6D',     # Light green
                                        '#F15BB5',     # Light pink
                                        '#9B5DE5',     # Light purple
                                        '#FEE440']),   # Light yellow
})

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
        
        # Color maps for sequential data
        self.cmap_light = mpl.colormaps['Blues'].copy()
        self.cmap_light.set_bad(color='white')
        
        # Default colors for comparison plots
        self.colors = {
            'data': '#00B4D8',      # Light blue for actual data
            'fitted': '#FF9F1C',    # Light orange for fitted values
            'trend': '#90BE6D'      # Light green for trends
        }

    def remove_frame(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def set_title(self, ax, title):
        ax.set_title(title, fontsize=self.fontsize, color='white')

    def add_grid(self, ax):
        ax.grid(True, linewidth=self.grid_linewidth, alpha=0.3)

    def set_legend(self, ax):
        if ax.get_legend():
            ax.legend(fontsize=self.labelsize, facecolor='#1C1C1C', 
                     edgecolor='white', labelcolor='white')

    def create_colorbar(self, ax, num_columns, label='Column'):
        sm = plt.cm.ScalarMappable(cmap=self.cmap_light, 
                                  norm=plt.Normalize(vmin=0, vmax=num_columns-1))
        sm.set_array([])
        cax = ax.inset_axes([0.3, 0.90, 0.4, 0.03])
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_ticks([0, num_columns-1])
        cbar.set_ticklabels(['1', str(num_columns)])
        cbar.set_label(label, labelpad=-10, fontsize=self.labelsize, color='white')
        cbar.ax.tick_params(labelsize=self.labelsize, labelcolor='white')
        return cbar 