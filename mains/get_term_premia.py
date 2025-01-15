import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from src.paths import Paths
from src.model.time.time_window import TimeWindow
from src.model.time.time_params import TimeParams
from src.model.estimation import Estimation

def initialize_time_window() -> TimeWindow:
    """Initialize the time window from 1980 to present."""
    return TimeWindow(
        init_date="1980-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d")
    )

def calculate_term_premia(time_window: TimeWindow, models: List[str]) -> Dict[str, pd.DataFrame]:
    """Calculate term premia for specified models."""
    term_premia_dict = {}
    
    for model in models:
        estimation = Estimation(
            init_date=time_window.init_date.strftime("%Y-%m-%d"),
            end_date=time_window.end_date.strftime("%Y-%m-%d"),
            model=model,
            freq="q",
            K=3,
            num_quarters=60
        )
        term_premia_dict[model] = estimation.TP
    
    return term_premia_dict

def save_term_premia(term_premia_dict: Dict[str, pd.DataFrame], data_path: Path) -> None:
    """Save term premia results to CSV files."""
    for model, tp in term_premia_dict.items():
        output_path = data_path / f"processed/term_premia_{model}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tp.to_csv(output_path)

def get_nber_recessions(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch NBER recession data from FRED."""
    return web.DataReader(
        'USREC', 'fred', 
        start=start_date, 
        end=end_date
    )

def setup_plot_style() -> None:
    """Configure plot styling with LaTeX formatting and larger fonts."""
    ticksize = 18
    title_size = 20
    label_size = 18
    legend_size = 18
    font_size = 18
    
    plt.rcParams.update({
        # Font settings
        "text.usetex": True,
        "font.family": "Latin Modern Roman",
        "font.size": font_size,
        
        # Axes settings
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "axes.edgecolor": "#555555",
        "axes.labelcolor": "#555555",
        
        # Tick settings
        "xtick.labelsize": ticksize,
        "ytick.labelsize": ticksize,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        
        # Legend settings
        "legend.fontsize": legend_size,
        
        # Figure settings
        "figure.figsize": (12, 6),
        "figure.facecolor": "none",
        "figure.edgecolor": "none",
        
        # Grid settings
        "grid.alpha": 0.3,
        "grid.color": "#555555"
    })

def create_figure() -> Tuple[plt.Figure, plt.Axes]:
    """Create and setup figure and axes with common styling."""
    fig, ax = plt.subplots()
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig, ax

def add_recession_shading(ax: plt.Axes, time_window: TimeWindow) -> None:
    """Add NBER recession shading to the plot."""
    recessions = get_nber_recessions(
        time_window.init_date.strftime("%Y-%m-%d"),
        time_window.end_date.strftime("%Y-%m-%d")
    )
    
    # Find recession periods
    recession_starts = recessions[recessions['USREC'].diff() == 1].index
    recession_ends = recessions[recessions['USREC'].diff() == -1].index
    
    # Add recession shading
    for start, end in zip(recession_starts, recession_ends):
        ax.axvspan(start, end, color='gray', alpha=0.2)

def plot_term_premia(term_premia_dict: Dict[str, pd.DataFrame], 
                    plot_path: Path,
                    time_window: TimeWindow) -> None:
    """Create and save comparison plot of term premia with LaTeX formatting and recession shading."""
    setup_plot_style()
    fig, ax = create_figure()
    
    # Set transparent background
    ax.set_facecolor("none")
    
    # Add recession shading
    add_recession_shading(ax, time_window)
    
    # Plot term premia with high-contrast colors
    colors = ['#0077BB', '#EE7733']  # Blue and Orange, colorblind-friendly
    for (model, tp), color in zip(term_premia_dict.items(), colors):
        ax.plot(tp.index, tp.loc[:,str(40)], label=f'{model.upper()} 10Y', 
                color=color, linewidth=2)
    
    # Customize plot
    ax.set_title('Term Premia Comparison: FF vs ACM', pad=20, color="#666666")
    ax.set_xlabel('')
    ax.set_ylabel('Term Premium')
    ax.legend(frameon=False)
    ax.grid(True)
    
    # Save plot in both SVG and PNG formats
    plot_path_svg = plot_path.with_suffix('.svg')
    plot_path_png = plot_path.with_suffix('.png')
    
    plot_path_svg.parent.mkdir(parents=True, exist_ok=True)
    plot_path_png.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(plot_path_svg, bbox_inches='tight', format='svg', transparent=True)
    plt.savefig(plot_path_png, bbox_inches='tight', format='png', transparent=True, dpi=300)
    plt.close()

def main():
    # Initialize paths and models
    paths = Paths()
    paths.ensure_directories_exist()
    models = ['ff', 'acm']
    
    # Calculate term premia
    time_window = initialize_time_window()
    term_premia_dict = calculate_term_premia(time_window, models)
    
    # Save results and create plot
    save_term_premia(term_premia_dict, paths.data_path)
    plot_path = paths.data_path / "figures/term_premia_comparison"  # removed extension
    plot_term_premia(term_premia_dict, plot_path, time_window)
    
    print("Term premia calculation completed.")
    print(f"Plots saved to: {plot_path}.svg and {plot_path}.png")

if __name__ == "__main__":
    main()
