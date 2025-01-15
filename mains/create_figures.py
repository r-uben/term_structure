import os
import sys
import pandas as pd
from pathlib import Path
import traceback
import matplotlib.pyplot as plt

from src.model.figures import Figures
from src.model.estimation import Estimation
from src.paths import Paths

def save_figure(fig, output_dir: Path, filename: str, close_fig: bool = True):
    """Helper function to save figures in both PNG and SVG formats."""
    fig.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f"{filename}.svg", bbox_inches='tight')
    if close_fig:
        plt.close(fig)

def create_figures(model: str):
    """
    Create and save figures for a given model.
    
    Args:
        model (str): Model type, either 'acm' or 'ff'
    """
    try:
        # Initialize estimation with all its components
        estimation = Estimation(
            init_date="1980-01-01",  # Start from 1980
            model=model,
            freq="QE-DEC"  # Ensure we're using quarterly frequency
        )
        
        # Create base output directory
        paths = Paths()
        base_dir = paths.data_path / 'figures'
        
        # Create model-specific directories
        model_dir = base_dir / model
        subdirs = ['model_parameters', 'returns', 'yield_curves']
        for subdir in subdirs:
            (model_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create figures instance
        figures = Figures(estimation, model)
        
        # Save individual plots
        # Model parameters
        fig_a = figures.model_parameters.plot_A_standalone()
        save_figure(fig_a, model_dir / 'model_parameters', 'A')
        
        fig_b = figures.model_parameters.plot_B_standalone()
        save_figure(fig_b, model_dir / 'model_parameters', 'B')
        
        fig_x = figures.model_parameters.plot_X_standalone()
        save_figure(fig_x, model_dir / 'model_parameters', 'X')
        
        fig_v = figures.model_parameters.plot_v_standalone()
        save_figure(fig_v, model_dir / 'model_parameters', 'v')
        
        # Returns
        fig_e = figures.returns.plot_E_standalone()
        save_figure(fig_e, model_dir / 'returns', 'E')
        
        fig_xr = figures.returns.plot_excess_returns_standalone()
        save_figure(fig_xr, model_dir / 'returns', 'excess_returns')
        
        # Yield curves
        fig_rates = figures.yield_curves.plot_yield_curve_vs_fitted_standalone()
        save_figure(fig_rates, model_dir / 'yield_curves', 'rates_comparison')
        
        fig_full = figures.yield_curves.plot_full_yield_curve_standalone()
        save_figure(fig_full, model_dir / 'yield_curves', 'full_yield_curve')
        
        if model == "ff":
            fig_cycle = figures.yield_curves.plot_yield_curve_cycle_standalone()
            save_figure(fig_cycle, model_dir / 'yield_curves', 'yield_curve_cycle')
        
        # Save combined figure
        fig_all = figures.results
        save_figure(fig_all, base_dir, f'model_results_{model}')
        
        return True
    except Exception as e:
        print(f"Error creating figures for {model} model:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        return False

def main():
    # Create figures for both models
    for model in ['acm', 'ff']:
        print(f"\nCreating figures for {model.upper()} model...")
        if create_figures(model):
            print(f"Figures saved successfully for {model.upper()} model!")
        else:
            print(f"Failed to create figures for {model.upper()} model.")

if __name__ == "__main__":
    main() 