import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from src.paths import Paths
from src.model.time.time_window import TimeWindow
from src.model.time.time_params import TimeParams
from src.model.estimation import Estimation

def main():
    # Initialize paths
    paths = Paths()
    paths.ensure_directories_exist()
    
    # Set up time parameters
    time_window = TimeWindow(
        init_date="1990-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d")
    )
    time_params = TimeParams(
        model="ff",  # Using the default model
        freq="q",    # quarterly frequency
        K=5,         # default value
        num_quarters=60  # This will give us roughly 10 years of data
    )
    
    # Initialize estimation
    estimation = Estimation(
        init_date=time_window.init_date.strftime("%Y-%m-%d"),
        end_date=time_window.end_date.strftime("%Y-%m-%d"),
        model="ff",
        freq="q",
        K=5,
        num_quarters=60
    )
    
    # Calculate term premia
    term_premia = estimation.TP
    
    # Save results
    output_path = paths.data_path / "processed/term_premia.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    term_premia.to_csv(output_path)
    
    # Optional: Plot the results
    plt.figure(figsize=(12, 6))
    for col in term_premia.columns:
        plt.plot(term_premia.index, term_premia[col], label=f'{col} months')
    plt.title('Term Premia Over Time')
    plt.xlabel('Date')
    plt.ylabel('Term Premium (%)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = paths.data_path / "figures/term_premia.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    print("Term premia calculation completed.")
    print(f"Results saved to: {output_path}")
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
