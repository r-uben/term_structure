import matplotlib.pyplot as plt
from .base import BaseFigure
from .model_parameters import ModelParametersFigures
from .returns import ReturnsFigures
from .yield_curves import YieldCurveFigures

class Figures(BaseFigure):
    def __init__(self, atsm, model):
        super().__init__(atsm, model)
        self.total_plots = 10  # A, B, X, v, E, xr, Short Rate, 10 Years, Full Yield Curve, Yield Curve Cycle
        self.rows = 5
        self.cols = 2
        
        # Initialize component classes
        self.model_parameters = ModelParametersFigures(atsm, model)
        self.returns = ReturnsFigures(atsm, model)
        self.yield_curves = YieldCurveFigures(atsm, model)

    def create_figure(self):
        return plt.figure(figsize=(20, 6 * self.rows), dpi=40)

    @property
    def results(self):
        fig = self.create_figure()
        
        # A plot
        ax_a = fig.add_subplot(self.rows, self.cols, 1)
        self.model_parameters.plot_A(ax_a)
        self.remove_frame(ax_a)
        self.add_grid(ax_a)
        
        # B plot
        ax_b = fig.add_subplot(self.rows, self.cols, 2)
        self.model_parameters.plot_B(ax_b)
        self.remove_frame(ax_b)
        self.add_grid(ax_b)
        self.set_legend(ax_b)
        
        # X plot
        ax_x = fig.add_subplot(self.rows, self.cols, 3)
        self.model_parameters.plot_X(ax_x)
        self.remove_frame(ax_x)
        self.add_grid(ax_x)
        
        # v plot
        ax_v = fig.add_subplot(self.rows, self.cols, 4)
        self.model_parameters.plot_v(ax_v)
        self.remove_frame(ax_v)
        self.add_grid(ax_v)
        
        # xr plot
        ax_xr = fig.add_subplot(self.rows, self.cols, 5)
        self.returns.plot_excess_returns(ax_xr)
        self.remove_frame(ax_xr)
        self.add_grid(ax_xr)
            
        # E plot
        ax_e = fig.add_subplot(self.rows, self.cols, 6)
        self.returns.plot_E(ax_e)
        self.remove_frame(ax_e)
        self.add_grid(ax_e)
        
        # Yield curve plots
        ax_short = fig.add_subplot(self.rows, self.cols, 7)
        ax_10y = fig.add_subplot(self.rows, self.cols, 8)
        self.yield_curves.plot_yield_curve_vs_fitted(ax_short, ax_10y)
        self.remove_frame(ax_short)
        self.remove_frame(ax_10y)
        self.add_grid(ax_short)
        self.add_grid(ax_10y)
        self.set_legend(ax_short)
        self.set_legend(ax_10y)
        
        # Full Yield Curve plot
        ax_full = fig.add_subplot(self.rows, self.cols, 9)
        self.yield_curves.plot_full_yield_curve(ax_full)
        self.remove_frame(ax_full)
        self.add_grid(ax_full)
        
        # Yield curve cycle plot
        if self.model == "ff":
            ax_cycle = fig.add_subplot(self.rows, self.cols, 10)
            self.yield_curves.plot_yield_curve_cycle(ax_cycle)
            self.remove_frame(ax_cycle)
            self.add_grid(ax_cycle)
        
        plt.tight_layout()
        return fig 