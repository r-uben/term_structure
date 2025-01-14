import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from src.core.models.forecast import Forecast
from src.core.models.estimation import Estimation
from src.paths import Paths

class Models:

    def __init__(self,
                init_date: str,
                end_date: str = None,
                freq: str = "QE-DEC",
                K: int = 5,
                num_quarters: int = 60,
                ):
        
        self.__paths = Paths()
        # The initial date of the estimation sample
        self.__init_date = init_date
        # The end date of the estimation sample
        self.__end_date = end_date
        # The frequency of the data
        self.__freq = freq
        # The number of (latent) factors
        self.__K = K
        # The number of quarters to be added when de-trending the yield curve (FF model only)
        self.__num_quarters = num_quarters

        self.__yield_curve = None

    @property
    def yield_curve(self):
        if self.__yield_curve is None:
            self.__yield_curve =pd.read_csv(self.__paths.raw_path / self.__freq[0].upper() / "yield_curve.csv", index_col=0)
            self.__yield_curve.index = pd.to_datetime(self.__yield_curve.index)
        return self.__yield_curve


    @property
    def ACM(self):
        model = Estimation(
            init_date=self.__init_date,
            end_date=self.__end_date,
            model="acm",
            freq=self.__freq,
            K=self.__K,
            num_quarters=self.__num_quarters,
        )
        return model
    
    @property
    def FF(self):
        model = Estimation(
            init_date=self.__init_date,
            end_date=self.__end_date,
            model="ff",
            freq=self.__freq,
            K=self.__K,
            num_quarters=self.__num_quarters,
        )
        return model
    
    def forecasts(self, h: int, maturity: int | str, PRINT=False):

        ACM = Forecast(self.ACM)
        FF  = Forecast(self.FF)
        
        ACM_forecasts = []
        FF_forecasts = []
        actual_values = []
        dates = []

        if type(maturity) == int:
            maturity = str(maturity)

        end_date = pd.to_datetime(self.__end_date)
        forecast_date = end_date
        last_date = self.FF.yield_curve["long"].index[-1]

        actual_date = forecast_date
        while actual_date <= last_date:
            forecast_date += pd.DateOffset(months=3)
  
            ACM_forecast_value = ACM.yield_curve(h, forecast_date=forecast_date).loc[:, str(maturity)].iloc[0]
            FF_forecast_value = FF.yield_curve(h, forecast_date=forecast_date).loc[:, str(maturity)].iloc[0]
            ACM_forecasts.append(ACM_forecast_value)
            FF_forecasts.append(FF_forecast_value)

            
            actual_date = forecast_date + pd.DateOffset(months=h*3)
            actual_date = actual_date + pd.offsets.MonthEnd(0)
            dates.append(actual_date)

            if actual_date in self.yield_curve.index:
                actual_value = self.yield_curve.loc[actual_date, str(maturity)]
                actual_values.append(actual_value)
            else:
                print(f"Actual date {actual_date} not in yield curve")
                actual_values.append(np.nan)

            if PRINT:
                print(f"Forecast Date: {forecast_date.strftime('%Y-%m-%d')}, "
                      f"Actual Date: {actual_date.strftime('%Y-%m-%d')}, "
                  f"ACM Forecast Value: {ACM_forecast_value:.4f}, "
                  f"FF Forecast Value: {FF_forecast_value:.4f}, "
                  f"Actual Value: {actual_value:.4f}")

        ACM_forecasts = pd.Series(ACM_forecasts, index=dates)
        FF_forecasts = pd.Series(FF_forecasts, index=dates)
        actual_values = pd.Series(actual_values, index=dates)

        # Remove NaN values
        actual_values = actual_values.dropna()

        return ACM_forecasts, FF_forecasts, actual_values


def build_latex_table(acm_rmse_dict, ff_rmse_dict, obs_count_dict, horizons):
    # Create LaTeX table
    latex_table = "\\begin{table}[h]\n\\centering\n"
    latex_table += "\\renewcommand{\\arraystretch}{1.2}\n"
    latex_table += "\\setlength{\\tabcolsep}{10pt}\n"
    latex_table += "\\begin{tabular}{l|ccccc}\n\\hline\n"
    latex_table += "\\rule{0pt}{3ex}Model & $h=1$ & $h=4$ & $h=8$ & $h=20$ & $h=40$ \\\\[0.5ex]\n\\hline\n"

    # Add ACM RMSE row
    latex_table += "\\rule{0pt}{3ex}ACM "
    for h in horizons:
        value = acm_rmse_dict.get(h)
        latex_table += f"& {value:.6f} " if value is not None else "& N/A "
    latex_table += "\\\\[0.5ex]\n"

    # Add FF RMSE row
    latex_table += "FF  "
    for h in horizons:
        value = ff_rmse_dict.get(h)
        latex_table += f"& {value:.6f} " if value is not None else "& N/A "
    latex_table += "\\\\[0.5ex]\n"

    # Add Observations row
    latex_table += "Obs. "
    for h in horizons:
        value = obs_count_dict.get(h)
        latex_table += f"& {value} " if value is not None else "& N/A "
    latex_table += "\\\\[0.5ex]\n"

    latex_table += "\\hline\n\\end{tabular}\n"
    latex_table += "\\caption{RMSE for ACM and FF models at different forecast horizons when predicting the one period yield}\n"
    latex_table += "\\label{tab:rmse_comparison}\n\\end{table}"
    return latex_table

if __name__ == "__main__":

    estimation_end_date = "2013-03-31"
    sample_end_date = "2018-06-30"  # Adjust this to your actual sample end date
    
    models = Models(
        init_date="1979-12-31",
        end_date=estimation_end_date,
        freq="QE-DEC",
        K=5,
        num_quarters=100,
    )

    # Initialize dictionaries to store RMSE values and observation counts
    acm_rmse_dict = {}
    ff_rmse_dict = {}
    obs_count_dict = {}
    horizons = [1, 4, 4*2, 4*5, 4*10]

    for h in horizons:
        ACM_forecasts, FF_forecasts, actual_values = models.forecasts(h=h, maturity=1)

        # Combine the three DataFrames
        combined_df = pd.concat([ACM_forecasts, FF_forecasts, actual_values], axis=1).dropna()
        combined_df.columns = ['ACM_forecast', 'FF_forecast', 'Actual']

        if not combined_df.empty:
            # Compute RMSE for ACM and FF forecasts
            acm_rmse = np.sqrt(mean_squared_error(combined_df['Actual'], combined_df['ACM_forecast']))
            ff_rmse = np.sqrt(mean_squared_error(combined_df['Actual'], combined_df['FF_forecast']))
            
            acm_rmse_dict[h] = acm_rmse
            ff_rmse_dict[h] = ff_rmse
            obs_count_dict[h] = len(combined_df)
    
    latex_table = build_latex_table(acm_rmse_dict, ff_rmse_dict, obs_count_dict, horizons)
    print("LaTeX Table:")
    print(latex_table)