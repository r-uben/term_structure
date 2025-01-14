import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.paths import Paths
from src.atsm.estimation import Estimation

from src.atsm.utils import Utils

class Forecast(Paths):

    def __init__(self, estimation: Estimation | None = None):
        super().__init__()
        self.__estimation = estimation

    def yield_curve(self, h, forecast_date = None, estimation = None):
        if estimation is None: 
            estimation = self.__estimation
            if estimation is None:
                raise("You must give me a model")
        
        A = estimation.A
        B = estimation.B
        X = estimation.pricing_factors.forecast(h, forecast_date)
        X = X.loc[X.Type == "Forecast"].drop(columns=["Type"])
        if estimation.model == "ff":
            trend_forecast = estimation.common_trend.forecast(h, forecast_date)
            trend_forecast = trend_forecast.loc[trend_forecast.Type == "Forecast", "value"].to_frame().sort_index()
            TREND = True
        else:
            TREND = False
            trend_forecast = None


        yield_curve = estimation.fit_yield_curve(A=A, B=B, X=X, TREND=TREND, trend=trend_forecast)
        return yield_curve
    
    def get_time_series(self, model="ff"):
        all_forecasts = []
        dates = [file.split('.')[0] 
                 for file in os.listdir(self.forecasters_path)
                 if "e" not in file.split('.')[0]]
        
        for date in tqdm(dates, desc="Processing dates"):
            if len(date) == 6:
                release_date = Utils.add_last_day_of_month(date)
                estimation = Estimation(
                    init_date="19800101",
                    end_date=release_date,
                    model=model,
                    freq="ME",
                    K=3
                )
                yield_curve = self.yield_curve(12+3, estimation)
                print(yield_curve)
                for h, maturity in tqdm([(3, 1*3), (3, 40*3), (12, 1*3), (12, 40*3)], 
                                        desc=f"Generating forecasts for {release_date}", 
                                        leave=False):

                    forecast_date = Utils.add_last_day_of_month(Utils.sum_months(release_date[:6], h))
                    forecast = yield_curve.loc[pd.to_datetime(forecast_date), str(maturity)]
                    forecast_dict = {
                        'release_date': release_date,
                        'forecast_date': forecast_date,
                        'maturity': '3m' if maturity == 1*3 else '10y',
                        'forecast': forecast
                    }
                    all_forecasts.append(forecast_dict)
            
        results_df = pd.DataFrame(all_forecasts)
        results_df = results_df.set_index(['release_date', 'forecast_date', 'maturity'])
        return results_df.sort_index(level='release_date')