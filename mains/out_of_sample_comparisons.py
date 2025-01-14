import matplotlib.pyplot as plt
import pandas as pd

from src.core.models.estimation import Estimation
from src.core.models.forecast import Forecast
from src.paths import Paths


class Comparisons:

    def __init__(self,
                 init_date: str,
                 end_date: str = None,
                 freq: str = "QE-DEC",
                 K: int = 5,
                 num_quarters: int = 60,
                 ):
        self.__paths = Paths()
        self.__init_date = init_date
        self.__end_date = end_date
        self.__freq = freq
        self.__K = K
        self.__num_quarters = num_quarters

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
    
    def forecasts(self, h):
        ACM = Forecast(self.ACM)
        FF = Forecast(self.FF)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot ACM.A
        axes[0].plot(self.ACM.A, marker='o')
        axes[0].set_title('ACM.A')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Value')

        # Plot FF.A
        axes[1].plot(self.FF.A, marker='o')
        axes[1].set_title('FF.A')
        axes[1].set_xlabel('Index')
        axes[1].set_ylabel('Value')

        plt.tight_layout()
        #plt.show()
        ACM_forecast = ACM.yield_curve(h)
        FF_forecast = FF.yield_curve(h)
        return ACM_forecast, FF_forecast
    
    def plot_forecasts(self, h, maturity, PLOT=False):
        ACM_forecast, FF_forecast = self.forecasts(h)
        yield_curve = self.ACM.yield_curve["long"]
        if type(maturity) == int:
            maturity = str(maturity)
        

        plt.figure(figsize=(10, 6))

        plt.plot(ACM_forecast.loc[:, maturity], label='ACM Forecast', marker='o')
        plt.plot(FF_forecast.loc[:, maturity], label='FF Forecast', marker='o')
        time_mask = yield_curve.index >= (pd.to_datetime(self.__end_date) - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
        time_mask = time_mask & (yield_curve.index <= ACM_forecast.index[-1])
        plt.plot(yield_curve.loc[time_mask, maturity], label='Short-Rate Yield', marker='o')

        # ACM fitted yield
        ACM_fitted = self.ACM.yield_curve_fitted
        fitted_time_mask = ACM_fitted.index >= (pd.to_datetime(self.__end_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        fitted_time_mask = fitted_time_mask & (ACM_fitted.index <= self.__end_date)
        plt.plot(ACM_fitted.loc[fitted_time_mask, maturity], label='ACM Fitted Yield', marker='o')

        # FF fitted yield
        FF_fitted = self.FF.yield_curve_fitted
        fitted_time_mask = FF_fitted.index >= (pd.to_datetime(self.__end_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        fitted_time_mask = fitted_time_mask & (FF_fitted.index <= self.__end_date)
        plt.plot(FF_fitted.loc[fitted_time_mask, maturity], label='FF Fitted Yield', marker='o')

        plt.xlabel('Date')
        plt.ylabel('Yield')
        plt.title('Yield Curve Forecasts')
        plt.legend()
        plt.show()
    
    
if __name__ == "__main__":


    forecasts = []
    maturities = ["1", "4", "20", "40"]
    for year in range(2005, 2023):
        forecast_date = f"{year}-06-30"
        comparisons = Comparisons(
            init_date="1979-12-31",
            end_date=forecast_date,
            freq="QE-DEC",
            K=5,
            num_quarters=60,
        )
        for maturity in maturities:
            for h in range(1, 4*5+1):  # from 1 quarter to 5 years
                ACM_forecast, FF_forecast = comparisons.forecasts(h=h)
                ACM_df = ACM_forecast.loc[:, maturity].reset_index()
                ACM_df["h"] = h
                ACM_df["model"] = "ACM"
                ACM_df["forecast_date"] = forecast_date
                ACM_df["maturity"] = maturity
                ACM_df.rename(columns={maturity: "forecasted_value"}, inplace=True)
                
                FF_df = FF_forecast.loc[:, maturity].reset_index()
                FF_df["h"] = h
                FF_df["model"] = "FF"
                FF_df["forecast_date"] = forecast_date
                FF_df["maturity"] = maturity
                FF_df.rename(columns={maturity: "forecasted_value"}, inplace=True)
                
                forecasts.append(ACM_df)
                forecasts.append(FF_df)
    
    forecasts_df = pd.concat(forecasts, ignore_index=True)
    forecasts_df = forecasts_df.set_index("index")
    forecasts_df.index.name = "date"
    forecasts_df.index = pd.to_datetime(forecasts_df.index)  # Ensure the index is datetime
    print(forecasts_df.head())

    yield_curve = pd.read_csv(Paths().data_path / "raw" / "Q" / "yield_curve.csv")
    yield_curve = yield_curve.set_index("date")
    yield_curve.index.name = "date"
    yield_curve.index = pd.to_datetime(yield_curve.index)  # Ensure the index is datetime
    print(yield_curve.head())

    # Reshape yield_curve to have a single column "actual_value" and a "maturity" column
    yield_curve_melted = yield_curve.reset_index().melt(id_vars=["date"], value_vars=maturities, var_name="maturity", value_name="actual_value")
    yield_curve_melted["maturity"] = yield_curve_melted["maturity"].astype(str)
    print(yield_curve_melted.head())

    # Merge forecasts_df with yield_curve_melted on the date and maturity columns
    merged_df = forecasts_df.reset_index().merge(yield_curve_melted, on=["date", "maturity"])
    
    # Calculate forecast error
    merged_df["forecast_error"] = merged_df["forecasted_value"] - merged_df["actual_value"]
    
    print(merged_df.head())

    # Get the last forecast error for each combination of model, horizon, forecast date, and maturity
    last_forecast_errors = merged_df.groupby(["model", "h", "forecast_date", "maturity"]).last().reset_index()
    
    # Get descriptive statistics of the last forecast errors by model, horizon, and maturity
    descriptive_stats = last_forecast_errors.groupby(["model", "h", "maturity"])["forecast_error"].agg(
        count='count',
        mean='mean',
        std='std',
        min='min',
        q25=lambda x: x.quantile(0.25),
        median=lambda x: x.quantile(0.5),
        q75=lambda x: x.quantile(0.75),
        max='max'
    )
    descriptive_stats.to_csv(Paths().data_path / "econ_rev" / "out_of_sample_comparisons.csv")