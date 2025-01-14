from src.utils.model_utils import Utils

from statsmodels.tsa.vector_ar.var_model import VAR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.paths import Paths

class PricingFactors(Utils):
    def __init__(self, 
                 yield_curve, 
                 macro_data, 
                 K, 
                 model,
                 freq = "QE-DEC",
                 freq_factor = 4):

        #
        self.yield_curve = yield_curve
        self.macro_data = macro_data
        self.K = K
        self.model = model
        self.freq = freq
        self.freq_factor = freq_factor

        #
        self.__X = None
        self.__X_VAR = None
        self.__mu = None
        self.__Phi = None
        self.__v = None
        self.__Sigma = None


    @property
    def data(self):
        if self.__X is None:
            self.__X = self.get_factors(yield_curve = self.yield_curve,
                                        macro_data  = self.macro_data,
                                        K           = self.K,
                                        model       = self.model,
                                        freq        = self.freq)
            self.plot_data()  # Plot the data after it's generated
        return self.__X

    def plot_data(self):
        """
        Plot the time series data for visual inspection.
        """
        plt.figure(figsize=(12, 6))
        for column in self.__X.columns:
            plt.plot(self.__X.index, self.__X[column], label=column)
        plt.title('Time Series Data')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig('pricing_factors_data.png')
        plt.close()

    def fit(self, data = None,p=1):
        if data is None: data = self.data
        # print("Fitting VAR model...")
        model = VAR(data)#.div(self.freq_factor))
        fitted_model = model.fit(maxlags=1, ic=None, trend='c')
        # print(fitted_model.summary())
        return fitted_model
    
    @property
    def X_VAR(self):
        if self.__X_VAR is None:
            self.__X_VAR = self.fit()
        return self.__X_VAR
    
    def forecast(self, h, forecast_date=None):
        """
        Make forecasts for h periods ahead.
        
        Args:
            h (int): Number of periods to forecast.
            forecast_date (str): The date to forecast from. If None, the last date in the data is used.
        
        Returns:
            pandas.DataFrame: Forecasted values for h periods.
        """

        if forecast_date is None:
            data = self.data
            # Generate in-sample predictions
            in_sample_predictions = self.X_VAR.fittedvalues

            # Use the last values from in-sample predictions for forecasting
            last_known_values = in_sample_predictions[-self.X_VAR.k_ar:].values
            
            # Make the forecast
            forecast_values = self.X_VAR.forecast(y=last_known_values, steps=h)
            
            # The forecast date is the last date in the estimation sample
            forecast_date = self.data.index[-1]
        else:
            # Since there's a forecast date given, we need to reestimate the VAR (just the VAR) to make a forecast over it. 
            # The coefficients of the ATSM are unchanged.
            yield_curve = pd.read_csv(Paths().raw_path / self.freq[0].upper() / "yield_curve.csv", index_col=0)
            yield_curve.index = pd.to_datetime(yield_curve.index)
            data = self.get_factors(yield_curve     = yield_curve.loc[:forecast_date+pd.Timedelta(days=1)],
                                        macro_data  = self.macro_data,
                                        K           = self.K,
                                        model       = self.model,
                                        freq        = self.freq)
            model = self.fit(data = data)
            in_sample_predictions = model.fittedvalues
            last_known_values = in_sample_predictions[-model.k_ar:].values
            forecast_values = model.forecast(y=last_known_values, steps=h)

        # Create a date range for the forecast periods
        if self.freq.lower().startswith('q'):
            forecast_dates = pd.date_range(start=forecast_date + pd.offsets.QuarterEnd(), periods=h, freq=self.freq)
        elif self.freq.lower().startswith('m'):
            forecast_dates = pd.date_range(start=forecast_date + pd.offsets.MonthEnd(), periods=h, freq=self.freq)
        else:
            raise ValueError("Unsupported frequency. Please use 'q' for quarters or 'm' for months.")
        # Create a DataFrame with the forecasted values
        forecast_df = pd.DataFrame(forecast_values, index=forecast_dates, columns=self.data.columns)
        
        # Create a DataFrame with in-sample predictions
        in_sample_df = pd.DataFrame(in_sample_predictions, index=self.data.index, columns=self.data.columns)
        
        # Combine the in-sample predictions with the forecast
        combined_df = pd.concat([in_sample_df, forecast_df], axis=0)
        
        # Add a column to distinguish between actual and forecasted values
        combined_df['Type'] = 'Actual'
        combined_df.loc[forecast_dates, 'Type'] = 'Forecast'
        
        return combined_df
    
    def get_residuals(self):
        """
        Calculates the residuals of the VAR model and stores them in a DataFrame.
        """
        residuals = self.X_VAR.resid
        return pd.DataFrame(data=residuals, index=self.data.index)
    
    @property
    def v(self):
        if self.__v is None:
            self.__v = self.get_residuals()
        return self.__v
    
    def get_residual_covariance(self):
        """
        Calculates the covariance matrix of the residuals.
        """
        v = self.v.dropna().values
        return np.dot(v.T, v) / len(v)
    

    def get_mu(self):
        mu = self.X_VAR.params.T["const"]
        return mu.values.reshape(-1, 1)

    @property
    def mu(self):
        if self.__mu is None:
            self.__mu = self.get_mu()
        return self.__mu

    def get_Phi(self):
        if self.__Phi is None:
            self.__Phi = self.X_VAR.params.T.loc[:, ~self.X_VAR.params.T.columns.isin(["const"])]
            self.__Phi.columns = self.__Phi.index
        return self.__Phi
    
    @property
    def Phi(self):
        if self.__Phi is None:
            self.__Phi = self.get_Phi()
        return self.__Phi
    
    
    @property
    def Sigma(self):
        if self.__Sigma is None:
            self.__Sigma = self.get_residual_covariance()
        return self.__Sigma

    def debug_info(self):
        """
        Print debug information about the data.
        """
        print("Data shape:", self.data.shape)
        print("\nData info:")
        print(self.data.info())
        print("\nData description:")
        print(self.data.describe())
        print("\nFirst few rows:")
        print(self.data.head())
        print("\nLast few rows:")
        print(self.data.tail())