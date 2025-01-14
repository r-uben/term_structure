import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.model_utils import Utils
from sklearn.linear_model import LinearRegression


ADJUST = True


class HoldingPeriodExcessReturns(Utils):


    def __init__(self, 
                 yield_curve,
                 time_params,
                 pricing_factors,
                 trend=None):
        
        self.__yield_curve = yield_curve
        self.__trend = trend
        self.__num_quarters = time_params.num_quarters
        self.__start_period = time_params.start_period
        self.__freq_factor = time_params.freq_factor
        self.__N = time_params.N
        self.__pricing_factors = pricing_factors
        self.__data  = None
        self.__maturities = None
        self.__model = None
        self.__residuals = None
        self.__freq = time_params.freq

    @property
    def maturities(self):
        if self.__maturities is None:
            self.__maturities = [1, 2, 3, 4, 5, 10, 15] # in years
            self.__maturities = [int(x * self.__freq_factor) for x in self.__maturities]
        return self.__maturities

    def compute(self, y_curve, N, start_period, freq_factor):
        p_curve = Utils.conversion_yields_to_prices(y_curve) #/ (freq_factor/4)
        hpr = (p_curve.iloc[:,:N-(start_period-1)-1].values \
            - p_curve.iloc[:,1:N-(start_period-1)].shift(1).values)
        hpr = hpr # freq_factor
        xr  = hpr - y_curve.loc[:, str(start_period)].shift(1).values.reshape(-1,1) #/ (freq_factor/4)
        # Annualize xr
        xr_curve = pd.DataFrame(xr, 
                                index=y_curve.index, 
                                columns=y_curve.columns[:N-(start_period-1)-1])
        return xr_curve
    
    def adjust_xr_curve(self, xr_curve, trend, start_period, N, freq_factor):
        xr_curve_adj = xr_curve.copy()
        common_index = xr_curve_adj.index.intersection(trend.index)
        trend = trend.loc[common_index, :]
        xr_curve_adj = xr_curve_adj.loc[common_index, :]
        
        # Calculate the trend difference
        trend_diff = trend.values - trend.shift(1).values
        
        # Create a range of n values
        n_values = np.arange(start_period + 1, N - (start_period-1) )
        
        # Create a matrix of adjustments
        adj_matrix = np.outer(trend_diff, n_values - 1)
        
        # Convert adj_matrix to a DataFrame with the same index as trend
        adj_matrix_df = pd.DataFrame(adj_matrix, 
                                     index=trend.index, 
                                     columns=[str(n-1) for n in range(start_period+1, N-(start_period-1))])
        
        
        # Apply the adjustment to all columns at once
        columns_to_adjust = [str(n-1) for n in range(start_period+1, N-(start_period-1))]
        
        xr_curve_adj.loc[:, columns_to_adjust] += adj_matrix_df.values
        return xr_curve_adj#*freq_factor

    @property
    def data(self):
        if self.__data is None:
            self.__data = self.compute( y_curve      = self.__yield_curve,
                                        N            = self.__N,
                                        start_period = self.__start_period,
                                        freq_factor  = self.__freq_factor)
            if  self.__trend is not None and ADJUST:
                self.__data = self.adjust_xr_curve(xr_curve=self.data, 
                                                   trend=self.__trend, 
                                                   start_period=self.__start_period,
                                                   N=self.__N,
                                                   freq_factor=self.__freq_factor)
                

        return self.__data#(self.freq_factor)
    

    def fit(self):
        X = pd.concat([self.__pricing_factors.v, 
                       self.__pricing_factors.data.shift(1)], 
                       axis=1).dropna()

        y = self.data.loc[:, [str(int(m) - 1) for m in self.maturities]].dropna()
        common_index = y.index.intersection(X.index)
        X_common = X.loc[common_index]
        y_common = y.loc[common_index]
        model = LinearRegression(fit_intercept=True)
        model.fit(X_common.values, y_common.values)

        # Calculate predictions and residuals
        y_pred = model.predict(X_common.values)
        residuals = y_common.values - y_pred
        
        residuals = pd.DataFrame(residuals, 
                                 index=y_common.index, 
                                 columns=y.columns)
        
        # Print R^2 score
        r2 = model.score(X_common.values, y_common.values)
        #print(f"R^2 Score: {r2}")

        # Print R^2 scores
       #Utils.print_r2(self.__pricing_factors, self.data.loc[:, [str(int(m) - 1) for m in self.maturities]].dropna())

        return model, residuals

    @property
    def model(self):
        if self.__model is None:
            self.__model, self.__residuals = self.fit()
        return self.__model

    @property
    def residuals(self):
        if self.__residuals is None:
            self.__model, self.__residuals = self.fit()

        return self.__residuals