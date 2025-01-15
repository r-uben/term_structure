import numpy as np
import pandas as pd
from src.utils.base_utils import Utils

class Trend:

    def __init__(self, 
                 macro_data=None,
                 yield_curve=None,
                 freq=None,
                 start_period=None):
        
        # Private properties
        self.__macro_data = macro_data
        self.__yield_curve = yield_curve
        self.__freq = freq
        self.__start_period = start_period
        self.__T = int(len(self.__macro_data))
        self.__N = len(self.__yield_curve.columns)
        self.__data = None
        self.__short_rate_trend = None


   
    
    @property
    def short_rate_trend(self): 
        if self.__short_rate_trend is None:
            self.__short_rate_trend = self.__macro_data.loc[:, "rf_LR"]
        return self.__short_rate_trend
    

    @property
    def data(self):

        if self.__data is None:
            self.__data = self.compute(
                short_rate_trend=self.short_rate_trend,
                N=self.__N,
                T=self.__T,
                start_period=self.__start_period,
                freq=self.__freq,
                macro_data=self.__macro_data,
                yield_curve=self.__yield_curve
            )
        return self.__data


    def compute(self,
                short_rate_trend,
                N,
                T,
                start_period,
                freq,
                macro_data,
                yield_curve):
        
        dim_y_trend_2 = int(N)#- (start_period - 1))

        y_trend = np.zeros((int(T), dim_y_trend_2))
        if (type(short_rate_trend) == pd.Series) or (type(short_rate_trend) == pd.DataFrame):
            y_trend[:,0] = short_rate_trend.values.reshape(-1)
        else:
            y_trend[:,0] = short_rate_trend


        for n in range(1, dim_y_trend_2):
            for t in range(T - n):
                y_trend[t,n] = ((n - 1) / n) * y_trend[t + 1,n - 1] \
                                + (1 / n) * y_trend[t,0]

        yield_curve_trend = pd.DataFrame(y_trend, index=macro_data.index, columns = yield_curve.columns)
        if "m" in freq.lower():
            yield_curve_trend = yield_curve_trend.resample(freq).interpolate(method='linear')
        return yield_curve_trend
