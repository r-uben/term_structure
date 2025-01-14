import numpy as np
import pandas as pd
from src.utils.model_utils import Utils

from src.model.time.time_params import TimeParams
from src.model.time.time_window import TimeWindow
from src.model.holding_period_excess_returns import HoldingPeriodExcessReturns
from src.model.params import Params
from src.model.data.yield_curve import YieldCurve
from src.model.data.macro_data import MacroData
from src.model.pricing_factors import PricingFactors
from src.model.common_trend import CommonTrend
from src.model.trend import Trend


class Estimation(Params, TimeWindow, TimeParams, Utils):

    def __init__(self,
                 init_date: str,
                 end_date: str = None,
                 model: str = "ff",
                 freq: str = "QE-DEC",
                 K: int = 5,
                 num_quarters: int = 60,
                 ):
        # Initialize base classes
        TimeWindow.__init__(self, init_date, end_date)
        Utils.__init__(self)
        TimeParams.__init__(self, model, freq, K, num_quarters)
        
        # Create YieldCurve and MacroData instances
        self.__yield_curve   = YieldCurve(time_params=self, 
                                          time_window=self)
        self.__yield_curve_cycle = None
        self.__macro_data    = MacroData(time_window=self)
        
        self.__short_rate_trend = None
        self.common_trend  = CommonTrend(time_params=self,time_window=self)
       
        self.__trend         = Trend(
            macro_data=self.macro_data["long"],
            yield_curve=self.yield_curve["long"],
            freq=self.freq,
            start_period=self.start_period
            )
        
        self.pricing_factors = PricingFactors(
            yield_curve=self.yield_curve["short"],
            macro_data=self.short_rate_trend,
            K=self.K,
            model=self.model.lower(),
            freq=self.freq,
            freq_factor=self.freq_factor)
        
        self.__xr  = HoldingPeriodExcessReturns(
            yield_curve=self.yield_curve["short"],
            time_params=self,
            pricing_factors=self.pricing_factors,
            trend=self.short_rate_trend)

        Params.__init__(
            self, 
            pricing_factors=self.pricing_factors,
            cross_sectional_regression=self.__xr,
            K=self.K, 
            num_quarters=self.num_quarters,
            freq_factor=self.freq_factor)

        self.__lambda0  = None
        self.__lambda1  = None
        self.__delta0   = None
        self.__delta1   = None
        self.__A        = None
        self.__B        = None
        self.__A_RF     = None
        self.__B_RF     = None
        self.__TP       = None
        self.__params   = None

        self.__yield_curve_fitted = None
        self.__yield_curve_fitted_RF = None


    @property
    def yield_curve(self):
        return self.__yield_curve.data 

    @property
    def yield_curve_cycle(self):
        if self.model.lower() == "ff":
            if self.__yield_curve_cycle is None:
                #print("Shape of yield_curve['short']:", self.yield_curve["short"].shape)
                #print("Shape of trend:", self.trend.shape)
                #common_index = self.yield_curve["short"].index.intersection(self.trend.index)   
                #self.__yield_curve_cycle = self.yield_curve["short"].loc[common_index, self.trend.columns] - self.trend.loc[common_index, self.trend.columns]
                self.__yield_curve_cycle = Utils.get_cycle(self.yield_curve["short"],
                                                           self.short_rate_trend, self.freq)
            return self.__yield_curve_cycle
        else:
            raise ValueError("Model not supported")

    @property
    def macro_data(self):
        if self.model.lower() == "ff":
            self.__macro_data.data["long"] = self.get_macro_data_with_trend(
                trend_model  = self.common_trend.model,
                macro_data   = self.__macro_data.data,
                end_date     = self.end_date,
                num_quarters = self.num_quarters,
                F_colnames   = self.common_trend.F_colnames)
        return self.__macro_data.data

    @property
    def short_rate_trend(self):
        if self.__short_rate_trend is None:
            if self.model.lower() == "ff":
                self.__short_rate_trend = self.macro_data["long"].loc[self.macro_data["short"].index, "rf_LR"].to_frame()
                if "m" in self.freq.lower():
                    self.__short_rate_trend = self.__short_rate_trend.resample(self.freq).interpolate(method='linear')
                return self.__short_rate_trend
            else:
                return None
        return self.__short_rate_trend
        
    @property
    def trend(self):
        if self.model.lower() == "ff":
            return self.__trend.data
        else:
            return None

    @property
    def X(self):
        return self.pricing_factors.data

    @property
    def xr(self):
        return self.__xr.data

    @property
    def lambda0(self) -> pd.DataFrame:
        if self.__lambda0 is None:
            self.__lambda0 = np.dot(self.Ginv_b, 
                                    (self.a \
                                    + 0.5 * (np.dot(self.B_star, self.Sigma.flatten()) \
                                    + self.sigma2))
                                    )
        return self.__lambda0
    
    @property
    def lambda1(self) -> pd.DataFrame:
        if self.__lambda1 is None:
            self.__lambda1 = np.dot(self.Ginv_b, self.c.T)
        return self.__lambda1

    @property
    def delta0(self) -> float:
        if self.__delta0 is None:
            if self.model == "ff":
                _, self.__delta0, _ = self.get_init_recursion(self.yield_curve_cycle \
                                                            .loc[:,str(self.start_period)], 
                                                            self.X)
            else:
                _, self.__delta0, _ = self.get_init_recursion(self.yield_curve["short"] \
                                                            .loc[:,str(self.start_period)], 
                                                            self.X)
        return self.__delta0
    
    @property
    def delta1(self) -> pd.DataFrame:
        if self.__delta1 is None:
            if self.model == "ff":
                _, _, self.__delta1, = self.get_init_recursion(self.yield_curve_cycle \
                                                               .loc[:,str(self.start_period)], 
                                                               self.X)
            else:
                _, _, self.__delta1 = self.get_init_recursion(self.yield_curve["short"] \
                                                                .loc[:,str(self.start_period)], 
                                                                self.X)
           
        return self.__delta1
    
    @property
    def A(self):
        if self.__A is None:
            self.__A, self.__B = self.get_ATSM_coeffs(
                delta0=self.delta0, 
                delta1=self.delta1, 
                mu=self.mu, 
                Phi=self.Phi, 
                lambda0=self.lambda0, 
                lambda1=self.lambda1, 
                Sigma=self.Sigma, 
                sigma2=self.sigma2, 
                N=self.N, 
                K=self.K, 
                start_period=self.start_period)
        return self.__A
    
    @property
    def A_RF(self):
        if self.__A_RF is None:
            self.__A_RF, self.__B_RF = self.get_ATSM_coeffs(
                delta0=self.delta0, 
                delta1=self.delta1, 
                mu=self.mu, 
                Phi=self.Phi, 
                lambda0=np.zeros_like(self.lambda0), 
                lambda1=np.zeros_like(self.lambda1), 
                Sigma=self.Sigma, 
                sigma2=self.sigma2, 
                N=self.N, 
                K=self.K, 
                start_period=self.start_period)
        return self.__A_RF

    @property
    def B(self):
        if self.__B is None:
            self.__A, self.__B = self.get_ATSM_coeffs(
                                            delta0=self.delta0, 
                                            delta1=self.delta1, 
                                            mu=self.mu, 
                                            Phi=self.Phi, 
                                            lambda0=self.lambda0, 
                                            lambda1=self.lambda1, 
                                            Sigma=self.Sigma, 
                                            sigma2=self.sigma2, 
                                            N=self.N, 
                                            K=self.K, 
                                            start_period=self.start_period)
        return self.__B
    
    @property
    def B_RF(self):
        if self.__B_RF is None:
            self.__A_RF, self.__B_RF = self.get_ATSM_coeffs(delta0=self.delta0,
                                               delta1=self.delta1,
                                               mu=self.mu,
                                               Phi=self.Phi,
                                               lambda0=np.zeros_like(self.lambda0),
                                               lambda1=np.zeros_like(self.lambda1),
                                               Sigma=self.Sigma,
                                               sigma2=self.sigma2,
                                               N=self.N,
                                               K=self.K,
                                               start_period=self.start_period)
        return self.__B_RF
    

    @property
    def yield_curve_fitted(self):

        if self.__yield_curve_fitted is None:
            if self.model == "ff":
                self.__yield_curve_fitted = self.fit_yield_curve(self.A, 
                                                                 self.B, 
                                                                 self.X, 
                                                                 TREND=True, 
                                                                 trend=self.short_rate_trend) 
            else:
                self.__yield_curve_fitted = self.fit_yield_curve(self.A, 
                                                                 self.B, 
                                                                 self.X) 
        return self.__yield_curve_fitted

    @property
    def yield_curve_fitted_RF(self):
        if self.__yield_curve_fitted_RF is None:
            if self.model == "ff":
                self.__yield_curve_fitted_RF    =   self.fit_yield_curve(self.A_RF, 
                                                                         self.B_RF, 
                                                                         self.X, 
                                                                         TREND=True, 
                                                                         trend=self.short_rate_trend) 
            else:
                self.__yield_curve_fitted_RF = self.fit_yield_curve(self.A_RF, 
                                                                   self.B_RF, 
                                                                   self.X) 
        return self.__yield_curve_fitted_RF

    @property
    def TP(self):
        if self.__TP is None:
           self.__TP = self.yield_curve_fitted - self.yield_curve_fitted_RF
        return self.__TP

    @property
    def params(self):
        if self.__params is None:
            params = {
                "A": self.A,
                "A_RF": self.A_RF,
                "B": self.B,
                "B_RF": self.B_RF,
                "lambda0": self.lambda0,
                "lambda1": self.lambda1,
                "delta0": self.delta0,
                "delta1": self.delta1,
                "mu": self.mu,
                "Phi": self.Phi,
                "Sigma": self.Sigma,
                "sigma2": self.sigma2,
                "N": self.N,
                "K": self.K,
            }
            self.__params = params
        return self.__params
