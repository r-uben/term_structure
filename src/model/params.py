import pandas as pd
import numpy as np

class Params():

    def __init__(self,
                 pricing_factors,
                 cross_sectional_regression,
                 K,
                 num_quarters,
                 freq_factor):
        

        ##

        self.pricing_factors            = pricing_factors
        self.cross_section_regression   = cross_sectional_regression
        self.K                          = K
        self.num_quarters               = num_quarters
        self.N                          = int(num_quarters*freq_factor/4)

        ###
        self.__mu       = None
        self.__Phi      = None
        self.__v        = None
        self.__Sigma    = None
        self.__E        = None
        self.__a        = None
        self.__b        = None
        self.__c        = None
        self.__sigma2   = None
        self.__B_star   = None
        self.__Ginv_b   = None

        

    @property
    def mu(self):
        if self.__mu is None:
            self.__mu = self.pricing_factors.mu
        return self.__mu

    @property
    def Phi(self):
        if self.__Phi is None:
            self.__Phi = self.pricing_factors.Phi
        return self.__Phi

    @property
    def v(self):
        if self.__v is None:
            self.__v = self.pricing_factors.v
        return self.__v
    
    @property
    def Sigma(self):
        if self.__Sigma is None:
            self.__Sigma = self.pricing_factors.Sigma
        return self.__Sigma

    @property
    def a(self):
        if self.__a is None:
            self.__a = self.cross_section_regression.model.intercept_
        return self.__a 
    
    @property
    def b(self) -> np.array:
        if self.__b is None:
            self.__b = self.cross_section_regression.model.coef_[:,:self.K].T 
        return self.__b

    @property 
    def c(self) -> np.array:
        if self.__c is None:
            self.__c = self.cross_section_regression.model.coef_[:,self.K:].T
        return self.__c

    @property
    def E(self):
        if self.__E is None:
            self.__E = self.cross_section_regression.residuals
            
        return self.__E     

    @property
    def sigma2(self) -> float:
        if self.__sigma2 is None:
            self.__sigma2 = np.trace(np.dot(self.E.T, self.E)) / (self.E.shape[0] * self.E.shape[1])
        return self.__sigma2

    @property
    def B_star(self) -> pd.DataFrame:
        if self.__B_star is None:
            self.__B_star = np.apply_along_axis(lambda x: np.outer(x, x).ravel(), axis=1, arr=self.b.T) 
        return self.__B_star
    
    @property
    def Ginv_b(self) -> pd.DataFrame:
        # Calculate Ginv_b
        if self.__Ginv_b is None:
            self.__Ginv_b = np.linalg.pinv(self.b.T)
        return self.__Ginv_b