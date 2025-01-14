import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm


from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.paths import Paths

class Utils(Paths):
    def __init__(self):
        pass

    @staticmethod
    def new_line():
        print("\n" + "="*50 + "\n")

    @staticmethod
    def get_recessions(start_date, end_date):
        
        # Download recessions data from FRED
        recessions = web.DataReader('USREC', 
                                    'fred', 
                                    start=start_date, 
                                    end=end_date)

        # Find recession periods (where USREC == 1)
        recession_periods = recessions[recessions['USREC'] == 1]

        # Group consecutive recession periods
        recession_groups = recession_periods.groupby((recession_periods.index.to_series().diff() > pd.Timedelta(days=90)).cumsum())
        return recession_groups

    @staticmethod
    def find_data(data):
        return Paths().data_path / data

    @staticmethod
    def set_time_window(start_date, end_date=None):
        if end_date is None:
            end_date = start_date + pd.Timedelta(days=365)
        return pd.date_range(start=start_date, end=end_date)
    
    @staticmethod
    def sum_N_quarters(start_date, num_quarters):
        start_date = pd.to_datetime(start_date)
        end_date = start_date + pd.DateOffset(months=num_quarters*3+1)
        return end_date
    
    
    @staticmethod
    def extend_macro_data(macro_data,
                          end_date,
                          num_quarters):
        extended_end_date = Utils.sum_N_quarters(start_date=end_date, 
                                                 num_quarters=num_quarters)
        
        extended_time_window = pd.date_range(start=macro_data["short"].index.min(), 
                                             end=extended_end_date, 
                                             freq='QE-DEC')
        
        extended_data = macro_data["long"].reindex(extended_time_window).ffill()
        return extended_data

    @staticmethod
    def compute_common_trend(trend_model,
                      F_colnames=None,
                      macro_data=None,
                      extended_macro_data=None,
                      end_date=None,
                      num_quarters=None, 
                      ):
        
        if extended_macro_data is None:
            if macro_data is None:
                raise ValueError("Either macro_data or extend_macro_data must be provided")
            extended_macro_data = Utils.extend_macro_data(macro_data,
                                                            end_date,
                                                            num_quarters)
        extended_X = pd.concat([pd.to_numeric(extended_macro_data[col]) 
                                for col in F_colnames], axis=1)
        gamma = trend_model.params

        rf_LR = 4 * (extended_X @ gamma)
        return rf_LR
    
    @staticmethod
    def get_macro_data_with_trend(trend_model,
                          macro_data,
                          end_date,
                          num_quarters, 
                          F_colnames):
        
        extended_data = Utils.extend_macro_data(macro_data,
                                                end_date,
                                                num_quarters)
        
        rf_LR = Utils.compute_common_trend( trend_model=trend_model,
                                     extended_macro_data=extended_data,
                                     F_colnames=F_colnames)
        
        extended_data['rf_LR'] = rf_LR
        
        return extended_data
    
    @staticmethod
    def get_principal_components(yield_curve, K):
        """
        Perform Principal Component Analysis (PCA) on the yield curve data.

        Parameters:
        yield_curve (pd.DataFrame): The yield curve data with dates as the index and maturities as columns.
        K (int): The number of principal components to extract.

        Returns:
        np.ndarray: The principal components of the yield curve data.
        """
        
        # Initialize PCA with the specified number of components
        pca = PCA(n_components=K)
        
        # Standardize the yield curve data by removing the mean (without scaling to unit variance)
        scaler = StandardScaler(with_mean=True, with_std=False)

        yield_curve_scaled = scaler.fit_transform(yield_curve)
        
        # Convert the scaled data back to a DataFrame to retain original column names and index
        yield_curve_scaled_df = pd.DataFrame(yield_curve_scaled, columns=yield_curve.columns, index=yield_curve.index)
        
        # Fit the PCA model on the scaled yield curve data
        pca.fit(yield_curve_scaled_df)
        
        # Transform the scaled yield curve data to its principal components
        principal_components = pca.transform(yield_curve_scaled_df)
        
        # Convert to pd.DataFrame and multiply the first column by -1
        principal_components = pd.DataFrame(data=principal_components, 
                                columns=[f'PC{k}' for k in range(1, K+1)],
                                index=yield_curve.index)
        # Check the sign of the first principal component and adjust if necessary
        if principal_components.iloc[0, 0] < 0:
            principal_components *= -1

        principal_components.index.freq = pd.infer_freq(principal_components.index)
        
            
        return principal_components
    
    @staticmethod
    def get_cycle(yield_curve, trend, freq):
        common_index = yield_curve.index.intersection(trend.index)
        if type(trend) == pd.Series:
            return yield_curve.loc[common_index].sub(trend.to_frame().loc[common_index].values)
        else:
            return yield_curve.loc[common_index].sub(trend.loc[common_index].values)
    
    @staticmethod
    def get_factors(yield_curve, macro_data, K, model, freq):
        if model == "ff":
            yield_curve_cycle = Utils.get_cycle(yield_curve, 
                                                macro_data["rf_LR"], 
                                                freq)
            if yield_curve_cycle.isna().any().any():
                print("NaN values found in yield_curve_cycle:")
                print(yield_curve_cycle[yield_curve_cycle.isna().any(axis=1)])
            X = Utils.get_principal_components(yield_curve_cycle, K)
        else:
            X = Utils.get_principal_components(yield_curve, K)
        return X


    @staticmethod
    def conversion_yields_to_prices(yields):
        """
            Convert bond yields to prices.
        """
        maturities = np.array([float(x)  for x in yields.columns])
        #maturities = np.array([float(x) / freq_factor for x in yields.columns])
        assert len(maturities) == yields.shape[1], "Maturities length must match the number of columns in yields"
        result = - (yields * maturities)
        return result
    
    @staticmethod
    def conversion_prices_to_yields(prices):
        """
            Convert bond prices to yields.
        """
        # Assuming `prices` is a DataFrame and `freq_factor` is defined
        #maturities = np.array([float(x) / freq_factor for x in prices.columns])
        maturities = np.array([float(x) for x in prices.columns])
        
        # Perform the element-wise multiplication
        result = - (prices / maturities)
        
        return result

    @staticmethod
    def get_init_recursion(short_term_rate, X):
        common_index = short_term_rate.index.intersection(X.index)
        model = sm.OLS((short_term_rate.loc[common_index]) , sm.add_constant(X.loc[common_index])).fit()
        delta0 = model.params.iloc[0]
        delta1 = model.params.iloc[1:].values
        return model, delta0, delta1
    

    @staticmethod
    def get_ATSM_coeffs(delta0, 
                        delta1, 
                        mu, 
                        Phi,
                        lambda0, 
                        lambda1, 
                        Sigma, 
                        sigma2, 
                        N, 
                        K, 
                        start_period):
        dim_2 = N - (start_period-1)

        # Initialise A and B
        A = np.zeros(dim_2)
        B = np.zeros((K, dim_2))
        A[0] = (-delta0 + 0.5 * (sigma2))
        B[:, 0] = (-delta1.reshape(-1))

        for n in range(1, dim_2):
            A[n] = A[n-1] \
                + np.dot(B[:, n-1].T, (mu.reshape(-1) - lambda0.reshape(-1))) \
                + 0.5 * (np.dot(np.dot(B[:, n-1].T, Sigma), B[:, n-1]) + sigma2) \
                - delta0
            B[:, n] = np.dot(B[:, n-1], (Phi.values - lambda1)) + B[:, 0]    
        return A, B
    
    @staticmethod
    def fit_price_curve(A,B,X):
        return (A.T.reshape(-1, 1) + B.T @ X.T).T
    
    @staticmethod
    def fit_yield_curve(A, B, X, TREND=False, trend=None):
        price_curve_fitted = Utils.fit_price_curve(A, B, X)
        price_curve_fitted.columns = [str(i) for i, _ in enumerate(price_curve_fitted.columns, start=1)]
        if TREND:
            yield_curve_cycle_fitted = Utils.conversion_prices_to_yields(price_curve_fitted)
            common_index = yield_curve_cycle_fitted.index.intersection(trend.index)
            yield_curve = yield_curve_cycle_fitted.loc[common_index, :].add(trend.loc[common_index].values, axis=0)
            return yield_curve
        else:
            return Utils.conversion_prices_to_yields(price_curve_fitted)

    @staticmethod
    def print_r2(pricing_factors, data):
        """
        Calculate and print R^2 scores for each column in the data.

        Parameters:
        pricing_factors (object): An object containing pricing factors data.
        data (pd.DataFrame): The target data for which to calculate R^2 scores.

        Returns:
        dict: A dictionary containing R^2 scores for each column.
        """
        X = pd.concat([pricing_factors.v, 
                       pricing_factors.data.shift(1)], 
                       axis=1).dropna()
        y = data.dropna()
        common_index = y.index.intersection(X.index)
        X_common = X.loc[common_index]
        y_common = y.loc[common_index]

        r2_scores = {}

        for column in y_common.columns:
            model = LinearRegression(fit_intercept=True)
            y_col = y_common[column].values.reshape(-1, 1)
            model.fit(X_common.values, y_col)
            
            r2 = model.score(X_common.values, y_col)
            r2_scores[column] = r2
            print(f"R^2 Score for {column}: {r2}")

        return r2_scores