import numpy as np


from scipy import stats
from sklearn.metrics import r2_score

import calendar
from datetime import datetime, timedelta

class Utils:

    @staticmethod
    def individual_r2_scores(X, y, model):
        predictions = model.predict(X)
        r2_scores = {}
        for i, col in enumerate(y):
            r2_scores[col] = r2_score(y[col], predictions[:, i])
        return r2_scores


    @staticmethod
    def regression_summary(X, y, model):
        predictions = model.predict(X)
        residuals = y - predictions
        mse = np.mean(residuals**2)
        n, k = X.shape
        var_b = mse * np.linalg.inv(X.T @ X)
        std_err = np.sqrt(np.diag(var_b))
        t_values = model.coef_ / std_err
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n-k-1)) for t in t_values]
        r_squared = r2_score(y, predictions)
        
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
        print("R-squared:", r_squared)
        print("Standard Errors:", std_err)
        print("P-values:", p_values)
        print("Mean Squared Error:", mse)

    @staticmethod
    def add_last_day_of_month(date_str):
        """
        Given a date in '%Y%m' format, adds the last day of the month.

        Args:
            date_str (str): Date string in '%Y%m' format.

        Returns:
            str: Date string in '%Y%m%d' format with the last day of the month.
        """

        # Parse the input date string
        date = datetime.strptime(date_str, '%Y%m')
        
        # Get the last day of the month
        _, last_day = calendar.monthrange(date.year, date.month)
        
        # Create a new date with the last day of the month
        new_date = date.replace(day=last_day)
        
        # Return the new date in '%Y%m%d' format
        return new_date.strftime('%Y%m%d')
    
    @staticmethod 
    def sum_months(date: str, months: int):
        date_obj = datetime.strptime(date, '%Y%m')
        date_sum = (date_obj + timedelta(days=(months+1)*30)).strftime('%Y%m')
        return date_sum

    