from src.model.data.macro_data import MacroData
from src.model.data.yield_curve import YieldCurve
from src.model.time.time_window import TimeWindow
from src.paths import Paths
from src.utils.model_utils import Utils

import statsmodels.api as sm

import pandas as pd
import numpy as np

class CommonTrend(TimeWindow):

    def __init__(self, 
                 time_params,
                 time_window=None, 
                 init_date=None, 
                 end_date=None):
        
        if time_window is not None:
            super().__init__(time_window.init_date, time_window.end_date)
        elif init_date is not None and end_date is not None:
            super().__init__(init_date, end_date)
        else:
            raise ValueError("Either time_window or both init_date and end_date must be provided")

        self.macro_data = MacroData(time_window=self)
        self.yield_curve = YieldCurve(time_params=time_params, time_window=self)
        
        self.__RHS = None
        self.__LHS = None
        self.__model = None
        self.freq = time_params.freq
        self.freq_factor = time_params.freq_factor

    @property
    def F_colnames(self):
        return ["MY", "dy_pot", "pi_LR"]
    
    @property
    def RHS(self):
        if self.__RHS is None:
            self.__RHS = pd.concat([pd.to_numeric(self.macro_data.data["short"][col]) \
                                    for col in self.F_colnames], axis=1)
        return self.__RHS
    
    @property
    def LHS(self):
        if self.__LHS is None:
            self.__LHS = pd.to_numeric(self.yield_curve.data["short"].loc[:, str(1)]).to_frame()
            self.__LHS = self.__LHS.div(4)  # not annualised...
        return self.__LHS
    
    @property
    def model(self):
        if self.__model is None:
            self.__model = self.get_model(self.LHS, self.RHS)
        return self.__model
    
    def get_model(self, LHS, RHS):
        common_index = LHS.index.intersection(RHS.index)
        X_common = RHS.loc[common_index]
        y_common = LHS.loc[common_index]
    
        
        model = sm.OLS(y_common, X_common)
        
        # Check in-sample predictions
        fitted_model = model.fit()
  
        return fitted_model
    
    def forecast(self, h=12, forecast_date=None):
        """
        Generate forecasts using the fitted model and add them as new rows with a 'Type' column.
        
        Parameters:
        h (int): Number of periods to forecast into the future. Default is 12.
        """

         # Determine the forecast start date
        if forecast_date is None:
            forecast_date = self.RHS.index[-1] + pd.offsets.QuarterEnd()

            # Get the fitted model
            fitted_model = self.model

            # Generate in-sample predictions
            in_sample_predictions = fitted_model.predict(self.RHS)*4

            # Combine the in-sample predictions with the RHS data. THIS IS ALWAYS QUARTERLY DATA
            combined_df = pd.concat([pd.Series(in_sample_predictions, index=self.RHS.index, name='value') , self.RHS], axis=1)
            combined_df = combined_df.loc[:, "value"].to_frame() 
            combined_df['Type'] = 'Actual'

        else:
            # Ensure forecast_start is at the end of a quarter
            forecast_date  = forecast_date + pd.offsets.QuarterEnd()

            # Get the fitted model
            RHS = pd.concat([pd.to_numeric(self.macro_data.data["long"][col]) \
                                    for col in self.F_colnames], axis=1).loc[:forecast_date]
            LHS =pd.to_numeric(self.yield_curve.data["long"].loc[:, str(1)]).to_frame().loc[:forecast_date]
            fitted_model = self.get_model(LHS, RHS)

            # Generate in-sample predictions
            in_sample_predictions = fitted_model.predict(RHS)*4
            # Combine the in-sample predictions with the RHS data. THIS IS ALWAYS QUARTERLY DATA
            combined_df = pd.concat([pd.Series(in_sample_predictions, index=RHS.index, name='value') , RHS], axis=1)
            combined_df = combined_df.loc[:, "value"].to_frame() 
            combined_df['Type'] = 'Actual'


        # Create future dates for forecasting. QUARTERLY FREQUENCY
        if "m" in self.freq.lower(): h /= 3 # adjust periods to quarterly frequency if monthly frequency
        future_dates = pd.date_range(start=forecast_date, periods=h, freq="QE-DEC")
        
        # Get future macro data. QUARTERLY FREQUENCY
        future_macro = self.macro_data.data["long"].reindex(future_dates)[self.F_colnames]
        # Generate out-of-sample predictions. QUARTERLY FREQUENCY
        future_predictions = fitted_model.predict(future_macro)*4 # annualised

        # Create a DataFrame with all predictions
        forecast_df = pd.DataFrame(index=future_dates, columns=['value'] + list(self.RHS.columns))
        forecast_df.loc[future_dates, 'value'] = future_predictions
        forecast_df = forecast_df.loc[:, "value"].to_frame()
        #forecast_df.loc[future_dates, self.RHS.columns] = future_macro
        forecast_df['Type'] = 'Forecast'
        

        # Combine actual and forecast data
        result_df = pd.concat([combined_df, forecast_df], axis=0)
        
        # Sort the DataFrame by index to ensure chronological order
        result_df = result_df.sort_index()
        
        # Interpolate all numeric columns at monthly frequency if needed
        if "m" in self.freq.lower():
            # Find the last 'Actual' date
            last_actual_date = result_df[result_df['Type'] == 'Actual'].index[-1]
            
            # Resample to monthly frequency
            result_df = result_df.resample('ME').last()
            
            # Ensure the last actual value is included
            if last_actual_date not in result_df.index:
                last_actual_row = result_df.loc[result_df.index < last_actual_date].iloc[-1].copy()
                last_actual_row.name = last_actual_date
                result_df = pd.concat([result_df.loc[:last_actual_date], 
                                       pd.DataFrame([last_actual_row]), 
                                       result_df.loc[last_actual_date:]]).drop_duplicates()
            
            # Interpolate the 'value' column
            result_df["value"] = result_df["value"].astype(float).interpolate(method='linear')
            
            # Set 'Type' based on the last actual date
            result_df['Type'] = np.where(result_df.index <= last_actual_date, 'Actual', 'Forecast')

        return result_df