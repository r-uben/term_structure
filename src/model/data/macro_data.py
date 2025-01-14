from src.paths import Paths
from src.utils.model_utils import Utils

from src.model.time.time_window import TimeWindow


import pandas as pd
class MacroData(TimeWindow, 
                Utils):
    def __init__(self,
                 time_window=None,
                 init_date=None,
                 end_date=None):
        if time_window is not None:
            super().__init__(time_window.init_date, time_window.end_date)
        elif init_date is not None and end_date is not None:
            super().__init__(init_date, end_date)
        else:
            raise ValueError("Either time_window or both init_date and end_date must be provided")
        
        Paths.__init__(self)
        Utils.__init__(self)
        self.__macro_data = None

    @property
    def data(self):
        if self.__macro_data is None:
            macro_data = pd.read_csv(Utils.find_data("raw/Q/macro_data.csv")).set_index("date")
            macro_data.index = pd.to_datetime(macro_data.index)
            macro_data.index.name = "date"
            self.__macro_data = {} 
            macro_data_short = macro_data.loc[(macro_data.index > self.init_date) & 
                                              (macro_data.index <= self.end_date)]
            macro_data_long  = macro_data.loc[(macro_data.index > self.init_date) & 
                                              (macro_data.index <= (pd.to_datetime(self.end_date) \
                                                                   + pd.DateOffset(years=10)).strftime('%Y-%m-%d'))]
            self.__macro_data["short"] = macro_data_short.sort_index()
            self.__macro_data["long"]  = macro_data_long.sort_index()
        return self.__macro_data
 
    

