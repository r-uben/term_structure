from datetime import datetime
import pandas as pd

class TimeWindow:
    def __init__(self, 
                 init_date, 
                 end_date):
        self.__init_date = init_date
        self.__end_date = end_date
        self.__init_year = None 
        self.__end_year = None 

    @property
    def init_date(self):
        if type(self.__init_date) == str:
            self.__init_date = pd.to_datetime(self.__init_date)
        return self.__init_date
    
    @property
    def end_date(self):
        if type(self.__end_date) == str:
            self.__end_date = pd.to_datetime(self.__end_date)
        elif self.__end_date is None:
            self.__end_date = datetime.today()
        return self.__end_date
    
    @property
    def time_window(self):
        return self.init_date, self.end_date
    
    @property
    def init_year(self):
        if self.__init_year is None:
            self.__init_year = self.init_date.year
        return self.__init_year
    
    @property
    def end_year(self):
        if self.__end_year is None:
            self.__end_year = self.end_date.year
        return self.__end_year