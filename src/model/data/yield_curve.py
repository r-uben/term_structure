import pandas as pd
from src.utils.model_utils import Utils
from src.model.time.time_params import TimeParams
from src.model.time.time_window import TimeWindow



class YieldCurve(Utils):

    def __init__(self, time_params, time_window):

        Utils.__init__(self)
        
        self.__init_date = time_window.init_date
        self.__end_date = time_window.end_date
        self.__N = time_params.N
        self.__freq = time_params.freq
        self.__yield_curve = None

    def load(self, freq=None):
        return pd.read_csv(Utils.find_data(f"raw/{self.__freq[0].upper()}/yield_curve.csv"))
    
    @property
    def data(self):
        if self.__yield_curve is None:
            self.__yield_curve = {}
            yield_curve_df = self.load()
            yield_curve_df = yield_curve_df.set_index(pd.to_datetime(yield_curve_df["date"])).drop(columns=["date"])
            yield_curve_df = yield_curve_df.loc[:, [x for x in yield_curve_df.columns if int(x) <= self.__N]]
            self.__yield_curve["short"] = yield_curve_df.loc[self.__init_date:self.__end_date]
            self.__yield_curve["long"]  = yield_curve_df.loc[self.__init_date:]

        return self.__yield_curve