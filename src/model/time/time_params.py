class TimeParams:
    
    def __init__(self, 
                 model: str = "ff", 
                 freq: str = "q", 
                 K: int = 5, 
                 num_quarters: int = 60):
        self.model = model
        self.freq = freq
        self.K = K
        self.num_quarters = num_quarters
        self.__freq_factor = None
        self.__freq_adjustment = None
        self.__start_period = None
        self.N = int(self.num_quarters * self.freq_adjustment)


    @property
    def freq(self):
        return self.__freq

    @freq.setter
    def freq(self, freq):
        self.__freq = freq

    @property
    def freq_factor(self):
        if self.__freq_factor is None:
            if "m" in self.freq.lower():
                self.__freq_factor = 12
            elif "q" in self.freq.lower():
                self.__freq_factor = 4
        return self.__freq_factor
 
    @property
    def freq_adjustment(self):
        if self.__freq_adjustment is None:
            self.__freq_adjustment = self.freq_factor / 4
        return self.__freq_adjustment
    
    @property
    def start_period(self):
        if self.__start_period is None:
            if "m" in self.freq.lower():
                self.__start_period = 1
            elif "q" in self.freq.lower():
                self.__start_period = 1
        return self.__start_period
    


