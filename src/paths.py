import os
from pathlib import Path


class Paths:
    def __init__(self):
        self.__data_path = None
        self.__results_path = None

    def ensure_directories_exist(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_path,
            self.data_path / "processed",
            self.data_path / "figures",
            self.raw_path,
            self.results_path,
            self.econ_rev
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def select_data_path(self) -> Path:
        path = Path("data")
        while not path.exists():
            path = Path("..") / path
        return path
    

    @property
    def data_path(self):
        if self.__data_path is None:
            self.__data_path = self.select_data_path()
        return self.__data_path
    
    @property
    def raw_path(self):
        return self.data_path / "raw"

    @property
    def results_path(self):
        if self.__results_path is None:
            self.__results_path = self.data_path.parent / "results"
        return self.__results_path
    
    @property
    def econ_rev(self):
        return self.data_path / "econ_rev"
