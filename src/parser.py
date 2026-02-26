import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Optional
from pathlib import Path


class Parser:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.data = DataFrame()
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        self.load()

    def load(self, sep: Optional[str] = "\t"):
        self.data = pd.read_csv(self.filepath, sep=sep)
        if "Time" in self.data.columns:
            self.data["Time"] = pd.to_datetime(self.data["Time"], unit="s")
            self.data.sort_values("Time", inplace=True)
        return self

    def get_metrics(self):
        exclude = {"phase", "Time", "wtime"}
        return [col for col in self.data.columns if col not in exclude]

    def get_feature_matrix(self, columns = None) -> np.ndarray:
        if columns is None:
            columns = self.get_metrics()
        return self.data[columns].to_numpy()
