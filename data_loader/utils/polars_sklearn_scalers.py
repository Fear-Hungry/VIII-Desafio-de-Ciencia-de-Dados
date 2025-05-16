import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from typing import List, Optional

class PolarsStandardScaler:
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, df: pl.DataFrame):
        cols = self.columns or df.columns
        X = df.select(cols).to_numpy()
        self.scaler.fit(X)
        self.columns = cols
        self.fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise RuntimeError("Scaler não foi ajustado (fit) ainda.")
        X = df.select(self.columns).to_numpy()
        X_scaled = self.scaler.transform(X)
        df_scaled = df.with_columns([
            pl.Series(name, X_scaled[:, idx]) for idx, name in enumerate(self.columns)
        ])
        return df_scaled

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str):
        joblib.dump({
            'scaler': self.scaler,
            'columns': self.columns
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls(columns=data['columns'])
        obj.scaler = data['scaler']
        obj.fitted = True
        return obj

class PolarsMinMaxScaler:
    def __init__(self, columns: Optional[List[str]] = None, feature_range=(-1, 1)):
        self.columns = columns
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.fitted = False

    def fit(self, df: pl.DataFrame):
        cols = self.columns or df.columns
        X = df.select(cols).to_numpy()
        self.scaler.fit(X)
        self.columns = cols
        self.fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise RuntimeError("Scaler não foi ajustado (fit) ainda.")
        X = df.select(self.columns).to_numpy()
        X_scaled = self.scaler.transform(X)
        df_scaled = df.with_columns([
            pl.Series(name, X_scaled[:, idx]) for idx, name in enumerate(self.columns)
        ])
        return df_scaled

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str):
        joblib.dump({
            'scaler': self.scaler,
            'columns': self.columns
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls(columns=data['columns'])
        obj.scaler = data['scaler']
        obj.fitted = True
        return obj
