import pandas as pd
from typing import Callable, Union
from typing_extensions import Protocol


class Mean:
    def __call__(self, df: pd.DataFrame):
        return df.mean()


class Median:
    def __call__(self, df: pd.DataFrame):
        return df.median()


class Std:
    def __call__(self, df: pd.DataFrame):
        return df.std()


class Var:
    def __call__(self, df: pd.DataFrame):
        return df.var()


class Quantile:
    def __init__(self, quantile=0.5):
        self.quantile = quantile

    def __call__(self, df: pd.DataFrame):
        return df.quantile(self.quantile)


class _StatFunc(Protocol):
    def __call__(
        self, *args, **kwargs
    ) -> Callable[[pd.core.groupby.DataFrameGroupBy], Union[float, pd.Series]]:
        pass
