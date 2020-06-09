import pandas as pd
from typing import Callable, Union
from typing_extensions import Protocol


def mean():
    def stat_func(df: pd.DataFrame):
        return df.mean()

    return stat_func


def median():
    def stat_func(df: pd.DataFrame):
        return df.median()

    return stat_func


def std():
    def stat_func(df: pd.DataFrame):
        return df.std()

    return stat_func


def var():
    def stat_func(df: pd.DataFrame):
        return df.var()

    return stat_func


def quantile(quantile=0.5):
    def stat_func(df: pd.DataFrame):
        return df.quantile(quantile)

    return stat_func


class _StatFunc(Protocol):
    def __call__(
        self, *args, **kwargs
    ) -> Callable[[pd.core.groupby.DataFrameGroupBy], Union[float, pd.Series]]:
        pass
