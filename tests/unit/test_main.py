import pytest
import pandas as pd
import numpy as np
import os
from target_statistic_encoding import Cat2Num, stat_funcs


def data_dir(file):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data", file)


data = pd.read_csv(data_dir("test_data.csv"))
n_splits = 5
credibility = 0.25
quantile = 0.5


@pytest.mark.parametrize(
    "stat_func",
    [
        stat_funcs.Mean,
        stat_funcs.Median,
        stat_funcs.Std,
        stat_funcs.Var,
        stat_funcs.Quantile,
    ],
)
def test_stat_funcs(stat_func):
    func_name = stat_func.__name__
    if func_name == "Quantile":
        stat_func = stat_func(quantile)
    else:
        stat_func = stat_func()
    cat2num = Cat2Num(["X1", "X2"], "target", stat_func)
    train = data[data.split == "train"]
    test = data[data.split == "test"]
    train = cat2num.fit_transform(train, n_splits=n_splits, credibility=credibility)
    test = cat2num.transform(test)
    test_data_gt = pd.read_csv(data_dir("test_data_test_{}.csv".format(func_name)))
    train_data_gt = pd.read_csv(data_dir("test_data_train_{}.csv".format(func_name)))
    assert np.isclose(
        test_data_gt[["X1_Cat2Num", "X2_Cat2Num"]].values, test[["X1_Cat2Num", "X2_Cat2Num"]].values
    ).all(), "Test data did not match for {}".format(func_name)
    assert np.isclose(
        train_data_gt[["X1_Cat2Num", "X2_Cat2Num"]].values, train[["X1_Cat2Num", "X2_Cat2Num"]].values
    ).all(), "Train data did not match for {}".format(func_name)
