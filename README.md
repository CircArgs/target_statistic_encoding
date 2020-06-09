# target statistic encoding

<div align="center">
<img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
<img alt="Language Python" src="https://img.shields.io/badge/language-Python-blue">
</div>

---

# Install

### from pypi

`pip install target_statistic_encoding`

### from source

`python -m pip install git+https://github.com/CircArgs/target_statistic_encoding.git`

# What?

There are many means to convert categorical features to numeric ones from one-hot to embeddings. Then there are target statistic methods. These methods take statistics based on the target feature.

# Why?

Even within this simple technique there is variation in implementations. Some implement a time-mimicking approach such as Catboost to gain robustness over target leakage. However, one issue with this approach is that while it introduces some variation to the encoding, for a some samples the statistic is possibly excessively biased. This small package takes a different approach for this reason. Instead, it uses stratified folds of the training set and aggregates target statistics on each fold independently.

# Benefits of this implementation

- stratified split target statistic helps prevent target leakage thus making your models more robust
- credibility factor allows categories with low support to be ignored additionally making your models more robust
- clean api
- variety of target statistic functions in addition to allowing custom implemented ones
- easy productionalization - everything is 100% serializable with pickle
    ex. 
    ```python
    #Ready for test time
    pd.to_pickle(cat2num, "cat2num_for_production.pkl")
    ```

# How?

This is just a simple utility library that performs the following sample operation:
[See this example notebook](examples/example.ipynb)

_keep in mind this is simply an example. The example target is random here so no real signal is expected_
![example usage](assets/example.png)

# API

## Instantiate

```python
Init signature:
Cat2Num(
    cat_vars: List[str],
    target_var: str,
    stat_func: target_statistic_encoding.stat_funcs.stat_funcs._StatFunc = <function mean.<locals>.stat_func at 0x7fea58a85950>,
)
Args:
    cat_vars (List[str]): a list of strings representing the categorical features to be encoded
    target_var (str): string of the name of the target feature in `data`
    stat_func (optional Function(*args, **kwargs) -> Function({pd.Series, pd.DataFrameGroupBy}) -> {float, pd.Series})): function which returns a closure to aggregate statistics over target - default stat_funcs.mean()
```

## fit

### prefer`.fit_transform` on your training set

**_Note: running `.fit` followed by `.transform` on your training set is not equivalent to simply running `.fit_transform`. There wil be no differentiation amongst category statistics as they will all be mapped to the mean._**

```python
cat2num.fit_transform(
    data: pandas.core.frame.DataFrame,
    split: str = None,
    n_splits: int = 5,
    credibility: Union[float, int] = 0,
    drop: bool = False,
    suffix: str = '_Cat2Num',
    inplace: bool = False,
)

Args:
    data (pd.DataFrame): pandas dataframe with categorical features to convert to numeric target statistic
    split (str): name of a column to use in the data for folding the data.
        - if this is use then n_splits is ignored
    n_splits (int): number of splits to use for target statistic
    credibility (float or int):
        - if float must be in [0, 1] as % of fitting data considered credible to fit statistic to
        - if int must be >=0 as number of records in fitting data level must exist within to be credible
        - levels not above this threshold will be given the overall target mean
    drop (bool): drop the original columns
    suffix (str): a string to append to the end of an encoded column, default `'_Cat2Num'`
    inplace (bool): whether the transformation should be done inplace or return the transformed data, default `False`

Returns:
    the passed dataframe with encoded columns added if inplace is `False` else `None`
```

```python
cat2num.fit(
    data: pandas.core.frame.DataFrame,
    credibility: Union[float, int] = 0,
)

Args:
    data (pd.DataFrame): pandas dataframe with categorical features to fit numeric target statistic from
    credibility (float or int):
        - if float must be in [0, 1] as % of fitting data considered credible to fit statistic to
        - if int must be >=0 as number of records in fitting data level must exist within to be credible
        - levels not above this threshold will be given the overall target mean

Returns:
    fit Cat2Num instance
```

### use `.transform` on your **non-training** set

```python
cat2num.transform(
    data: pandas.core.frame.DataFrame,
    drop: bool = False,
    suffix: str = '_Cat2Num',
    inplace: bool = False,
)

Args:
    data (pd.DataFrame): pandas dataframe with categorical features to convert to numeric target statistic
    drop (bool): drop the original columns
    suffix (str): a string to append to the end of an encoded column, default `'_Cat2Num'`
    inplace (bool): whether the transformation should be done inplace or return the transformed data, default `False`

Returns:
    the passed dataframe with encoded columns added if inplace is `False` else `None`
```

# Custom target statistic functions

You may optionally opt for a target statistic based on a statistic other than the mean although this is usually unwanted/unnecessary.

Several are included and you can implement your own with a few considerations.

### Given:

- mean (`target_statistic_encoding.stat_funcs.Mean()`) - the default
- median (`target_statistic_encoding.stat_funcs.Median()`)
- std (`target_statistic_encoding.stat_funcs.Std()`)
- var (`target_statistic_encoding.stat_funcs.Var()`)
- quantile (`target_statistic_encoding.stat_funcs.Quantile(quantile=0.5)`)

### Implement your own:

You may optionally implement your own target statistic function. It must be a callable that operates on the `pandas.core.groupby.DataFrameGroupby` type i.e. the result of a `pandas.DataFrame.groupby` e.g.: something akin to

<table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>target</th>    </tr>    <tr>      <th>X1</th>      <th></th>    </tr>  </thead>  <tbody>    <tr>      <th>a</th>      <td>0.287356</td>    </tr>    <tr>      <th>b</th>      <td>0.298795</td>    </tr>    <tr>      <th>c</th>      <td>0.336879</td>    </tr>    <tr>      <th>d</th>      <td>0.287037</td>    </tr>  </tbody></table>
