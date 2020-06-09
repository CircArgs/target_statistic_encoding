import pandas as pd
import numpy as np
from typing import List, Union, Callable
import warnings
from .stat_funcs import stat_funcs


class Cat2Num:
    """converts categorical features into numeric based on the frequency of the target statistic. 
        utilizes stratified splits to avoid target leakage
        
        Example:
            #given datasets for training a model - train
            #and for testing - test
            
            cat2num = Cat2Num(['example_cat_var1', 'example_cat_var1'], 'my_target')
            #for the true utility of splits as descibed above (avoiding target leakage) you should only
            #use `fit_transform` and not `fit` then `transform` on your training data
            
            fitted_train=cat2num.fit_transform(train)
            fitted_test=cat2num.transform(test)
            
            #now you may train your model using the train dataset and validate with the test
            
            model.fit(X=fitted_train.drop('my_target', axis=1), y=fitted_train['my_target'],
                        X_test=fitted_test.drop('my_target', axis=1), y_test=fitted_test['my_target']
                        )
	
    """

    def __init__(
        self,
        cat_vars: List[str],
        target_var: str,
        stat_func: stat_funcs._StatFunc = stat_funcs.Mean(),
    ):
        """
        Args:
            cat_vars (List[str]): a list of strings representing the categorical features to be encoded
            target_var (str): string of the name of the target feature in `data`
            stat_func (optional Function(*args, **kwargs) -> Function({pd.Series, pd.DataFrameGroupBy}) -> {float, pd.Series})): function which returns a closure to aggregate statistics over target - default stat_funcs.mean()
        
        Returns:
            Cat2Num object
        """
        assert (
            cat_vars
        ), "You must supply some categorical variables. `cat_vars` was empty."
        self.cat_vars = cat_vars
        self.target_var = target_var
        self.stat_func = stat_func
        self.__maps = {}
        self.__fit_transform = False
        self.__fit = False
        self.__cred_per = None

    def __assign_split(self, d, n):
        tgt = d[[self.target_var]].sort_values(self.target_var)
        tgt["__CAT2NUM_SPLIT_TEMP_VAR__"] = np.array(
            list(range(n)) * (int(len(tgt) / n) + 1)
        ).flatten()[0 : len(tgt)]
        return tgt.drop(self.target_var, axis=1)

    def __get_split_mean(self, d, v, cred):
        temp = d[[v, self.target_var, "__CAT2NUM_SPLIT_TEMP_VAR__"]]
        # check if credibility is a percentage or frequency
        cred_per = self.__check_credibility(cred)
        # get counts of levels
        n_levels = temp[v].value_counts(dropna=False, normalize=cred_per)
        # get levels that have too low credibility and wil be given the mean
        mean_levels = [v + "__CSTV__" for v in n_levels[n_levels < cred].index.tolist()]
        # design the data
        temp[v + "__CAT2NUM_SPLIT_TEMP_VAR__"] = (
            temp[v].astype(str)
            + "__CSTV__"
            + temp["__CAT2NUM_SPLIT_TEMP_VAR__"].astype(str)
        )
        temp = temp.drop([v, "__CAT2NUM_SPLIT_TEMP_VAR__"], axis=1)
        # get target statistics by level
        maps = self.stat_func(temp.groupby(v + "__CAT2NUM_SPLIT_TEMP_VAR__"))
        # get mean value for uncredible cats and replace uncreds with mean
        if mean_levels:
            map_mean = float(maps.mean())
            maps = maps[self.target_var]
            for v in mean_levels:
                maps.loc[maps.index.str.contains(v)] = map_mean
        return maps

    def __agg_split_mean(self, m):
        m = m.reset_index()
        m[m.columns[0]] = np.array([i.split("__CSTV__")[0] for i in m[m.columns[0]]])
        m = m.groupby(m.columns[0]).mean()
        return m[self.target_var]

    def __check_cats(self, d):
        miss_cat_vars = [c for c in self.cat_vars if not c in d.columns]
        if miss_cat_vars:
            raise ValueError(f"{str(miss_cat_vars)[1:-1]} not found in data.")

    def __check_target(self, d):
        if not self.target_var in d.columns:
            raise ValueError(f"`target_var` {self.target_var} not found in data.")
        assert set(d[self.target_var].unique()) == set(
            (0, 1)
        ), f"`target_var` {self.target_var} has values other than 0,1."

    def __check_column(self, v, d):
        if not v in d.columns:
            raise ValueError(f"`split` {v} not found in data.")

    def __check_credibility(self, credibility):
        if self.__cred_per is None:
            if type(credibility) == int and credibility >= 0:
                self.__cred_per = False
            elif type(credibility) == float and credibility >= 0 and credibility <= 1:
                self.__cred_per = True
            else:
                raise ValueError(
                    "`credibility` must be nonnegative int or a float in [0,1]."
                )
        return self.__cred_per

    def __str__(self):
        return f"""{'Fit' if (self.__fit or self.__fit_transform) and self.__maps else 'Unfit'}: Cat2Num(cat_vars = {self.cat_vars}, target_var = {self.target_var}, stat_func = {self.stat_func})"""

    def __repr__(self):
        return self.__str__()

    def fit(self, data: pd.DataFrame, credibility: Union[float, int] = 0):
        """
        ***This method is not typically advised.***
        
        Fit the object using solely the mean by level for the categorical variables.
        
        Args:
            data (pd.DataFrame): pandas dataframe with categorical features to fit numeric target statistic from
            credibility (float or int): 
                - if float must be in [0, 1] as % of fitting data considered credible to fit statistic to
                - if int must be >=0 as number of records in fitting data level must exist within to be credible
                - levels not above this threshold will be given the overall mean of the target statistic
                
        Returns:
            fit Cat2Num instance
        """
        if not self.__fit_transform:
            warnings.warn(
                """\nYou have yet to run `fit_transform`. \nYou are not utilizing the n-split feature and so you may see greater bias as a result of target leakage.\n"""
            )

        self.__check_cats(data)
        self.__check_target(data)
        temp = data[self.cat_vars + [self.target_var]]
        temp["__CAT2NUM_SPLIT_TEMP_VAR__"] = 0
        for c in self.cat_vars:
            m = self.__get_split_mean(temp, c, credibility)
            m = self.__agg_split_mean(m)
            self.__maps[c] = m
        self.__fit = True
        return self

    def transform(
        self,
        data: pd.DataFrame,
        drop: bool = False,
        suffix: str = "_Cat2Num",
        inplace: bool = False,
    ):
        """
        Transform a dataframes categorical features to mean target rate by category using mean over splits from 
        
        Args:
            data (pd.DataFrame): pandas dataframe with categorical features to convert to numeric target statistic
            drop (bool): drop the original columns
            suffix (str): a string to append to the end of an encoded column, default `'_Cat2Num'`
            inplace (bool): whether the transformation should be done inplace or return the transformed data, default `False`
            
        Returns:
            the passed dataframe with encoded columns added if inplace is `False` else `None`
        """
        assert (
            self.__fit or self.__fit_transform
        ) and self.__maps, "You must run fit or fit_transform first."

        self.__check_cats(data)

        temp = data
        if not inplace:
            temp = data.copy()

        for c in self.cat_vars:
            m = self.__maps[c]
            temp[c + suffix] = temp[c].astype(str).map(m)

        if drop:
            temp.drop(self.cat_vars, axis=1, inplace=True)

        return temp if not inplace else None

    def fit_transform(
        self,
        data: pd.DataFrame,
        split: str = None,
        n_splits: int = 5,
        credibility: Union[float, int] = 0,
        drop: bool = False,
        suffix: str = "_Cat2Num",
        inplace: bool = False,
    ):
        """
        Args:
            data (pd.DataFrame): pandas dataframe with categorical features to convert to numeric target statistic
            split (str): name of a column to use in the data for folding the data.
                - if this is use then n_splits is ignored
            n_splits (int): number of splits to use for target statistic
            credibility (float or int): 
                - if float must be in [0, 1] as % of fitting data considered credible to fit statistic to
                - if int must be >=0 as number of records in fitting data level must exist within to be credible
                - levels not above this threshold will be given the overall mean of the target statistic
            drop (bool): drop the original columns
            suffix (str): a string to append to the end of an encoded column, default `'_Cat2Num'`
            inplace (bool): whether the transformation should be done inplace or return the transformed data, default `False`
            
        Returns:
            the passed dataframe with encoded columns added if inplace is `False` else `None`
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.__check_cats(data)
            self.__check_target(data)

            temp = data.copy()
            if not split is None:
                self.__check_column(split, temp)
                temp["__CAT2NUM_SPLIT_TEMP_VAR__"] = (
                    temp[split].astype("category").cat.codes
                )
            else:
                temp = temp.merge(
                    self.__assign_split(temp, n_splits),
                    right_index=True,
                    left_index=True,
                    how="left",
                    copy=False,
                )
            for c in self.cat_vars:
                m = self.__get_split_mean(temp, c, credibility)
                temp[c + suffix] = (
                    temp[c].astype(str)
                    + "__CSTV__"
                    + temp["__CAT2NUM_SPLIT_TEMP_VAR__"].astype(str)
                ).map(m)
                m = self.__agg_split_mean(m)
                self.__maps[c] = m

            if drop:
                temp.drop(self.cat_vars, axis=1, inplace=True)
            temp.drop("__CAT2NUM_SPLIT_TEMP_VAR__", axis=1, inplace=True)
            self.__fit_transform = True

            if inplace:
                for v in self.cat_vars:
                    data[v + suffix] = temp[v + suffix]
                return
            return temp
