# target statistic encoding

<div align="center">
<img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
<img alt="Language Python" src="https://img.shields.io/badge/language-Python-blue">
</div>

---



# What?

There are many means to convert categorical features to numeric ones from one-hot to embeddings. Then there are target statistic methods. These methods take statistics based on the target feature.

# Why?

Even within this simple technique there is variation in implementations. Some implement a time-mimicking approach such as Catboost to gain robustness over target leakage. However, one issue with this approach is that while it introduces some variation to the encoding, for a some samples the statistic is possibly excessively biased. This small package takes a different approach for this reason. Instead, it uses stratified folds of the training set and aggregates target statistics on each fold independently.

# How?

This is just a simple utility library that performs the following sample operation:

