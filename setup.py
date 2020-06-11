
# -*- coding: utf-8 -*-

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


import os.path

readme = ''
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'README.rst')
if os.path.exists(readme_path):
    with open(readme_path, 'rb') as stream:
        readme = stream.read().decode('utf8')


setup(
    long_description=readme,
    name='target_statistic_encoding',
    version='0.1.1',
    description='A lightweight library for encoding categorical features in your dataset with robust k-fold target statistics in training.',
    python_requires='==3.*,>=3.6.1',
    project_urls={"homepage": "https://github.com/CircArgs/target_statistic_encoding"},
    author='CircArgs',
    author_email='quebecname@gmail.com',
    license='MIT',
    packages=['target_statistic_encoding', 'target_statistic_encoding.stat_funcs'],
    package_dir={"": "."},
    package_data={},
    install_requires=['pandas==0.*,>=0.23.0', 'typing-extensions==3.*,>=3.7.4'],
    extras_require={"dev": ["black==19.*,>=19.10.0.b0", "dephell==0.*,>=0.8.3", "pylint==2.*,>=2.5.2", "pytest==3.*,>=3.0.0", "pytest-cov==2.*,>=2.4.0"]},
)
