#!/usr/bin/env python

"""
References:
https://scikit-learn.org/stable/developers/develop.html
https://sklearn-template.readthedocs.io/en/latest/quick_start.html
"""

import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from lad.binarizer.cutpoint import CutpointBinarizer
from lad.featureselection.greedy import GreedySetCover
from lad.rulegenerator.eager import MaxPatterns

# Docs
__author__ = "Vaux Gomes"
__version__ = "0.4"


class LADClassifier(BaseEstimator, ClassifierMixin):
    """
    LAD Classifier

    Implements the Maximized Prime Patterns heuristic described in the
    "Maximum Patterns in Datasets" paper. It generates one pattern (rule)
    per observation, while attempting to: (i) maximize the coverage of other
    observations belonging to the same class, and (ii) preventing the
    coverage of too many observations from outside that class. The amount of
    "outside" coverage allowed is controlled by the minimum purity parameter
    (from the main LAD classifier).

    Attributes
    ---------
    tolerance: float
        Tolerance for cutpoint generation. A cutpoint will only be generated
        between two values if they differ by tat least this value. (Default = 1.0)

    fp_tolerance: float
        (Default = 0.5)

    fn_tolerance: float
        (Default = 0.5)
    """

    def __init__(self, tolerance=1.0, fp_tolerance=0.5, fn_tolerance=0.5):
        self.tolerance = tolerance
        self.__fp_tolerance = fp_tolerance
        self.__fn_tolerance = fn_tolerance
        self.model = None

    def fit(self, X: pl.DataFrame, y: pl.Series):
        # X, y = check_X_y(X, y.to_list(), accept_sparse=True)
        self.is_fitted_ = True

        print("# Binarization")
        cpb = CutpointBinarizer(self.tolerance)
        Xbin = cpb.fit_transform(X, y)

        print(Xbin.shape)
        print(Xbin.columns)

        print("# Feature Selection")
        gsc = GreedySetCover()
        Xbin = gsc.fit_transform(Xbin, y)

        print(Xbin.shape)
        print(Xbin.columns)

        print("# Rule building")
        self.model = MaxPatterns(cpb, gsc, self.__fp_tolerance, self.__fn_tolerance)
        rules = self.model.fit(Xbin, y)

        print(rules)

        return self  # `fit` should always return `self`

    def predict(self, X):
        # X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        return self.model.predict(X)

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        return self.model.predict_proba(X)

    def __str__(self):
        return self.model.__str__()
