#!/usr/bin/env python

import polars as pl

from lad.binarizer.cutpoint import CutpointBinarizer
from lad.featureselection.greedy import GreedySetCover
from lad.rulegenerator.eager import MaxPatterns

# Docs
__author__ = "Bha Gu"
__version__ = "0.9"


class LADClassifier:
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

    base_precision: float

        (Default = 0.5)

    base_recall: float
        (Default = 0.5)
    """

    def __init__(
        self,
        bin_size: float = 1.0,
        base_precision: float = 0.5,
        base_recall: float = 0.5,
        max_terms_in_patterns: int = 4,
        new_test: bool = False,
    ):
        self.bin_size = bin_size
        self.__base_precision = base_precision
        self.__base_recall = base_recall
        self.__max = max_terms_in_patterns
        self.model: MaxPatterns | None = None
        self.__labels = pl.Series()
        self.__new_test = new_test

    def __handle_labels(self, y: pl.Series) -> pl.Series:
        self.__labels = y.unique()
        return y.map_elements(
            lambda s: self.__labels.to_list().index(s), return_dtype=pl.UInt64
        )

    def get_labels(self) -> pl.Series:
        return self.__labels

    def fit(self, X: pl.DataFrame, y: pl.Series):
        self.is_fitted_ = True

        y = self.__handle_labels(y)

        print("# Binarization")
        cpb = CutpointBinarizer(self.bin_size)
        Xbin = cpb.fit_transform(X, y)

        print("# Feature Selection")
        gsc = GreedySetCover()
        Xbin = gsc.fit_transform(Xbin, y)

        print(Xbin.shape)
        print(Xbin.columns)

        print("# Rule building")
        self.model = MaxPatterns(
            cpb,
            gsc,
            self.__base_precision,
            self.__base_recall,
            self.__max,
            self.__new_test,
        )
        rules = self.model.fit(Xbin, y)

        print(rules)

        return self  # `fit` should always return `self`

    def predict(self, X):
        return self.model.predict(X).map_elements(
            lambda x: self.__labels[x], return_dtype=self.__labels.dtype
        )

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __str__(self):
        return self.model.__str__()
