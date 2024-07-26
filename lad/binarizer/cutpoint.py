#!/usr/bin/env python
import numpy as np


class CutpointBinarizer:
    # TODO implement Double Sided Binarisation
    def __init__(self, tolerance=0.0):
        self.__tolerance = tolerance
        self.__cutpoints = []

    def get_cutpoints(self):
        return self.__cutpoints

    def fit(self, X, y):
        self.__cutpoints = []

        cc = X.shape[1]

        for att in range(cc):
            row = X[:, att]
            labels = None  # Previuos labels
            u = None  # Previuos xi

            __cutpoints = []

            values = sorted(np.unique(row))
            print(values)
            delta = 0
            last = values[0]
            for v in values[1:]:
                delta += v - last
                last = v

            __tolerance = delta / len(values)

            if len(values) <= 2:
                __tolerance = 0.0

            # Finding transitions
            for v in values:
                # Classes where v appears
                indexes = np.where(row == v)[0]
                __labels = set(y[indexes])

                # Main condition
                if labels is not None:
                    variation = v - u  # Current - Previous
                    if variation > __tolerance * self.__tolerance:

                        # Testing for transition
                        if (len(labels) > 1 or len(__labels) > 1) or labels != __labels:
                            __cutpoints.append(u + variation / 2.0)

                labels = __labels
                u = v

            self.__cutpoints.append(__cutpoints)

        return self.__cutpoints

    def transform(self, X):
        Xbin = np.empty((X.shape[0], 0), bool)

        for att, cutpoints in enumerate(self.__cutpoints):
            for cutpoint in cutpoints:
                # Binarizing
                row = X[:, att]
                row = row.reshape(X.shape[0], 1) <= cutpoint
                Xbin = np.hstack((Xbin, row))

        return Xbin

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
