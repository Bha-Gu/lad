#!/usr/bin/env python
import numpy as np


class CutpointBinarizer:
    # TODO implement Double Sided Binarisation
    def __init__(self, tolerance=1.0, double_binarization=True):
        self.__tolerance = tolerance
        self.__types_number = []
        self.__cutpoints = []
        self.__mutator = []
        self.__db = double_binarization

    def get_cutpoints(self):
        return self.__cutpoints

    def fit(self, X, y):
        self.__cutpoints = []

        cc = X.shape[1]

        for att in range(cc):
            row = X.T[att]
            labels = None  # Previuos labels
            u = None  # Previuos xi
            y = np.array(y)

            values = np.unique(row)

            values_type = values.dtype

            if np.issubdtype(values_type, np.number):
                self.__types_number.append(True)
                self.__mutator.append([])
            else:
                self.__types_number.append(False)
                self.__mutator.append(values)
                self.__cutpoints.append([])
                continue

            __cutpoints = []

            sorted_values = sorted(values)
            print("values", sorted_values)
            delta = 0
            last = sorted_values[0]
            for v in sorted_values[1:]:
                delta += v - last
                last = v

            __tolerance = delta / len(sorted_values)

            if len(sorted_values) <= 2:
                __tolerance = 0.0

            count = 1
            # Finding transitions
            for v in sorted_values:
                # Classes where v appears
                indexes = np.where(row == v)[0]
                print(indexes)
                __labels = set(y[indexes].flatten())
                print(__labels)
                # Main condition
                if labels is not None:
                    variation = v - u  # Current - Previous
                    if variation > __tolerance * self.__tolerance / count:

                        # Testing for transition
                        if (len(labels) > 1 or len(__labels) > 1) or labels != __labels:
                            count = 1
                            __cutpoints.append(u + variation / 2.0)
                        else:
                            count += 1
                    else:
                        count += 1

                labels = __labels
                u = v

            self.__cutpoints.append(__cutpoints)

        return self.__cutpoints

    def transform(self, X):
        Xbin = np.empty((X.shape[0], 0), bool)

        for att, (cutpoints, type_data, values) in enumerate(
            zip(self.__cutpoints, self.__types_number, self.__mutator)
        ):
            if type_data:
                for cutpoint in cutpoints:
                    # Binarizing
                    row = X.T[att]
                    row = row.reshape(X.shape[0], 1) <= cutpoint
                    Xbin = np.hstack((Xbin, row))
                if self.__db:
                    length = len(cutpoints)
                    for i in range(length):
                        for j in range(i + 1, length):
                            row = X.T[att]
                            row = (
                                row.reshape(X.shape[0], 1) > cutpoints[i]
                            ) <= cutpoints[j]
                            Xbin = np.hstack((Xbin, row))
            else:
                for value in values:
                    row = X.T[att]
                    row = row == value
                    Xbin = np.hstack((Xbin, row))

        return Xbin

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
