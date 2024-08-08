#!/usr/bin/env python
import numpy as np


class CutpointBinarizer:
    # TODO implement Double Sided Binarisation
    def __init__(self, tolerance=1.0, double_binarization=True):
        self.__tolerance = tolerance
        self.__cutpoints = []
        self.__db = double_binarization

    def get_cutpoints(self):
        return self.__cutpoints

    def fit(self, X, y):
        self.__cutpoints = []

        feature_count = X.shape[1]
        y = np.array(y)
        X = np.array(X)

        for feature in range(feature_count):
            column = X.T[feature]
            values = np.unique(column)

            values_type = values.dtype

            if np.issubdtype(values_type, np.number):
                sorted_values = sorted(values)
                delta = 0
                prev_value = sorted_values[0]
                for value in sorted_values[1:]:
                    delta += value - prev_value
                    prev_value = value

                tolerance = delta / len(sorted_values)

                if len(sorted_values) <= 2:
                    tolerance = 0.0

                cutpoints = []
                prev_labels = None  # Previuos labels
                prev_value = None  # Previuos xi
                count = 1
                # Finding transitions
                for value in sorted_values:
                    # Classes where v appears
                    indexes = np.where(column == value)
                    labels = set(y[indexes[0]].flatten())
                    # Main condition
                    if prev_labels is not None:
                        variation = value - prev_value  # Current - Previous
                        if variation > tolerance * self.__tolerance / count:
                            # Testing for transition
                            if (
                                len(prev_labels) > 1 or len(labels) > 1
                            ) or prev_labels != labels:
                                count = 0
                                cutpoints.append(prev_value + variation / 2.0)

                        count += 1

                    prev_labels = labels
                    prev_value = value

                self.__cutpoints.append((True, cutpoints))

            else:
                self.__cutpoints.append((False, values))

        return self.__cutpoints

    def transform(self, X):
        Xbin = np.empty((X.shape[0], 0), bool)
        X = np.array(X)
        print(self.__cutpoints)
        for att, (type_data, cutpoints) in enumerate(self.__cutpoints):
            print("transform", att)
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
                for value in cutpoints:
                    row = X.T[att]
                    row = row == value
                    Xbin = np.hstack((Xbin, row))
        print(Xbin.shape)
        return Xbin

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
        # self.__types_number = []
        # self.__cutpoints = []
        # self.__mutator = []
        #
        # feature_count = X.shape[1]
        # y = np.array(y)
        # X = np.array(X)
        #
        # for feature in range(feature_count):
        #     column = X[:, feature]
        #     prev_label = None  # Previuos label
        #     prev_value = None  # Previuos xi
        #
        #     values = np.unique(column)
        #
        #     values_type = values.dtype
        #
        #     if np.issubdtype(values_type, np.number):
        #         self.__types_number.append(True)
        #         self.__mutator.append([])
        #     else:
        #         self.__types_number.append(False)
        #         self.__mutator.append(values)
        #         self.__cutpoints.append([])
        #         continue
        #
        #     __cutpoints = []
        #
        #     sorted_values = sorted(values)
        #     print("values", sorted_values)
        #     delta = 0
        #     prev = sorted_values[0]
        #     for v in sorted_values[1:]:
        #         delta += v - prev
        #         prev = v
        #
        #     __tolerance = delta / len(sorted_values)
        #
        #     if len(sorted_values) <= 2:
        #         __tolerance = 0.0
        #
        #     count = 1
        #     # Finding transitions
        #     for value in sorted_values:
        #         # Classes where v appears
        #         indexes = np.where(column == value)
        #         label = set(y[indexes[0]].flatten())
        #         # Main condition
        #         if prev_label is not None:
        #             variation = value - prev_value  # Current - Previous
        #             if variation > __tolerance * self.__tolerance / count:
        #                 # Testing for transition
        #                 if (
        #                     len(prev_label) > 1 or len(label) > 1
        #                 ) or prev_label != label:
        #                     count = 1
        #                     __cutpoints.append(prev_value + variation / 2.0)
        #                 else:
        #                     count += 1
        #             else:
        #                 count += 1
        #
        #         prev_label = label
        #         prev_value = value
        #
        #     self.__cutpoints.append(__cutpoints)
        #
        # Xbin = np.empty((X.shape[0], 0), bool)
        # X = np.array(X)
        # for att, (cutpoints, type_data, values) in enumerate(
        #     zip(self.__cutpoints, self.__types_number, self.__mutator)
        # ):
        #     print("transform", att)
        #     if type_data:
        #         for cutpoint in cutpoints:
        #             # Binarizing
        #             row = X.T[att]
        #
        #             row = row.reshape(X.shape[0], 1) <= cutpoint
        #             Xbin = np.hstack((Xbin, row))
        #         if self.__db:
        #             length = len(cutpoints)
        #             for i in range(length):
        #                 for j in range(i + 1, length):
        #                     row = X.T[att]
        #                     row = (
        #                         row.reshape(X.shape[0], 1) > cutpoints[i]
        #                     ) <= cutpoints[j]
        #                     Xbin = np.hstack((Xbin, row))
        #     else:
        #         for value in values:
        #             row = X.T[att]
        #             print(value)
        #             row = row == value
        #             Xbin = np.hstack((Xbin, row))
        #
        # return Xbin
