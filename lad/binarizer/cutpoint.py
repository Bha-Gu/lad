#!/usr/bin/env python
import polars as pl


class CutpointBinarizer:
    """
    CutpointBinarizer

    Implements a cutpoint generation method that is performed per column
    The base algorithm is subject to change

    Attributes
    ---------
    tolerance: float
    Tolerance for cutpoint generation. A cutpoint will only be generated
    between two values if they compoundly differ by at least this value. (Default = 1.0)
    """

    def __init__(self, tolerance=1.0):
        self.__cutpoints = []
        self.__tolerance = tolerance

    def get_cutpoints(self):
        return self.__cutpoints

    def fit(self, X: pl.DataFrame, y: pl.Series):
        self.__cutpoints = []
        self.__tolerance = 1.0
        X_tmp = X.hstack([y])
        schema = X.schema
        # print(schema)
        features = X.columns
        for feature in features:
            if feature == "label":
                continue
            col = X_tmp.select(pl.col(feature), pl.col("label"))
            if schema[feature].is_numeric():
                sorted_values = col.sort(feature)
                delta = sorted_values[feature].diff(null_behavior="drop").sum()
                tolerance = (
                    delta / len(sorted_values) if len(sorted_values) > 2 else 0.0
                )
                cutpoints = []
                prev_labels = None
                prev_value = None
                count = 1
                labels = set()

                for value, label in sorted_values.rows():
                    if value != prev_value:
                        labels = set()
                    labels.add(label)

                    if prev_labels is not None:
                        variation = value - prev_value
                        if variation > tolerance * self.__tolerance / count:
                            if (
                                len(prev_labels) > 1 or len(labels) > 1
                            ) or prev_labels != labels:
                                count = 0
                                cutpoints.append(prev_value + variation / 2.0)

                        count += 1

                    prev_labels = labels
                    prev_value = value
                if len(cutpoints) == 0:
                    cutpoints.append(sorted_values[feature].mean())
                self.__cutpoints.append((True, cutpoints))
            else:
                self.__cutpoints.append((False, col[feature].to_list()))

        return self.__cutpoints

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        Xbin = pl.DataFrame()

        for column_name, (type_val, cutpoints) in zip(X.columns, self.__cutpoints):
            column = X[column_name]
            if type_val:
                col = (column <= cutpoints[0]).alias(f"{column_name}<={cutpoints[0]}")
                Xbin = Xbin.hstack([col])

                length = len(cutpoints)
                for i in range(length - 1):
                    j = i + 1
                    col = ((column > cutpoints[i]) & (column <= cutpoints[j])).alias(
                        f"{column_name}->({cutpoints[i]}<->{cutpoints[j]})"
                    )
                    Xbin = Xbin.hstack([col])

                col = (cutpoints[-1] <= column).alias(f"{cutpoints[-1]}<={column_name}")
                Xbin = Xbin.hstack([col])
            else:
                for value in cutpoints:
                    col = (column == value).alias(f"{column_name}={value}")
                    Xbin = Xbin.hstack([col])

        return Xbin

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
