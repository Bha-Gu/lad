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
        self.__cp__idx = pl.DataFrame(
            [[i for i in range(len(features))]], schema=schema, orient="row"
        )
        print(self.__cp__idx)
        for feature in features:
            if feature == "label":
                continue
            col = X_tmp.select(pl.col(feature), pl.col("label"))
            a = col[feature].n_unique() == 2
            if schema[feature].is_numeric() and not a:
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
                if a:
                    self.__cutpoints.append((False, [col[feature].unique()[0]]))
                else:
                    self.__cutpoints.append((False, col[feature].unique().to_list()))

        return self.__cutpoints

    def transform_column(self, col: pl.Series, filter=None) -> pl.DataFrame:
        Xbin = pl.DataFrame()
        column_name = col.name
        (type_val, cutpoints) = self.__cutpoints[self.__cp__idx[column_name][0]]
        if type_val:
            name = f"{column_name}<={cutpoints[0]}"
            c = (col <= cutpoints[0]).alias(name)
            if filter is None or name in filter:
                Xbin = Xbin.hstack([c])

            length = len(cutpoints)
            for i in range(length - 1):
                j = i + 1
                name = f"{column_name}->({cutpoints[i]}<->{cutpoints[j]})"
                c = ((col > cutpoints[i]) & (col <= cutpoints[j])).alias(name)

                if filter is None or name in filter:
                    Xbin = Xbin.hstack([c])

            name = f"{cutpoints[length - 1]}<={column_name}"
            c = (cutpoints[length - 1] <= col).alias(name)

            if filter is None or name in filter:
                Xbin = Xbin.hstack([c])
        else:
            for value in cutpoints:
                name = f"{column_name}={value}"
                c = (col == value).alias(name)

                if filter is None or name in filter:
                    Xbin = Xbin.hstack([c])

        return Xbin

    def transform(self, X: pl.DataFrame, filter=None) -> pl.DataFrame:
        Xbin = []

        for column_name in X.columns:
            column = X[column_name]
            Xbin.append(self.transform_column(column, filter))

        return pl.concat(Xbin, how="horizontal")

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
