#!/usr/bin/env python
import polars as pl


class CutpointBinarizer:
    def __init__(self, tolerance=1.0):
        self.__tolerance = tolerance
        self.__cutpoints = []

    def get_cutpoints(self):
        return self.__cutpoints

    def fit(self, X: pl.DataFrame, y: pl.Series):
        self.__cutpoints = []

        for column in X.get_columns():
            values_type = column.dtype.is_numeric()

            if values_type:
                sorted_values = column.unique().sort()
                delta = sorted_values.diff(null_behavior="drop").sum()

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
                    indexes = (column == value).arg_true()
                    labels = set(y[indexes].to_list())
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
                self.__cutpoints.append((False, column.to_list()))

        return self.__cutpoints

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        Xbin = pl.DataFrame()

        print(self.__cutpoints)
        for column, (type_data, cutpoints) in zip(X.get_columns(), self.__cutpoints):
            column_name = column.name

            if type_data:
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

        print(Xbin.shape)
        return Xbin

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
