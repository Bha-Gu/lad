#!/usr/bin/env python
import polars as pl


class CutpointBinarizer:
    def __init__(self):
        self.__cutpoints = []

    def get_cutpoints(self):
        return self.__cutpoints

    def fit(self, X: pl.DataFrame, y: pl.Series):
        self.__cutpoints = []

        schema = X.schema

        nominal_df = X.select(
            pl.col(col) for col, dtype in schema.items() if not dtype.is_numeric()
        )
        combined_df = X.hstack([y])

        numeric_df = combined_df.group_by("label").agg(
            [
                pl.concat_list([pl.col(col).min(), pl.col(col).max()])
                for col, dtype in schema.items()
                if dtype.is_numeric()
            ]
        )

        numeric_cols = numeric_df.columns
        nominal_cols = nominal_df.columns

        for col in X.columns:
            if col in numeric_cols:
                flattened_series = numeric_df[col].explode().sort().unique()
                trimmed_series = flattened_series[1:-1]
                self.__cutpoints.append(trimmed_series)
            if col in nominal_cols:
                self.__cutpoints.append(nominal_df[col].unique())

        return self.__cutpoints

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        Xbin = pl.DataFrame()

        for cutpoints in self.__cutpoints:
            column_name = cutpoints.name
            column = X[column_name]
            if cutpoints.dtype.is_numeric():
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
