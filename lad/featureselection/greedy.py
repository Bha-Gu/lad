import polars as pl


class GreedySetCover:
    """
    Feature Selection

    Basic class based feture selection optimized for binary data
    Threshold is based of class count (subject to change)
    """

    def __init__(self, in_place: bool = False):
        self.__removed = []
        self.__inplace = in_place

    def get_removed(self):
        return pl.Series("Removed", self.__removed)

    def fit(self, Xbin: pl.DataFrame, y: pl.Series):
        features = Xbin.columns
        feature_count: int = len(features)

        for i in range(feature_count):
            if i in self.__removed:
                continue
            current_col: pl.Series = Xbin[features[i]]

            for j in range(i + 1, feature_count):
                if j in self.__removed:
                    continue
                comparison_col: pl.Series = Xbin[features[j]]

                # Vectorized XOR and NOT operations
                a: pl.Series = current_col ^ comparison_col  # XOR operation
                b: pl.Series = ~a  # NOT XOR

                if a.all() or b.all():
                    self.__removed.append(j)

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        list_of_rejected = [X.columns[i] for i in self.__removed]
        if self.__inplace:
            for col_name in list_of_rejected:
                X.drop_in_place(col_name)
            return X
        else:
            return X.select(pl.col("*").exclude(list_of_rejected))

    def fit_transform(self, Xbin, y) -> pl.DataFrame:
        features = Xbin.columns
        feature_count: int = len(features)

        for i in range(feature_count):
            if i in self.__removed:
                continue
            current_col: pl.Series = Xbin[features[i]]

            for j in range(i + 1, feature_count):
                if j in self.__removed:
                    continue
                comparison_col: pl.Series = Xbin[features[j]]

                # Vectorized XOR and NOT operations
                a: pl.Series = current_col ^ comparison_col  # XOR operation
                b: pl.Series = ~a  # NOT XOR

                if a.all() or b.all():
                    self.__removed.append(j)
                    if self.__inplace:
                        Xbin.drop_in_place(features[j])

        if self.__inplace:
            return Xbin
        else:
            return self.transform(Xbin)
