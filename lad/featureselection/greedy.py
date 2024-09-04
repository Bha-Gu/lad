import polars as pl
from tqdm.auto import tqdm


class GreedySetCover:
    """
    Feature Selection

    Basic class based feture selection optimized for binary data
    Threshold is based of class count (subject to change)
    """

    def __init__(self):
        self.__removed = []

    def get_removed(self):
        return pl.Series("Removed", self.__removed)

    def fit(self, Xbin: pl.DataFrame, y: pl.Series):
        features = Xbin.columns
        feature_count: int = len(features)

        for i in tqdm(range(feature_count), desc="Feature selection"):
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
        return X.select(pl.col("*").exclude(list_of_rejected))

    def fit_transform(self, Xbin, y) -> pl.DataFrame:
        self.fit(Xbin, y)
        return self.transform(Xbin)
