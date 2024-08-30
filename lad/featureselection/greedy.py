import polars as pl

from lad.binarizer.cutpoint import CutpointBinarizer


class GreedySetCover:
    """
    Feature Selection

    Basic class based feture selection optimized for binary data
    Threshold is based of class count (subject to change)
    """

    def __init__(self, binarizer: CutpointBinarizer):
        self.__selected = []
        self.__labels = []
        self.__binarizer = binarizer

    def get_selected(self):
        return pl.Series("Selected", self.__selected)

    def __check_column_quality(self, col_y: pl.DataFrame) -> pl.Series:
        class_count = len(self.__labels)
        df = col_y
        total = df.group_by("label", maintain_order=True).sum().drop("label")
        y_t = (
            df.group_by("label", maintain_order=True)
            .len()
            .select([pl.col("len").alias("y_t")])
            .to_series()
        )

        T = total.sum()
        final = T * y_t.sum()

        for classs in range(class_count):
            tc = total[classs]
            final -= y_t[classs] * tc
            for subclass in range(classs + 1, class_count):
                final -= 2 * tc * total[subclass]

        # Calculate upper bound as per your described method
        y_t_sorted = y_t.to_list()
        y_t_sorted.sort()

        while len(y_t_sorted) > 2:
            y_t_sorted[0] = y_t_sorted[0] + y_t_sorted[1]
            y_t_sorted.pop(1)
            y_t_sorted.sort()

        upper_bound = y_t_sorted[0] * y_t_sorted[1]

        final = pl.Series(final.row(0)) / upper_bound

        return final <= 1 / class_count

    def fit(self, Xbin: pl.DataFrame, y: pl.Series):
        self.__selected = []

        Xbin_prune = Xbin
        y_series = y
        features = Xbin_prune.columns

        self.__labels = y.unique().sort().to_list()

        for feature in features:
            df = self.__binarizer.transform_column(Xbin[feature])
            columns = df.columns
            rejected = self.__check_column_quality(df.hstack([y_series]))
            for i, f in enumerate(columns):
                if not rejected[i]:
                    self.__selected.append(f)

        return self.__selected

    def transform(self, X: pl.DataFrame):
        return self.__binarizer.transform(X, filter=self.__selected)

    def fit_transform(self, Xbin, y):
        self.fit(Xbin, y)
        return self.transform(Xbin)
