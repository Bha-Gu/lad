#!/usr/bin/env python
import polars as pl
from tqdm.auto import tqdm


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

    def __init__(self, bin_size: float = 1.0):
        self.__cutpoints = []
        self.__bin_size: float = bin_size

    def get_cutpoints(self):
        return self.__cutpoints

    def test(self, col: pl.Series, y: pl.Series) -> bool:
        labels: list[int] = y.unique().sort().to_list()
        class_count: int = len(labels)
        df = pl.DataFrame([col, y])
        total: pl.Series = df.group_by("label", maintain_order=True).sum()[col.name]

        y_t: pl.Series = (
            df.group_by("label", maintain_order=True)
            .len()
            .select([pl.col("len").alias("y_t")])
            .to_series()
        )

        T: float = total.sum()
        # Initialize the final array
        final: float = T * y_t.sum()

        for classs in range(class_count):
            tc: float = total[classs]
            final -= y_t[classs] * tc
            for subclass in range(classs + 1, class_count):
                final -= 2 * tc * total[subclass]

        y_t_sorted: list[int] = y_t.to_list()
        y_t_sorted.sort()

        while len(y_t_sorted) > 2:
            y_t_sorted[0] = y_t_sorted[0] + y_t_sorted[1]
            y_t_sorted.pop(1)
            y_t_sorted.sort()

        upper_bound: int = y_t_sorted[0] * y_t_sorted[1]

        normalized_final: float = final / upper_bound if upper_bound != 0 else 0

        passed: bool = normalized_final > 1 / class_count

        return passed

    def fit(self, X: pl.DataFrame, y: pl.Series):
        self.__cutpoints: list[  # pyright: ignore [reportRedeclaration]
            tuple[bool, list[float] | list[str]]
        ] = []
        self.__selected: set[str] = set()  # pyright: ignore [reportRedeclaration]
        schema = X.schema
        features = X.columns

        for feature in tqdm(features, desc="Cutpoint Generation"):
            print(feature)
            col_y: pl.DataFrame = pl.DataFrame([X[feature], y])
            if schema[feature].is_numeric():
                sorted_values: pl.DataFrame = col_y.sort(feature)
                delta: float = sorted_values[feature].diff(null_behavior="drop").sum()
                tolerance: float = delta / (len(sorted_values) - 1)
                cutpoints: set[float] = set()
                prev_cutpoint: float | None = None
                first_cutpoint: bool = False
                prev_labels: set[str] = set()
                prev_value: float | None = None
                labels: set[str] = set()

                for value, label in sorted_values.rows():
                    labels.add(label)

                    if prev_value is not None:
                        variation: float = value - prev_cutpoint
                        if variation > tolerance * self.__bin_size:
                            if (
                                len(prev_labels) > 1 or len(labels) > 1
                            ) or prev_labels != labels:
                                cp: float = (  # pyright: ignore [reportRedeclaration]
                                    prev_value + (value - prev_value) / 2.0
                                )
                                if first_cutpoint:
                                    name: (  # pyright: ignore [reportRedeclaration]
                                        str
                                    ) = f"{feature}<={cp}"
                                    col: (  # pyright: ignore [reportRedeclaration]
                                        pl.Series
                                    ) = (col_y[feature] <= cp).alias(name)
                                    if self.test(col, y):
                                        self.__selected.add(name)
                                        cutpoints.add(cp)
                                    first_cutpoint = False
                                else:
                                    # fmt: off
                                    name: str = f"{prev_cutpoint}<={feature}<{cp}" # pyright: ignore [reportRedeclaration]

                                    col: pl.Series = ( # pyright: ignore [reportRedeclaration]
                                        (col_y[feature] <= cp)
                                        & (col_y[feature] > prev_cutpoint)
                                    ).alias(name)
                                    # fmt: on
                                    if self.test(col, y):
                                        self.__selected.add(name)
                                        cutpoints.add(cp)
                                        (
                                            cutpoints.add(prev_cutpoint)
                                            if prev_cutpoint is not None
                                            else None
                                        )
                                prev_cutpoint = cp
                                prev_labels = labels
                                labels = set()

                    if prev_cutpoint is None:
                        prev_cutpoint = value
                        first_cutpoint = True
                    prev_value = value
                name: str = (  # pyright: ignore [reportRedeclaration]
                    f"{prev_cutpoint}<{feature}"
                )
                col: pl.Series = (  # pyright: ignore [reportRedeclaration]
                    X[feature] > prev_cutpoint
                ).alias(name)
                if self.test(col, y):
                    self.__selected.add(name)

                if len(cutpoints) == 0:
                    cp_tmp = sorted_values[feature].mean()
                    try:
                        cp = float(str(cp_tmp)) if cp_tmp is not None else 0.0
                    except (ValueError, TypeError):
                        print("Error at error check 1")
                        cp = 0.0

                    cutpoints.add(cp)
                self.__cutpoints.append((True, sorted(cutpoints)))
            else:
                values = col_y[feature].unique().to_list()
                cps = []
                for value in values:
                    name: str = f"{feature}={value}"
                    col: pl.Series = (col_y[feature] == value).alias(name)
                    if self.test(col, y):
                        self.__selected.add(name)
                        cps.append(name)
                self.__cutpoints.append((False, cps))
        return (self.__cutpoints, self.__selected)

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        Xbin = pl.DataFrame()
        print("# Transform")
        for column_name, (type_val, cutpoints) in tqdm(
            zip(X.columns, self.__cutpoints), desc="Transforming data"
        ):
            print(column_name)
            column: pl.Series = X[column_name]
            if type_val:
                name: str = f"{column_name}<={cutpoints[0]}"
                col: pl.Series = (column <= cutpoints[0]).alias(name)
                if name in self.__selected:
                    Xbin = Xbin.hstack([col])

                length: int = len(cutpoints)
                for i in range(length - 1):
                    j = i + 1
                    name = f"{cutpoints[i]}<={column_name}<{cutpoints[j]}"
                    col = ((column > cutpoints[i]) & (column <= cutpoints[j])).alias(
                        name
                    )
                    if name in self.__selected:
                        Xbin = Xbin.hstack([col])
                name = f"{cutpoints[-1]}<{column_name}"
                col = (cutpoints[-1] <= column).alias(name)
                if name in self.__selected:
                    Xbin = Xbin.hstack([col])
            else:
                for value in cutpoints:
                    col = (column == value).alias(f"{column_name}={value}")
                    Xbin = Xbin.hstack([col])

        return Xbin

    def fit_transform(self, X: pl.DataFrame, y: pl.Series) -> pl.DataFrame:
        self.fit(X, y)
        return self.transform(X)
