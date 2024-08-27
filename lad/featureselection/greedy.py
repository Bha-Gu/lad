import numpy as np
import polars as pl


class GreedySetCover:
    """Set covering problem solver"""

    def __init__(self):
        self.__selected = []

    def get_selected(self):
        return np.array(self.__selected)

    def fit(self, Xbin: pl.DataFrame, y: pl.Series):
        self.__selected = []

        Xbin_prune = Xbin
        y_series = y
        features = Xbin_prune.columns

        labels = y.unique().sort().to_list()
        class_count = len(labels)
        # Create a DataFrame with the feature values and the corresponding labels

        df = Xbin_prune.hstack([y_series])

        # Group by label and sum the occurrences of True values for each feature
        total = df.group_by("label", maintain_order=True).sum().drop("label")

        y_t = (
            df.group_by("label", maintain_order=True)
            .len()
            .select([pl.col("len").alias("y_t")])
            .to_series()
        )

        # Calculate the total occurrence of True values for each feature across all classes
        T = total.sum()
        # Initialize the final array
        final = T * y_t.sum()

        # Subtract the contributions from each class
        for classs in range(class_count):
            tc = total[classs]
            final -= y_t[classs] * tc
            for subclass in range(classs + 1, class_count):
                final -= 2 * tc * total[subclass]

        # Convert final to a Series
        final_series = pl.Series(final.row(0))

        # Identify features to reject based on the threshold
        rejected = final_series / final_series.max() <= 1 / class_count

        # Create a mask for the selected features
        features = [f for i, f in enumerate(features) if not rejected[i]]

        feature_count = len(features)

        removed = []

        for i in range(feature_count):
            if i in removed:
                continue
            current_col = Xbin_prune[features[i]]

            for j in range(i + 1, feature_count):
                if j in removed:
                    continue
                comparison_col = Xbin_prune[features[j]]

                # Vectorized XOR and NOT operations
                a = current_col ^ comparison_col  # XOR operation
                b = ~a  # NOT XOR

                if a.all() or b.all():
                    removed.append(j)

        self.selected = [f for f in features if f not in removed]
        print(self.selected)
        print(features)
        print(removed)
        return self.__selected

    def transform(self, Xbin):
        return Xbin.select(self.__selected)

    def fit_transform(self, Xbin, y):
        self.fit(Xbin, y)
        return self.transform(Xbin)
