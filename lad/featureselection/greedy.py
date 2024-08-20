import numpy as np
import polars as pl


class GreedySetCover:
    """Set covering problem solver"""

    def __init__(self, max_features=0):
        self.__selected = []
        # self.__ranking = []
        self.__scp = None
        self.__max = max_features

    def get_selected(self):
        return np.array(self.__selected)

    def fit(self, Xbin: pl.DataFrame, y: pl.Series):
        self.__selected = []

        Xbin_prune = Xbin.clone()

        labels = y.unique().sort().to_list()
        class_count = len(labels)
        sample_count = Xbin_prune.shape[0]
        y_idx = [labels.index(val) for val in y]

        subset_index = np.zeros(sample_count, dtype=int)

        invalids = []

        while True:
            subset_count = 2 ** len(self.__selected)
            features = Xbin_prune.columns
            feature_count = len(features)

            total = np.zeros((subset_count, class_count, feature_count), dtype=int)
            y_t = np.zeros((subset_count, class_count), dtype=int)

            for i in range(sample_count):
                for feature in range(feature_count):
                    if Xbin_prune[i, feature]:
                        total[subset_index[i], y_idx[i], feature] += 1
                y_t[subset_index[i], y_idx[i]] += 1

            T = np.sum(total, axis=1)
            YT = np.sum(y_t, axis=1)

            final = np.zeros((subset_count, feature_count), dtype=int)
            for feature in range(feature_count):
                for subset in range(subset_count):
                    final[subset, feature] += YT[subset] * T[subset, feature]
                    for classs in range(class_count):
                        final[subset, feature] -= (
                            y_t[subset, classs] * total[subset, classs, feature]
                        )
                        for subclass in range(classs + 1, class_count):
                            final[subset, feature] -= (
                                2
                                * total[subset, classs, feature]
                                * total[subset, subclass, feature]
                            )

            final_rank = np.sum(final, axis=0)

            if np.sum(final_rank) == 0:
                break

            best = np.argmax(final_rank)
            base_best = features[best]

            self.__selected.append(base_best)
            if len(self.__selected) == self.__max:
                break

            for sample in range(sample_count):
                if Xbin_prune[base_best][sample]:
                    subset_index[sample] += 2 ** (len(self.__selected) - 1)

            rejected = np.where(final_rank == 0)[0]

            mask = np.ones(feature_count, dtype=bool)
            mask[rejected] = False
            mask[best] = False
            features = np.array(features)[mask]
            Xbin_prune = Xbin_prune.select(features)

    def transform(self, Xbin):
        return Xbin.select(self.__selected)

    def fit_transform(self, Xbin, y):
        self.fit(Xbin, y)
        return self.transform(Xbin)
