import numpy as np


class GreedySetCover:
    """Set covering problem solver"""

    def __init__(self, max_features=0):
        self.__selected = []
        self.__scp = None
        self.__max = max_features

    def get_selected(self):
        return np.array(self.__selected)

    def fit(self, Xbin, y):
        self.__selected = []

        Xbin_prune = Xbin.copy()
        labels = np.unique(y)
        class_count = len(labels)
        sample_count = Xbin_prune.shape[0]
        y_idx = []
        for i in y:
            y_idx.append(np.where(labels == i)[0][0])

        subset_index = np.zeros(sample_count, dtype=int)

        invalids = []

        while True:
            subset_count = 2 ** len(self.__selected)
            feature_count = Xbin_prune.shape[1]

            total = np.zeros((subset_count, class_count, feature_count), dtype=int)
            y_t = np.zeros((subset_count, class_count), dtype=int)
            for i in range(sample_count):
                total[subset_index[i]][y_idx[i]] += Xbin_prune[i]
                y_t[subset_index[i]][y_idx[i]] += 1

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

            base_best = best
            for invalid in invalids:
                if invalid <= base_best:
                    base_best += 1

            self.__selected.append(base_best)
            if len(self.__selected) == self.__max:
                break
            self.__selected.sort()

            for sample in range(sample_count):
                if Xbin_prune[sample, best]:
                    subset_index[sample] += 2 ** (len(self.__selected) - 1)

            rejected = np.where(final_rank == 0)[0]
            base_rejected = rejected.copy()

            for invalid in invalids:
                for i in range(len(rejected)):
                    if invalid <= base_rejected[i]:
                        base_rejected[i] += 1

            invalids.append(base_best)
            for r in base_rejected:
                if r not in invalids:
                    invalids.append(r)
                else:
                    print("Debug print, should be impossible")
            invalids.sort()

            mask = np.ones(feature_count, dtype=bool)
            mask[rejected] = False
            mask[best] = False
            Xbin_prune = Xbin_prune[:, mask]

    def transform(self, Xbin):
        return Xbin[:, self.__selected]

    def fit_transform(self, Xbin, y):
        self.fit(Xbin, y)
        return self.transform(Xbin)
