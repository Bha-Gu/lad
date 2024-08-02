import time  # Debug import

import numpy as np

# TODO restruct feature selection


class UnWeightedSetCoveringProblem:
    """Set covering problem builder"""

    def __init__(self, selected=[]):
        self.__scp = []
        self.__selected = selected

    def fit(self, Xbin, y):
        length = 2 ** len(self.__selected)
        labels = np.unique(y)
        class_count = len(labels)
        feature_count = Xbin.shape[1]
        sample_count = Xbin.shape[0]
        total = np.zeros((length, class_count, feature_count))
        y_t = np.zeros((length, class_count))

        a = Xbin[:, self.__selected]
        b = []
        y_idx = []
        for i in y:
            y_idx.append(np.where(labels == i)[0][0])
        for i in a:
            idx = 0
            for jdx, j in enumerate(i):
                idx += (jdx + 1) * j
            b.append(idx)

        for i in range(sample_count):
            total[b[i]][y_idx[i]] += Xbin[i]
            y_t[b[i]][y_idx[i]] += 1

        YT = np.sum(y_t, axis=1)
        T = np.sum(total, axis=1)

        pos = np.zeros((length, feature_count))

        for i in range(length):
            pos[i] += YT[i] * T[i]
            for j in range(class_count):
                pos[i] -= y_t[i, j] * total[i, j, :]
                for k in range(j + 1, class_count):
                    pos[i] -= 2 * total[i, j, :] * total[i, k, :]

        pos = np.sum(pos, axis=0)
        print(pos)
        invalid = np.where(pos == 0)
        if np.sum(pos) == 0:
            return None, invalid
        return np.argmax(pos), invalid


class GreedySetCover:
    """Set covering problem solver"""

    def __init__(self, max_features=0):
        self.__selected = []
        self.__scp = None
        self.__max = max_features

    def get_selected(self):
        return np.array(self.__selected)

    def fit(self, Xbin, y):
        self.__selected.clear()

        builder = UnWeightedSetCoveringProblem(self.__selected)
        scp, invalid = builder.fit(Xbin, y)

        invalids = []

        effective_selected = []
        Xbin_prune = Xbin.copy()

        while scp:
            if scp is None:
                break
            print(invalid)
            mask = np.ones(Xbin_prune.shape[1], dtype=bool)
            mask[invalid] = False
            mask[effective_selected] = True
            invalid = np.where(mask == False)
            invalid = invalid[0]
            Xbin_prune = Xbin_prune[:, mask]
            actual_next_feature = scp
            for i in invalids:
                if i < actual_next_feature:
                    actual_next_feature += 1
                for j in range(len(invalid)):
                    if i < invalid[j]:
                        invalid[j] += 1
            for i in invalid:
                invalids.append(i)
            invalids.sort()

            self.__selected.append(actual_next_feature)
            if len(self.__selected) == self.__max:
                break

            self.__selected.sort()

            effective_selected = []
            for i in range(len(self.__selected)):
                effective_selected.append(self.__selected[i])
                for j in range(len(invalids)):
                    if self.__selected[i] >= invalids[j]:
                        effective_selected[i] -= 1
            print(scp)

            print(self.__selected)
            builder = UnWeightedSetCoveringProblem(effective_selected)
            scp, invalid = builder.fit(Xbin_prune, y)

        self.__selected.sort()

    def transform(self, Xbin):
        return Xbin[:, self.__selected]

    def fit_transform(self, Xbin, y):
        self.fit(Xbin, y)
        return self.transform(Xbin)
