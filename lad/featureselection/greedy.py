import time  # Debug import

import numpy as np

# TODO restruct feature selection


class UnWeightedSetCoveringProblem:
    """Set covering problem builder"""

    def __init__(self, selected=[]):
        self.__scp = []
        self.__selected = selected

    def fit(self, Xbin, y):
        self.__scp = np.zeros(Xbin.shape[1])
        labels = np.unique(y)

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):

                # Crossover
                for u in Xbin[y == labels[i]]:
                    for v in Xbin[y == labels[j]]:
                        inc = np.bitwise_xor(u, v)
                        if not np.any(inc[self.__selected]):
                            self.__scp += inc

        self.__scp = np.array(self.__scp)
        print(self.__scp)
        return self.__scp


class GreedySetCover:
    """Set covering problem solver"""

    def __init__(self):
        self.__selected = []
        self.__scp = None

    def get_selected(self):
        return np.array(self.__selected)

    def fit(self, Xbin, y):
        self.__selected.clear()

        builder = UnWeightedSetCoveringProblem()
        scp = builder.fit(Xbin, y)

        print("SCP", scp.shape)

        while True:
            # sum_ = scp.sum(axis=0)
            att = np.argmax(scp)

            if scp[att] == 0:
                break

            # scp = np.delete(scp, np.where(scp[:, att]), axis=0)
            self.__selected.append(att)
            builder = UnWeightedSetCoveringProblem(self.__selected)
            scp = builder.fit(Xbin, y)

        self.__selected.sort()

    def transform(self, Xbin):
        return Xbin[:, self.__selected]

    def fit_transform(self, Xbin, y):
        self.fit(Xbin, y)
        return self.transform(Xbin)
