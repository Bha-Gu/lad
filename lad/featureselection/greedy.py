import time  # Debug import

import numpy as np

# TODO restruct feature selection


class UnWeightedSetCoveringProblem:
    """Set covering problem builder"""

    def __init__(self, selected=[], y_important=[]):
        self.__scp = []
        self.__selected = selected
        self.__yi = y_important

    def fit(self, Xbin, y):
        self.__scp = []
        labels = np.unique(y)

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if len(self.__yi):
                    if not ((i in self.__yi) or (j in self.__yi)):
                        print("Skip ", i, j)
                        continue
                # Crossover
                for u in Xbin[y == labels[i]]:
                    for v in Xbin[y == labels[j]]:
                        inc = np.bitwise_xor(u, v)
                        if not np.any(inc[self.__selected]):
                            self.__scp.append(inc)

        self.__scp = np.array(self.__scp)

        return self.__scp


class GreedySetCover:
    """Set covering problem solver"""

    def __init__(self, y_importances=[]):
        self.__selected = []
        self.__scp = None
        self.__yis = y_importances
        self.__yis.append([])

    def get_selected(self):
        return np.array(self.__selected)

    def fit(self, Xbin, y):
        self.__selected.clear()

        for yi in self.__yis:
            builder = UnWeightedSetCoveringProblem(self.__selected, yi)
            scp = builder.fit(Xbin, y)

            print("SCP", scp.shape)

            while len(scp):
                sum_ = scp.sum(axis=0)
                att = np.argmax(sum_)

                if sum_[att] == 0:
                    break

                scp = np.delete(scp, np.where(scp[:, att]), axis=0)
                self.__selected.append(att)

        self.__selected.sort()

    def transform(self, Xbin):
        return Xbin[:, self.__selected]

    def fit_transform(self, Xbin, y):
        self.fit(Xbin, y)
        return self.transform(Xbin)
