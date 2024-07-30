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

        # Convert Xbin and y to numpy arrays if they aren't already
        Xbin = np.asarray(Xbin)
        y = np.asarray(y)

        # Ensure that self.__selected is a numpy array for efficient indexing
        self.__selected = np.asarray(self.__selected).astype(int)

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                # Get binary samples for the current label pair
                X_i = Xbin[y == labels[i]]
                X_j = Xbin[y == labels[j]]

                for u in X_i:
                    # Calculate XOR with each v in X_j in batches
                    batch_size = 1000  # Adjust batch size based on available memory
                    for start in range(0, len(X_j), batch_size):
                        end = start + batch_size
                        X_j_batch = X_j[start:end]

                        xor_matrix = np.bitwise_xor(u, X_j_batch)

                        # Check the condition across the selected indices
                        mask = ~np.any(xor_matrix[:, self.__selected], axis=1)

                        # Sum the increments where the mask is True
                        self.__scp += xor_matrix[mask].sum(axis=0)

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
