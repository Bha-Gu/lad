import numpy as np


class MaxPatterns:
    def __init__(self, binarizer, selector, purity):
        self.__min_purity = purity
        self.__rules = []

        self.__cutpoints = binarizer.get_cutpoints()
        self.__selected = selector.get_selected()

    def predict(self, X):
        weights = {}

        for r in self.__rules:
            label = r["label"]
            weight = r["weight"]

            indexes = np.arange(X.shape[0])

            for i, condition in enumerate(r["conditions"]):
                att = r["attributes"][i]
                val = r["values"][i]

                if condition:
                    # print(f'att{att} <= {val}', end=', ')
                    indexes = indexes[np.where(X.T[att, indexes] <= val)]
                else:
                    # print(f'att{att} > {val}', end=', ')
                    indexes = indexes[np.where(X.T[att, indexes] > val)]

            # print(r['label'])

            for i in indexes:
                weights[i] = weights.get(i, {})
                weights[i][label] = weights[i].get(label, 0) + weight

        predictions = []
        for i in range(X.shape[0]):
            if i not in weights:
                predictions.append(self.__most_frequent_label)
            else:
                predictions.append(max(weights[i], key=weights[i].get))

        return np.array(predictions)

    def predict_proba(self, X):
        predictions = self.predict(X)
        output = np.zeros((len(X), self.__labels))

        for i in range(len(X)):
            output[i][predictions[i]] = 1

        return output

    def fit(self, Xbin, y):
        #
        unique, counts = np.unique(y, return_counts=True)

        #
        self.__rules.clear()
        self.__labels = len(np.unique(y))
        self.__most_frequent_label = unique[np.argmax(counts)]

        #
        rules_weights = []
        labels_weights = {}

        for sample in np.unique(Xbin, axis=0):
            print("Instance: ", sample)
            features = list(np.arange(sample.shape[0]))
            print("Attributes: ", features)
            # Stats
            repet, _, purity, label, discrepancy = self.__get_stats(
                Xbin, y, sample, features
            )
            print("Stats: ", repet, "_", purity, label, discrepancy)

            # Choosing rule's attributes
            while len(features) > 1:
                worst = None  # Actually, the worst
                tmp_attributes = features.copy()

                # Find the best attribute to be removed
                for feature in features:
                    # Candidate
                    tmp_attributes.remove(feature)

                    # Stats
                    _, _, __purity, _, __discrepancy = self.__get_stats(
                        Xbin, y, sample, tmp_attributes
                    )

                    # Testing candidate
                    if __purity >= self.__min_purity:
                        if __purity > purity or (
                            __purity == purity and __discrepancy < discrepancy
                        ):
                            worst = feature
                            purity = __purity
                            discrepancy = __discrepancy

                    #
                    tmp_attributes.append(feature)

                if worst is None:
                    break

                # Update rule
                features.remove(worst)

                # Turn's stats
                _, _, purity, label, discrepancy = self.__get_stats(
                    Xbin, y, sample, features
                )

            # Forming rule object
            r = {
                "label": label,
                "attributes": features.copy(),
                "conditions": list(sample[features]),
                "purity": purity,
            }

            # Storing rule
            if r not in self.__rules:
                self.__rules.append(r)
                rules_weights.append(repet)
            else:
                # When the same rule as build more than once
                rules_weights[self.__rules.index(r)] += repet

            labels_weights[label] = labels_weights.get(label, 0) + repet

        # Reweighting
        for i, r in enumerate(self.__rules):
            r["weight"] = rules_weights[i] / labels_weights[r["label"]]
        print(self.__rules)
        self.__adjust()

    def __adjust(self):
        for r in self.__rules:
            conditions = {}
            print(self.__cutpoints)
            cutpoints = [self.__cutpoints[i] for i in self.__selected[r["attributes"]]]

            for i, (att, value) in enumerate(cutpoints):
                condition = conditions.get(att, {})
                symbol = r["conditions"][i]  # True: <=, False: >

                if symbol:
                    condition[symbol] = min(value, condition.get(symbol, value))
                else:
                    condition[symbol] = max(value, condition.get(symbol, value))

                conditions[att] = condition

            r["attributes"].clear()
            r["conditions"].clear()
            r["values"] = []

            for att in conditions:
                for condition in conditions[att]:
                    r["attributes"].append(att)
                    r["conditions"].append(condition == "<=")
                    r["values"].append(conditions[att][condition])

        self.__rules.sort(key=lambda x: x["label"])

    def __get_stats(self, Xbin, y, sample, features):
        covered = np.where((Xbin[:, features] == sample[features]).all(axis=1))
        uncovered = np.setdiff1d(np.arange(Xbin.shape[0]), covered[0])

        unique, counts = np.unique(y[covered], return_counts=True)
        argmax = np.argmax(counts)
        purity = counts[argmax] / len(covered[0])
        label = unique[argmax]

        uncovered_class = uncovered[y[uncovered] == label]
        uncovered_other = uncovered[y[uncovered] != label]

        distance_class = np.sum(
            np.bitwise_xor(Xbin[uncovered_class][:, features], sample[features])
        )

        distance_other = np.sum(
            np.bitwise_xor(Xbin[uncovered_other][:, features], sample[features])
        )

        discrepancy = (
            max(1.0, distance_class)
            / max(1.0, len(uncovered_class))
            / max(1.0, distance_other)
            / max(1.0, len(uncovered_other))
        )

        return len(covered[0]), counts[argmax], purity, label, discrepancy

    def __str__(self):
        s = f"MaxPatterns Set of Rules [{len(self.__rules)}]:\n"

        for r in self.__rules:
            label = r["label"]
            # weight = r['weight']
            conditions = []

            for i, condition in enumerate(r["conditions"]):
                att = r["attributes"][i]
                val = r["values"][i]

                if condition:
                    conditions.append(f"att{att} <= {val:.4}")
                else:
                    conditions.append(f"att{att} > {val:.4}")

            # Label -> CONDITION_1 AND CONDITION_2 AND CONDITION_n
            s += f'{label} \u2192 {" AND ".join(conditions)}\n'

        return s
