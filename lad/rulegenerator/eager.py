from typing import List, Optional, Set

import numpy as np


class MaxPatterns:
    def __init__(
        self,
        binarizer,
        selector,
        fp_tolerance=0.5,
        fn_tolerance=0.5,
        max_terms_in_patterns=4,
    ):
        self.__rules = []

        self.__fp_tolerance = fp_tolerance
        self.__fn_tolerance = fn_tolerance
        self.__max_terms = max_terms_in_patterns

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

    def __base_fit(self, X_pos, X_neg):
        feature_count = len(X_pos[0])
        prime_patterns = set()
        prev_degree_non_prime_patterns = set([0])
        for d in range(1, self.__max_terms):
            curr_degree_non_prime_patterns = set()
            print("D: ", d)
            for curr_base_patterns in prev_degree_non_prime_patterns:
                print("Pattern: ", curr_base_patterns)
                largets_idx_of_terms_in_curr_patterns = -1
                tmp_value = curr_base_patterns
                while tmp_value > 0:
                    tmp_value = tmp_value // 4
                    largets_idx_of_terms_in_curr_patterns += 1
                start_of_range = 0
                if largets_idx_of_terms_in_curr_patterns != -1:
                    start_of_range = largets_idx_of_terms_in_curr_patterns
                for i in range(start_of_range, feature_count):
                    print("I, ", i)
                    for possible_term in [3, 2]:
                        print("Possible, ", possible_term)
                        should_break = False
                        possible_next_pattern = curr_base_patterns
                        possible_next_pattern += possible_term * (4**i)
                        tmp_possible = possible_next_pattern
                        idx = -1
                        while tmp_possible > 0:
                            idx += 1
                            value = tmp_possible % 4
                            tmp_possible = tmp_possible // 4
                            if value == 0:
                                continue
                            test_pattern = possible_next_pattern
                            test_pattern -= value * (4**idx)
                            if not prev_degree_non_prime_patterns.__contains__(
                                test_pattern
                            ):
                                should_break = True
                                break
                        if should_break:
                            print("Continueing loop")
                            continue
                        pos_count_prime = 0
                        for sample_t in X_pos:
                            if self.__match_terms(
                                sample_t,
                                self.__gen_pattern(
                                    possible_next_pattern, feature_count
                                ),
                            ):
                                pos_count_prime += 1
                        if self.__fn_tolerance <= 2 * pos_count_prime / len(X_pos):
                            pos_count = 0
                            neg_count = 0
                            for sample in X_pos:
                                if self.__match_terms(
                                    sample,
                                    self.__gen_pattern(
                                        curr_base_patterns, feature_count
                                    ),
                                ):
                                    pos_count += 1

                            for smaple in X_neg:
                                if self.__match_terms(
                                    sample,
                                    self.__gen_pattern(
                                        curr_base_patterns, feature_count
                                    ),
                                ):
                                    neg_count += 1

                            pos_pct = pos_count / len(X_pos)
                            neg_pct = neg_count / len(X_neg)

                            hd = pos_pct / (pos_pct + neg_pct)

                            if hd >= self.__fp_tolerance:
                                prime_patterns.union(set([curr_base_patterns]))
                                print(curr_base_patterns)
                            else:
                                curr_degree_non_prime_patterns.union(
                                    set([curr_base_patterns])
                                )
            prev_degree_non_prime_patterns = curr_degree_non_prime_patterns
        return prime_patterns

    def __gen_pattern(self, a, n):
        out: List[Optional[bool]] = [None for _ in range(n)]
        tmp = a
        idx = 0
        while tmp > 0:
            value = tmp % 4
            tmp = tmp // 4
            val = None
            if value == 4:
                val = True
            if value == 3:
                val = False
            out[idx] = val
            idx += 1
        return out

    def __match_terms(self, a, b):
        out = True
        for i in range(len(a)):
            if b[i] is not None:
                if a[i] == b[i]:
                    pass
                else:
                    out = False
        return out

    def fit(self, Xbin, y):
        # #
        unique, counts = np.unique(y, return_counts=True)
        #
        # #
        self.__rules.clear()
        self.__labels = unique
        self.__most_frequent_label = unique[np.argmax(counts)]

        # #
        for lable in unique:
            X_pos = []
            X_neg = []
            for idx in range(len(Xbin)):
                if lable == y[idx]:
                    X_pos.append(Xbin[idx])
                else:
                    X_neg.append(Xbin[idx])

            patterns = self.__base_fit(X_pos, X_neg)
            print(patterns)
            self.__rules.append(patterns)
        print(self.__rules)
        # rules_weights = []
        # labels_weights = {}
        #
        # for sample in np.unique(Xbin, axis=0):
        #     features = list(np.arange(sample.shape[0]))
        #     # Stats
        #     repet, _, purity, label, discrepancy = self.__get_stats(
        #         Xbin, y, sample, features
        #     )
        #
        #     # Choosing rule's attributes
        #     while len(features) > 1:
        #         worst = None  # Actually, the worst
        #         tmp_attributes = features.copy()
        #
        #         # Find the best attribute to be removed
        #         for feature in features:
        #             # Candidate
        #             tmp_attributes.remove(feature)
        #
        #             # Stats
        #             _, _, __purity, _, __discrepancy = self.__get_stats(
        #                 Xbin, y, sample, tmp_attributes
        #             )
        #
        #             # Testing candidate
        #             if __purity >= self.__min_purity:
        #                 if __purity > purity or (
        #                     __purity == purity and __discrepancy < discrepancy
        #                 ):
        #                     worst = feature
        #                     purity = __purity
        #                     discrepancy = __discrepancy
        #
        #             #
        #             tmp_attributes.append(feature)
        #
        #         if worst is None:
        #             break
        #
        #         # Update rule
        #         features.remove(worst)
        #
        #         # Turn's stats
        #         _, _, purity, label, discrepancy = self.__get_stats(
        #             Xbin, y, sample, features
        #         )
        #
        #     # Forming rule object
        #     r = {
        #         "label": label,
        #         "attributes": features.copy(),
        #         "conditions": list(sample[features]),
        #         "purity": purity,
        #     }
        #
        #     # Storing rule
        #     if r not in self.__rules:
        #         self.__rules.append(r)
        #         rules_weights.append(repet)
        #     else:
        #         # When the same rule as build more than once
        #         rules_weights[self.__rules.index(r)] += repet
        #
        #     labels_weights[label] = labels_weights.get(label, 0) + repet
        #
        # # Reweighting
        # for i, r in enumerate(self.__rules):
        #     r["weight"] = rules_weights[i] / labels_weights[r["label"]]
        # self.__adjust()

    # def __adjust(self):
    #     for r in self.__rules:
    #         conditions = {}
    #         print(self.__selected)
    #         print(self.__cutpoints)
    #         # cutpoints = [self.__cutpoints[i] for i in self.__selected[r["attributes"]]]
    #         cutpoints = []
    #         for i in self.__selected[r["attributes"]]:
    #             print(i)
    #             a = self.__cutpoints[i]
    #             print(a)
    #             cutpoints.append(a)
    #
    #         for i, (att, value) in enumerate(cutpoints):
    #             condition = conditions.get(att, {})
    #             symbol = r["conditions"][i]  # True: <=, False: >
    #
    #             if symbol:
    #                 condition[symbol] = min(value, condition.get(symbol, value))
    #             else:
    #                 condition[symbol] = max(value, condition.get(symbol, value))
    #
    #             conditions[att] = condition
    #
    #         r["attributes"].clear()
    #         r["conditions"].clear()
    #         r["values"] = []
    #
    #         for att in conditions:
    #             for condition in conditions[att]:
    #                 r["attributes"].append(att)
    #                 r["conditions"].append(condition == "<=")
    #                 r["values"].append(conditions[att][condition])
    #
    #     self.__rules.sort(key=lambda x: x["label"])
    #
    # def __get_stats(self, Xbin, y, sample, features):
    #     covered = np.where((Xbin[:, features] == sample[features]).all(axis=1))
    #     uncovered = np.setdiff1d(np.arange(Xbin.shape[0]), covered[0])
    #
    #     unique, counts = np.unique(y[covered], return_counts=True)
    #     argmax = np.argmax(counts)
    #     purity = counts[argmax] / len(covered[0])
    #     label = unique[argmax]
    #
    #     uncovered_class = uncovered[y[uncovered] == label]
    #     uncovered_other = uncovered[y[uncovered] != label]
    #
    #     distance_class = np.sum(
    #         np.bitwise_xor(Xbin[uncovered_class][:, features], sample[features])
    #     )
    #
    #     distance_other = np.sum(
    #         np.bitwise_xor(Xbin[uncovered_other][:, features], sample[features])
    #     )
    #
    #     discrepancy = (
    #         max(1.0, distance_class)
    #         / max(1.0, len(uncovered_class))
    #         / max(1.0, distance_other)
    #         / max(1.0, len(uncovered_other))
    #     )
    #
    #     return len(covered[0]), counts[argmax], purity, label, discrepancy
    # #
    # def __str__(self):
    #     s = f"MaxPatterns Set of Rules [{len(self.__rules)}]:\n"
    #
    #     for r in self.__rules:
    #         label = r["label"]
    #         # weight = r['weight']
    #         conditions = []
    #
    #         for i, condition in enumerate(r["conditions"]):
    #             att = r["attributes"][i]
    #             val = r["values"][i]
    #
    #             if condition:
    #                 conditions.append(f"att{att} <= {val:.4}")
    #             else:
    #                 conditions.append(f"att{att} > {val:.4}")
    #
    #         # Label -> CONDITION_1 AND CONDITION_2 AND CONDITION_n
    #         s += f'{label} \u2192 {" AND ".join(conditions)}\n'
    #
    #     return s
