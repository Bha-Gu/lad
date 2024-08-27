# from typing import List, Optional

import copy

import numpy as np
import polars as pl


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

    # def __sec_fit(self, X: pl.DataFrame, y: pl.Series):
    #     prime_patterns = []
    #     prev_degree_non_prime_patterns = [set()]
    #     features = X.columns
    #     feature_count = len(features)
    #     max = self.__max_terms
    #     if max > feature_count or max == 0:
    #         max = feature_count
    #     for d in range(1,max):
    #         if len(X) == 0:
    #             break
    #         curr_degree_patterns = []
    #         for base_pattern in prev_degree_non_prime_patterns:
    #             for feature in features:
    #
    #
    #
    #     pass

    def __base_fit(self, X_pos: pl.DataFrame, X_neg: pl.DataFrame, feature_count):
        prime_patterns = []
        prev_degree_non_prime_patterns = [set()]
        features = X_pos.columns
        max = self.__max_terms
        if max > feature_count or max == 0:
            max = feature_count
        for d in range(1, max):
            print("Loop1 Index: ", d)
            if len(X_pos) == 0:
                break
            curr_degree_non_prime_patterns = []
            for curr_base_patterns in prev_degree_non_prime_patterns:
                if len(X_pos) == 0:
                    break
                print("  Loop2 CBP: ", curr_base_patterns)
                largets_idx_of_terms_in_curr_patterns = -1

                for idx, feature in enumerate(features):
                    if (True, feature) in curr_base_patterns or (
                        False,
                        feature,
                    ) in curr_base_patterns:
                        largets_idx_of_terms_in_curr_patterns = idx

                start_of_range = largets_idx_of_terms_in_curr_patterns + 1

                for i in range(start_of_range, feature_count):
                    print("    Loop3 Index: ", i)
                    for possible_term in [True, False]:
                        if len(X_pos) == 0:
                            break
                        print("      Loop4 Term", possible_term)
                        should_break = False
                        possible_next_pattern = copy.deepcopy(curr_base_patterns)
                        possible_next_pattern.add((possible_term, features[i]))
                        print(possible_next_pattern)
                        for term in possible_next_pattern:
                            test_pattern = copy.deepcopy(possible_next_pattern)
                            test_pattern.discard(term)
                            print(test_pattern)
                            if test_pattern not in prev_degree_non_prime_patterns:
                                should_break = True
                                break
                        if should_break:
                            print("      Loop4 Continue")
                            continue
                        filters = [
                            pl.col(column_name) == desired_value
                            for desired_value, column_name in possible_next_pattern
                        ]
                        filter = filters[0]
                        for f in filters[1:]:
                            filter &= f
                        print(filter)
                        pos_count_prime = len(X_pos.filter(filter))

                        if self.__fn_tolerance <= 2 * pos_count_prime / len(X_pos):
                            print("        Cond1 Pass: ", possible_next_pattern)
                            pos_count = len(X_pos.filter(filter))
                            neg_count = len(X_neg.filter(filter))

                            print(pos_count, neg_count)
                            pos_pct = pos_count
                            neg_pct = neg_count
                            base = pos_pct + neg_pct
                            hd = 0.0
                            if base > 0.0:
                                hd = pos_pct / base

                            if hd >= self.__fp_tolerance:
                                print("          Cond2 Pass: ", hd)
                                prime_patterns.append(possible_next_pattern)
                                X_pos = X_pos.filter(filter)
                                X_neg = X_neg.filter(filter)
                                print(len(X_pos), len(X_neg))
                            else:
                                print("          Cond2 Fail: ", hd)
                                curr_degree_non_prime_patterns.append(
                                    possible_next_pattern
                                )
            prev_degree_non_prime_patterns = curr_degree_non_prime_patterns
        return prime_patterns

    def fit(self, Xbin: pl.DataFrame, y: pl.Series):
        # #
        unique, counts = np.unique(y, return_counts=True)
        #
        # #
        features = Xbin.columns
        feature_count = len(features)

        self.__rules.clear()
        self.__labels = unique
        self.__most_frequent_label = unique[np.argmax(counts)]
        # #
        for lable in unique:
            print(lable)
            X = Xbin.hstack([y])
            X_pos = X.filter(pl.col("label") == lable).drop("label")
            X_neg = X.filter(pl.col("label") != lable).drop("label")

            patterns = self.__base_fit(X_pos, X_neg, feature_count)
            self.__rules.append(patterns)
        print(self.__rules)
        return self.__rules

        # num_zeros = 2 * feature_count
        # for i in self.__rules:
        #     for j in i:
        #         print(f"{{:0{num_zeros}b}}".format(j))
        #     print()
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
