import copy

import polars as pl
from tqdm.auto import tqdm

from lad.binarizer.cutpoint import CutpointBinarizer
from lad.featureselection.greedy import GreedySetCover


class MaxPatterns:
    """
    Rule Generation

    Attributes
    ---------
    fp_tolerance: float
        (Default = 0.5)

    fn_tolerance: float
        (Default = 0.5)
    """

    def __init__(
        self,
        binarizer: CutpointBinarizer,
        selector: GreedySetCover,
        base_precision: float = 0.5,
        base_recall: float = 0.5,
        max_terms_in_patterns: int = 4,
        new_test: bool = False,
        multi_class_balance: str = "1v1",
    ):
        self.__rules = []

        self.__base_precision = base_precision
        self.__base_recall = base_recall
        self.__max_terms = max_terms_in_patterns
        self.__new_test = new_test

        self.__binarizer = binarizer
        self.__selector = selector
        self.__balance = multi_class_balance

    def predict(self, X: pl.DataFrame) -> pl.Series:
        proba = pl.DataFrame(self.predict_proba(X))
        return (
            proba.select(pl.concat_list(pl.all()))
            .to_series()
            .map_elements(lambda x: pl.Series(x).arg_max(), return_dtype=pl.Int64)
        )

    def multi_predict(self, X: pl.DataFrame):
        X = self.__selector.transform(self.__binarizer.transform(X))
        prediction = []
        for r in self.__rules:
            out = pl.Series([False for _ in range(len(X))])
            for rule in r:
                tmp = pl.Series([True for _ in range(len(X))])
                for v, f in rule:
                    tmp &= X[f] == v
                out |= tmp

            prediction.append(out.cast(pl.Int8))
        prediction = pl.DataFrame(prediction)
        return (
            prediction.select(pl.concat_list(pl.all()))
            .to_series()
            .map_elements(lambda x: pl.Series(x).arg_max(), return_dtype=pl.Int64)
        )

    def predict_proba(self, X: pl.DataFrame) -> list[pl.Series]:
        X = self.__selector.transform(self.__binarizer.transform(X))
        prediction = []
        for r in self.__rules:
            out = pl.Series([False for _ in range(len(X))])
            t_c = pl.Series([1.0 for _ in range(len(X))])
            for c, rule in r:
                tmp = pl.Series([True for _ in range(len(X))])
                for v, f in rule:
                    tmp &= X[f] == v
                out |= tmp
                t_c *= 1 - c * tmp.cast(pl.Int8)

            prediction.append(1 - t_c)
        return prediction

    def multi_fit(self, X: pl.DataFrame, y: pl.Series):
        features = X.columns
        feature_count = len(features)
        X_base = X.hstack([y])
        self.__rules.clear()
        max = self.__max_terms
        if max > feature_count or max == 0:
            max = feature_count + 1
        else:
            max += 1

        labels = y.unique(maintain_order=True)
        self.__labels = labels
        lens = y.unique_counts()
        removed_sizes = pl.Series([0 for _ in range(len(labels))])

        prime_patterns = [[] for _ in range(len(labels))]
        prev_degree_non_prime_patterns = [set()]

        for d in range(1, max):
            curr_degree_non_prime_patterns = []
            for curr_base_patterns in tqdm(
                prev_degree_non_prime_patterns,
                desc=f"{d} depth rule generation",
            ):
                if len(X_base) == 0:
                    break
                largets_idx_of_terms_in_curr_patterns = -1

                for idx, feature in enumerate(features):
                    if (True, feature) in curr_base_patterns or (
                        False,
                        feature,
                    ) in curr_base_patterns:
                        largets_idx_of_terms_in_curr_patterns = idx

                start_of_range = largets_idx_of_terms_in_curr_patterns + 1

                for i in range(start_of_range, feature_count):
                    for possible_term in [True, False]:
                        if len(X_base) == 0:
                            break
                        should_break = False
                        possible_next_pattern = copy.deepcopy(curr_base_patterns)
                        possible_next_pattern.add((possible_term, features[i]))
                        for term in possible_next_pattern:
                            test_pattern = copy.deepcopy(possible_next_pattern)
                            test_pattern.discard(term)
                            if test_pattern not in prev_degree_non_prime_patterns:
                                should_break = True
                                break
                        if should_break:
                            continue
                        curr_degree_non_prime_patterns.append(possible_next_pattern)
            size = len(curr_degree_non_prime_patterns)
            scores = [
                pl.Series(
                    [0.0 for _ in range(size)],
                    dtype=pl.Float64,
                )
                for _ in range(len(labels))
            ]
            selected = [[] for _ in range(len(labels))]
            got = True
            while got:
                got = False
                for i in range(size):
                    if i in selected:
                        for l in range(len(labels)):
                            scores[l][i] = -1
                        continue
                    filters = [
                        pl.col(column_name) == desired_value
                        for desired_value, column_name in curr_degree_non_prime_patterns[
                            i
                        ]
                    ]
                    filter = filters[0]
                    for f in filters[1:]:
                        filter &= f
                    # print(X_base)
                    matches = X_base.filter(filter).select(pl.col("target"))["target"]
                    # print(labels, matches)
                    counts = copy.deepcopy(labels).append(matches).unique_counts()
                    for l in range(len(labels)):
                        counts[l] -= 1

                    print(counts)

                    recalls = [
                        (count + r_s) / (counts.sum() + removed_sizes.sum())
                        if (counts.sum() + removed_sizes.sum()) > 0
                        else 0.0
                        for count, r_s in zip(counts, removed_sizes)
                    ]

                    r_pass = [recall >= self.__base_recall for recall in recalls]

                    print(recalls)

                    for l in range(len(labels)):
                        if r_pass[l] and lens[l] > 0 and (lens.sum() - lens[l]) > 0:
                            scores[l][i] = counts[l] / lens[l] - (
                                counts.sum() - counts[l]
                            ) / (lens.sum() - lens[l])
                            if scores[l][i] >= self.__base_precision:
                                got = True
                        else:
                            scores[l][i] = 0.0

                    print(scores)
                bests = [
                    score.arg_max() if float(str(score.max())) > 0.0 else None
                    for score in scores
                ]
                for l in range(len(labels)):
                    best = bests[l]
                    if best is None or scores[l][best] < self.__base_precision:
                        continue
                    selected[l].append(bests[l])
                    filters = [
                        pl.col(column_name) == desired_value
                        for desired_value, column_name in curr_degree_non_prime_patterns[
                            best
                        ]
                    ]
                    filter = filters[0]
                    for f in filters[1:]:
                        filter &= f

                    X_base = X_base.filter(~filter)
                    matches = X_base["target"]
                    removed_sizes = (
                        lens - copy.deepcopy(labels).append(matches).unique_counts()
                    )
            selected_flat = [item for sublist in selected for item in sublist]
            prev_degree_non_prime_patterns = [
                v
                for i, v in enumerate(curr_degree_non_prime_patterns)
                if i not in selected_flat
            ]
            for l in range(len(labels)):
                for i in selected_flat:
                    prime_patterns.append(curr_degree_non_prime_patterns[i])
        self.__rules = prime_patterns
        return prime_patterns

    def __my_fit(self, X_pos: pl.DataFrame, X_neg: pl.DataFrame):
        pos_shape = X_pos.shape[0]
        neg_shape = X_neg.shape[0]
        p_s = 0
        n_s = 0
        features = X_pos.columns
        feature_count = len(features)
        max = self.__max_terms
        prime_patterns = []
        prev_degree_non_prime_patterns = [set()]
        if max > feature_count or max == 0:
            max = feature_count + 1
        else:
            max += 1
        for d in range(1, max):
            curr_degree_non_prime_patterns = []
            for curr_base_patterns in tqdm(
                prev_degree_non_prime_patterns,
                desc=f"{d} depth rule generation",
            ):
                if len(X_pos) == 0:
                    break
                largets_idx_of_terms_in_curr_patterns = -1

                for idx, feature in enumerate(features):
                    if (True, feature) in curr_base_patterns or (
                        False,
                        feature,
                    ) in curr_base_patterns:
                        largets_idx_of_terms_in_curr_patterns = idx

                start_of_range = largets_idx_of_terms_in_curr_patterns + 1

                for i in range(start_of_range, feature_count):
                    for possible_term in [True, False]:
                        if len(X_pos) == 0:
                            break
                        should_break = False
                        possible_next_pattern = copy.deepcopy(curr_base_patterns)
                        possible_next_pattern.add((possible_term, features[i]))
                        for term in possible_next_pattern:
                            test_pattern = copy.deepcopy(possible_next_pattern)
                            test_pattern.discard(term)
                            if test_pattern not in prev_degree_non_prime_patterns:
                                should_break = True
                                break
                        if should_break:
                            continue
                        curr_degree_non_prime_patterns.append(possible_next_pattern)
            size = len(curr_degree_non_prime_patterns)
            scores = pl.Series([0.0 for _ in range(size)])
            selected = []
            got = True
            while got:
                got = False
                for i in range(size):
                    if i in selected:
                        scores[i] = -1
                        continue
                    filters = [
                        pl.col(column_name) == desired_value
                        for desired_value, column_name in curr_degree_non_prime_patterns[
                            i
                        ]
                    ]
                    filter = filters[0]
                    for f in filters[1:]:
                        filter &= f
                    TP = len(X_pos.filter(filter))
                    pos_len = len(X_pos)
                    FP = len(X_neg.filter(filter))
                    neg_len = len(X_neg)
                    base = TP + p_s + FP + n_s
                    recall = (TP + p_s) / base if base > 0.0 else 0.0
                    if self.__base_recall <= recall and pos_len > 0.0:
                        precision = TP / pos_len - FP / neg_len
                        scores[i] = precision
                        got = True
                    else:
                        scores[i] = 0.0
                best = scores.arg_max()
                if best is None or scores[best] <= self.__base_precision:
                    break
                selected.append(best)
                filters = [
                    pl.col(column_name) == desired_value
                    for desired_value, column_name in curr_degree_non_prime_patterns[
                        best
                    ]
                ]
                filter = filters[0]
                for f in filters[1:]:
                    filter &= f
                X_pos = X_pos.filter(~filter)
                p_s = pos_shape - X_pos.shape[0]
                X_neg = X_neg.filter(~filter)
                n_s = neg_shape - X_neg.shape[0]
            prev_degree_non_prime_patterns = [
                v
                for i, v in enumerate(curr_degree_non_prime_patterns)
                if i not in selected
            ]
            for i in selected:
                prime_patterns.append(curr_degree_non_prime_patterns[i])
        return prime_patterns

    def __base_fit(self, X_pos: pl.DataFrame, X_neg: pl.DataFrame, feature_count):
        size = X_pos.shape[0]
        prime_patterns = []
        prev_degree_non_prime_patterns = [set()]
        features = X_pos.columns
        max = self.__max_terms
        if max > feature_count or max == 0:
            max = feature_count
        print(max)
        pos = 0
        neg = 0
        for d in range(1, max + 1):
            if len(X_pos) == 0:
                break
            curr_degree_non_prime_patterns = []
            for curr_base_patterns in tqdm(
                prev_degree_non_prime_patterns,
                desc=f"{d} depth rule generation",
            ):
                if len(X_pos) == 0:
                    break
                largets_idx_of_terms_in_curr_patterns = -1

                for idx, feature in enumerate(features):
                    if (True, feature) in curr_base_patterns or (
                        False,
                        feature,
                    ) in curr_base_patterns:
                        largets_idx_of_terms_in_curr_patterns = idx

                start_of_range = largets_idx_of_terms_in_curr_patterns + 1

                for i in range(start_of_range, feature_count):
                    for possible_term in [True, False]:
                        if len(X_pos) == 0:
                            break
                        should_break = False
                        possible_next_pattern = copy.deepcopy(curr_base_patterns)
                        possible_next_pattern.add((possible_term, features[i]))
                        for term in possible_next_pattern:
                            test_pattern = copy.deepcopy(possible_next_pattern)
                            test_pattern.discard(term)
                            if test_pattern not in prev_degree_non_prime_patterns:
                                should_break = True
                                break
                        if should_break:
                            continue
                        filters = [
                            pl.col(column_name) == desired_value
                            for desired_value, column_name in possible_next_pattern
                        ]
                        filter = filters[0]
                        for f in filters[1:]:
                            filter &= f
                        pos_count_prime = len(X_pos.filter(filter))

                        if self.__base_recall <= (pos_count_prime + pos) / size:
                            pos_count = len(X_pos.filter(filter))
                            neg_count = len(X_neg.filter(filter))

                            pos_pct = pos_count + pos
                            neg_pct = neg_count + neg
                            base = pos_pct + neg_pct
                            hd = 0.0
                            if base > 0.0:
                                hd = pos_pct / base

                            if hd >= self.__base_precision:
                                pos = pos_pct
                                neg = neg_pct
                                prime_patterns.append(possible_next_pattern)
                                X_pos = X_pos.filter(~filter)
                                X_neg = X_neg.filter(~filter)
                            else:
                                curr_degree_non_prime_patterns.append(
                                    possible_next_pattern
                                )
            prev_degree_non_prime_patterns = curr_degree_non_prime_patterns
        return prime_patterns

    def fit(self, Xbin: pl.DataFrame, y: pl.Series):
        unique = y.unique()
        features = Xbin.columns
        feature_count = len(features)

        self.__rules.clear()
        X = Xbin.hstack([y])
        if self.__balance == "1va":
            for lable in unique:
                X_pos = X.filter(pl.col("label") == lable).drop("label")
                X_neg = X.filter(pl.col("label") != lable).drop("label")

                patterns = (
                    self.__my_fit(X_pos, X_neg)
                    if self.__new_test
                    else self.__base_fit(X_pos, X_neg, feature_count)
                )
                patts = []
                for p in patterns:
                    filters = [
                        pl.col(column_name) == desired_value
                        for desired_value, column_name in p
                    ]
                    filter = filters[0]
                    for f in filters[1:]:
                        filter &= f
                    pos_count = len(X_pos.filter(filter))
                    neg_count = len(X_neg.filter(filter))
                    acc = (pos_count + neg_count) / X.shape[0]
                    patts.append((acc, p))
                self.__rules.append(patts)
        elif self.__balance == "1v1":
            for lable in unique:
                for lable1 in unique:
                    if lable == lable1:
                        continue
                    X_pos = X.filter(pl.col("label") == lable).drop("label")
                    X_neg = X.filter(pl.col("label") == lable1).drop("label")
                    patterns = (
                        self.__my_fit(X_pos, X_neg)
                        if self.__new_test
                        else self.__base_fit(X_pos, X_neg, feature_count)
                    )
                    patts = []
                    for p in patterns:
                        filters = [
                            pl.col(column_name) == desired_value
                            for desired_value, column_name in p
                        ]
                        filter = filters[0]
                        for f in filters[1:]:
                            filter &= f
                        pos_count = len(X_pos.filter(filter))
                        neg_count = len(X_neg.filter(filter))
                        acc = (pos_count + neg_count) / X.shape[0]
                        patts.append((acc, p))
        else:
            lable1 = self.__balance.split("=", 1)[1]
            for lable in unique:
                if lable == lable1:
                    continue
                X_pos = X.filter(pl.col("label") == lable1).drop("label")
                X_neg = X.filter(pl.col("label") == lable).drop("label")
                patterns = (
                    self.__my_fit(X_pos, X_neg)
                    if self.__new_test
                    else self.__base_fit(X_pos, X_neg, feature_count)
                )
                patts = []
                for p in patterns:
                    filters = [
                        pl.col(column_name) == desired_value
                        for desired_value, column_name in p
                    ]
                    filter = filters[0]
                    for f in filters[1:]:
                        filter &= f
                    pos_count = len(X_pos.filter(filter))
                    neg_count = len(X_neg.filter(filter))
                    acc = (pos_count + neg_count) / X.shape[0]
                    patts.append((acc, p))

        return self.__rules
