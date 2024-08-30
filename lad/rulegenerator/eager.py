import copy

import polars as pl


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
        selector,
        base_precision=0.5,
        base_recall=0.5,
        max_terms_in_patterns=4,
    ):
        self.__rules = []

        self.__base_precision = base_precision
        self.__base_recall = base_recall
        self.__max_terms = max_terms_in_patterns

        self.__selector = selector

    def predict(self, X: pl.DataFrame) -> pl.Series:
        X = self.__selector.transform(X)
        y = []
        columns = X.columns
        for sample in X.rows():
            prediction = []
            for r in self.__rules:
                out = False
                t_c = 1.0
                for c, rule in r:
                    tmp = True
                    for v, f in rule:
                        tmp &= sample[columns.index(f)] == v
                    out |= tmp
                    if tmp:
                        t_c *= 1 - c

                prediction.append(1 - t_c)
            y.append((pl.Series(prediction)).arg_max())
        return pl.Series("label", y)

    def predict_proba(self, X):
        pass

    def __base_fit(self, X_pos: pl.DataFrame, X_neg: pl.DataFrame, feature_count):
        size = X_pos.shape[0] + X_neg.shape[0]
        prime_patterns = []
        prev_degree_non_prime_patterns = [set()]
        features = X_pos.columns
        max = self.__max_terms
        if max > feature_count or max == 0:
            max = feature_count
        for d in range(1, max):
            if len(X_pos) == 0:
                break
            curr_degree_non_prime_patterns = []
            for curr_base_patterns in prev_degree_non_prime_patterns:
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

                        if self.__base_recall <= 2 * pos_count_prime / len(X_pos):
                            pos_count = len(X_pos.filter(filter))
                            neg_count = len(X_neg.filter(filter))

                            pos_pct = pos_count
                            neg_pct = neg_count
                            base = pos_pct + neg_pct
                            hd = 0.0
                            if base > 0.0:
                                hd = pos_pct / base

                            if hd >= self.__base_precision:
                                hd *= len(X_pos) + len(X_neg)
                                hd /= size
                                prime_patterns.append((hd, possible_next_pattern))
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
        for lable in unique:
            X_pos = X.filter(pl.col("label") == lable).drop("label")
            X_neg = X.filter(pl.col("label") != lable).drop("label")

            patterns = self.__base_fit(X_pos, X_neg, feature_count)
            self.__rules.append(patterns)
        return self.__rules
