"""Логистическая регрессия с L2 и подбором gamma через 5-fold CV."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def loo_logreg_gamma(X, y, gammas):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for g in gammas:
        C = 1.0 / g if g > 0 else 1.0
        scores = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(
                C=C, penalty="l2", max_iter=1000, random_state=42,
            )
            clf.fit(X[tr], y[tr])
            scores.append(accuracy_score(y[te], clf.predict(X[te])))
        results[g] = float(np.mean(scores))
    best_g = max(results, key=results.get)
    return results, best_g