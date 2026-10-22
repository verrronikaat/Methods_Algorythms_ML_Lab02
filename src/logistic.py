"""Логистическая регрессия с L2 и LOO-подбором gamma."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score


def loo_logreg_gamma(X, y, gammas):
    """LOO-подбор gamma (C = 1/gamma)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    loo = LeaveOneOut()
    results = {}
    for g in gammas:
        C = 1.0 / g if g > 0 else 1.0
        preds = np.zeros(len(y), dtype=int)
        for tr, te in loo.split(X):
            clf = LogisticRegression(
                C=C, penalty="l2", max_iter=3000, random_state=42,
            )
            clf.fit(X[tr], y[tr])
            preds[te] = clf.predict(X[te])
        results[g] = accuracy_score(y, preds)
    best_g = max(results, key=results.get)
    return results, best_g