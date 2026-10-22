"""Главный пайплайн лабораторной работы №2, вариант 17."""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from src import config as cfg
from src.data_loader import load_data
from src.preprocessing import build_features, select_features, split_and_scale
from src.knn import KNNClassifier, loo_knn
from src.parzen import ParzenClassifier, loo_parzen
from src.potential import PotentialClassifier
from src.compactness import compactness_profile, greedy_remove
from src.logistic import loo_logreg_gamma
from src.utils import print_metrics

warnings.filterwarnings("ignore")

cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name):
    path = cfg.RESULTS_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved {path}")


def main():
    print("=" * 70)
    print(f" Лабораторная работа №2 — Вариант {cfg.SEED}")
    print("=" * 70)

    # ------- Часть 1 -------
    print("\n[Часть 1] Данные и предобработка")
    train, test = load_data()
    train_p, test_p = build_features(train, test)
    X_df, y = select_features(train_p, is_train=True)
    X_test_df, _ = select_features(test_p, is_train=False)
    X_test_df = X_test_df.reindex(columns=X_df.columns, fill_value=0)

    (X_train, X_val, X_test,
     y_train, y_val, y_test, scaler) = split_and_scale(
        X_df.values, y, seed=cfg.SEED
    )
    print(f"  train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # ------- Часть 2. kNN -------
    print(f"\n[Часть 2] kNN ({cfg.METRIC}, uniform)")
    loo_err, best_k = loo_knn(
        X_train, y_train, k_max=cfg.K_MAX,
        metric=cfg.METRIC, weights=cfg.WEIGHTS,
    )
    print(f"  LOO best_k = {best_k}")

    ks = sorted(loo_err.keys())
    plt.figure(figsize=(8, 4))
    plt.plot(ks, [1 - loo_err[k] for k in ks], marker="o")
    plt.axvline(best_k, color="r", ls="--", label=f"best k={best_k}")
    plt.xlabel("k")
    plt.ylabel("LOO accuracy")
    plt.title("kNN: зависимость LOO-accuracy от k")
    plt.grid(alpha=0.3)
    plt.legend()
    save_fig("knn_loo_vs_k")

    print("  Сравнение весов при лучшем k:")
    for w in ["uniform", "distance", "linear", "exponential"]:
        clf = KNNClassifier(k=best_k, metric=cfg.METRIC, weights=w)
        clf.fit(X_train, y_train)
        print_metrics(f"kNN (w={w})", y_val, clf.predict(X_val))

    # sklearn baseline
    sk = KNeighborsClassifier(n_neighbors=best_k, metric=cfg.METRIC)
    sk.fit(X_train, y_train)
    print_metrics("sklearn KNN", y_val, sk.predict(X_val))

    # ------- Часть 3. aNN (Парзен) -------
    print(f"\n[Часть 3] Окно Парзена (kernel={cfg.KERNEL})")
    h_grid = np.linspace(0.05, cfg.H_MAX, 20)
    _, best_h = loo_parzen(
        X_train, y_train, h_grid, kernel=cfg.KERNEL,
        metric=cfg.METRIC, mode="fixed",
    )
    print(f"  LOO best_h = {best_h:.3f}")

    parz = ParzenClassifier(h=best_h, kernel=cfg.KERNEL,
                            metric=cfg.METRIC, mode="fixed")
    parz.fit(X_train, y_train)
    print_metrics(f"Parzen fixed h={best_h:.2f}", y_val, parz.predict(X_val))

    parz_var = ParzenClassifier(kernel=cfg.KERNEL, metric=cfg.METRIC,
                                mode="variable", k=best_k)
    parz_var.fit(X_train, y_train)
    print_metrics("Parzen variable", y_val, parz_var.predict(X_val))

    # ------- Часть 5. Потенциальные функции -------
    print("\n[Часть 5] Метод потенциальных функций")
    pot = PotentialClassifier(h=best_h, kernel=cfg.KERNEL, metric=cfg.METRIC)
    pot.fit(X_train, y_train)
    print_metrics("Potential", y_val, pot.predict(X_val))

    # ------- Часть 4. Профиль компактности + жадное удаление -------
    print("\n[Часть 4] Профиль компактности")
    profile = compactness_profile(X_train, y_train, metric=cfg.METRIC)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(profile) + 1), profile)
    plt.xlabel("m (число эталонов)")
    plt.ylabel("Ошибка 1NN")
    plt.title("Профиль компактности Π(m)")
    plt.grid(alpha=0.3)
    save_fig("compactness_profile")

    print("  Жадное удаление эталонов (max_iter=30)...")
    ref_idx, history = greedy_remove(
        X_train, y_train, metric=cfg.METRIC, max_iter=30,
    )
    print(f"  Итоговое число эталонов: {len(ref_idx)} из {len(X_train)}")
    if history:
        hist_arr = np.array(history)
        plt.figure(figsize=(8, 4))
        plt.plot(hist_arr[:, 0], hist_arr[:, 1], marker="o")
        plt.gca().invert_xaxis()
        plt.xlabel("Размер Ω")
        plt.ylabel("LOO-ошибка 1NN")
        plt.title("Жадное удаление эталонов")
        plt.grid(alpha=0.3)
        save_fig("greedy_removal")

    # PCA-визуализация эталонов
    print("  Визуализация эталонов через PCA...")
    pca = PCA(n_components=2, random_state=cfg.SEED)
    X_pca = pca.fit_transform(X_train)
    ref_mask = np.zeros(len(X_train), dtype=bool)
    if len(ref_idx) > 0:
        ref_mask[ref_idx] = True

    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[~ref_mask, 0], X_pca[~ref_mask, 1],
                c=y_train[~ref_mask], cmap="coolwarm",
                s=15, alpha=0.3, label="Удалённые объекты")
    plt.scatter(X_pca[ref_mask, 0], X_pca[ref_mask, 1],
                c=y_train[ref_mask], cmap="coolwarm",
                s=60, edgecolors="black", linewidths=0.5,
                label="Оставшиеся эталоны")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Эталонные объекты после жадного удаления (PCA)")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig("pca_reference_set")

    # ------- Часть 7. Логистическая регрессия -------
    print("\n[Часть 7] Логистическая регрессия (L2)")
    gammas = [0.01, 0.1, 1.0]
    cv_res, best_g = loo_logreg_gamma(X_train, y_train, gammas)
    print(f"  CV best gamma = {best_g}")

    logreg = LogisticRegression(
        C=1.0 / best_g, penalty="l2", max_iter=1000, random_state=cfg.SEED,
    )
    logreg.fit(X_train, y_train)
    print_metrics("LogReg", y_val, logreg.predict(X_val))

    # ------- Часть 8. Submission -------
    print("\n[Часть 8] Формирование submission")
    candidates = {
        "kNN": KNNClassifier(k=best_k, metric=cfg.METRIC, weights="linear"),
        "Parzen": ParzenClassifier(h=best_h, kernel=cfg.KERNEL,
                                   metric=cfg.METRIC, mode="fixed"),
        "Potential": PotentialClassifier(h=best_h, kernel=cfg.KERNEL,
                                         metric=cfg.METRIC),
        "LogReg": LogisticRegression(C=1.0 / best_g, penalty="l2",
                                     max_iter=1000, random_state=cfg.SEED),
    }
    best_name, best_acc = None, -1
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        acc = (pred == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_name = name
    print(f"  Лучшая модель: {best_name} (val acc={best_acc:.4f})")

    best_clf = candidates[best_name]
    X_test_full = X_test_df.values
    X_test_scaled = scaler.transform(X_test_full)
    test_pred = best_clf.predict(X_test_scaled)

    submission = pd.DataFrame({
        "S.No": test["S.No"].values,
        "isAlive": test_pred.astype(int),
    })
    submission.to_csv(cfg.SUBMISSION_PATH, index=False)
    print(f"  Сохранено: {cfg.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()