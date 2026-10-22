
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from . import config as cfg


CULTURES_GROUPED = {
    "Old Nations": ["valyrian", "first men", "andal", "andals", "rhoynar"],
    "the North": ["northmen", "northern mountain clans", "crannogmen"],
    "the Iron Islands": ["ironborn", "ironmen"],
    "the Mountain and the Vale": [
        "valemen", "vale", "vale mountain clans", "sistermen",
    ],
    "the Isles and Rivers": ["riverlands", "rivermen"],
    "the Rock": ["westerman", "westermen", "westerlands"],
    "the Stormlands": ["stormlander", "stormlands"],
    "the Reach": ["reach", "reachmen", "the reach"],
    "Dorne": ["dornish", "dornishmen", "dorne"],
    "Essos Nations": [
        "astapor", "astapori", "braavosi", "braavos", "tyroshi",
        "lysene", "lyseni", "myrish", "pentoshi", "qartheen", "qarth",
        "dothraki", "lhazarene", "lhazareen", "meereen", "meereenese",
        "norvoshi", "qohor", "summer isles", "summer islands",
        "summer islander", "asshai", "asshai'i", "norvos",
        "ghiscari", "ghiscaricari",
    ],
    "Other Nations": [
        "ibbenese", "westeros", "free folk", "wildling", "wildlings",
        "naathi",
    ],
}


def _invert_cultures():
    inv = {}
    for k, vals in CULTURES_GROUPED.items():
        for v in vals:
            inv[v] = k
    return inv


def preprocess_common(df):
    """Общая предобработка: age, numDeadRelations, culture."""
    df = df.copy()

    # --- age: разбиваем на age_value и age_no_data ---
    if "age" in df.columns:
        df["age_value"] = df["age"].fillna(0)
        df["age_no_data"] = df["age"].isna().astype(int)

    # --- numDeadRelations -> boolDeadRelations ---
    if "numDeadRelations" in df.columns:
        df["boolDeadRelations"] = (df["numDeadRelations"] > 0).astype(int)

    # --- culture -> culture_grouped ---
    if "culture" in df.columns:
        inv = _invert_cultures()
        df["culture_grouped"] = (
            df["culture"].astype(str).str.lower().map(inv)
        )
        df["culture_grouped"] = df["culture_grouped"].fillna("culture_no_data")

    return df


def merge_rare_culture(train_df, test_df, threshold=None):
    """Объединение редких категорий culture_grouped в 'culture_other'."""
    if threshold is None:
        threshold = cfg.RARE_THRESHOLD

    # считаем частоты по train, чтобы категории совпадали в train/test
    counts = train_df["culture_grouped"].value_counts()
    keep = set(counts[counts >= threshold].index)

    def _apply(df):
        df = df.copy()
        df["culture_grouped"] = df["culture_grouped"].where(
            df["culture_grouped"].isin(keep), "culture_other"
        )
        return df

    return _apply(train_df), _apply(test_df)


def encode_popularity_qcut(train_df, test_df, q=10):
    """Popularity -> qcut(q) -> One-Hot."""
    if "popularity" not in train_df.columns:
        return train_df, test_df

    train_df = train_df.copy()
    test_df = test_df.copy()

    _, bins = pd.qcut(train_df["popularity"], q=q, retbins=True,
                      duplicates="drop")
    bins[0] = -np.inf
    bins[-1] = np.inf

    train_df["pop_bin"] = pd.cut(train_df["popularity"], bins=bins,
                                 labels=False)
    test_df["pop_bin"] = pd.cut(test_df["popularity"], bins=bins,
                                labels=False).fillna(-1).astype(int)

    train_df = pd.get_dummies(train_df, columns=["pop_bin"],
                              prefix="pop", dtype=int)
    test_df = pd.get_dummies(test_df, columns=["pop_bin"],
                             prefix="pop", dtype=int)
    train_df, test_df = train_df.align(test_df, join="left",
                                       axis=1, fill_value=0)
    return train_df, test_df


def encode_house_onehot(train_df, test_df):
    """House -> One-HotEncoder."""
    if "house" not in train_df.columns:
        return train_df, test_df

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["house"] = train_df["house"].fillna("house_no_data")
    test_df["house"] = test_df["house"].fillna("house_no_data")

    train_df = pd.get_dummies(train_df, columns=["house"],
                              prefix="house", dtype=int)
    test_df = pd.get_dummies(test_df, columns=["house"],
                             prefix="house", dtype=int)
    train_df, test_df = train_df.align(test_df, join="left",
                                       axis=1, fill_value=0)
    return train_df, test_df


def encode_culture_onehot(train_df, test_df):
    """culture_grouped -> One-Hot, с выравниванием колонок."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df = pd.get_dummies(train_df, columns=["culture_grouped"],
                              prefix="cult", dtype=int)
    test_df = pd.get_dummies(test_df, columns=["culture_grouped"],
                             prefix="cult", dtype=int)
    train_df, test_df = train_df.align(test_df, join="left",
                                       axis=1, fill_value=0)
    return train_df, test_df


def build_features(train_df, test_df):
    """Полный пайплайн предобработки."""
    train_df = preprocess_common(train_df)
    test_df = preprocess_common(test_df)

    # Popularity: qcut(10)
    train_df, test_df = encode_popularity_qcut(
        train_df, test_df, q=cfg.POPULARITY_Q
    )

    # Culture: объединение редких
    train_df, test_df = merge_rare_culture(train_df, test_df)

    # Culture: One-Hot
    train_df, test_df = encode_culture_onehot(train_df, test_df)

    # House: One-Hot
    train_df, test_df = encode_house_onehot(train_df, test_df)

    return train_df, test_df


DROP_COLS = [
    "name", "title", "culture", "dateOfBirth", "mother", "father",
    "heir", "age", "numDeadRelations", "S.No", "isAlive",
]


def select_features(df, is_train=True):
    """Оставляет только числовые признаки."""
    cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[cols].select_dtypes(include=[np.number]).copy()
    X = X.fillna(0)
    y = None
    if is_train and "isAlive" in df.columns:
        y = df["isAlive"].values.astype(int)
    return X, y


def split_and_scale(X, y, seed=None):
    """70/15/15 + StandardScaler."""
    if seed is None:
        seed = cfg.SEED
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=cfg.TEST_RATIO, random_state=seed, stratify=y,
    )
    val_ratio = cfg.VAL_RATIO / (cfg.TRAIN_RATIO + cfg.VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, random_state=seed, stratify=y_tmp,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler