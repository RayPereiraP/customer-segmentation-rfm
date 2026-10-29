import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def scale(rfm: pd.DataFrame) -> np.ndarray:
    # log1p reduz a assimetria (poucos clientes gastam muito)
    return StandardScaler().fit_transform(np.log1p(rfm))


def evaluate_k(X: np.ndarray, k_range=range(2, 9)) -> dict[int, float]:
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
    return scores


def fit_clusters(X: np.ndarray, rfm: pd.DataFrame, k: int) -> pd.Series:
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    # Reordena para o segmento 0 ser o de maior gasto médio
    order = (
        rfm.assign(raw=labels)
        .groupby("raw")["Monetary"].mean()
        .sort_values(ascending=False).index
    )
    mapping = {old: new for new, old in enumerate(order)}
    return pd.Series(labels, index=rfm.index).map(mapping)