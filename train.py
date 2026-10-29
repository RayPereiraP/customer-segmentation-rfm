import argparse
import json
from pathlib import Path

from src.clustering import evaluate_k, fit_clusters, scale
from src.data import clean, load_raw
from src.features import build_rfm

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
OUT_DIR = BASE_DIR / "outputs"


def main(k: int):
    OUT_DIR.mkdir(exist_ok=True)

    df = clean(load_raw(DATA_PATH))
    rfm = build_rfm(df)
    print(f"{len(rfm)} clientes analisados")

    X = scale(rfm)
    scores = evaluate_k(X)
    print("Silhouette por k:", {k_: round(v, 3) for k_, v in scores.items()})

    rfm["Segment"] = fit_clusters(X, rfm, k)
    rfm.to_csv(OUT_DIR / "rfm_segments.csv")

    profile = rfm.groupby("Segment").agg(
        Clientes=("Recency", "size"),
        Recency_medio=("Recency", "mean"),
        Frequency_medio=("Frequency", "mean"),
        Monetary_medio=("Monetary", "mean"),
    ).round(1)
    print(f"\nPerfil dos segmentos (k={k}):\n{profile}")

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({"k": k, "silhouette_by_k": scores}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4, help="nº de segmentos")
    main(parser.parse_args().k)