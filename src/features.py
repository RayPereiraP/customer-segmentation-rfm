import pandas as pd


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Recency (dias desde a última compra), Frequency (nº de compras)
    e Monetary (total gasto) por cliente."""
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    return df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )