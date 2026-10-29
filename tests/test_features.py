import pandas as pd

from src.features import build_rfm


def test_build_rfm():
    df = pd.DataFrame({
        "CustomerID": [1, 1, 2],
        "InvoiceNo": ["A1", "A2", "B1"],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-10", "2011-01-05"]),
        "Revenue": [10.0, 20.0, 5.0],
    })
    rfm = build_rfm(df)
    assert rfm.loc[1, "Frequency"] == 2
    assert rfm.loc[1, "Monetary"] == 30.0
    assert rfm.loc[1, "Recency"] == 1
    assert rfm.loc[2, "Recency"] == 6