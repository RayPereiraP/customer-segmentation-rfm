from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Segmentação de Clientes", page_icon="🧩")
st.title("🧩 Segmentação de clientes (RFM + K-Means)")

path = Path(__file__).parent / "outputs" / "rfm_segments.csv"
if not path.exists():
    st.error("Rode `python train.py` primeiro para gerar os segmentos.")
    st.stop()

df = pd.read_csv(path)
df["Segment"] = "Segmento " + df["Segment"].astype(str)

st.subheader("Perfil médio por segmento")
profile = df.groupby("Segment").agg(
    Clientes=("CustomerID", "count"),
    Recency=("Recency", "mean"),
    Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean"),
).round(1)
st.dataframe(profile)

st.subheader("Clientes por segmento")
st.bar_chart(profile["Clientes"])

st.subheader("Frequência × Gasto")
st.scatter_chart(df, x="Frequency", y="Monetary", color="Segment")

st.subheader("Explorar um segmento")
choice = st.selectbox("Segmento", sorted(df["Segment"].unique()))
st.dataframe(df[df["Segment"] == choice].sort_values("Monetary", ascending=False))