# Customer Segmentation (RFM + K-Means)

Segmentação de clientes de um e-commerce com base em Recência, Frequência e Valor monetário (RFM), usando K-Means, com dashboard em Streamlit.

## Dataset
[E-Commerce Data (Kaggle)](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
Baixe o `data.csv` e coloque em `data/`.

## Abordagem
- Limpeza: remoção de clientes sem ID, cancelamentos e valores inválidos
- Features RFM por cliente
- `log1p` + `StandardScaler`, porque os gastos são muito assimétricos
- K-Means com k escolhido considerando silhouette e interpretabilidade de negócio
- Segmentos reordenados pelo gasto médio (segmento 0 = maior valor)

## Estrutura
- `src/data.py`: carga e limpeza
- `src/features.py`: cálculo do RFM
- `src/clustering.py`: escala, avaliação de k e clusterização
- `train.py`: pipeline completo
- `app.py`: dashboard
- `tests/`: testes unitários

## Como rodar
```bash
pip install -r requirements.txt
python -m pytest
python train.py --k 4
streamlit run app.py
```

## Perfil dos segmentos
| Segmento | Clientes | Recency | Frequency | Monetary | Interpretação |
|---|---|---|---|---|---|
| 0 | XX | XX | XX | XX | ... |

![Dashboard](dashboard.png)

## Stack
Python, pandas, scikit-learn, Streamlit, pytest