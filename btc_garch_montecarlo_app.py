import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import pandas as pd
import datetime

st.set_page_config(page_title="BTC GARCH + Monte Carlo", layout="wide")
st.title("📈 Simulación BTC: GARCH + Monte Carlo")

# ========== PARÁMETROS ==========
st.sidebar.header("⚙️ Parámetros")
dias = st.sidebar.slider("Días a simular", min_value=10, max_value=90, value=30)
simulaciones_n = st.sidebar.slider("Número de simulaciones", min_value=100, max_value=2000, value=1000)
umbral_factor = st.sidebar.slider("Multiplicador de umbral de volatilidad", min_value=1.0, max_value=3.0, value=1.5)
interval = st.sidebar.selectbox("Temporalidad", options=["1d", "1h", "15m"], index=0)

# ========== AJUSTE DE FECHAS SEGÚN INTERVALO ==========
hoy = datetime.datetime.today()
if interval == "1d":
    start_date = "2020-01-01"
elif interval == "1h":
    start_date = (hoy - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
elif interval == "15m":
    start_date = (hoy - datetime.timedelta(days=60)).strftime('%Y-%m-%d')

# ========== DESCARGA DE DATOS ==========
with st.spinner("Descargando datos de BTC..."):
    btc = yf.download("BTC-USD", start=start_date, interval=interval)

# Mostrar columnas disponibles
st.write("Columnas disponibles:", btc.columns.tolist())

# Validación básica
if btc.empty or 'Close' not in btc.columns:
    st.error("No se pudieron obtener datos válidos de BTC. Intenta con otro intervalo.")
    st.stop()

returns = 100 * btc['Close'].pct_change().dropna()

# ========== MODELO GARCH ==========
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
res = garch_model.fit(disp="off")
forecast = res.forecast(horizon=1)
volatility_today = np.sqrt(forecast.variance.values[-1][0]) / 100
mean
