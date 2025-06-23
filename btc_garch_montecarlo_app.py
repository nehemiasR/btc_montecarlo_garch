import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import pandas as pd

st.set_page_config(page_title="BTC GARCH + Monte Carlo", layout="wide")
st.title("📈 Simulación BTC: GARCH + Monte Carlo")

# ========== PARÁMETROS ==========
st.sidebar.header("⚙️ Parámetros")
dias = st.sidebar.slider("Días a simular", min_value=10, max_value=90, value=30)
simulaciones_n = st.sidebar.slider("Número de simulaciones", min_value=100, max_value=2000, value=1000)
umbral_factor = st.sidebar.slider("Multiplicador de umbral de volatilidad", min_value=1.0, max_value=3.0, value=1.5)

with st.spinner("Descargando datos de BTC..."):
    btc = yf.download("BTC-USD", start="2020-01-01")

# Debug: mostrar columnas
st.write("Columnas disponibles:", btc.columns.tolist())

# Verifica si existe 'Adj Close'
if 'Adj Close' not in btc.columns:
    st.error("No se encontró la columna 'Adj Close'. La descarga puede estar fallando.")
    st.stop()

returns = 100 * btc['Adj Close'].pct_change().dropna()

# ========== MODELO GARCH ==========
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
res = garch_model.fit(disp="off")
forecast = res.forecast(horizon=1)
volatility_today = np.sqrt(forecast.variance.values[-1][0]) / 100
mean_return = returns.mean() / 100

vol_threshold = returns.rolling(30).std().mean() / 100 * umbral_factor
anomaly = volatility_today > vol_threshold

st.subheader("🔍 Estado actual del mercado:")
st.write(f"**Volatilidad estimada actual:** {volatility_today:.4f}")
st.write(f"**Umbral de volatilidad:** {vol_threshold:.4f}")
if anomaly:
    st.warning("⚠️ Volatilidad anómala detectada. Ejecutando simulación...")
else:
    st.success("✅ Volatilidad dentro de rango normal. No se recomienda operar impulsivamente.")

# ========== SIMULACIÓN ==========
if anomaly:
    S0 = btc['Adj Close'][-1]
    T = dias
    N = simulaciones_n

    simulaciones = np.zeros((T, N))
    for i in range(N):
        precios = [S0]
        for t in range(1, T):
            drift = (mean_return - 0.5 * volatility_today**2)
            shock = volatility_today * np.random.normal()
            precios.append(precios[-1] * np.exp(drift + shock))
        simulaciones[:, i] = precios

    expected_path = simulaciones.mean(axis=1)
    p05 = np.percentile(simulaciones, 5, axis=1)
    p95 = np.percentile(simulaciones, 95, axis=1)

    # ========== GRÁFICO ==========
    st.subheader("📊 Simulación Monte Carlo (con GARCH)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(simulaciones, color='skyblue', alpha=0.02)
    ax.plot(expected_path, color='black', label='Promedio esperado')
    ax.plot(p05, color='red', linestyle='--', label='5% percentil')
    ax.plot(p95, color='green', linestyle='--', label='95% percentil')
    ax.set_title("Simulación Monte Carlo de BTC")
    ax.set_xlabel("Días futuros")
    ax.set_ylabel("Precio BTC (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ========== RESULTADOS ==========
    st.subheader("📌 Resultados:")
    st.write(f"- Precio actual: **{S0:.2f} USD**")
    st.write(f"- Precio esperado en {T} días: **{expected_path[-1]:.2f} USD**")
    st.write(f"- Rango 90% esperado: **{p05[-1]:.2f} USD - {p95[-1]:.2f} USD**")
else:
    st.info("No se generó simulación porque no hay anomalía significativa en la volatilidad.")
