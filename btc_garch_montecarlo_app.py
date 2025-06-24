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
interval = st.sidebar.selectbox("Temporalidad", options=["1d"], index=0)
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
mean_return = returns.mean() / 100

vol_std = returns.rolling(30).std()
if vol_std.isna().all().all():
    st.error("No hay suficientes datos recientes para calcular la volatilidad.")
    st.stop()

vol_threshold = vol_std.mean().mean() / 100 * umbral_factor
anomaly = volatility_today > vol_threshold

# ========== MOSTRAR ESTADO ==========
st.subheader("🔍 Estado actual del mercado:")
st.write(f"**Volatilidad estimada actual:** {volatility_today:.4f}")
st.write(f"**Umbral de volatilidad:** {vol_threshold:.4f}")
if anomaly:
    st.warning("⚠️ Volatilidad anómala detectada. Ejecutando simulación...")
else:
    st.success("✅ Volatilidad dentro de rango normal. No se recomienda operar impulsivamente.")

# ========== GRÁFICO DE VOLATILIDAD GARCH ==========
st.subheader("📉 Volatilidad estimada por GARCH")

garch_vol = res.conditional_volatility / 100

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(garch_vol, label="Volatilidad GARCH", color='blue')
ax2.axhline(vol_threshold, color='red', linestyle='--', label="Umbral de volatilidad")
ax2.set_title("Volatilidad diaria estimada (modelo GARCH)")
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Volatilidad (σ)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# ========== SIMULACIÓN MONTE CARLO ==========
if anomaly:
    S0 = btc['Close'].iloc[-1]
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

    # ========== GRÁFICO DE SIMULACIÓN ==========
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
