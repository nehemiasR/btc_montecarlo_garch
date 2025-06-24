import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import pandas as pd
import datetime

st.set_page_config(page_title="BTC GARCH + Monte Carlo", layout="wide")
st.title("📈 Simulación BTC: GARCH + Monte Carlo")

# ====== Parámetros ======
st.sidebar.header("⚙️ Parámetros")
dias = st.sidebar.slider("Días a simular", min_value=10, max_value=90, value=30)
simulaciones_n = st.sidebar.slider("Número de simulaciones", min_value=100, max_value=2000, value=1000)
umbral_factor = st.sidebar.slider("Multiplicador de umbral de volatilidad", min_value=1.0, max_value=3.0, value=1.5)

interval = st.sidebar.selectbox("Temporalidad", options=["1d", "1h", "15m"], index=0)

# Ajustar fecha inicio según interval para asegurar suficiente historia
hoy = datetime.datetime.now()
if interval == "1d":
    start_date = (hoy - datetime.timedelta(days=365*3)).strftime('%Y-%m-%d')  # 3 años para diario
elif interval == "1h":
    start_date = (hoy - datetime.timedelta(days=365)).strftime('%Y-%m-%d')    # 1 año para horario
elif interval == "15m":
    start_date = (hoy - datetime.timedelta(days=90)).strftime('%Y-%m-%d')     # 3 meses para 15min

with st.spinner(f"Descargando datos BTC-USD con intervalo {interval}..."):
    btc = yf.download("BTC-USD", start=start_date, interval=interval)

# Revisar que haya datos
if btc.empty:
    st.error("No se descargaron datos. Intenta cambiar la temporalidad o fecha.")
    st.stop()

# Detectar y aplanar MultiIndex en columnas si existe
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

# Verificar que 'Close' exista
if 'Close' not in btc.columns:
    st.error("La columna 'Close' no está disponible en los datos descargados.")
    st.stop()

# Calcular retornos (porcentaje)
returns = 100 * btc['Close'].pct_change().dropna()

# Ajustar ventana rolling según interval para volatilidad
if interval == "1d":
    rolling_window = 30
elif interval == "1h":
    rolling_window = 24 * 7  # 7 días horarios
elif interval == "15m":
    rolling_window = 4 * 24 * 7  # 7 días de 15min

if len(returns) < rolling_window:
    st.error(f"No hay suficientes datos para la ventana de {rolling_window} períodos en intervalo {interval}.")
    st.stop()

vol_std = returns.rolling(rolling_window).std()

# Modelo GARCH
try:
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    res = garch_model.fit(disp="off")
except Exception as e:
    st.error(f"Error al ajustar modelo GARCH: {e}")
    st.stop()

forecast = res.forecast(horizon=1)
volatility_today = np.sqrt(forecast.variance.values[-1][0]) / 100
mean_return = returns.mean() / 100

# Calcular umbral de volatilidad asegurando que sea escalar float
vol_threshold = float(vol_std.mean()) / 100 * umbral_factor
anomaly = volatility_today > vol_threshold

# Mostrar estado mercado con colores y barras
st.subheader("🔍 Estado actual del mercado:")

col1, col2 = st.columns([1,4])
with col1:
    st.metric("Volatilidad estimada actual", f"{volatility_today:.4f}")
    st.metric("Umbral de volatilidad", f"{vol_threshold:.4f}")

with col2:
    fig_garch, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.sqrt(res.conditional_volatility) / 100, label='Volatilidad condicional GARCH')
    ax.axhline(vol_threshold, color='red', linestyle='--', label='Umbral volatilidad')
    ax.set_title("Volatilidad Condicional GARCH vs Umbral")
    ax.set_xlabel("Períodos")
    ax.set_ylabel("Volatilidad")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig_garch)

if anomaly:
    st.warning("⚠️ Volatilidad anómala detectada. Ejecutando simulación Monte Carlo...")
else:
    st.success("✅ Volatilidad dentro de rango normal. No se recomienda operar impulsivamente.")

# Simulación solo si hay anomalía
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

    # Gráfico Monte Carlo
    st.subheader("📊 Simulación Monte Carlo (con GARCH)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(simulaciones, color='skyblue', alpha=0.02)
    ax.plot(expected_path, color='black', label='Promedio esperado')
    ax.plot(p05, color='red', linestyle='--', label='5% percentil')
    ax.plot(p95, color='green', linestyle='--', label='95% percentil')
    ax.set_title(f"Simulación Monte Carlo de BTC ({interval})")
    ax.set_xlabel("Días futuros")
    ax.set_ylabel("Precio BTC (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Resultados
    st.subheader("📌 Resultados:")
    st.write(f"- Precio actual: **{S0:.2f} USD**")
    st.write(f"- Precio esperado en {T} días: **{expected_path[-1]:.2f} USD**")
    st.write(f"- Rango 90% esperado: **{p05[-1]:.2f} USD - {p95[-1]:.2f} USD**")
else:
    st.info("No se generó simulación porque no hay anomalía significativa en la volatilidad.")
