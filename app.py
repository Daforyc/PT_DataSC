# app.py (versión completa)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import openai
import os
import io
import base64

# CONFIGURACIÓN
openai.api_key = os.getenv("sk-proj-uJ7z6Q54sPl1lS3FZjADT3BlbkFJAeEirwzGd6pJ5aAVhDB7")

# ----------------------- FUNCIONES ------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_parquet("coffee_db.parquet")
    df = df.rename(columns=lambda x: x.strip())
    df_long = df.melt(id_vars=['Country', 'Coffee type'],
                      value_vars=[col for col in df.columns if "/" in col],
                      var_name='Year', value_name='Consumption')
    df_long['Year'] = df_long['Year'].str[:4].astype(int)
    df_long['Country'] = df_long['Country'].str.strip()
    df_long['Coffee type'] = df_long['Coffee type'].str.strip()
    df_long = df_long.dropna(subset=['Consumption'])
    return df_long

def entrenar_modelo(df, columna_fecha='Year', columna_valor='Consumption', horizonte=15):
    df_model = df.rename(columns={columna_valor: 'y'})
    df_model['ds'] = pd.to_datetime(df_model[columna_fecha], format='%Y')
    df_model = df_model[['ds', 'y']]
    model = Prophet()
    model.fit(df_model)
    future = model.make_future_dataframe(periods=horizonte, freq='YE')
    forecast = model.predict(future)
    forecast['Year'] = forecast['ds'].dt.year
    return model, forecast

def graficar_y_describir(titulo, forecast, ult_ano):
    fig = plt.figure(figsize=(10,4))
    plt.plot(forecast['Year'], forecast['yhat'], label='Predicción')
    plt.fill_between(forecast['Year'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3)
    plt.title(titulo)
    plt.xlabel("Año")
    plt.ylabel("Consumo")
    plt.grid(True)
    st.pyplot(fig)

    # Descripción
    inicio, fin = forecast['yhat'].iloc[ult_ano - forecast['Year'].min()], forecast['yhat'].iloc[-1]
    variacion = fin - inicio
    porcentaje = (variacion / inicio) * 100 if inicio != 0 else 0
    tendencia = "aumentó" if variacion > 0 else "disminuyó" if variacion < 0 else "se mantuvo estable"
    st.markdown(f"**\u2705 Entre {forecast['Year'].min()} y {forecast['Year'].max()}, el consumo {tendencia} de {inicio:.0f} a {fin:.0f} ({porcentaje:.1f}%).**")

def responder_chat(mensaje, df):
    resumen_df = df.groupby(['Country', 'Coffee type', 'Year'])['yhat'].sum().reset_index().head(10).to_string(index=False)
    prompt = f"""
Eres un analista de datos sobre consumo de café. Aquí hay un ejemplo de datos proyectados:

{resumen_df}

El usuario pregunta: {mensaje}

Responde con base en los datos, en español.
"""
    respuesta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente que analiza datos proyectados de consumo de café por país, tipo de café y año."},
            {"role": "user", "content": prompt}
        ]
    )
    return respuesta.choices[0].message.content.strip()

# ----------------------- APP STREAMLIT ------------------------
st.set_page_config(layout="wide")
st.title("☕ Análisis Global y Proyecciones del Consumo de Café")
df_long = cargar_datos()

# ---------- GRÁFICO 1: Consumo Global Anual ----------
st.header("🌎 Consumo Global Total Anual")
consumo_global = df_long.groupby('Year')['Consumption'].sum().reset_index()
model_global, forecast_global = entrenar_modelo(consumo_global, 'Year', 'Consumption', horizonte=2035 - consumo_global['Year'].max())
graficar_y_describir("Consumo Global Total hasta 2035", forecast_global, consumo_global['Year'].max())

# ---------- GRÁFICO 2: Consumo Total por Tipo de Café ----------
st.header("🌿 Consumo Total por Tipo de Café")
tipo_selected = st.selectbox("Selecciona tipo de café para ver proyección:", sorted(df_long['Coffee type'].unique()))
df_tipo = df_long[df_long['Coffee type'] == tipo_selected]
consumo_tipo = df_tipo.groupby('Year')['Consumption'].sum().reset_index()
model_tipo, forecast_tipo = entrenar_modelo(consumo_tipo, 'Year', 'Consumption', horizonte=2035 - consumo_tipo['Year'].max())
graficar_y_describir(f"Consumo de {tipo_selected} hasta 2035", forecast_tipo, consumo_tipo['Year'].max())

# ---------- GRÁFICO 3: Top 5 Países por Consumo Total ----------
st.header("🏆 Consumo Anual de Café en los 5 Países con Mayor Consumo")
top5 = df_long.groupby('Country')['Consumption'].sum().nlargest(5).index.tolist()
df_top5 = df_long[df_long['Country'].isin(top5)]
consumo_top5 = df_top5.groupby(['Year', 'Country'])['Consumption'].sum().reset_index()
top_forecast_df = pd.DataFrame()
for pais in top5:
    pais_data = consumo_top5[consumo_top5['Country'] == pais]
    model, forecast = entrenar_modelo(pais_data, 'Year', 'Consumption', horizonte=2035 - pais_data['Year'].max())
    forecast['Country'] = pais
    forecast['Coffee type'] = 'Total'
    top_forecast_df = pd.concat([top_forecast_df, forecast])

fig, ax = plt.subplots(figsize=(10,5))
for pais in top5:
    data = top_forecast_df[top_forecast_df['Country'] == pais]
    ax.plot(data['Year'], data['yhat'], label=pais)
ax.set_title("Consumo Proyectado hasta 2035 - Top 5 Países")
ax.set_xlabel("Año")
ax.set_ylabel("Consumo")
ax.grid(True)
ax.legend()
st.pyplot(fig)
st.markdown("**\u2705 Se proyecta que estos países mantendrán el liderazgo en consumo hasta 2035, con variaciones de crecimiento por país.**")

# ---------- CHATBOT ANALÍTICO ----------
st.header("\ud83d\udcac Haz preguntas al asistente")
df_chat = top_forecast_df.copy()
mensaje = st.chat_input("Pregúntame sobre las proyecciones de consumo de café...")
if mensaje:
    with st.spinner("Analizando proyecciones..."):
        respuesta = responder_chat(mensaje, df_chat)
    st.chat_message("user").write(mensaje)
    st.chat_message("assistant").write(respuesta)
