# app.py (versión completa)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import openai
import os
import io
import base64
from transformers import pipeline
import openpyxl

# CONFIGURACIÓN
# Activar si NO estás usando OpenAI
USE_TRANSFORMERS = True  # Cambia a False si prefieres OpenAI

# Cargar modelo gratuito
if USE_TRANSFORMERS:
    @st.cache_resource
    def cargar_chatbot_local():
        return pipeline("text2text-generation", model="google/flan-t5-base")

    chatbot = cargar_chatbot_local()
else:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
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

#def responder_chat(mensaje, df):
 #   resumen_df = df.groupby(['Country', 'Coffee type', 'Year'])['yhat'].sum().reset_index().head(10).to_string(index=False)
  #  prompt = f"""
#Eres un analista de datos sobre consumo de café. Aquí hay un ejemplo de datos proyectados:

#{resumen_df}

#El usuario pregunta: {mensaje}

#Responde con base en los datos, en español.
#"""
 #   respuesta = openai.ChatCompletion.create(
  #      model="gpt-3.5-turbo",
   #     messages=[
    #        {"role": "system", "content": "Eres un asistente que analiza datos proyectados de consumo de café por país, tipo de café y año."},
     #       {"role": "user", "content": prompt}
      #  ]
    #)
    #return respuesta.choices[0].message.content.strip()

def responder_chat(mensaje, df):
    resumen_df = df.groupby(['Country', 'Coffee type', 'Year'])['yhat'].sum().reset_index().head(10)
    ejemplo = resumen_df.to_string(index=False)

    prompt = f"""
Actúa como un analista de datos de consumo de café. Aquí tienes algunas predicciones de consumo:

{ejemplo}

Con base en eso, responde esta pregunta en español: {mensaje}
"""

    if USE_TRANSFORMERS:
        salida = chatbot(prompt, max_length=512, do_sample=False)[0]["generated_text"]
        return salida.strip()
    else:
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un analista que responde preguntas sobre consumo de café."},
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

# ---------- Sección: Modelo por país y tipo (interactivo) -----------

st.header("🔮 Predicción por país y tipo de café (2021–2030)")

# Asegurar limpieza
df_long['Country'] = df_long['Country'].str.strip()
df_long['Coffee type'] = df_long['Coffee type'].str.strip()
df_long = df_long.dropna(subset=['Consumption'])

# Desplegable dinámico
paises = sorted(df_long['Country'].unique())
pais = st.selectbox("🌍 Selecciona país", paises)

tipos_disponibles = sorted(df_long[df_long['Country'] == pais]['Coffee type'].unique())
tipo = st.selectbox("☕ Selecciona tipo de café", tipos_disponibles)

# Botón de acción
if st.button("🔮 Predecir consumo"):
    df_filtered = df_long[
        (df_long['Country'] == pais) &
        (df_long['Coffee type'] == tipo)
    ]

    if df_filtered.shape[0] < 2:
        st.warning(f"❌ No hay suficientes datos para {pais} - {tipo}")
    else:
        df_model = df_filtered.rename(columns={'Consumption': 'y'})
        df_model['ds'] = pd.to_datetime(df_model['Year'], format='%Y')
        df_model = df_model[['ds', 'y']]

        model = Prophet()
        model.fit(df_model)

        ult_ano = df_model['ds'].dt.year.max()
        future = model.make_future_dataframe(periods=2030 - ult_ano, freq='Y')
        forecast = model.predict(future)

        # Mostrar gráfico
        fig = model.plot(forecast)
        plt.title(f"Predicción de consumo en {pais} - {tipo} hasta 2030")
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

        # Mostrar tabla
        forecast['Year'] = forecast['ds'].dt.year
        tabla = forecast[['Year', 'yhat', 'yhat_lower', 'yhat_upper']]
        tabla = tabla[tabla['Year'] > ult_ano].round(2)

        tabla_mostrada = tabla.rename(columns={
            'Year': 'Año',
            'yhat': 'Predicción',
            'yhat_lower': 'Límite inferior',
            'yhat_upper': 'Límite superior'
        })

        st.subheader("📊 Tabla de predicción (2021–2030)")
        st.dataframe(tabla_mostrada.set_index('Año'))

        # Botón de descarga
        towrite = io.BytesIO()
        tabla_mostrada.to_excel(towrite, index=False, sheet_name='Predicción')
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="prediccion_cafe.xlsx">📥 Descargar predicción en Excel</a>'
        st.markdown(href, unsafe_allow_html=True)
        
# ---------- CHATBOT ANALÍTICO ----------

st.subheader("💬 Hazle preguntas al asistente")
mensaje = st.chat_input("Pregúntame sobre los datos proyectados...")

if mensaje:
    with st.spinner("Analizando proyecciones..."):
        respuesta = responder_chat(mensaje, df_pred)
    st.chat_message("user").write(mensaje)
    st.chat_message("assistant").write(respuesta)
    
#st.header("\ud83d\udcac Haz preguntas al asistente")
#df_chat = top_forecast_df.copy()
#mensaje = st.chat_input("Pregúntame sobre las proyecciones de consumo de café...")
#if mensaje:
#    with st.spinner("Analizando proyecciones..."):
#        respuesta = responder_chat(mensaje, df_chat)
#    st.chat_message("user").write(mensaje)
#    st.chat_message("assistant").write(respuesta)
