# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
import io
import base64
from transformers import pipeline
#import openpyxl

# CONFIGURACIÓN
USE_TRANSFORMERS = True  # Cambia a False si se utilizara OpenAI

# Chatbot (con Transformers)
if USE_TRANSFORMERS:
    from transformers import pipeline

    @st.cache_resource
    def cargar_chatbot_local():
        return pipeline("text2text-generation", model="google/flan-t5-large")

    chatbot = cargar_chatbot_local()
else:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

def responder_chat(mensaje, df_pred, df_real):
    mensaje = mensaje.lower()

    # --------- Análisis por reglas simples ---------
    respuesta_directa = None

    # ¿Año con mayor consumo proyectado?
    if "año con mayor consumo" in mensaje or "mayor consumo" in mensaje:
        anio = df_pred.loc[df_pred['yhat'].idxmax(), 'Year']
        valor = int(df_pred['yhat'].max())
        respuesta_directa = f"📈 El año con mayor consumo proyectado es **{anio}**, con un total de aproximadamente **{valor:,} tazas** de café."

    # ¿País con menor consumo?
    elif "país que menos" in mensaje:
        pais = df_real.groupby('Country')['Consumption'].sum().idxmin()
        val = int(df_real.groupby('Country')['Consumption'].sum().min())
        respuesta_directa = f"📉 El país que menos café consumió en total es **{pais}**, con aproximadamente **{val:,} tazas**."

    # ¿Consumo total en un año específico?
    elif "consumo total" in mensaje and any(str(a) in mensaje for a in df_real['Year'].unique()):
        for year in df_real['Year'].unique():
            if str(year) in mensaje:
                total = int(df_real[df_real['Year'] == year]['Consumption'].sum())
                respuesta_directa = f"📊 En el año **{year}**, se consumieron aproximadamente **{total:,} tazas** de café en total."
                break

    # ¿Tipos de café en un país?
    elif "tipo de café" in mensaje and "en" in mensaje:
        for pais in df_real['Country'].unique():
            if pais.lower() in mensaje:
                tipos = df_real[df_real['Country'].str.lower() == pais.lower()]['Coffee type'].unique()
                tipos_str = ', '.join(sorted(tipos))
                respuesta_directa = f"☕ En **{pais}** se consumen los siguientes tipos de café: {tipos_str}."
                break

    # --------- Si hubo una respuesta directa, formatearla ---------
    if respuesta_directa:
        prompt = f"""
Convierte esta información técnica en una respuesta clara y profesional en español:

{respuesta_directa}
"""
        salida = chatbot(prompt, max_length=256, do_sample=False)[0]["generated_text"]
        return salida.strip()

    # --------- Si no hay respuesta directa, usa datos para redactar con contexto ---------
    resumen_real = df_real.groupby(['Country', 'Coffee type', 'Year'])['Consumption'].sum().reset_index().head(10).to_string(index=False)
    resumen_pred = df_pred.groupby(['Country', 'Coffee type', 'Year'])['yhat'].sum().reset_index().head(10).to_string(index=False)

    prompt = f"""
Actúa como un analista de datos de café. Usa los siguientes datos para responder la pregunta final:

📊 Datos históricos:
{resumen_real}

📈 Proyecciones:
{resumen_pred}

Pregunta del usuario:
\"\"\"{mensaje}\"\"\"
"""

    salida = chatbot(prompt, max_length=512, do_sample=False)[0]["generated_text"]
    return salida.strip()
    
# Carga de datos reales
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

def descripcion_tendencia(forecast):
    inicio, fin = forecast['yhat'].iloc[0], forecast['yhat'].iloc[-1]
    variacion = fin - inicio
    tendencia = "aumentó" if variacion > 0 else "disminuyó" if variacion < 0 else "se mantuvo estable"
    porcentaje = (variacion / inicio) * 100 if inicio != 0 else 0
    recomendacion = {
        "aumentó": "ampliar presencia en el mercado y reforzar campañas de fidelización.",
        "disminuyó": "ajustar estrategias y buscar nuevos segmentos de mercado.",
        "se mantuvo estable": "conservar esfuerzos actuales e investigar oportunidades emergentes."
    }[tendencia]
    st.markdown(f"""
**📌 Entre {forecast['Year'].min()} y {forecast['Year'].max()}, el consumo {tendencia} de {int(inicio):,} a {int(fin):,} tazas ({porcentaje:.1f}%).**  
**💡 Recomendación de ventas:** {recomendacion}
""")

# APP
st.set_page_config(layout="wide")
st.title("☕ Análisis Global y Proyecciones del Consumo de Café (en tazas)")

df_long = cargar_datos()

# Consumo global total
st.header("🌎 Consumo Global Total Anual")
consumo_global = df_long.groupby('Year')['Consumption'].sum().reset_index()
model_global, forecast_global = entrenar_modelo(consumo_global, 'Year', 'Consumption', horizonte=2035 - consumo_global['Year'].max())
forecast_global["Country"] = "Global"
forecast_global["Coffee type"] = "Total"

fig1 = plt.figure(figsize=(10,4))
plt.plot(forecast_global['Year'], forecast_global['yhat'], label='Predicción')
plt.fill_between(forecast_global['Year'], forecast_global['yhat_lower'], forecast_global['yhat_upper'], alpha=0.3)
plt.title("Consumo Global Total hasta 2035")
plt.xlabel("Año")
plt.ylabel("Tazas de café")
plt.grid(True)
st.pyplot(fig1)
descripcion_tendencia(forecast_global)

# Por tipo de café
st.header("🌿 Consumo Total por Tipo de Café")
tipo_selected = st.selectbox("Selecciona tipo de café:", sorted(df_long['Coffee type'].unique()))
df_tipo = df_long[df_long['Coffee type'] == tipo_selected]
consumo_tipo = df_tipo.groupby('Year')['Consumption'].sum().reset_index()
model_tipo, forecast_tipo = entrenar_modelo(consumo_tipo, 'Year', 'Consumption', horizonte=2035 - consumo_tipo['Year'].max())

fig2 = plt.figure(figsize=(10,4))
plt.plot(forecast_tipo['Year'], forecast_tipo['yhat'], label='Predicción', color='green')
plt.fill_between(forecast_tipo['Year'], forecast_tipo['yhat_lower'], forecast_tipo['yhat_upper'], alpha=0.3, color='lightgreen')
plt.title(f"Consumo de {tipo_selected} hasta 2035")
plt.xlabel("Año")
plt.ylabel("Tazas de café")
plt.grid(True)
st.pyplot(fig2)
descripcion_tendencia(forecast_tipo)

# Top 5 países
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

fig3, ax = plt.subplots(figsize=(10,5))
for pais in top5:
    data = top_forecast_df[top_forecast_df['Country'] == pais]
    ax.plot(data['Year'], data['yhat'], label=pais)
ax.set_title("Consumo Proyectado hasta 2035 - Top 5 Países")
ax.set_xlabel("Año")
ax.set_ylabel("Tazas de café")
ax.grid(True)
ax.legend()
st.pyplot(fig3)
st.markdown("**📌 Se proyecta que estos países mantendrán el liderazgo en consumo hasta 2035.**")

# Interactivo por país y tipo
st.header("🔮 Predicción Interactiva por País y Tipo (2021–2030)")
paises = sorted(df_long['Country'].unique())
pais = st.selectbox("🌍 País", paises)
tipos_disponibles = sorted(df_long[df_long['Country'] == pais]['Coffee type'].unique())
tipo = st.selectbox("☕ Tipo de café", tipos_disponibles)

if st.button("🔮 Predecir consumo"):
    df_filtered = df_long[
        (df_long['Country'] == pais) &
        (df_long['Coffee type'] == tipo)
    ]
    if df_filtered.shape[0] < 2:
        st.warning(f"❌ No hay suficientes datos para {pais} - {tipo}")
    else:
        model = Prophet()
        df_model = df_filtered.rename(columns={'Consumption': 'y'})
        df_model['ds'] = pd.to_datetime(df_model['Year'], format='%Y')
        df_model = df_model[['ds', 'y']]
        model.fit(df_model)

        ult_ano = df_model['ds'].dt.year.max()
        future = model.make_future_dataframe(periods=2030 - ult_ano, freq='Y')
        forecast = model.predict(future)
        forecast['Year'] = forecast['ds'].dt.year

        fig = model.plot(forecast)
        plt.title(f"Predicción de consumo en {pais} - {tipo} hasta 2030")
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        descripcion_tendencia(forecast)

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

        # Descargar como Excel/ CSV
        towrite = io.StringIO()
        tabla_mostrada.to_csv(towrite, index=False)
        b64 = base64.b64encode(towrite.getvalue().encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediccion_cafe.csv">📥 Descargar predicción en CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# CHATBOT
st.subheader("💬 Asistente conversacional sobre café")
st.markdown("Selecciona una pregunta sugerida o escribe la tuya:")

# Lista de preguntas sugeridas
sugerencias = [
    "¿Qué país consume menos café?",
    "¿Cuál es el tipo de café más consumido en Colombia?",
    "¿Cuántas tazas se consumieron en el año 2010?",
    "¿Qué tipo de café crecerá más hasta 2035?",
    "¿El consumo global está aumentando o disminuyendo?",
    "¿En qué año se espera el mayor consumo?",
    "¿Cuál es el país con mayor proyección en 2030?"
]

# Mostrar como botones
cols = st.columns(2)
for i, pregunta in enumerate(sugerencias):
    if cols[i % 2].button(f"💬 {pregunta}"):
        st.session_state["chat_sugerencia"] = pregunta


mensaje = st.chat_input("Haz una pregunta sobre los datos o predicciones...")

# Usar sugerencia si fue seleccionada
if "chat_sugerencia" in st.session_state:
    mensaje = st.session_state.pop("chat_sugerencia")

if mensaje:
    with st.spinner("Analizando..."):
        respuesta = responder_chat(mensaje, df_pred=forecast_global, df_real=df_long)
    st.chat_message("user").write(mensaje)
    st.chat_message("assistant").write(respuesta)
