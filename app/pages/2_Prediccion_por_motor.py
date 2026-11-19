import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predicción por motor", page_icon="⚓", layout="wide")

from carga import ARTEFACTOS, ESTADOS, NOMBRES, NUM, columnas_prob, datos, predicciones

st.title("Predicción por motor")
st.caption("Solo motores de prueba, que los modelos no vieron al entrenar. Predicciones de la semilla 42, desde la semana 12 de cada motor.")

pred = predicciones()
motor = st.selectbox("Motor de prueba", sorted(pred["engine_id"].unique()))
g = pred[pred["engine_id"] == motor].set_index("timestamp").sort_index()

# Alerta de la última semana, como la vería un cliente.
ultima = g.iloc[-1]
c1, c2, c3 = st.columns(3)
c1.metric("Última semana", ultima.name.strftime("%Y-%m-%d"))
for col, clave in ((c2, "mlp"), (c3, "rnn")):
    prob = max(ultima[columnas_prob(clave)[e]] for e in ESTADOS)
    col.metric(NOMBRES[clave], ultima[f"pred_{clave}"], f"probabilidad {prob:.0%}", delta_color="off")
st.caption(f"Estado real de la última semana: **{ultima['estado_real']}**.")

st.subheader("Estado real")
st.line_chart(g["estado_real"].map({e: i for i, e in enumerate(ESTADOS)}).rename("estado (0 Normal, 1 Requires Maintenance, 2 Critical)"),
              height=180)

colores = ["#2ca02c", "#ff7f0e", "#d62728"]
for clave in ("mlp", "rnn"):
    st.subheader(f"Probabilidad de cada estado: {NOMBRES[clave]}")
    probs = g[[columnas_prob(clave)[e] for e in ESTADOS]]
    probs.columns = ESTADOS
    st.line_chart(probs, color=colores, height=250)

st.subheader("Aciertos en este motor")
filas = []
for clave in ("mlp", "rnn"):
    filas.append({"Modelo": NOMBRES[clave],
                  "Exactitud": f"{(g[f'pred_{clave}'] == g['estado_real']).mean():.1%}",
                  "Critical tomados como Normal": int(((g["estado_real"] == "Critical") & (g[f"pred_{clave}"] == "Normal")).sum())})
st.dataframe(pd.DataFrame(filas), hide_index=True)

st.subheader("Últimas 8 semanas")
tabla = g[["estado_real", "pred_mlp", "pred_rnn"]].tail(8)
if (ARTEFACTOS / "marine_engine_data_modificado.csv").exists():
    sensores = datos().set_index(["engine_id", "timestamp"])[NUM]
    tabla = tabla.join(sensores.loc[motor].round(2), how="left")
st.dataframe(tabla)
