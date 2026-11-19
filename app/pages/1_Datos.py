import pandas as pd
import streamlit as st

st.set_page_config(page_title="Datos", page_icon="⚓", layout="wide")

from carga import ESTADOS, NUM, datos, metricas, particion

st.title("Datos")
df = datos()
conjunto = particion(df["engine_id"])
df["conjunto"] = df["engine_id"].map(conjunto)

assert sorted(m for m, c in conjunto.items() if c == "Prueba") == sorted(metricas()["motores_prueba"]), \
    "La partición no coincide con la del notebook"

c1, c2, c3 = st.columns(3)
c1.metric("Semanas", f"{len(df):,}".replace(",", "."))
c2.metric("Motores", df["engine_id"].nunique())
c3.metric("Semanas por motor", int(df.groupby("engine_id").size().median()))

st.subheader("Estados de mantenimiento")
st.caption("Después de la modificación del dataset. En el original, las tres clases tenían un tercio cada una y no se relacionaban con los sensores.")
tabla = pd.crosstab(df["maintenance_status"], df["conjunto"]).reindex(ESTADOS)[["Entrenamiento", "Validación", "Prueba"]]
st.bar_chart(tabla)
st.dataframe(tabla)

st.subheader("Partición por motor")
st.caption("Ningún motor aparece en más de un conjunto: 35 de entrenamiento, 8 de validación y 7 de prueba (semilla 42).")
part = (df.groupby("conjunto")["engine_id"].agg(lambda s: ", ".join(sorted(s.unique())))
        .reindex(["Entrenamiento", "Validación", "Prueba"]))
for nombre, motores in part.items():
    st.markdown(f"**{nombre}:** {motores}")

st.subheader("Sensores de un motor")
c1, c2 = st.columns(2)
motor = c1.selectbox("Motor", sorted(df["engine_id"].unique()))
sensor = c2.selectbox("Sensor", NUM, index=NUM.index("coolant_temp"))
g = df[df["engine_id"] == motor]
st.scatter_chart(g, x="timestamp", y=sensor, color="maintenance_status")
st.caption(f"{motor} pertenece al conjunto de {conjunto[motor].lower()}. Cada punto es una semana, coloreada por su estado.")
