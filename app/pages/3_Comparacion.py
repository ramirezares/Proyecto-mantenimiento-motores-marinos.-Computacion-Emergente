import pandas as pd
import streamlit as st

st.set_page_config(page_title="Comparación", page_icon="⚓", layout="wide")

from carga import ARTEFACTOS, ESTADOS, NOMBRES, con_desviacion, metricas, predicciones

st.title("Comparación de los dos modelos")
m = metricas()
st.caption(f"Conjunto de prueba: {m['filas_prueba']} semanas de {len(m['motores_prueba'])} motores. "
           "Media ± desviación de cinco semillas.")

etiquetas = {"exactitud": "Exactitud", "precision_macro": "Precisión macro", "exhaustividad_macro": "Exhaustividad macro",
             "f1_macro": "F1 macro", "recall_critical": "Exhaustividad Critical",
             "critical_como_normal": "Critical tomados como Normal"}
resumen = pd.DataFrame({modelo: {etiquetas[k]: con_desviacion(v) for k, v in valores.items() if k in etiquetas}
                        for modelo, valores in m["resumen"].items()})
st.dataframe(resumen)

st.subheader("Por clase (cada clase contra las otras dos)")
cols = st.columns(2)
for col, clave in zip(cols, ("mlp", "rnn")):
    tabla = pd.DataFrame(m["por_clase"][clave]["media"]).reindex(ESTADOS)
    col.markdown(f"**{NOMBRES[clave]}**")
    col.dataframe(tabla.round(3))

st.subheader("Matrices de confusión (semilla 42)")
if (ARTEFACTOS / "matrices_prueba.png").exists():
    st.image(str(ARTEFACTOS / "matrices_prueba.png"))
pred = predicciones()
cols = st.columns(2)
for col, clave in zip(cols, ("mlp", "rnn")):
    matriz = pd.crosstab(pd.Categorical(pred["estado_real"], ESTADOS), pd.Categorical(pred[f"pred_{clave}"], ESTADOS),
                         rownames=["real"], colnames=["predicho"], dropna=False)
    col.markdown(f"**{NOMBRES[clave]}**")
    col.dataframe(matriz)

st.subheader("¿La diferencia es real? Prueba de McNemar")
st.caption("Semanas que acierta un modelo y falla el otro. Con p < 0,05 la diferencia no se explica por azar.")
st.dataframe(pd.DataFrame(m["mcnemar"]), hide_index=True)

st.subheader("Semanas Critical que un modelo tomó como Normal (semilla 42)")
graves = pred[(pred["estado_real"] == "Critical") & ((pred["pred_mlp"] == "Normal") | (pred["pred_rnn"] == "Normal"))]
st.dataframe(graves[["engine_id", "timestamp", "estado_real", "pred_mlp", "pred_rnn"]], hide_index=True)
