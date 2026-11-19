"""Portada de la demostración. Ejecutar desde la raíz del repositorio con: streamlit run app/app.py"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mantenimiento de motores marinos", page_icon="⚓", layout="wide")

from carga import con_desviacion, metricas

st.title("Alerta de mantenimiento para motores marinos")
st.markdown(
    "Demostración de un servicio que lee los sensores semanales de un motor y estima si está **Normal**, "
    "si **requiere mantenimiento** o si está en estado **crítico**. Compara dos redes neuronales: "
    "un perceptrón multicapa, que mira cada semana por separado, y una red recurrente, que mira las últimas semanas."
)
st.warning(
    "Los datos tienen una tendencia de deterioro **simulada**. El dataset original no relacionaba los sensores con el estado; "
    "con autorización del profesor se añadió un patrón de desgaste (celdas 13 a 18 del notebook). "
    "Los resultados muestran cómo funcionan los modelos sobre ese patrón, no sobre motores reales."
)

m = metricas()
st.subheader(f"Resultados en prueba: {m['filas_prueba']} semanas de {len(m['motores_prueba'])} motores, cinco semillas")
filas = []
for modelo, valores in m["resumen"].items():
    filas.append({"Modelo": modelo,
                  "Exactitud": con_desviacion(valores["exactitud"]),
                  "F1 macro": con_desviacion(valores["f1_macro"]),
                  "Exhaustividad Critical": con_desviacion(valores["recall_critical"]),
                  "Critical tomados como Normal": con_desviacion(valores["critical_como_normal"])})
st.dataframe(pd.DataFrame(filas), hide_index=True)

st.markdown(
    "**Páginas** (menú de la izquierda):\n"
    "- **Datos:** clases, partición por motor y sensores de cada motor.\n"
    "- **Predicción por motor:** probabilidad de cada estado semana a semana en los motores de prueba.\n"
    "- **Comparación:** métricas por clase, matrices de confusión y prueba de McNemar.\n"
    "- **Simulador:** mueve los sensores y mira cómo responde el perceptrón."
)
st.caption("Todos los números salen del notebook público de Kaggle; esta interfaz solo los muestra.")
