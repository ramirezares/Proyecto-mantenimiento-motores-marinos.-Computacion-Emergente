import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Simulador", page_icon="⚓", layout="wide")

from carga import CAT, ESTADOS, NUM, datos, preprocesador, ruta

st.title("Simulador")
st.markdown("Mueve los sensores de una semana y mira qué estado estima el **perceptrón multicapa**. "
            "La red recurrente no aparece aquí porque necesita las últimas semanas de un motor, no una sola.")

try:
    import torch
    from modelos import PerceptronMulticapa
except ImportError:
    st.error("Esta página necesita PyTorch. Instálalo con `pip install -r requirements.txt`.")
    st.stop()


@st.cache_resource
def cargar_mlps(ruta_archivo):
    ckpt = torch.load(ruta_archivo, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    modelos = []
    for estado in ckpt["estados"]:
        m = PerceptronMulticapa(ckpt["n_entradas"], [cfg["neuronas"]] * cfg["capas"], cfg["dropout"])
        m.load_state_dict(estado)
        m.eval()
        modelos.append(m)
    return modelos


prep = preprocesador()            # el mismo escalador y codificador del notebook; aquí no se ajusta nada
mlps = cargar_mlps(str(ruta("mlp.pt")))
df = datos()

# Punto de partida: una semana real, para no empezar con una combinación imposible.
c1, c2 = st.columns(2)
motor = c1.selectbox("Partir de un motor", sorted(df["engine_id"].unique()))
g = df[df["engine_id"] == motor].reset_index(drop=True)
semana = c2.slider("Semana", 1, len(g), len(g))
base = g.iloc[semana - 1]
st.caption(f"Estado real de esa semana: **{base['maintenance_status']}**.")

valores = {}
izq, der = st.columns(2)
for i, col in enumerate(NUM):
    destino = izq if i % 2 == 0 else der
    lo, hi = float(df[col].quantile(0.005)), float(df[col].quantile(0.995))
    if hi <= lo:                                  # columnas casi constantes (p. ej. exhaust_temp)
        lo, hi = float(df[col].min()), float(df[col].max())
    if hi <= lo:
        valores[col] = lo
        destino.caption(f"{col}: constante en los datos ({lo:g})")
        continue
    inicial = float(np.clip(base[col], lo, hi))
    valores[col] = destino.slider(col, lo, hi, inicial, key=f"{motor}-{semana}-{col}")
categorias = dict(zip(CAT, prep.named_transformers_["cat"].categories_))
for col in CAT:
    opciones = list(categorias[col])
    valores[col] = st.selectbox(col, opciones, index=opciones.index(base[col]) if base[col] in opciones else 0)

X = prep.transform(pd.DataFrame([valores])[NUM + CAT]).astype(np.float32)
with torch.no_grad():
    probs = np.stack([torch.softmax(m(torch.from_numpy(X)), 1).numpy()[0] for m in mlps])
media, desv = probs.mean(0), probs.std(0)

estado = ESTADOS[int(media.argmax())]
st.metric("Estado estimado", estado, f"probabilidad {media.max():.0%}", delta_color="off")
st.bar_chart(pd.DataFrame({"probabilidad": media}, index=ESTADOS), horizontal=True, height=200)
st.caption("Promedio de los cinco perceptrones entrenados con distintas semillas. "
           "Desviación entre semillas: " + ", ".join(f"{e} {d:.2f}" for e, d in zip(ESTADOS, desv)) + ".")
