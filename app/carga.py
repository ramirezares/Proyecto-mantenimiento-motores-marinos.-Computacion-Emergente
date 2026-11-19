"""Lectura de los artefactos exportados de Kaggle. Ninguna página entrena ni ajusta nada."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
ARTEFACTOS = RAIZ / "artifacts"

ESTADOS = ["Normal", "Requires Maintenance", "Critical"]
NUM = ["engine_temp", "oil_pressure", "fuel_consumption", "vibration_level", "rpm",
       "engine_load", "coolant_temp", "exhaust_temp", "running_period", "fuel_consumption_per_hour"]
CAT = ["engine_type", "fuel_type", "manufacturer"]
SEMILLA = 42
NOMBRES = {"mlp": "Perceptrón multicapa", "rnn": "Red recurrente"}


def ruta(nombre):
    """Ruta de un artefacto; si falta, avisa y detiene la página."""
    p = ARTEFACTOS / nombre
    if not p.exists():
        st.error(f"Falta `artifacts/{nombre}`. Descárgalo de la pestaña Output de la versión guardada en Kaggle.")
        st.stop()
    return p


@st.cache_data
def _csv(ruta_archivo):
    return pd.read_csv(ruta_archivo, parse_dates=["timestamp"])


@st.cache_data
def _json(ruta_archivo):
    with open(ruta_archivo, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def _joblib(ruta_archivo):
    return joblib.load(ruta_archivo)


def datos():
    """Dataset con la tendencia simulada (celda 15 del notebook)."""
    df = _csv(str(ruta("marine_engine_data_modificado.csv")))
    return df.sort_values(["engine_id", "timestamp"]).reset_index(drop=True)


def predicciones():
    """Predicciones de prueba de la semilla 42 (celda 40 del notebook)."""
    return _csv(str(ruta("predicciones_prueba.csv")))


def metricas():
    return _json(str(ruta("metricas.json")))


def preprocesador():
    return _joblib(str(ruta("preprocesador.joblib")))


def particion(ids_motores):
    """Repite la partición por motor del notebook (celdas 17 y 18): 35 / 8 / 7 con semilla 42."""
    motores = np.random.default_rng(SEMILLA).permutation(np.sort(np.unique(ids_motores)))
    conjunto = {m: "Entrenamiento" for m in motores[:35]}
    conjunto.update({m: "Validación" for m in motores[35:43]})
    conjunto.update({m: "Prueba" for m in motores[43:]})
    return conjunto


def columnas_prob(clave):
    """Nombres de las columnas de probabilidad de un modelo en predicciones_prueba.csv."""
    return {e: f"prob_{clave}_{e.replace(' ', '_')}" for e in ESTADOS}


def con_desviacion(valor):
    """Da formato a {"media": x, "desv": y} como 'x ± y'."""
    if valor["desv"] is None:
        return f"{valor['media']:.3f}"
    return f"{valor['media']:.3f} ± {valor['desv']:.3f}"
