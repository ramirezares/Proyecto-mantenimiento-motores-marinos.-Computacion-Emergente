

# Clasificación del estado de mantenimiento de motores marinos
## Proyecto mantenimiento motores marinos. Computación Emergente



Proyecto 2 de Computación Emergente (UNIMET). Compara un perceptrón multicapa y una red recurrente (LSTM) para clasificar el estado de mantenimiento de motores marinos en tres clases: Normal, Requires Maintenance y Critical.

- **Notebook público de Kaggle (donde se ejecuta todo):** [Notebook en Kaggle](https://www.kaggle.com/code/aresramirez/notebook-proyecto-parte-2-computaci-n-emergente)
- **Informe (Google Doc):** [Informe](https://docs.google.com/document/d/1zJh0yUo8IMJbF-TdEAEpt-4ijl0j3UUlZUEHaH1V_ps/edit?usp=sharing)
- **Integrantes:** Giselle Esclasans, Daniela Zambrano, Ares Ramírez

## Datos

Dataset [Preventive Maintenance for Marine Engines](https://www.kaggle.com/datasets/jeleeladekunlefijabi/preventive-maintenance-for-marine-engines): 5.200 registros semanales de 50 motores.

**Modificación declarada.** El dataset original no tiene relación entre los sensores y el estado (los modelos de referencia quedan en el azar). Con autorización del profesor, el notebook añade una tendencia de deterioro simulada (celdas 13 a 18, semilla 42). Los resultados corresponden a ese patrón simulado, no a motores reales.

## Resultados en prueba (7 motores, 651 semanas, cinco semillas)

| Modelo | Exactitud | F1 macro | Critical tomados como Normal |
|---|---|---|---|
| Clase mayoritaria | 0,551 | 0,237 | 135 |
| Perceptrón multicapa | 0,692 ± 0,004 | 0,633 ± 0,007 | 14,2 ± 1,9 |
| Red recurrente (LSTM, T = 8) | 0,813 ± 0,016 | 0,776 ± 0,020 | 1,8 ± 1,3 |

## Estructura

    notebook/    copia del notebook de Kaggle
    artifacts/   modelos, preprocesador, métricas, predicciones y figuras exportados de Kaggle
    requirements.txt

Los modelos se entrenan solo en Kaggle. Este repositorio guarda copias de los resultados; para reproducirlos, ejecutar el notebook en Kaggle con Save & Run All.
