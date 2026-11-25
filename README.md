# ApuCast v3

ApuCast transforma pronósticos globales GFS en mapas de 3 km de humedad, temperatura y viento para la Cordillera Blanca. La rama de trabajo es `production`. Las versiones anteriores están en `versions/`.

## CSV para la hackathon

El archivo listo para subir es [`submission_v3.csv`](versions/v3/submission_v3.csv). Tiene 646 800 filas, conserva los IDs y el orden de la plantilla oficial y limita la humedad a 0–100 %. La temperatura está en Kelvin y el viento en m/s. El [manifiesto](versions/v3/submission_v3.json) registra su SHA-256 y los pesos de mezcla.

El candidato promedió **0,421261** en validación retrospectiva de tres cortes y tres semillas. La mezcla 50/50 promedió **0,420624** con el mismo protocolo. El score público de v3 está pendiente. [Scores por corte y semilla](versions/v3/scores_temporadas_semillas.csv)

## Versiones

| Carpeta | Contenido |
| --- | --- |
| [`v3`](versions/v3/) | CSV, manifiesto, resultados del backtest y pesos ResNet del candidato. |
| [`v2.1-production`](versions/v2.1-production/) | Mezcla fija 50/50 con guías, scripts y validación exploratoria. |
| [`v2-top`](versions/v2-top/) | Tres CNN archivadas, preprocesamiento y registro del envío de referencia. |
| [`v3-experimental`](versions/v3-experimental/) | Experimentos y diagnósticos de la rama histórica `modelo/phaletasss-max-v3`. |

## Regeneración en Kaggle

1. Abre [`train/submission_notebook.ipynb`](train/submission_notebook.ipynb) con un kernel de Kaggle. Agrega los datos oficiales de `hackathon-ia-pucp-2026` y selecciona **GPU T4 x2**.
2. Ejecuta todas las celdas con `PRUEBA_RAPIDA = False`. El notebook carga las tres CNN archivadas, entrena tres ResNet durante 25 épocas y aplica los pesos de mezcla versionados.
3. Descarga `/kaggle/working/submission_v3.csv` junto con su JSON. Usa el hash del manifiesto de esa ejecución para identificar el archivo si vuelves a entrenar.

El backtest está en [`train/backtest_mezcla.py`](train/backtest_mezcla.py). Entrena antes de cada ventana de calibración, ajusta una mezcla por variable en los siete días previos y evalúa en el corte que sigue. Los datos oficiales se adjuntan desde Kaggle o se colocan en `data/`.

La demo de heladas en `app/` usa la interpolación de la plantilla. La integración del modelo v3 en la demo sigue pendiente.
