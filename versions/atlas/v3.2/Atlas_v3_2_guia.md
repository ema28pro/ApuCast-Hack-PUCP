# Atlas v3.2 | ShadowBytes — manual reproducible

## Alcance y estado
- Récord público comunicado por el equipo: Ultralisk v2.7 = 0.49619. No se ha obtenido score público de Atlas v3.2.
- Código completo: `Atlas_v3_2.py`. Notebook autónomo: `Atlas_v3_2.ipynb` (escribe los mismos bytes del script y lo invoca).
- El proxy CPU ha sido ejecutado con el dataset oficial y la caché previa de Narrow; no representa el ensemble completo. El smoke CPU completó las cinco variantes, escribió su reporte y exportó los checkpoints; el comando externo agotó su tiempo después de imprimir el resultado. Se comprobó posteriormente la integridad de su manifiesto y una inferencia sin Y. No se ejecutó CUDA, refit final ni CSV competitivo.
- Use un directorio nuevo para cada etapa; no tome archivos de un directorio incompleto de una ejecución previa como resultados nuevos. Los manifiestos verifican integridad, no certifican el origen de una caché suministrada por terceros.

## Tareas y funciones del script

| Tarea | Funciones | Contraste |
|---|---|---|
| T1 | `run_audit`, `evaluate` | score global/diario/semanal, hora UTC, extremos, máscaras |
| T2 | `baseline`, `optimum`, `nested`, `fit_chain`, `chain` | Narrow/Ultralisk vs Atlas, pesos y correlaciones fuera de calibración |
| T3 | `fit_anchor`, `anchor_predict`, `crossfit`, `train_net` | AtlasPhysicsInSample vs AtlasPhysicsOOF: mismas filas OOF, seeds, ancho, épocas y pérdida |
| T4 | `train_net(kind='wind')`, `MixWind`, `MixFull` | MixPhysics vs MixFull; el viento no cambia RH/T |
| T5 | `history`, `design`, `model_new` | AnchorRaw vs Anchor; RawOOF vs PhysicsOOF; PhysicsOOF vs AuxOOF |
| T6 | `manifest`, `verify_manifest`, `refit`, `infer` | recarga Anchor, checkpoint FP32, inferencia sin Y |

## Ejecución

Requisitos: Python 3, numpy, pandas, scipy, scikit-learn, PyTorch 2.x, matplotlib, IPython. Para full/refit: dos Tesla T4, entorno de Kaggle, dataset oficial extraído (`train.npz`, `test.npz`, `static.npz`, `sample_submission.csv`). No descarga datos externos. El script conserva las arquitecturas de Ultralisk mediante el código de auditoría previo incrustado. Usa el método Stack + NarrowSpatial para reconstruir la referencia del nuevo corte, no el CSV del test como validación.

```bash
# 1. Proxy CPU, con la caché previa que contiene NarrowSinSesgo y AtlasAnchor.
python Atlas_v3_2.py --mode proxy --data /ruta/dataset --cache /ruta/validation_predictions.npz --out ./atlas32_proxy
# 2. Smoke CPU (submuestreo, 1 época, 1 seed, ancho 16).
python Atlas_v3_2.py --mode smoke --data /ruta/dataset --out ./atlas32_smoke
# 3. Validación completa T4 x2 (una única ventana).
python Atlas_v3_2.py --mode full --data /kaggle/input --out /kaggle/working/Atlas-v3.2-full --epochs 48
# 4. Reutilizar caché exacta del corte, si existe (nuevo directorio).
python Atlas_v3_2.py --mode full --data /kaggle/input --cache /kaggle/working/Atlas-v3.2-full/ultra_family_cache.npz --out /kaggle/working/Atlas-v3.2-repeat --epochs 48
```

La reconstrucción completa usa 10 redes Ultralisk y 10 redes Atlas (5 variantes × 2 seeds), dos GPU en paralelo; 48 épocas por variante Atlas. Estimación de planificación, no benchmark: 15–45 min para full, según coste real de ConvLSTM, hardware y escritura de artefactos; refit añade aproximadamente 5–20 min. Medir por log antes de reservar GPU. No usar smoke como estimación de score ni de velocidad GPU.

## Protocolo y gate
- Train `T < 2024052100`; calibración `[2024052100,2024052800)`; evaluación `[2024052800,2024070100)` horas múltiplos de 3. 1999/56/256 muestras; 32 días observados. No hay etiquetas del 31/05 ni 22/06. Sin recalibrar con Y de evaluación. Las horas anteriores se toman de los datos GFS oficiales y las faltantes llevan máscara.
- Calibración de referencia expansiva: los dos primeros días inicializan el baseline; cada día posterior se predice con pesos/bias ajustados solo con días anteriores. En esos cinco días prequential se selecciona contracción de pesos mediante validación expansiva interna. Se ajustan los pesos finales con esos cinco días y se congelan. El baseline primario se calibra con los siete días antes de evaluar. Esto introduce una pequeña diferencia entre el baseline prequential y el final, deliberada para evitar ajuste secundario in-sample. No se seleccionan pesos por semana de evaluación.
- Pesos por enlace [0, 0.5]; cada mezcla convexa preserva suma 1, pero las contribuciones totales de Atlas en cadenas pueden superar 0.5. MixFull está predeclarado como candidato principal, no se elige por score retrospectivo. Se conserva el peso cero si gana el baseline dentro de la calibración.
- Bootstrap pareado móvil, 5000 réplicas para bloques de 3 y 7 días calendario; agrega SSE y recuenta píxeles-hora antes de recalcular RMSE global. Los días ausentes conservan cero observaciones, no inventan etiquetas. Umbral max(0.01311, 1.96 * max(SE3,SE7)). 0.01311 es una penalización conservadora por la discrepancia histórica observada (0.01874 - 0.00563), no una estimación estadística del ruido público. Exige además IC inferior >= +0.003 en ambos, mejora >= 65% de días observados y ganancia no negativa en últimos 14 días. Un solo candidato primario; ablations exploratorias sin ajuste por comparaciones múltiples.
- Una única ventana histórica reutilizada no permite certificar generalización ciega. El bootstrap mide variabilidad temporal condicional al modelo y ventana, no incorpora incertidumbre por selección, seed ni dominio público/privado. El gate habilita revisión humana, nunca envío automático.

## Resultados CPU reales (proxy)

| Modelo | Score |
|---|---:|
| NarrowProxy | 0.415114 |
| AtlasAnchorRaw | 0.398189 |
| AtlasAnchor | 0.402972 |
| MixAnchorRaw | 0.419253 |
| MixAnchorUV | 0.421645 |
| MixAnchor | 0.424106 |

MixAnchor: +0.00899127; 30/32 días; IC95% 3 días [0.00512327,0.01358467], 7 días [0.00410140,0.01529014]; últimos 14 días +0.00505652. Gate no superado; además no es Ultralisk completo. Anchor exportado reproduce el Anchor antiguo con diferencia absoluta máxima 3.05e-5 y delta skill ~4e-10. Las máscaras indican historia incompleta en 1.15% train, 0% cal, 0.78% eval. El contraste físico Anchor vs Raw es +0.004783; no prueba todavía el aporte de la red. No usar las puntuaciones smoke para evaluar capacidad predictiva.

## Cross-fitting y ablation
- Bloques de 7 días, ventana expansiva desde 28 días de burn-in y gap de 24 horas. Se excluyen primeras semanas sin historia suficiente. Todos los normalizadores, EOF y Ridge se ajustan dentro del entrenamiento de cada bloque. Las redes in-sample y OOF entrenan exactamente con las mismas filas cubiertas. Semillas 42,123, ancho 48, 48 épocas por defecto, misma pérdida base; sin dropout (fijo para controlar la comparación paralela).
- Raw: RH,T,U,V + DEM, coordenadas, reloj y máscara; elimina variables atmosféricas derivadas del Anchor y red y canales topográficos derivados de la red. Physics: variables físicas, relieve e insolación. Aux: añade pérdidas auxiliares a Physics. Wind: pérdidas U/V 0.35/0.65 y gradiente, descarta salidas RH/T al inferir. Estos indicadores son aproximaciones, no dinámica WRF explícita. El contraste Raw/Physics cambia la representación de ambos bloques, no identifica causalmente cada variable individual.
- La red aprende una corrección sin recibir el Anchor como entrada explícita (se suma al final). Es una limitación de esta arquitectura; no confundir con un operador físico completo. El cambio OOF puede aumentar el tamaño de residuos y el ruido; se valida con mismas filas, no con entrenamiento mayor para una rama.

## Artefactos y despliegue

Una vez que `report.json` de full diga `review_eligible=true`, revisar las métricas y aprobar explícitamente:

```bash
python Atlas_v3_2.py --mode refit --data /kaggle/input --audit /kaggle/working/Atlas-v3.2-full --out /kaggle/working/Atlas-v3.2-refit --approve
python Atlas_v3_2.py --mode infer --bundle /kaggle/working/Atlas-v3.2-refit --input /ruta/test.npz --base-csv /ruta/submission_ultralisk_v2_7.csv --template /ruta/sample_submission.csv --out /kaggle/working/Atlas-v3.2-inference
```

Refit ajusta Anchor + ramas Physics/Wind con todas las etiquetas oficiales de train; congela la política derivada de auditoría. La calibración del nuevo entrenamiento puede cambiar, por eso la transferencia de pesos es una incertidumbre explícita. Refit verifica hash de código, datos y todos los archivos de auditoría. Requiere GPU y gate; no hay bypass smoke/proxy. El script de inferencia no abre `train.npz`, no lee Y, y produce `submission_atlas_v3_2_REVIEW.csv` solo si recibe un bundle refit y el CSV v2.7 de hash exacto. Alternativamente `--base-npz` con arrays `times`, `prediction` permite evaluar un forecast operativo; sin base se exporta Atlas únicamente. El CSV histórico es válido para competición, pero no es un modelo operativo: reconstruir Ultralisk para fechas nuevas requiere sus propios artefactos de inferencia (no incluidos en el bundle Atlas). No afirmar despliegue operativo del ensemble sin ese componente.

`anchor_physics.npz`: clima GFS/WRF, normalización, centro/componentes/escala PCA, mapa de celdas, medias/escalas Ridge y coeficientes por celda. `*.pt`: EMA, normalizadores, ancho, arquitectura, seed, epochs. `static.npz`: geometría. `bundle.json`: orden de mezcla y pesos. `manifest.json`: SHA-256 de datos, código, artefactos, semillas, cortes, versiones y tolerancias. `validation_predictions.npz` incluye Y histórico solo para auditoría, no se usa en inferencia. Tolerancias propuestas: replay por valor 1e-4 (FP32), reproducción skill 5e-4 entre entornos (no garantizada). CUDA puede tener operaciones no deterministas aun con configuración determinista en modo warning; se debe medir la desviación en dos ejecuciones independientes. No se ha validado el refit final ni la ruta CSV completa con CUDA.

## Archivos de resultados
`validation_metrics.csv`, `weekly_scores.csv`, `daily_scores.csv`, `daily_scores.png`, `diagnostics.csv`, `error_correlations.csv`, `paired_ablations.json`, `crossfit_manifest.csv`, `report.json`, `manifest.json`. `diagnostics.csv`: RMSE global por hora y estado de máscara, y RMSE individual en colas (percentiles 5/95 de train). La caché de referencia evita repetir entrenamiento solo si coincide fingerprint exacto de datos, corte y código Ultralisk. Para medir sensibilidad a seeds, comparar checkpoints y repetir ejecución en directorio nuevo; no se ha estimado dispersión por seed con GPU.
