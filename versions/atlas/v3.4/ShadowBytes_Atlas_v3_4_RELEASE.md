# ShadowBytes — Atlas v3.4 RegimeFlow

**Registro de release competitivo | Hackathon IA PUCP 2026 / ApuCast**  
**Fecha:** 26 de noviembre de 2026 (Perú)  
**Estado:** récord público del equipo; referencia congelada para Atlas v3.5.  
**Score público:** `0.49741` · **Posición:** `6.º` en la captura textual comunicada por el equipo.

## 1. Resultado oficial comunicado por el equipo

Kaggle: “New personal best! Your latest submission scored 0.49741, improving on your previous best of 0.49619.”

| Indicador | Valor |
|---|---:|
| Récord anterior: Ultralisk v2.7 | 0.49619 |
| Nuevo récord: Atlas v3.4 | 0.49741 |
| Ganancia absoluta | +0.00122 |
| Mejora relativa del score anterior | +0.2459 % |
| Error normalizado medio anterior (1 − skill) | 0.50381 |
| Error normalizado medio nuevo | 0.50259 |
| Reducción relativa del error normalizado | 0.2422 % |
| Posición anterior comunicada | 8.º |
| Nueva posición comunicada | 6.º |
| Distancia a 0.50000 | 0.00259 |
| Distancia a 0.51000 | 0.01259 |

Se consumió el primer intento propio de los tres presupuestados para hoy. Quedan dos según el reparto indicado por el usuario, sujeto a los envíos de los compañeros y al límite real de Kaggle. El score se registra a partir del mensaje del equipo, no de una consulta independiente a Kaggle.

## 2. Historial público confirmado

| Modelo | Score público | Incremento frente al anterior de esta tabla |
|---|---:|---:|
| ApuCast v3 T4x2 | 0.46485 | — |
| Ultralisk v2.1 Optimized | 0.46025 | −0.00460 |
| Ultralisk v2.3 Hybrid | 0.49056 | +0.03031 |
| Ultralisk v2.7 NarrowSpatial | 0.49619 | +0.00563 |
| **Atlas v3.4 RegimeFlow** | **0.49741** | **+0.00122** |

Ganancia respecto a ApuCast v3: +0.03256 (7.0044 % de su score, 6.0843 % de reducción relativa de su error normalizado). Atlas v3.2.5 y v3.3 no tienen score público confirmado en este registro. No asignar scores de auditoría a submissions.

## 3. Leaderboard comunicado el 26/11/2026

| Puesto | Equipo | Score | Diferencia frente a ShadowBytes |
|---:|---|---:|---:|
| 1 | SciTeX | 0.51938 | +0.02197 |
| 2 | wslm | 0.51025 | +0.01284 |
| 3 | NoTeX | 0.50912 | +0.01171 |
| 4 | GrupoZ | 0.50570 | +0.00829 |
| 5 | PASSWORD | 0.49844 | +0.00103 |
| 6 | ShadowBytes | 0.49741 | — |

Las diferencias indican cuánto falta para empatar, no para superar. La tabla puede cambiar. No interpretar los puestos no incluidos como resultados desconocidos de esos equipos.

## 4. Contrato de datos y métrica

Downscaling contemporáneo GFS → WRF, no predicción del siguiente paso temporal. Entradas GFS t−2, t−1, t sobre 17×13×4 y terreno DEM; salida WRF 70×35×4 en t. Train X 2935×17×13×4; test X 792×17×13×4. La evaluación emplea 264 instantes a intervalos de 3 h y 2450 píxeles por instante (646800 filas). Variables RH (%), T (K), U y V (m/s). No hay presión como entrada ni etiquetas Y en test.

`skill = 1 − mean(RMSE_RH/25.360, RMSE_T/2.0652, RMSE_U/2.6776, RMSE_V/2.0876)`

RMSE global por variable, calculado a partir de SSE / número total de píxeles-instantes, no promedio simple de scores diarios ni de folds. El loss de entrenamiento de las redes no es esta métrica.

## 5. Receta congelada de v3.4

La base es Ultralisk v2.7 NarrowSpatial, un ensemble de diez redes (CNN ×3, ResNet ×3, Mutalisk ×2, Viper ×2), más kernels y corrección espacial. Atlas aporta un ancla estadística física con corrección de sesgo y deriva, sin cambiar la arquitectura de las redes originales.

Por variable: `pred = (1-w)*Ultralisk + w*Atlas_component`, con clipping de RH a [0,100].

| Variable | Componente Atlas | Peso Atlas | Peso Ultralisk |
|---|---|---:|---:|
| RH | DriftBias | 0.1676328571 | 0.8323671429 |
| T | DriftBias | 0.2051881805 | 0.7948118195 |
| U | AdaptiveFlow | 0.2093805205 | 0.7906194795 |
| V | DriftBias | 0.1862552111 | 0.8137447889 |

DriftBias corrige deriva temporal y sesgo del ancla; AdaptiveFlow condiciona la corrección al régimen del viento. Los componentes fueron seleccionados tras examinar los resultados retrospectivos O1–O3. Los pesos se congelaron antes de inferir test. No hay selección automática por día ni modificación con etiquetas del test.

O4, componente Atlas: 2767 muestras de entrenamiento hasta 23/06/2024 23:00 UTC; calibración independiente de 56 instantes entre 24/06/2024 00:00 y 30/06/2024 21:00 UTC; test 264 instantes. La reconstrucción de Ultralisk usa sus propios cortes y receta de entrenamiento, no confundirlos con el corte O4 del componente Atlas.

## 6. Evidencia histórica FULL

Los valores proceden de `atlas34/Atlas34_AUDIT.csv`, calculados a partir de los SSE por variable de `Atlas33_FULL_AUDIT.csv`. No corresponden a una nueva evaluación independiente de v3.4.

| Modelo | O1 | O2 | O3 | Pooled |
|---|---:|---:|---:|---:|
| Ultralisk reconstruido | 0.502686 | 0.473991 | 0.430914 | 0.465401 |
| Reference (Atlas v3.2.5) | 0.503397 | 0.476140 | 0.438847 | 0.469491 |
| MixAdaptiveFlow | 0.503462 | 0.476333 | 0.439168 | 0.469698 |
| MixDriftBias | 0.503338 | 0.476769 | 0.441419 | 0.470713 |
| MixRecentBias | 0.503408 | 0.476818 | 0.438914 | 0.469728 |
| **RegimeFlow v3.4** | **0.503481** | **0.476721** | **0.441126** | **0.470625** |

RegimeFlow mejora en los tres orígenes; DriftBias puro gana más acumulado pero pierde O1. La elección por variable sacrifica parte de esa ganancia a favor de la consistencia histórica. El gate de v3.4 registra:

| Indicador | Resultado |
|---|---:|
| Ganancia O1 / O2 / O3 vs Reference | +0.00008304 / +0.00058084 / +0.00227902 |
| Ganancia pooled vs Reference | +0.00113475276 |
| Ganancia pooled vs Ultralisk de la misma auditoría | ≈+0.005225 |
| Días ganados acumulados | 70/84 (83.33 %) |
| Días ganados O3 | 30/32 (93.75 %) |
| Semanas 3–5 O3 | +0.00273921 |
| Últimos 14 días O3 | +0.00257377 |
| IC95 bootstrap 3 días (10000 réplicas) | [0.00084421, 0.00143366] |
| IC95 bootstrap 7 días (10000 réplicas) | [0.00079370, 0.00145124] |
| Leave-one-origin-out (sin O1/O2/O3) | +0.00156176 / +0.00137128 / +0.00033767 |
| statistical_gate_passed_retrospective | true |
| review_eligible | false |

O1: 666 train, 32 cal, 208 eval (26 días observados); O2: 1215 train, 56 cal, 206 eval (26 días); O3: 1999 train, 56 cal, 256 eval (32 días). El bootstrap usa pocos bloques aproximadamente no solapados (30 de tres días, 12 de siete); los intervalos pueden ser optimistas por autocorrelación y selección retrospectiva. La confirmación pública es evidencia nueva del submission completo, no prueba confirmatoria de cada componente ni del leaderboard privado.

## 7. Ejecución real en Kaggle y Mutalisk

Notebook autónomo de una celda, Internet OFF, dos Tesla T4 (14.56 GiB cada una), PyTorch 2.11.0+cu130. Se reconstruyó v2.7 por falta del CSV original en la sesión. Fase neuronal v2.4 imprimió el CSV a 1229.6 s (20 min 29.6 s); fase v2.7 imprimió el suyo a 29.7 s adicionales. No hay tiempo total exacto del pipeline completo en el log.

| Fold de Ultralisk | Skill original | Reconstruido |
|---|---:|---:|
| 1 | 0.459901 | 0.459908 |
| 2 | 0.460417 | 0.460391 |
| 3 | 0.465080 | 0.465100 |
| April | 0.558458 | 0.558460 |

No confundir estos folds con O1–O3 de Atlas. Diferencia máxima redondeada de skill: 0.000026. Los hashes de las predicciones base son distintos; la reproducción es cercana, no idéntica.

Mutalisk: GeoWind, CNN residual multiescala, 456324 parámetros, AMP, 36 épocas, semillas 42 y 123, OneCycleLR. No LSTM, GRU ni atención. La rama adicional baja y recupera resolución para ampliar contexto espacial.

| Periodo | Loss final seed 42 | Loss final seed 123 |
|---|---:|---:|
| Fold 1 | 0.32905 | 0.32374 |
| Fold 2 | 0.31782 | 0.31422 |
| Fold 3 | 0.30215 | 0.30534 |
| April | 0.33685 | 0.33494 |
| Final | 0.29730 | 0.29664 |

El loss final de seed 42 cae desde 0.84367; seed 123 desde 0.85187. Tiempos finales: 88.1 y 88.2 s. Peso de la familia Mutalisk dentro de Ultralisk reconstruido (antes de mezclar Atlas): RH 31.751 %, T 25.616 %, U 20.302 %, V 21.666 %. No se ha medido en este log el skill individual de cada semilla; no justificar más peso, ancho o épocas por el loss. Se debe comparar por validación temporal y diversidad de residuos.

## 8. Identidad del submission y verificación independiente del archivo

Archivo recibido: `Atlas34_REVIEW(1).csv` (37 013 134 bytes); salida Kaggle `Atlas34_REVIEW.csv`.

- SHA-256 del archivo recibido, recalculado: `4af8d9249b3ff20f8c5209f1698fd3eff71bd31dd99d5fa00bd518a2764d4870`. Coincide con el log Kaggle.
- 646800 filas, columnas exactas `id,RH,T,U,V`, IDs únicos y orden idéntico a `sample_submission.csv` dentro del ZIP oficial.
- Todos los valores numéricos finitos; RH mínimo 0, máximo 100. T [261.088093, 300.415894], U [−17.330185, 11.374638], V [−16.737725, 10.491331].
- Error máximo absoluto de replay del componente Atlas, según log: 3.0517578125e−05.
- No se modificó el CSV ni se efectuó submission automático durante esta documentación.

| Variable | MAE de predicciones vs CSV v2.7 público | MAE vs v3.4 local |
|---|---:|---:|
| RH | 0.781567 | 0.005689 |
| T | 0.073043 | 0.000415 |
| U | 0.112709 | 0.000724 |
| V | 0.099395 | 0.000621 |

Estas son diferencias entre predicciones, no errores frente a WRF. No hay Y de test disponible.

### Huellas SHA-256 para reproducibilidad

| Artefacto | SHA-256 |
|---|---|
| CSV Kaggle v3.4 recibido | `4af8d9249b3ff20f8c5209f1698fd3eff71bd31dd99d5fa00bd518a2764d4870` |
| Base v2.7 pública original | `9ce6e7594d0edbf884e56902fd67e1fa02de1a94ba2f6815fc5c8b5115a801e1` |
| Base v2.7 reconstruida (log) | `123648e1dd131f5e221984d833dc7f2793005b94f81bac907f9cfab5c2ace33f` |
| CSV v3.4 local (no confundir con el enviado) | `905dc77511ac6aad847ec08a918f8f1a022547ef106a649bba46e010e46c6c6e` |
| Notebook Kaggle autónomo | `bd7309b48b8c17ebbd06dd0432179ee6ce6793defff260eee7ca7900ac8735e3` |
| Atlas_v3_4.py | `d2cb790e59832ad52b3173de496681d39ad4dcd84864cae3940941a4f60bf6bf` |
| Atlas33_FULL_AUDIT.csv | `78568dd2c40e4e5d4317fc113fe992eff4e8f4143b9d8c0bbec6d6cbcd9d83a7` |
| Atlas325_FULL_FINAL_AUDIT.csv | `724ca9c3b31bd82852e2019e3db4dc3d2439d0188092c9868f7a1ba8ecfb17c7` |
| Log completo: Markdown pegado.md | `4f76a3d543e0266376afcb469350a6ddbf5f20951ae181721c6d79f35b75cf20` |

El hash reconstruido se extrae del log; no se dispone aquí de ese CSV base concreto para recalcularlo. El resto de las huellas anteriores se recalcularon sobre los archivos locales.

## 9. Interpretación del resultado público

El incremento público +0.00122 equivale a aproximadamente 23.35 % del incremento retrospectivo acumulado +0.005225 frente a Ultralisk. Esta razón describe un único par de resultados; no constituye una tasa de transferencia predictiva y no puede extrapolarse a la próxima versión. No se puede separar cuánto procede de DriftBias, AdaptiveFlow, v3.2.5 o la reconstrucción de v2.7 porque no hay submissions públicos de esos componentes por separado. No usar el leaderboard como objetivo de ajuste iterativo ni asumir mejora privada.

## 10. Protocolo propuesto para v3.5 (no implementado)

Congelar v3.4 como referencia pública. Preservar el CSV exacto, notebook autónomo, auditorías y log antes de cerrar sesiones. Evaluar v3.5 contra la receta v3.4, con comparaciones emparejadas por fecha, mismos cortes y misma política de calibración. No cambiar la referencia original de Ultralisk.

1. Mutalisk: aislar variantes 36 épocas/ancho original, 36 épocas/ancho 64 y 54 épocas/ancho original; mismas semillas y resto de configuración. Medir skill de calibración y evaluación, no solo loss. Son hipótesis, no cambios aprobados.
2. Diversidad: comparar errores y ensembles de 2 frente a más semillas, incluyendo sesgo, correlación y ganancia efectiva. Evitar costes sin señal.
3. Viento: evaluar magnitud y dirección, hora, intensidad y terreno; la dirección requiere controlar velocidades cercanas a cero. No imponer divergencia nula sin información física suficiente.
4. Segunda diferencia temporal: señal previa pequeña en anchor individual; exigir prueba de complementariedad con el ensemble.
5. Gate: preregistrar candidatos, usar O1–O3, bootstrap por bloques 3/7 días, LOO positivo, magnitud útil y comprobar deterioro temporal. No presentar una auditoría repetida como ciega.
6. Entrega: Kaggle una celda autónoma, Internet OFF, T4×2 si requiere redes; CSV competitivo solo tras auditoría. No autoenviar; preservar dos intentos propios restantes hasta tener evidencia suficiente.

## 11. Fuentes y límites de la documentación

- Resultado y leaderboard públicos: texto comunicado por el usuario el 26/11/2026.
- `Markdown pegado.md`: log de la corrida final Kaggle; incluye resultados de Mutalisk, reconstrucción, gate y manifiesto de despliegue.
- `Atlas34_REVIEW(1).csv`: archivo competitivo recibido; verificación independiente de dimensiones, IDs, finitud, rangos y SHA-256.
- `hackathon-ia-pucp-2026.zip`: plantilla oficial usada para comparar IDs.
- `atlas34/Atlas34_AUDIT.csv`, `atlas34/manifest.json`, `Atlas33_FULL_AUDIT.csv`, `Atlas325_FULL_FINAL_AUDIT.csv`: evidencia histórica retrospectiva.
- `Atlas_v3_4.py` y `Atlas_v3_4_KAGGLE_AUTONOMO.ipynb`: código y receta del release.
- Los scores anteriores se registran de los resultados públicos comunicados en el proyecto. No se consultó Kaggle ni se entrenaron modelos nuevos durante esta documentación.

**Decisión:** Atlas v3.4 es el nuevo récord público de ShadowBytes, no un resultado privado confirmado. Mantener v2.7 como respaldo y v3.2.5 como candidato sin score público confirmado. Toda modificación posterior debe recibir una versión y auditoría distintas.
