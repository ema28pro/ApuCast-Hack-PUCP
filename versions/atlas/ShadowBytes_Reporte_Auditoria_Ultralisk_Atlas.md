# Ultralisk → Atlas

SHADOWBYTES  /  HACKATHON IA PUCP 2026
Reporte técnico y base de auditoría · 25 noviembre 2026

## 01  Resumen ejecutivo

Ultralisk optimiza un ensemble de predicciones para reducir el RMSE normalizado. Atlas plantea un modelo independiente, condicionado por la dinámica atmosférica y el relieve, con una base estadística y una red residual temporal. El objetivo es superar 0.50 y aspirar a 0.51 sin comprometer el leaderboard privado.

| Hito | Skill público | Δ vs anterior |
| --- | --- | --- |
| ApuCast v3 T4x2 | 0.46485 | — |
| Ultralisk v2.3 | 0.49056 | +0.02571 |
| Ultralisk v2.7 | 0.49619 | +0.00563 |
| Atlas v3.1 | No evaluado | Pendiente |

Ganancia acumulada: +0.03134 puntos (+6.74% del score inicial). Reducción del error normalizado: 5.86%. Desde el récord de v2.7, alcanzar 0.50 exige reducir el error restante un 0.76%; llegar a 0.51 exige un 2.74%.

## Alcance y nivel de evidencia

Se inspeccionaron los scripts disponibles, registros, CSV de validación, el ZIP oficial y la guía. Se recalcularon hashes y se verificaron los tres CSV históricos contra la plantilla. Los scores públicos son los comunicados por el equipo; no se ha consultado el leaderboard ni ejecutado un nuevo entrenamiento. Atlas v3.1 solo dispone de una base estadística evaluada y una prueba reducida de integración.

## Dictamen preliminar

Ultralisk v2.7: referencia competitiva conservada. Atlas v3.1: candidato experimental, no autorizado todavía. La decisión final requiere evaluación completa, pruebas de generalización y reproducción del pipeline de inferencia. No se garantiza un score de 0.50 o 0.51.


---

# 02  Datos, métrica y reglas

Contrato de entrada y salida verificado en el ZIP oficial

| Elemento | Especificación |
| --- | --- |
| Train | X: 2935 × 17 × 13 × 4; Y: 2935 × 70 × 35 × 4. Del 18/02 al 30/06/2024. |
| Test | X: 792 × 17 × 13 × 4; sin Y. Del 02/07 al 04/08/2024. |
| Estáticos | DEM, latitudes y longitudes GFS/WRF; cuatro variables RH, T, U, V. |
| Inferencia | GFS de t−2, t−1, t; salida WRF en t, 70 × 35 celdas a 3 km. |
| CSV | 646.800 filas, id,RH,T,U,V; 264 instantes evaluados; UTC cada 3 h. |

## Métrica oficial

Score = 1 − (RMSE_RH / 25.360 + RMSE_T / 2.0652 + RMSE_U / 2.6776 + RMSE_V / 2.0876) / 4. El RMSE de cada variable se calcula globalmente sobre todos los píxeles y horas del conjunto evaluado. RH en %, T en K y U/V en m/s.

## Restricciones de competición

Solo datos oficiales para entrenar y predecir en Kaggle. No compartir código ni predicciones entre equipos. La guía exige validación temporal, un notebook reproducible, pesos, dependencias y un producto que cargue el mismo modelo presentado. Límite: cinco envíos diarios y elección de dos finales; el privado (~70% de los días) determina el resultado. [1–2]

## Cortes temporales utilizados

| Periodo | Entrena antes de | Calibración | Evaluación (fin excl.) |
| --- | --- | --- | --- |
| Abril | 09/04 | 09–15/04 | 16/04–01/05 |
| Mayo | 09/05 | 09–15/05 | 16/05–01/06 |
| Junio I | 25/05 | 25–31/05 | 01/06–16/06 |
| Junio II | 09/06 | 09–15/06 | 16/06–01/07 |

Fechas UTC de 2024. Las ventanas se reconstruyen por timestamp y no por posición. Los periodos conocidos son retrospectivos; abril no es una prueba ciega adicional.


---

# 03  Ultralisk: optimización del ensemble

La serie 2.x explota diversidad de errores y calibración supervisada en semanas anteriores a la evaluación.

| Versión | Aporte técnico | Evidencia |
| --- | --- | --- |
| 2.3 | 3 CNN + 3 ResNet + 2 Mutalisk + 2 Viper; kernel estadístico; mezcla por variable. | 0.49056 público |
| 2.4 | Kernel horario Clock; selección temporal y caché de predicciones. | Validación completa |
| 2.5 | Kernel regional: alpha 1.2, sigma 0.65, contexto regional 70%. | RegionalFixed evaluado |
| 2.6 | Narrow: alpha 1.2, sigma 0.40, contexto regional 85%, escala 90 días. | Kernel individual |
| 2.7 | NarrowSpatial: pesos convexos por variable + sesgo suavizado según píxel y hora. | 0.49619 público |

## Resultados comparables: ensembles completos

| Modelo | Mayo | Junio I | Junio II | Abril |
| --- | --- | --- | --- | --- |
| v2.3 (repro.) | 0.43984 | 0.43966 | 0.44968 | 0.55090 |
| v2.4 Anchor | 0.44473 | 0.44743 | 0.45463 | 0.55352 |
| v2.5 RegionalFixed | 0.44768 | 0.45353 | 0.45643 | 0.55569 |
| v2.7 NarrowSpatial | 0.45990 | 0.46042 | 0.46508 | 0.55846 |

La ganancia media mayo-junio de v2.7 frente a v2.3 reproducida es +0.01874, pero la ganancia pública fue +0.00563. Este contraste muestra que la extrapolación de validación a Kaggle no es fiable. [3–4]

## Cuello de botella: viento

| Variable | RMSE v2.7 junio II | RMSE normalizado |
| --- | --- | --- |
| RH | 9.18969 | 36.24% |
| T | 0.91281 | 44.20% |
| U | 1.59630 | 59.62% |
| V | 1.54305 | 73.92% |

El mayor error relativo corresponde a V; constituye una prioridad experimental, no una justificación para sacrificar las otras variables.


---

# 04  Atlas: modelo físico-informado

Nueva arquitectura independiente; Ultralisk se utiliza solo como referencia de evaluación.

| Componente | Implementación en Atlas v3.1 |
| --- | --- |
| Entrada temporal | Tres frames, 15 canales por frame; máscara de ausencias, hora cíclica y declinación solar. |
| Variables físicas | RH/T/U/V, depresión de punto de rocío, presión de vapor, velocidad, divergencia, vorticidad, deformaciones, advecciones y gradientes. |
| Topografía | DEM, pendientes, relieve a tres escalas, curvatura y coordenadas; flujo ascendente/transversal e incidencia solar potencial. |
| Anchor | Climatología horaria, PCA (24 EOF), Ridge local por celda GFS más vecinos; alpha 0.08, 40 regiones observadas. |
| Red residual | ConvLSTM ancho 48, atención 4 cabezas, fusión con topografía; decodificadores RH/T y U/V; corrección sobre Anchor. |
| Entrenamiento | 48 épocas, batch 32, semillas 42/123, AdamW, cosine LR, EMA, AMP en CUDA. Pérdida 60% RMSE normalizado + 40% MSE; variante física añade velocidad, vapor y gradientes. |

## Evidencia actual

| Periodo | Ultralisk v2.7 | Atlas Anchor | Δ Atlas |
| --- | --- | --- | --- |
| Mayo | 0.45990 | 0.43651 | −0.02339 |
| Junio I | 0.46042 | 0.42882 | −0.03160 |
| Junio II | 0.46508 | 0.41569 | −0.04939 |
| Abril | 0.55846 | 0.52618 | −0.03228 |

Piloto v3.0, 573 muestras / 3 épocas: Raw 0.22487; Physics 0.22352; PhysicsReg 0.22350. Smoke v3.1, 1 época: Anchor 0.32053; mejor residual 0.32056. Estas pruebas no estiman el rendimiento completo ni demuestran mejora física. [5–7]

La arquitectura usa aproximaciones físicas, no resuelve las ecuaciones de WRF. RawResidual conserva la misma base estadística enriquecida: la ablación compara solo la rama neuronal.


---

# 05  Matriz de auditoría técnica

Prioridad alta = resolver antes de aprobar la versión para competencia/producto.

| ID / prioridad | Hallazgo y riesgo | Prueba o acción requerida |
| --- | --- | --- |
| A01 / Alta | Atlas Anchor se ajusta con Y[tr] y devuelve predicciones sobre tr; la red aprende residuos in-sample. No es fuga del test demostrada, pero puede distorsionar el aprendizaje. | Generar residuos mediante cross-fitting temporal; comparar contra in-sample con idénticos cortes. [6:167–185, 201–206] |
| A02 / Alta | Selección repetida sobre mayo/junio/abril. El gate usa referencias históricas fijas; no hay evaluación ciega nueva. | Congelar parámetros; evaluar bloques no usados para selección o validación anidada. Reportar dispersión por día. [6:59, 268–285] |
| A03 / Alta | Atlas guarda pesos EMA y normalización, pero Anchor no exporta PCA, coeficientes Ridge ni climatología. | Guardar artefactos completos; implementar inferencia sin Y; reproducir CSV y salida de una ventana real. [6:167–185, 230] |
| A04 / Alta | Ablación Raw/Physics comparte Anchor con física; no mide aporte de física en todo el sistema. | Comparar base y red con/sin física, igual semilla, cortes y presupuesto. [6:76–83, 114–137, 272–279] |
| A05 / Media | Huecos temporales se sustituyen por frame actual, con máscaras. El decoder recibe los frames interpolados. | Medir frecuencia de huecos y rendimiento con máscaras; verificar t−2/t−1/t y orden lat/lon. [6:27–32, 119–136] |
| A06 / Media | Métrica de entrenamiento es aproximada por batch y auxiliares; clipping RH altera errores. | Recalcular score global con float64; comparar por variable, hora, altitud y extremos. [6:48–51, 213–215] |
| A07 / Alta | Caché v2.7 verifica tiempos/objetivos pero no acredita por sí misma origen de cada modelo. | Manifestar hash de datos/código, seeds, cortes, pesos, dependencias y cachés; reconstruir desde cero. [4:235–258] |

Estado: hallazgos de revisión estática, pendientes de pruebas correctivas. No se observó uso explícito de etiquetas del test en los scripts inspeccionados; no equivale a certificar ausencia absoluta de fuga.


---

# 06  Protocolo de aprobación y evidencias

Criterios para una revisión independiente de Atlas v3.1

| Control | Criterio de aceptación |
| --- | --- |
| Integridad | Solo ZIP oficial; hashes de datos, código y salida; columnas y orden idénticos a plantilla, valores finitos y RH 0–100. |
| Validación | Mismos cortes y preprocesamiento train-only; comparación Anchor, Raw, Physics, Physical y v2.7; reporte RMSE/skill por fold. |
| Generalización | Parámetros congelados; auditoría temporal adicional cuando sea viable; bootstrap por bloques de días, no píxeles independientes. |
| Gate de código | Mayo y junio I positivos; media de ambos ≥ 0.003; junio II ≥ 0.003; abril ≥ −0.002; nunca smoke. No garantiza score público. |
| Reproducción | Notebook limpio en Kaggle T4 x2; generar CSV desde cero; guardar pesos y Anchor; comparar predicciones con tolerancia documentada. |
| Despliegue | Inferencia con tres horas GFS y DEM; verificar orientación, UTC, latencia y consistencia con el CSV final. |
| Decisión | Conservar v2.7; usar Atlas solo tras revisar métricas, manifiesto y CSV. Registrar dos finales según bases. |

## Trazabilidad verificada

Los tres CSV tienen 646.800 filas, columnas y orden correctos, valores finitos y RH válido. SHA-256 (prefijo de 16 caracteres): ApuCast v3 cd64d3e0c2884ce7; v2.3 3f3a72a9ee25604b; v2.7 9ce6e7594d0edbf8. Los hashes completos de v2.3 y v2.7 coinciden con los registros compartidos.

## Fuentes internas y puntos de revisión

[1] Guia_Participante_Hackathon_IA_PUCP_2026.pdf, secciones Qué recibe y qué predice, Kaggle y entregables. [2] Bases_Hackathon_IA_PUCP_2026.pdf, reglas c–d de uso de datos. [3] Ultralisk_v2_3.py y Ultralisk_v2_4.py. [4] Ultralisk_v2_7.py; validation_metrics(2).csv; training(2).log. [5] Atlas_v3_0_PhysicalLSTM.py; Atlas_v3_0_piloto_CPU.csv. [6] Atlas_v3_1.py (líneas citadas en la matriz). [7] atlas31_anchor/validation_metrics.csv; atlas31_smoke/validation_metrics.csv y report.json. Scores públicos: historial comunicado por ShadowBytes.

Conclusión: mantener Ultralisk como referencia; corregir los hallazgos A01–A04 y completar validación CUDA de Atlas antes de decidir un envío.
