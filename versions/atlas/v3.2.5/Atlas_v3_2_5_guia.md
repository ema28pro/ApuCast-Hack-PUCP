# Atlas v3.2.5 | Manual técnico y de auditoría

## Alcance y archivos
`Atlas_v3_2_5.py` importa `Atlas_v3_2.py` sin modificarlo. Incluye T1–T8; `Atlas_v3_2_5_resumen.py` agrega resultados persistentes de tres orígenes sin reentrenar. El preregistro tiene hash fijo y debe permanecer intacto. Los datos deben estar extraídos en una carpeta con train.npz, test.npz, static.npz y sample_submission.csv. No se requieren servicios externos. La auditoría no produce ni envía un CSV de competición. El refit competitivo y la inferencia integral Ultralisk+Atlas se mantienen pendientes de una revisión humana; inferencia individual Atlas desde artefactos sí implementada.

## Evidencia real
CPU: se reconstruyó Narrow desde los índices de entrenamiento de cada origen (no se aceptó la caché legacy); se ejecutaron los cinco Anchors y los seis contrastes de mezcla, incluyendo factor común/variable. O1/O2/O3: 26/26/32 días con datos, respectivamente. O1 tiene 32 horas de calibración (4 días observados), solo 2 prequential tras el warmup; se aplica el fallback pre-registrado. O2 tiene 206 horas de evaluación, dos horas menos que 26 días completos. O3 tiene 256. En O3 MixBiasVar 0.430093 vs 0.415114, pooled gain +0.00816255; gate de GPU verdadero, gate de submission falso. El control placebo supera a Physics en O1 pero no O2/O3. Los factores individuales no superaron a común en O3. Los resultados son retrospectivos y exclusivamente proxy. El código neural se probó con 4 muestras oficiales, 1 época, ancho 16, en CPU, verificando entrenamiento/recarga/forma en ambas variantes; no se midió capacidad predictiva. Inferencia Bias sin Y reproducida en 8 tiempos evaluables de calibración: diferencia máxima 0.0, no garantía de igualdad CUDA. No hay benchmark CUDA ni evaluación v2.7 completa en v3.2.5.

## Kaggle — pasos
Coloca los tres .py y el preregistro en la misma carpeta (por ejemplo, descomprime el ZIP como dataset adjunto y copia los archivos a /kaggle/working/atlas325-code). Ejecuta desde ella. Sustituye DATA por la ruta oficial extraída. Usa GPU T4 x2 para modos full/benchmark, CPU es suficiente para proxy. Nunca reutilices un directorio de una versión de código distinta; el hash invalida caches. En una sesión nueva conserva el directorio de resultados como dataset privado si quieres reutilizarlo, sin incluir datos externos.

```bash
DATA=/kaggle/input/competitions/hackathon-ia-pucp-2026
P=/kaggle/working/atlas325-proxy
F=/kaggle/working/atlas325-full
# 1. CPU, origen por origen para poder reanudar.
for o in O1 O2 O3; do
  python Atlas_v3_2_5.py --mode proxy --data "$DATA" --out "$P" --origin "$o" || break
done
python Atlas_v3_2_5_resumen.py "$P"
# 2. GPU: reconstruir únicamente v2.7 O3, sin rama Atlas neural.
python Atlas_v3_2_5.py --mode full --data "$DATA" --out "$F" --origin O3
# 3. Si O3 justifica continuar: completar los otros orígenes, 10 redes cada uno.
for o in O1 O2; do
  python Atlas_v3_2_5.py --mode full --data "$DATA" --out "$F" --origin "$o" || break
done
python Atlas_v3_2_5_resumen.py "$F"
# 4. Solo si interesa medir las ramas neuronales; reutiliza caché full O3.
python Atlas_v3_2_5.py --mode benchmark --epochs 2 --origin O3 --data "$DATA" --out "$F" --cpu-report "$P/pooled_report.json"
cp "$F/report.json" /kaggle/working/atlas325-benchmark.json
# 5. Después de examinar benchmark (no antes):
python Atlas_v3_2_5.py --mode full --neural --epochs 48 --origin O3 --data "$DATA" --out "$F" --cpu-report "$P/pooled_report.json" --benchmark-report /kaggle/working/atlas325-benchmark.json
python Atlas_v3_2_5_resumen.py "$F"
```

Los modos benchmark y full reutilizan la misma caché de las familias Ultralisk (25/36 épocas constantes, no dependen de `--epochs` Atlas). El benchmark sobrescribe el reporte de O3; guardar el reporte del benchmark fuera de F y volver a ejecutar full antes de resumir. T6 se ejecuta solo O3 (O1 no alcanza warmup de 28 días). Si un proceso se interrumpe, los Anchors completos con metadata se reutilizan. Un archivo truncado falla al cargar; elimínalo explícitamente y vuelve a ejecutar. No modificar cache ni usar --fit-only con Bias, que es una corrección calibrada y no un Anchor independiente.

Inferencia de auditoría, no refit de competición:
```bash
python Atlas_v3_2_5.py --mode infer --bundle "$P/O3" --component Bias --input /ruta/X_y_times.npz --out /kaggle/working/atlas325-replay
```
`--component` admite Raw, Placebo, Physics, Recent, Drift, Bias y los checkpoints neuronales si existen. Verifica hashes del manifiesto del directorio padre, lee únicamente X y times. Conserva todas las horas de entrada; filtrar UTC múltiplo de 3 para evaluación. Usa los dos frames previos cuando estén disponibles. Predicciones de un Anchor histórico no constituyen un refit final.

## Hipótesis / confirmación / riesgos
T1: si el primario gana en la cola y todos los controles del preregistro, habilita revisión, no envío. Las ventanas no son independientes; IC retrospectivo.
T2: bias 35% fijo, recencia 30d y deriva regularizada son contrastes aislados. Evaluar delta semanal frente a Narrow/v2.7 y frente a mezcla sin bias; la deriva se limita a 34 días desde fin del train. La tendencia lineal puede extrapolar mal. El placebo no aporta información externa.
T3: prior cero con penalización del 25% de la energía del contraste; factores por variable se contraen 50% hacia factor común. Cinco días prequential como máximo, muy pocos para selección; no variar por semana. Contrastar MixBiasVar con MixBiasCommon; nunca elegir por evaluación ya observada.
T4: Raw12 vs Placebo29 vs Physics29; mismo número de columnas locales/vecinas para Placebo y Physics, EOF24 y Ridge alpha .08. El placebo usa tanh y productos genéricos fijos, no una distribución idéntica a los canales físicos. No es un control causal completo, puede necesitar varios placebos/regularización cruzada.
T5: cache metadata guarda train_times completos, max<CUT, consulta, origen, modo y hashes. Comparación exacta, sin aceptar caché legacy. Hashes no son firma criptográfica ni prueba contra manipulación deliberada; sirven para detectar incoherencias accidentales.
T6: OOF/in-sample x input/no input con mismo número de parámetros (canales Anchor a cero en control), 2 semillas, 48 épocas, U/V adicional. La mezcla directa y la mezcla en dos etapas con Bias se calibran sin evaluar Y fuera de calibración. OOF usa Anchors más pequeños, diferencia de distribución permanece. Las redes son secundarias, sin selección automática ni autorización propia.
T7: benchmark 2 épocas, tiempos por seed/dispositivo y estimación 48 épocas con segunda época de cada par paralelo; extrapolación solo entrenamiento Atlas O3, excluye overhead; revisar coste total y memoria antes de continuar. CUDA grid_sample backward y otros kernels pueden no ser deterministas, warn_only=True; tolerancia de replay FP32 1e-4, no igualdad entre entrenamientos. Reportar dispersión entre semillas y, si se dispone, repeticiones de misma semilla (objetivo operativo de delta score <=.001 no validado).
T8: v2.7 reconstruido = CNN3, ResNet3, Mutalisk2, Viper2, Kernel, Clock, Regional/Narrow y calibración NarrowSpatial; no es el CSV Kaggle evaluado, es reproducción en cortes históricos. Si O3 empeora, detener rama neural, conservar v2.7; si Anchor gana y neural no aporta, usar únicamente Anchor tras revisar refit; si neural aporta, preregistrar nueva comparación independiente antes de adoptar. Ningún resultado garantiza 0.50/0.51.

## Salidas
`pooled_report.json`, `pooled_metrics.csv`, `pooled_weekly.csv`, `pooled_daily.csv`, por origen métricas y correlaciones, `daily_sse.npz`, `reference_cache.npz`, `anchor_*.npz`, `anchor_bias.npz`, `static.npz`, `validation_predictions.npz`, checkpoints opcionales, `manifest.json`. El ZIP de entrega contiene resultados tabulares y manifiesto histórico; arrays/cachés/checkpoints grandes no se incluyen y deben regenerarse. `manifest.json` de la corrida local referencia todos los archivos de la corrida completa, no solo los que se incluyen en el ZIP.
