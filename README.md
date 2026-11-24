# ApuCast v2.1 production

Candidato de ShadowBytes para la Hackathon IA PUCP 2026. Combina la referencia congelada v2 con el nuevo ensamble ResNet. El nombre de la rama identifica la línea de integración. El rendimiento del candidato en Kaggle queda pendiente hasta enviar y registrar el CSV.

## Lógica del modelo

- **v2:** tres CNN de semillas 0, 1 y 2. Predicen anomalías sobre la climatología horaria. Se mantiene el baseline público archivado de **0,433510**.
- **ResNet:** mismas entradas y salida que v2. Añade conexiones de salto en el encoder y el decoder. Entrena tres semillas fijas con 25 épocas y todos los datos oficiales etiquetados hasta el 30 de junio de 2024.
- **Mezcla:** peso fijo 50/50 por variable entre la predicción de v2 y el promedio de las tres ResNet. RH se limita a 0–100 %. Los campos usan RH en porcentaje, T en Kelvin y U y V en m/s.

Cada predicción usa GFS de t−2, t−1 y t, DEM y hora UTC. Los rezagos ausentes se rellenan con el mapa de la hora actual, como en el v2. La climatología y normalización de ResNet se ajustan con sus datos de entrenamiento y se guardan junto a los pesos. Cada arquitectura usa sus propias escalas antes de mezclar las salidas finales.

## Qué nos llevó a esta versión

La mezcla fija ganó **+0,014261**, **+0,022541** y **+0,006882** frente a la CNN en los tres pilotos pareados de ocho épocas y una semilla. El primer par de 25 épocas de mayo, con semilla 0 y selección de checkpoints separada, dio **+0,015003**, con IC del 95 % de **[+0,012713, +0,017566]**. La tabla está en models/v2.1-production/validacion_exploratoria.csv. Son comparaciones locales con las horas y los denominadores de Kaggle. Queda pendiente el benchmark completo de tres semillas en los tres cortes.

En el histórico de junio, V fue la variable con mayor error relativo. A las 16:00 de Lima, entre 2.000 y 4.000 m, buena parte del error varió por día y celda. Eso motivó comparar una arquitectura capaz de responder a cambios del campo de viento y conservar la referencia v2 dentro del ensamble.

## Cómo generar el CSV

Instala requirements.txt y coloca los tres NPZ oficiales y la plantilla sample_submission.csv bajo data/. Los comandos aceptan --data para usar otra carpeta.

```bash
python -m pip install -r requirements.txt
python -X utf8 train/exportar_v2.py --output submission_v2_top.csv
python -X utf8 train/generar_submission_v3.py --work trabajo_resnet --output submission_v3_resnet.csv
python -X utf8 train/generar_mezcla_v3.py --v2 submission_v2_top.csv --v2-manifest submission_v2_top.json --resnet submission_v3_resnet.csv --output submission_v3_mezcla_50_50.csv
```

Para reproducir la mezcla con el v2 exacto que se subió, usa el CSV archivado con models/v2-top/registro_kaggle.json. El hash de ese CSV se verifica antes de mezclar. Si regeneras el v2 en otra máquina, usa su nuevo JSON. Una diferencia de redondeo puede cambiar el hash y deja ese nuevo archivo sin score público asociado.

El entrenamiento guarda cada semilla terminada en trabajo_resnet y permite reanudar con la misma carpeta y el mismo presupuesto. El proceso comprueba las **646.800 filas**, los encabezados id,RH,T,U,V, el orden exacto de IDs, que todo sea finito y el rango de RH. Cada CSV nuevo queda asociado a un SHA-256. Los datos oficiales y los CSV de predicciones se conservan fuera de Git.

## Integración con el producto

app/gfs_forecast.py obtiene el GFS de Open-Meteo para el pronóstico actual. Para conectar esta rama, aplica las mismas ventanas y la normalización correspondiente a cada modelo, restaura la climatología y mezcla los cuatro campos antes de ordenar los sectores por temperatura mínima. La demo de la plantilla sirve como punto de partida para esa integración. A escala de 3 km, el producto prioriza sectores que conviene revisar.
