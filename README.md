# ApuCast v2 top

Referencia congelada de ShadowBytes para la Hackathon IA PUCP 2026. El envío archivado usa un ensamble CNN de tres semillas y limita la humedad relativa a 0–100 %.

| Medida | Valor |
|---|---:|
| Público vinculado al CSV archivado | 0,433510 |
| Proxy de Kaggle en junio | 0,448374 |
| Validación histórica informada por el equipo | 0,348012 |

El equipo también informó 0,433550 en un mensaje posterior. El registro del archivo identificado por hash conserva 0,433510 hasta que se vincule el otro valor al envío correspondiente.

## Cómo funciona

Cada miembro toma los mapas GFS oficiales de las horas t−2, t−1 y t, junto con la hora UTC y el relieve. Aprende anomalías de RH, T, U y V sobre la climatología horaria de WRF. El modelo suma esa climatología, el ensamble promedia las tres predicciones y al final se recorta RH. Si falta un rezago, se rellena con el mapa de la hora actual.

Los pesos son de las semillas 0, 1 y 2. Se entrenaron con datos anteriores al 16 de junio de 2024. Los checkpoints se escogieron en el corte de junio. El proxy histórico usa las horas UTC cada tres horas y los denominadores 25,360000, 2,065200, 2,677600 y 2,087600. La ventana de junio da contexto sobre el error y el score público indica el resultado en los días públicos de la competencia.

## Regenerar con pesos congelados

Instala las dependencias y coloca train.npz, test.npz, static.npz y sample_submission.csv oficiales en data/. También puedes pasar su ruta con --data.

```bash
python -m pip install -r requirements.txt
python -X utf8 train/exportar_v2.py --output submission_v2_top.csv
```

El exportador verifica **646.800 filas**, los encabezados id,RH,T,U,V, IDs únicos en el mismo orden que la plantilla, valores finitos y RH dentro del rango. T está en Kelvin y U y V en m/s. El hash del envío original está en models/v2-top/registro_kaggle.json. Los CSV y datos oficiales se guardan fuera del historial de Git.

## Archivos y casuísticas

- models/v2-top/ contiene los tres checkpoints, la climatología y las escalas archivadas, y el registro de Kaggle.
- train/starter_notebook.ipynb define la arquitectura, las ventanas y la normalización que reutiliza el exportador.
- train/exportar_v2.py reconstruye el envío desde los pesos. Cuando la versión o el dispositivo de PyTorch cambian, verifica el CSV y compara su hash con el registrado.
- app/gfs_forecast.py descarga GFS actual para la aplicación. app/ejemplo_producto.py es el ejemplo base. Para usar el v2 en la app hay que cargar sus tres pesos y aplicar el preprocesamiento archivado.
- La salida tiene resolución de 3 km. El producto propone priorizar sectores fríos para una ronda de campo. El rendimiento en una parcela requiere observaciones de campo.
