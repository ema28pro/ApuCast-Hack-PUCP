# ApuCast v3 calibrada

- [CSV para Kaggle](submission_v3.csv) y [manifiesto SHA-256](submission_v3.json).
- [Resultados por corte y semilla](scores_temporadas_semillas.csv) e [informe detallado](backtest.json).
- [Pesos de las tres ResNet y preprocesamiento](models/resnet/). Las tres CNN usadas en la mezcla están en [v2-top](../v2-top/models/v2-top/).

El CSV reúne las predicciones de ambas familias con pesos ResNet de 56,10 % para RH, 64,75 % para T, 68,18 % para U y 64,71 % para V. El resto de cada predicción proviene del ensamble CNN. Para regenerar el archivo, usa el [notebook de production](../../train/submission_notebook.ipynb).
