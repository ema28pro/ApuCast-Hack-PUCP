# Atlas v3.5 — Mutalisk Evolution
ShadowBytes | 26/11/2026 | Protocolo fijado antes de ejecutar v3.5 FULL

## Objetivo
Récord público comunicado: Atlas v3.4 = 0.49741, sexto. Tercer puesto comunicado = 0.50912. Brecha +0.01171 (reducción relativa del error normalizado medio de 2.33%). No se garantiza el objetivo. La mejora observada de v3.4 sobre v2.7 fue +0.00122; dividirla por la mejora retrospectiva +0.005225 da ~23.35%, no un factor predictivo calibrado. No usar el score público como objetivo de entrenamiento.

## Experimentos
Referencia: reconstrucción de Ultralisk v2.7 NarrowSpatial + ruta Atlas v3.4 (DriftBias RH/T/V; AdaptiveFlow U), pesos por origen calibrados con el procedimiento prequential de v3.2.5/v3.3. Mutalisk original: 48 canales, 36 épocas, semillas 42 y 123. Long12: conserva las tres entradas originales y añade t-3, t-6 y t-12 con anomalías horarias, delta t-6 y máscaras de disponibilidad, sin usar datos futuros. No usa presión. Wide64: ancho principal 64, rama baja 80, 36 épocas. Epoch54: arquitectura original, 54 épocas con OneCycle reescalado. Semillas 42/123, batch y regularización originales. Las tres variantes se entrenan en todos los orígenes. No ajustar el modelo usando el loss de train.

Políticas por variante: reemplazo del 50% o 100% de la contribución Mutalisk dentro del Ultralisk calibrado (mapa de bias y pesos congelados); ADD: añadir la variante al Ultralisk mediante mezcla convexa por variable, pesos calibrados prequentialmente con el mismo método regularizado y cap 0.5. En todas ellas la rama Atlas queda intacta. No son nueve nuevos entrenamientos.

## División y selección
O1: corte 17 marzo, cal hasta 24 marzo, evaluación hasta 21 abril; O2: corte 14 abril, cal hasta 21 abril, evaluación hasta 21 mayo; O3: corte 21 mayo, cal hasta 28 mayo, evaluación hasta 1 julio. Intervalos semiabiertos; evaluación y cal cada tres horas. Elegir el candidato positivo en ambos O1/O2 con mayor gain pooled desde SSE; congelarlo antes de O3. O3 también imprime las alternativas, pero no se permite cambiar el elegido. O3 se conoce de versiones anteriores, por lo que la evaluación es retrospectiva. Sin candidato positivo: conservar v3.4.

Gate: candidato positivo en cada origen y leave-one-origin-out, pooled >=0.0015, O3 >=0.0015, semanas 3–5 positivas, últimos 14 días no negativos, >=65% días ganados en O3 y pooled, IC95 bootstrap emparejado por bloques de 3 y 7 días con límite inferior >0. Este umbral es una decisión de ingeniería, no una equivalencia al score público. Los bloques no son plenamente independientes. `review_eligible=False` refleja falta de confirmación ciega aunque pase `retrospective_deploy_gate`.

## Deploy
Solo si pasa gate: reconstruir Ultralisk v2.7 con las tres semanas originales de calibración (verificar sus tres folds históricos a tolerancia 0.001), entrenar sobre todos los datos oficiales y entrenar las dos semillas de la variante elegida. Atlas usa su corte O4 original y pesos congelados de v3.4; ADD usa pesos de calibración O3. La base puede diferir del CSV que obtuvo 0.49741 por no determinismo CUDA; si el archivo exacto del récord está adjunto opcionalmente, registrar MAE del control reconstruido, sin utilizarlo como etiqueta. No se requiere ese archivo. Verificar 646800 filas, orden de IDs, valores finitos, RH válida y SHA-256. No hacer submit automático. El gate puede fallar y en ese caso no se produce CSV competitivo.

## Evidencia actual
CPU smoke: una época, 32 ejemplos de train y 4 de consulta, una semilla, cuatro arquitecturas, en datos oficiales. Comprueba entrenamiento e inferencia, máscaras, combinación identidad y calibración básica. No mide generalización. No hay resultados FULL nuevos ni score público v3.5 antes de ejecutar Kaggle. Validación GPU y ruta completa de exportación pendientes. No se han probado más semillas ni residual OOF del ensemble. No gastar los dos intentos propios restantes si el gate falla.
