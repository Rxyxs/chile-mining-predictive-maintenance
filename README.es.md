# Mantenimiento predictivo con datos reales de camiones pesados (SCANIA Component X)

Español · [English version](README.md)

![tests](https://github.com/Rxyxs/heavy-truck-predictive-maintenance/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

![Costo de la decisión por regla](reports/figures/01_costo_decision.png)

## Por qué este proyecto

El mantenimiento predictivo de una flota de transporte es un problema de decisión, no de clasificación: mandar a taller un camión que no lo necesitaba cuesta poco; no detectar una falla cuesta mucho. Una versión anterior de este repositorio corría sobre un generador de datos que escribí yo mismo, y un modelo que funciona bien sobre datos que simuló su propio autor está midiendo el simulador. Lo reemplacé por un benchmark público real, **SCANIA Component X**, y reconstruí el pipeline en torno a la estructura de costos que ese benchmark define.

La pregunta que respondo es concreta: **dado el historial de lecturas de un camión hasta hoy, ¿qué acción de mantenimiento minimiza el costo esperado, y cuánto mejor es eso que no hacer nada?**

También debo ser claro sobre el límite de esa pregunta. Son camiones pesados *de carretera* de un fabricante europeo, no camiones de extracción fuera de ruta de la minería chilena. Los datos tienen la misma forma que el problema minero (contadores de uso acumulados, fallas raras y caras, censura), pero aquí no tengo evidencia de que los números se transfieran a una faena.

## Datos

[SCANIA Component X](https://researchdata.se/en/catalogue/dataset/2024-34) — público, CC BY 4.0, sin registro. Descrito en *SCANIA Component X dataset: a real-world multivariate time series dataset for predictive maintenance* ([Scientific Data, 2025](https://www.nature.com/articles/s41597-025-04802-6)). Es el dataset del IDA 2024 Industrial Challenge.

| | |
|---|---|
| Vehículos (train) | 23.550, de los cuales 2.272 tuvieron el Component X reparado durante el estudio (9,6%); el resto está censurado |
| Lecturas (train) | 1.122.452 lecturas de muestreo irregular, 105 columnas de 14 variables anonimizadas |
| Especificaciones | 8 columnas categóricas por vehículo |
| Validación / test | 5.046 / 5.045 vehículos, **una lectura elegida al azar por vehículo** (simula estar parado en "hoy", sin información del futuro) |
| Etiquetas | 0 = sin falla en 48 unidades de tiempo, 1–4 = falla en 24–48, 12–24, 6–12, 0–6 unidades |

Cosas que importan al leer los números:

- **Las 105 columnas son contadores acumulados y bins de histograma** (100% no decrecientes). El valor crudo mide sobre todo la edad del camión, así que las features son tasas y fracciones (abajo).
- **Los nombres de las variables están anonimizados.** Puedo decir *cuál* variable importa, no *qué* mide.
- **La clase 0 es el 97%** de los vehículos de validación y test. Predecir "sano" para todos es una referencia que se ve fuerte.
- Los autores advierten que las frecuencias de reparación y de lectura pueden haber sido modificadas y no son necesariamente representativas del uso real.

## El problema de decisión

El reto puntúa una recomendación con una matriz de costos asimétrica:

| | recomendar 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **el camión está sano (0)** | 0 | 7 | 8 | 9 | 10 |
| **falla en 24–48 (1)** | 200 | 0 | 7 | 8 | 9 |
| **falla en 12–24 (2)** | 300 | 200 | 0 | 7 | 8 |
| **falla en 6–12 (3)** | 400 | 300 | 200 | 0 | 7 |
| **falla en 0–6 (4)** | 500 | 400 | 300 | 200 | 0 |

Una revisión innecesaria cuesta 7–10; una falla no detectada cuesta 200–500. La referencia a superar es "no recomendar nada para nadie": **57.400** en validación y **56.100** en test.

## Método

1. **Features** (`src/features/engineering.py`, 407 por lectura + 8 especificaciones). Usan solo las lecturas hasta la actual: tasa de uso desde el inicio, tasa de uso en las últimas 5 lecturas, fracciones de cada bin de histograma en toda la vida y en la ventana reciente, número de lecturas y separación entre ellas. Un test recalcula cada feature sobre historiales truncados y verifica que no cambie, así que nada mira hacia adelante.
2. **Clasificador de riesgo.** LightGBM multiclase sobre todas las lecturas con riesgo más un 5% de las sanas (re-ponderadas ×20). Los vehículos se separan en train y hold-out *por vehículo*, con parada temprana sobre el hold-out. Los hiperparámetros se fijan de antemano; nada se ajusta con validación ni test.
3. **Regla de decisión.** El modelo entrega probabilidades por clase; la acción es la de **menor costo esperado** según la matriz de arriba, no la clase más probable.
4. **Tiempo hasta la falla.** Regresión sobre las lecturas de los vehículos que sí fallaron (objetivo `log1p`), separada por vehículo.
5. **Supervivencia.** Modelo de Cox con *entrada tardía* en un landmark de 50 unidades de tiempo (todos los vehículos sobrevivieron más allá; los 217 camiones sin lectura antes de ese punto quedan fuera, así que entran 23.333), con uso temprano y especificaciones.
6. **SHAP** sobre el modelo de tiempo hasta la falla.

## Resultados

### Costo de la decisión (métrica oficial, menor es mejor)

| Regla | Validación | Test |
|---|---|---|
| No recomendar nada | 57.400 | 56.100 |
| Clase más probable (argmax) | 52.726 | 54.781 |
| **Costo esperado mínimo** | **41.141** (−28%) | **38.625** (−31%) |

Discriminación, independiente de la regla de decisión: el AUC ROC de "algún riesgo" es **0,709** en validación y **0,699** en test. Es señal real y es modesta.

| Regla de costo esperado mínimo | Validación | Test |
|---|---|---|
| Camiones que van a fallar y reciben alarma | 127 de 136 (93%) | 109 de 142 (77%) |
| Camiones sanos que reciben alarma | 3.640 de 4.910 (74%) | 2.447 de 4.903 (50%) |

![Matriz de confusión en test](reports/figures/02_matriz_confusion.png)

**Qué dice esto y qué no.**

- La regla reduce el costo oficial en cerca de 30% frente a no hacer nada. Lo logra alarmando a una parte grande de los camiones sanos, porque con una razón de costos de 7–10 contra 200–500 una falsa alarma es barata. Si tu costo real de una visita a taller es mayor que 7–10, la política óptima alarma menos y el ahorro se achica.
- En la práctica el sistema funciona como un interruptor binario "alarma / sin alarma". La mayoría de sus alarmas (cerca de 78% en test) son clase 4, así que **no resuelve las ventanas 24–48 / 12–24 / 6–12** (ver la matriz).
- El argmax casi no ayuda (casi nunca alarma), que es justamente la razón de usar la matriz de costos para decidir.
- El recall y la tasa de falsas alarmas difieren bastante entre validación y test (93% / 74% contra 77% / 50%) con el mismo modelo y la misma regla. Hay solo 136 y 142 camiones que fallan en cada conjunto, así que no lo leería como nada distinto de ruido en torno a un AUC de cerca de 0,70.
- No comparo contra la tabla de posiciones del reto: no he verificado esas cifras y la comparación requeriría el mismo protocolo.

### Tiempo hasta la falla (vehículos que fallan)

Vehículos retenidos: 568 camiones con falla, 26.020 lecturas.

| | Modelo | Baseline (mediana) |
|---|---|---|
| MAE, todas las lecturas | 45,4 | 67,1 |
| MAE, lecturas a menos de 48 unidades de la falla | 37,8 | 80,4 |

El modelo supera al baseline, pero un error de unas 38 unidades dentro de una ventana de 48 significa que no ubica la falla con precisión. Además es condicional a que el camión falle: responde "cuánto falta, si falla", no "¿va a fallar?".

![SHAP](reports/figures/03_shap_tiempo_restante.png)

El ranking SHAP lo encabezan `time_step`, `n_readouts` y `Spec_7`: la regresión se apoya sobre todo en cuánto tiempo lleva operando el camión y en su configuración. Las variables operacionales anonimizadas aportan una señal más débil.

### Supervivencia (vehículos retenidos, índice de concordancia)

| Covariables | C-index |
|---|---|
| Solo especificaciones | 0,648 |
| Solo uso temprano | 0,673 |
| Ambas | **0,704** |

## Limitaciones

- **No son datos mineros.** Ver arriba; la transferencia a camiones de extracción fuera de ruta es un supuesto.
- **Variables anonimizadas**, así que no hay diagnóstico físico.
- **Pocos positivos.** 136 y 142 camiones que fallan en validación y test; los intervalos en torno a cualquier cifra de costo son anchos y no los calculé.
- **La matriz de costos es la del benchmark.** El ahorro vale lo que valgan esos 7–10 y 200–500.
- **El dataset no especifica las unidades de tiempo**, así que "6 unidades" no tiene un significado declarado en horas o kilómetros.
- **Una sola familia de modelos** (gradient boosting) e hiperparámetros fijos. No probé modelos secuenciales ni afiné más las features.

## API

`POST /score` recibe el historial de lecturas de un camión y sus `Spec_0…Spec_7`, y devuelve las probabilidades por clase, el costo esperado de cada acción y la recomendada. Requiere el header `X-API-Key` y limita las solicitudes por IP. `GET /health` es abierto.

```bash
export RISK_API_KEY=change-me
uvicorn src.api.main:app
```

El modelo se carga en el primer uso; antes de entrenar el endpoint responde `503` con el comando a ejecutar.

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -c "from src.data.scania_loader import download; download()"   # ~1,65 GB en data/raw/scania
python -m src.models.train_pipeline                                   # features, modelos, figuras (~3 min)
pytest                                                                # 52 tests, sin necesitar el dataset
```

Salidas: `reports/results.json`, `reports/figures/`, `reports/shap_time_to_failure.csv`; modelos en `data/processed/models/`.

## Estructura

```
src/data/scania_loader.py       descarga, carga, etiquetas oficiales y matriz de costos
src/features/engineering.py     features por lectura a partir de contadores acumulados
src/models/cost_decision.py     decisión de costo esperado mínimo, confusión, métricas de alarma
src/models/train_pipeline.py    clasificador, tiempo hasta la falla, Cox, SHAP
src/models/scorer.py            puntúa un camión a partir de su historial
src/models/make_figures.py      figuras
src/api/main.py                 servicio FastAPI
tests/                          tests unitarios sobre camiones sintéticos; ninguno necesita los datos reales
```

Una versión anterior de este repositorio también contenía un generador de datos sintéticos, una red multitarea en PyTorch y un dashboard en Streamlit. Los eliminé porque dependían del esquema sintético; siguen en el historial de git.

## Licencia

MIT. Los datos son © Scania CV AB, publicados bajo CC BY 4.0.
