# chile-mining-predictive-maintenance

Sistema hibrido de Mantenimiento Predictivo para flotillas de **camiones de
extraccion (CAEX)** y **chancadores primarios** en faenas mineras chilenas
(Chuquicamata, Escondida, Los Bronces, El Teniente, Radomiro Tomic).

**Fase 1** implementa el pipeline completo: generacion sintetica de la base
relacional (telemetria + metadatos + mantenimiento, con censura de datos) →
feature engineering de degradacion (Polars + FFT) → modelo hibrido de
Survival Analysis (CoxPH) + RUL (LightGBM) + clasificacion de tipo de falla,
con explicabilidad SHAP → servicio FastAPI de scoring → dashboard Streamlit
con heatmap de riesgo de flota.

## 🎯 Problema de negocio

Predecir la **probabilidad de falla catastrofica** y la **Vida Util
Restante (RUL, en horas)** de cada equipo, para reducir paradas no
planificadas en faena.

## 🛠️ Stack

- Python 3.11+
- Polars + PyArrow — procesamiento de datos en memoria de alta velocidad
- LightGBM — regresion de RUL + clasificacion multiclase de tipo de falla
- Lifelines (CoxPH) — Survival Analysis con datos censurados
- SHAP — explicabilidad tecnica para mecanicos e ingenieros de mantenimiento
- FastAPI — servicio de score de riesgo operacional
- Streamlit + Plotly — dashboard de telemetria de flota con heatmap de riesgo
- Pytest — validacion de datos, features y modelos

> **Nota de diseno:** el enunciado permitia LightGBM *o* PyTorch para el
> multi-task learning. Se eligio LightGBM para ambas cabezas (regresion RUL
> + clasificacion de falla) por ser mas liviano de entrenar/servir y mas
> facil de explicar con SHAP en un pipeline de Fase 1; un modelo conjunto en
> PyTorch queda como extension natural de Fase 2.

## 📁 Estructura

```
chile-mining-predictive-maintenance/
├── data/
│   ├── raw/
│   └── processed/                        # parquet + modelos entrenados (generados)
├── src/
│   ├── data/
│   │   └── mining_data_generator.py      # base relacional sintetica (3 tablas)
│   ├── features/
│   │   └── engineering.py                # rolling stats, deltas, var. acumulada, FFT
│   ├── models/
│   │   ├── train_survival_pipeline.py    # CoxPH + LightGBM (RUL + clasificacion) + SHAP
│   │   └── scoring.py                    # carga de artefactos + scoring compartido
│   ├── api/
│   │   └── main.py                       # FastAPI: score de riesgo operacional
│   └── app/
│       └── dashboard.py                  # Streamlit: heatmap de riesgo de flota
├── tests/
├── requirements.txt
└── README.md
```

## 🗄️ Esquema relacional sintetico

- **`equipment_metadata`**: `equipment_id`, `equipment_type` (CAEX /
  Chancador Primario), `model`, `faena`, `manufacture_year`, `install_date`,
  `hours_in_current_cycle` (reloj de supervivencia), `event_observed`
  (1=fallo observado, 0=censurado/aun operando).
- **`sensor_telemetry`**: lecturas de `engine_temp_c`, `vibration_rms_mm_s`,
  `hydraulic_pressure_bar`, `rpm`, `fuel_consumption_lph` que cubren el
  **ciclo de vida completo** de cada equipo (0 horas hasta
  `hours_in_current_cycle`), a resolucion adaptativa: hora a hora si el
  ciclo dura menos que `DEFAULT_TARGET_READINGS_PER_EQUIPMENT = 600` horas,
  o con intervalo creciente si es mas largo, para acotar el volumen de
  datos sin perder cobertura del ciclo completo.
- **`maintenance_logs`**: eventos de `mantenimiento_programado` y
  `falla_no_planificada` (con `component`/`failure_type` y `downtime_hours`).

El reloj de supervivencia (`hours_in_current_cycle`) modela **horas desde la
ultima gran intervencion**, no horas de vida total del equipo — asi se
simula un proceso de renovacion realista: `duration = min(T, C)`,
`event = 1{T <= C}`, con `T` (tiempo real de falla, Weibull) y `C` (corte de
observacion) generados por equipo segun su tipo, faena y antiguedad.

## 🚀 Instalacion

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Uso

Ejecutar en orden desde la raiz del repositorio:

### 1. Generar la base relacional sintetica

```powershell
python -m src.data.mining_data_generator
```

Genera 520+ equipos, ~175k lecturas de telemetria y su historial de
mantenimiento en `data/processed/*.parquet`.

### 2. Feature engineering

```powershell
python -m src.features.engineering
```

Calcula, por equipo (Polars, vectorizado): medias/std rodantes (6h/24h),
deltas de temperatura y presion vs. tendencia de 168h, varianza acumulada
(expanding) de vibracion, y features FFT de vibracion en ventanas
deslizantes de 24 lecturas (amplitud dominante + energia espectral, para
detectar componentes periodicas de defectos de rodamiento).

### 3. Entrenar el pipeline hibrido

```powershell
python -m src.models.train_survival_pipeline
```

Entrena y evalua (siempre con split train/test **a nivel de equipo**, nunca
por fila, para evitar fuga de datos):

- **CoxPH** (supervivencia con censura) → C-Index en holdout.
- **LightGBM Regressor** (RUL, solo equipos con falla observada) → MAE en
  horas.
- **LightGBM Classifier** (tipo de falla mas probable) → Accuracy / F1-macro.
- **SHAP** sobre el modelo de RUL → `data/processed/shap_rul_importance.csv`
  y `data/processed/models/shap_rul_summary.png`.
- **SHAP multiclase** sobre el clasificador de tipo de falla → importancia
  global (`shap_failure_classifier_importance.csv`) y desglosada por clase
  (`shap_failure_classifier_importance_by_class.csv` +
  `data/processed/models/shap_failure_classifier_summary.png`), para que un
  mecanico vea que sensor explica *cada* modo de falla especifico, no solo
  el RUL agregado.

Guarda modelos en `data/processed/models/` y metricas en
`data/processed/metrics.json`.

> El RUL se entrena sobre el ciclo de vida completo de cada equipo fallado
> (no una ventana reciente acotada), por lo que las predicciones van desde
> 0 horas (en la falla) hasta miles de horas en equipos jovenes dentro de
> su ciclo. Los umbrales de riesgo (`src/models/scoring.py`) son horizontes
> de negocio reales: CRITICO < 1 semana, ALTO < 1 mes, MEDIO < 3 meses,
> BAJO en adelante.

### 4. Servicio de scoring (FastAPI)

```powershell
uvicorn src.api.main:app --reload
```

Endpoints principales (documentacion interactiva en `/docs`):

- `GET /health`
- `GET /equipment` — lista de la flota (filtros `faena`, `equipment_type`)
- `GET /equipment/{equipment_id}/risk` — RUL, tipo de falla probable,
  probabilidad por clase y supervivencia condicional a 30 dias
- `GET /fleet/risk-summary` — conteo por nivel de riesgo, global y por faena
- `POST /score/raw` — scoring ad-hoc a partir de un vector de features
  (sin necesidad de un `equipment_id` registrado)

### 5. Dashboard de flota (Streamlit)

```powershell
streamlit run src/app/dashboard.py
```

Filtros por faena/tipo/riesgo, KPIs de flota, **heatmap de riesgo por
equipo** (una celda = un equipo, coloreado por nivel de riesgo), tabla
filtrable y panel de detalle por equipo (probabilidades de falla + tendencia
de sensores).

### 6. Tests

```powershell
pytest
```

Cubre: integridad del esquema relacional sintetico y consistencia de la
censura, ausencia de fuga en el feature engineering (nulos de warm-up de
rolling/FFT), rango valido de C-Index/MAE/Accuracy, y contrato del modulo de
scoring compartido (`FleetScorer`).

## 🔭 Siguientes pasos

- Reemplazar el generador Weibull por datos historicos reales de faena.
- Multi-task learning conjunto (PyTorch) para RUL + clasificacion con
  representacion compartida, en vez de dos LightGBM independientes.
- Autenticacion y rate-limiting en la API para uso productivo en faena.
