# Predictive maintenance on real heavy-duty truck data (SCANIA Component X)

[Versión en español](README.es.md) · English

![tests](https://github.com/Rxyxs/chile-mining-predictive-maintenance/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

![Decision cost by rule](reports/figures/01_costo_decision.png)

## Why this project

Predictive maintenance for haul fleets is a decision problem, not a classification problem: sending a truck to the workshop that did not need it costs a little, missing a failure costs a lot. An earlier version of this repository ran on a data generator I wrote myself, and a model that scores well on data its author simulated is measuring the simulator. I replaced it with a real public benchmark, **SCANIA Component X**, and rebuilt the pipeline around the cost structure that benchmark defines.

The question I answer is concrete: **given the readout history of a truck up to today, which maintenance action minimises expected cost, and how much better is that than doing nothing?**

I also have to be upfront about the limit of that question. These are heavy-duty *road* trucks from a European manufacturer, not Chilean off-highway mining haul trucks. The data has the same shape as the mining problem (cumulative usage counters, rare and expensive failures, censoring), but I have no evidence here that the numbers transfer to a mine.

## Data

[SCANIA Component X](https://researchdata.se/en/catalogue/dataset/2024-34) — public, CC BY 4.0, no registration. Described in *SCANIA Component X dataset: a real-world multivariate time series dataset for predictive maintenance* ([Scientific Data, 2025](https://www.nature.com/articles/s41597-025-04802-6)). It is the dataset of the IDA 2024 Industrial Challenge.

| | |
|---|---|
| Vehicles (train) | 23,550, of which 2,272 had Component X repaired during the study (9.6%); the rest are censored |
| Readouts (train) | 1,122,452 irregularly sampled readouts, 105 columns from 14 anonymised variables |
| Specifications | 8 categorical columns per vehicle |
| Validation / test | 5,046 / 5,045 vehicles, **one randomly chosen readout per vehicle** (it simulates standing at "today" with no future information) |
| Labels | 0 = no failure within 48 time units, 1–4 = failure in 24–48, 12–24, 6–12, 0–6 time units |

Things that matter when reading the numbers:

- **All 105 columns are cumulative counters and histogram bins** (100% non-decreasing). The raw value mostly measures the truck's age, so the features are rates and fractions (below).
- **Variable names are anonymised.** I can say *which* variable matters, not *what* it measures.
- **Class 0 is 97%** of validation and test vehicles. Predicting "healthy" for everyone is a strong-looking baseline.
- The authors state that repair and readout frequencies may have been modified and are not necessarily representative of real usage.

## The decision problem

The challenge scores a recommendation with an asymmetric cost matrix:

| | recommend 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **truck is healthy (0)** | 0 | 7 | 8 | 9 | 10 |
| **fails in 24–48 (1)** | 200 | 0 | 7 | 8 | 9 |
| **fails in 12–24 (2)** | 300 | 200 | 0 | 7 | 8 |
| **fails in 6–12 (3)** | 400 | 300 | 200 | 0 | 7 |
| **fails in 0–6 (4)** | 500 | 400 | 300 | 200 | 0 |

An unnecessary check costs 7–10; a missed failure costs 200–500. The reference to beat is "recommend nothing for everyone": **57,400** on validation and **56,100** on test.

## Method

1. **Features** (`src/features/engineering.py`, 407 per readout + 8 specifications). Uses only readouts up to the current one: usage rate since start, usage rate over the last 5 readouts, histogram bin fractions over the lifetime and over the recent window, readout count and gap. A test recomputes every feature on truncated histories and checks it does not change, so nothing looks ahead.
2. **Risk classifier.** LightGBM multiclass on all risky readouts plus 5% of healthy ones (re-weighted ×20). Vehicles are split into train and hold-out *by vehicle*, with early stopping on the hold-out. Hyperparameters are fixed up front; nothing is tuned on validation or test.
3. **Decision rule.** The model outputs class probabilities; the action is the one with the **lowest expected cost** under the matrix above, not the most probable class.
4. **Time to failure.** Regression on readouts of vehicles that did fail (`log1p` target), split by vehicle.
5. **Survival.** Cox model with *delayed entry* at a landmark of 50 time units (every vehicle survived past it; the 217 trucks with no readout before that point are left out, so 23,333 enter), on early-life usage and specifications.
6. **SHAP** on the time-to-failure model.

## Results

### Decision cost (official metric, lower is better)

| Rule | Validation | Test |
|---|---|---|
| Recommend nothing | 57,400 | 56,100 |
| Most probable class (argmax) | 52,726 | 54,781 |
| **Minimum expected cost** | **41,141** (−28%) | **38,625** (−31%) |

Discrimination, independent of the decision rule: ROC AUC for "any risk" is **0.709** on validation and **0.699** on test. That is real signal and it is modest.

| Minimum-expected-cost rule | Validation | Test |
|---|---|---|
| Failing trucks that get an alarm | 127 of 136 (93%) | 109 of 142 (77%) |
| Healthy trucks that get an alarm | 3,640 of 4,910 (74%) | 2,447 of 4,903 (50%) |

![Confusion matrix on test](reports/figures/02_matriz_confusion.png)

**What this says, and what it does not.**

- The rule cuts the official cost by about 30% against doing nothing. It does so by alarming a large share of healthy trucks, because with a 7–10 versus 200–500 cost ratio a false alarm is cheap. If your real cost of a workshop visit is higher than 7–10, the optimal policy alarms less and the saving shrinks.
- In practice the system behaves as a binary "alarm / no alarm" switch. Most of its alarms (about 78% on test) are class 4, so **it does not resolve the 24–48 / 12–24 / 6–12 windows** (see the matrix).
- Argmax barely helps (it almost never alarms), which is the point of using the cost matrix to decide.
- Recall and false-alarm rate differ noticeably between validation and test (93% / 74% against 77% / 50%) with the same model and rule. There are only 136 and 142 failing trucks in each set, so I would not read the difference as anything but noise around an AUC of about 0.70.
- I do not compare against the challenge leaderboard: I have not verified those figures and the comparison would need the same protocol.

### Time to failure (vehicles that fail)

Held-out vehicles: 568 failed trucks, 26,020 readouts.

| | Model | Median baseline |
|---|---|---|
| MAE, all readouts | 45.4 | 67.1 |
| MAE, readouts within 48 units of failure | 37.8 | 80.4 |

The model is better than the baseline, but an error of about 38 units inside a 48-unit window means it cannot locate the failure precisely. It is also conditional on the truck failing, so it answers "how long, if it fails", not "will it fail".

![SHAP](reports/figures/03_shap_tiempo_restante.png)

The SHAP ranking is led by `time_step`, `n_readouts` and `Spec_7`: the regression leans mostly on how long the truck has been running and on its configuration. The anonymised operating variables contribute a weaker signal.

### Survival (held-out vehicles, concordance index)

| Covariates | C-index |
|---|---|
| Specifications only | 0.648 |
| Early-life usage only | 0.673 |
| Both | **0.704** |

## Limitations

- **Not mining data.** See above; transfer to off-highway mining trucks is an assumption.
- **Anonymised variables**, so no physical diagnosis.
- **Few positives.** 136 and 142 failing trucks in validation and test; the intervals around any cost figure are wide and I did not compute them.
- **The cost matrix is the benchmark's.** The saving is only as good as those 7–10 and 200–500 figures.
- **Time units are unspecified** by the dataset, so "6 units" has no stated meaning in hours or kilometres.
- **A single model family** (gradient boosting) and fixed hyperparameters. I did not try sequence models or tune the features further.

## API

`POST /score` takes the readout history of one truck and its `Spec_0…Spec_7`, and returns class probabilities, the expected cost of each action and the recommended one. It needs an `X-API-Key` header and is rate limited per IP. `GET /health` is open.

```bash
export RISK_API_KEY=change-me
uvicorn src.api.main:app
```

The model is loaded on first use; before training the endpoint answers `503` with the command to run.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -c "from src.data.scania_loader import download; download()"   # ~1.65 GB into data/raw/scania
python -m src.models.train_pipeline                                   # features, models, figures (~3 min)
pytest                                                                # 52 tests, no dataset needed
```

Outputs: `reports/results.json`, `reports/figures/`, `reports/shap_time_to_failure.csv`; models in `data/processed/models/`.

## Layout

```
src/data/scania_loader.py       download, load, official labels and cost matrix
src/features/engineering.py     per-readout features from cumulative counters
src/models/cost_decision.py     minimum-expected-cost decision, confusion, alarm metrics
src/models/train_pipeline.py    classifier, time to failure, Cox, SHAP
src/models/scorer.py            score one truck from its history
src/models/make_figures.py      figures
src/api/main.py                 FastAPI service
tests/                          unit tests on synthetic trucks; none need the real data
```

An earlier version of this repository also contained a synthetic data generator, a multi-task PyTorch network and a Streamlit dashboard. I removed them because they depended on the synthetic schema; they remain in the git history.

## License

MIT. The data is © Scania CV AB, released under CC BY 4.0.
