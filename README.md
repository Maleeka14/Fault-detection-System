# GridSense — AI-Based Power Grid Fault Detection

A machine learning system that detects and locates faults on the IEEE 33-bus radial distribution network. Three models (Random Forest, XGBoost, Neural Network) classify six fault types and pinpoint the faulted bus from 165 voltage measurements.

**Live demo:** [umerrr123-faults-in-grid.hf.space](https://umerrr123-faults-in-grid.hf.space)

---

## What it does

- **Fault type classification** — identifies one of six fault categories (Normal, LG, LL, LLG, LLLG, HIF)
- **Fault location prediction** — predicts which of the 32 buses is faulted
- **SHAP attribution** — explains which voltage features drove each prediction
- **Batch processing** — upload a CSV of scenarios and download results
- **Model comparison** — side-by-side accuracy breakdown for all three models

---

## Model performance

| Model | Fault Type Accuracy | Fault Location Accuracy |
|---|---|---|
| Random Forest | **99.48%** | 88.75% |
| XGBoost | 99.43% | 87.81% |
| Neural Network | 93.49% | **91.87%** |

Evaluated on a held-out test set of 384 samples (fault type) and 320 samples (fault location).

---

## Fault types

| Code | Name | Description |
|---|---|---|
| — | Normal | Grid within design parameters |
| LG | Line-to-ground | Single phase contacts earth — most common (~70–80% of faults) |
| LL | Line-to-line | Two phases shorted without ground involvement |
| LLG | Double line-to-ground | Two phases simultaneously grounded |
| LLLG | Three-phase-to-ground | All three phases to ground — most severe |
| HIF | High-impedance fault | Very low fault current, often missed by conventional protection |

---

## Input features

165 features per scenario — five measurement types across all 33 buses:

| Feature group | Count | Description |
|---|---|---|
| `V_bus[1–33]` | 33 | Voltage magnitude (per unit) |
| `Angle_bus[1–33]` | 33 | Phase angle (degrees) |
| `V0_bus[1–33]` | 33 | Zero-sequence component |
| `V1_bus[1–33]` | 33 | Positive-sequence component |
| `V2_bus[1–33]` | 33 | Negative-sequence component |

---

## Dataset

- **1,920 scenarios** balanced across 6 fault classes (320 per class)
- Faults injected via Y-bus matrix modification using symmetrical components theory
- Load variations: 55–145% of base loading for Normal class
- Fault impedance (`Rf`): 0–1.0 pu for standard faults, 2–2000 pu for HIF
- Network: IEEE 33-bus Baran & Wu radial feeder, 12.66 kV, 3.715 MW

---

## Project structure

```
├── api/
│   ├── main.py              # FastAPI backend
│   └── requirements.txt
├── dashboard/
│   └── index.html           # Frontend (single-page app)
├── data/
│   ├── raw/                 # Original CSV datasets
│   └── processed/           # Scaled arrays and label encoders
├── models/
│   ├── rf_taskA.pkl         # Random Forest — fault type
│   ├── rf_taskB.pkl         # Random Forest — fault location
│   ├── xgb_taskA.pkl        # XGBoost — fault type
│   ├── xgb_taskB.pkl        # XGBoost — fault location
│   ├── ann_taskA.keras      # Neural Network — fault type
│   └── ann_taskB.keras      # Neural Network — fault location
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_random_forest.ipynb
│   ├── 04_xgboost.ipynb
│   ├── 05_ann.ipynb
│   └── 06_SHAP.ipynb
├── results/                 # Confusion matrices, SHAP plots
├── Dockerfile
└── README.md
```

---

## Run locally

**Requirements:** Python 3.11+

```bash
# Install dependencies
pip install -r api/requirements.txt

# Start the API server
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000` — the dashboard loads automatically.

API docs available at `http://localhost:8000/docs`.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server and model status |
| `GET` | `/models` | Accuracy metrics for all models |
| `GET` | `/features` | Full list of 165 feature names |
| `POST` | `/predict` | Single prediction with optional SHAP |
| `POST` | `/predict/batch` | Batch prediction (up to 100 rows) |

**Single prediction example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 0.998, ...], "model": "rf", "include_shap": false}'
```

---

## Tech stack

- **Backend:** FastAPI, scikit-learn, XGBoost, TensorFlow/Keras, SHAP
- **Frontend:** Vanilla HTML/CSS/JS (no framework)
- **Deployment:** Docker, Hugging Face Spaces

---

## Limitations

- Trained on steady-state simulation data — real grid measurements may differ due to noise and harmonic distortion
- Specific to the IEEE 33-bus topology — will not transfer to other networks without retraining
- Zero-sequence impedance approximated as Z0 = 3·Z1 uniformly across all lines
