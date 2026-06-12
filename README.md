<div align="center">

# ⚡ GridSense

### AI-Based Power Grid Fault Detection on the IEEE 33-Bus Network

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1F77B4?style=flat-square)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8E44AD?style=flat-square)](https://shap.readthedocs.io/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2025b-0076A8?style=flat-square&logo=mathworks&logoColor=white)](https://www.mathworks.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)

**Three ML models classify six fault types and pinpoint the faulted bus from 165 voltage measurements.**

[🌐 Live Demo](https://umerrr123-faults-in-grid.hf.space) · [📖 API Docs](http://localhost:8000/docs) · [🧪 Notebooks](./notebooks)

</div>

---

## 🎯 What It Does

> A complete end-to-end pipeline from physics-based simulation through deployed inference.

| Capability | Description |
|------------|-------------|
| 🔍 **Fault Type Classification** | Identifies one of six fault categories (Normal, LG, LL, LLG, LLLG, HIF) |
| 📍 **Fault Location Prediction** | Predicts which of 32 buses is faulted |
| 💡 **SHAP Attribution** | Explains which voltage features drove each prediction |
| 📊 **Batch Processing** | Upload CSV scenarios and download results |
| ⚖️ **Model Comparison** | Side-by-side accuracy across Random Forest, XGBoost, and ANN |

---

## 📈 Model Performance

<div align="center">

| Model | Fault Type | Fault Location |
|:---|:---:|:---:|
| 🌲 **Random Forest** | 🟢 **99.48%** | 88.75% |
| 🚀 **XGBoost** | 99.43% | 87.81% |
| 🧠 **Neural Network** | 93.49% | 🟢 **91.87%** |

*Evaluated on a held-out test set of 384 samples (type) and 320 samples (location).*

</div>

> 🌲 Tree-based models excel at fault **type** classification. 🧠 The neural network leads on spatial **localization**.

---

## ⚡ Fault Categories

| Code | Name | Description |
|:---:|:---|:---|
| 🟢 **—** | Normal | Grid within design parameters |
| 🟠 **LG** | Line-to-ground | Single phase contacts earth (~70–80% of faults) |
| 🟠 **LL** | Line-to-line | Two phases shorted, no ground involvement |
| 🔴 **LLG** | Double line-to-ground | Two phases simultaneously grounded |
| 🔴 **LLLG** | Three-phase-to-ground | All three phases to ground — most severe |
| 🟡 **HIF** | High-impedance fault | Very low fault current, often missed by protection |

---

## 🧬 Input Features

> **165 features per scenario** — five measurement types across all 33 buses.

| Feature Group | Count | Description | Physical Role |
|:---|:---:|:---|:---|
| `V_bus[1–33]` | 33 | Voltage magnitude (per unit) | Captures voltage sag from fault current |
| `Angle_bus[1–33]` | 33 | Phase angle (degrees) | Critical for HIF detection |
| `V0_bus[1–33]` | 33 | Zero-sequence component | Spikes during ground faults |
| `V1_bus[1–33]` | 33 | Positive-sequence component | Fundamental operating voltage |
| `V2_bus[1–33]` | 33 | Negative-sequence component | Rises during unbalanced faults |

---

## 📦 Dataset

```
🔢 1,920 scenarios            ✅ Perfectly balanced (320 per class)
⚙️  Y-bus matrix modification  📐 Symmetrical components theory
🔌 IEEE 33-bus Baran-Wu feeder ⚡ 12.66 kV, 3.715 MW
📊 Load variation: 55–145%     🌳 HIF resistance: 2–2000 pu
```

Generated using sequence-network fault analysis with backward-forward sweep load flow (tolerance 1e-10). All five fault types simulated across 32 fault buses with 10 resistance levels each.

---

## 📁 Project Structure

```
GridSense/
│
├── 📂 api/
│   ├── main.py                    # FastAPI backend with 6 endpoints
│   └── requirements.txt
│
├── 📂 dashboard/
│   └── index.html                 # Single-page interactive frontend
│
├── 📂 data/
│   ├── raw/                       # Original CSV dataset
│   └── processed/                 # Scaled arrays + label encoders
│
├── 📂 matlab/
│   └── generate_dataset.m         # Physics-based simulation script
│
├── 📂 models/
│   ├── rf_taskA.pkl     rf_taskB.pkl      🌲 Random Forest
│   ├── xgb_taskA.pkl    xgb_taskB.pkl     🚀 XGBoost
│   └── ann_taskA.keras  ann_taskB.keras   🧠 Neural Network
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb               # Exploratory analysis
│   ├── 02_preprocessing.ipynb     # Splits & scaling
│   ├── 03_random_forest.ipynb     # 🌲 RF training
│   ├── 04_xgboost.ipynb           # 🚀 XGBoost training
│   ├── 05_ann.ipynb               # 🧠 Neural network training
│   └── 06_SHAP.ipynb              # 💡 Feature attribution
│
├── 📂 results/                    # Confusion matrices, SHAP plots
├── 🐳 Dockerfile
└── 📖 README.md
```

---

## 🚀 Run Locally

**Requirements:** Python 3.11+

```bash
# 1. Clone the repository
git clone https://github.com/Maleeka14/Fault-detection-System.git
cd Fault-detection-System

# 2. Install dependencies
pip install -r api/requirements.txt

# 3. Start the API server
uvicorn api.main:app --reload --port 8000
```

🌐 **Dashboard:** Open `dashboard/index.html` in your browser
📖 **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/health` | Server and model status |
| `GET` | `/models` | Accuracy metrics for all models |
| `GET` | `/features` | Full list of 165 feature names |
| `POST` | `/predict` | Single prediction with optional SHAP |
| `POST` | `/predict/batch` | Batch prediction (up to 100 rows) |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.0, 0.998, 0.991, ...],
    "model": "rf",
    "include_shap": true
  }'
```

**Example response:**

```json
{
  "fault_type": {
    "predicted_class": "LG",
    "confidence": 1.0,
    "all_probabilities": { "LG": 1.0, "HIF": 0.0 }
  },
  "fault_location": {
    "predicted_bus": 2,
    "confidence": 0.945,
    "top_3_buses": []
  }
}
```

---

## 💡 SHAP Explainability

Validated that models learned **physically correct** features rather than spurious correlations:

| Fault Type | Dominant Features | Physical Justification |
|:---|:---|:---|
| 🟠 **LG** | Zero-sequence (V0) | Ground fault current path |
| 🟠 **LL** | Negative-sequence (V2) | Phase imbalance |
| 🟡 **HIF** | Voltage angles | Magnitude changes are subtle at high impedance |
| 🔴 **LLLG** | V2 components | Captures small asymmetries in balanced fault |

This confirms the ML models converged on the same features that classical power-systems theory predicts.

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|:---|:---|
| **Simulation** | MATLAB R2025b · IEEE 33-bus model |
| **ML / Data** | Python 3.11 · scikit-learn · XGBoost · TensorFlow · SHAP · pandas · NumPy |
| **Backend** | FastAPI · Uvicorn · Pydantic |
| **Frontend** | Vanilla HTML / CSS / JavaScript |
| **Deployment** | Docker · Hugging Face Spaces |

</div>

---

## ⚠️ Limitations


- 📊 Trained on **steady-state simulation data** — real grid measurements may differ due to noise and harmonic distortion
- 🔌 Specific to the **IEEE 33-bus topology** — will not transfer to other networks without retraining
- ⚙️ Zero-sequence impedance approximated as **Z0 = 3·Z1** uniformly across all lines (standard assumption when actual data unavailable)
- 📈 LG resistance restricted to 0–0.005 pu to ensure separation from HIF — documented in methodology

---

## 📜 License

This project is released under the [MIT License](LICENSE). Use it, modify it, learn from it.

---

<div align="center">

**Built end-to-end** · MATLAB simulation → ML pipeline → REST API → Web dashboard

⭐ Star this repo if you found it useful

</div>
