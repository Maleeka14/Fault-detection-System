"""
AI-Based Power System Fault Detection API
IEEE 33-Bus Distribution Network

Endpoints:
    GET  /                  → API info and status
    GET  /health            → Health check
    GET  /models            → Available models and their accuracy
    GET  /features          → List of all 165 feature names
    POST /predict           → Single prediction with SHAP explanation
    POST /predict/batch     → Batch prediction (multiple rows)

Models served:
    - Random Forest (best for fault type classification)
    - XGBoost
    - ANN / Neural Network (best for fault location prediction)

Usage:
    uvicorn main:app --reload --port 8000
    Swagger docs at http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
import pickle
import os
import shap
from contextlib import asynccontextmanager

# ──────────────────────────────────────────────
#  Paths — adjust if your folder layout differs
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")


# ──────────────────────────────────────────────
#  Feature names (165 total)
# ──────────────────────────────────────────────
FEATURE_NAMES = (
    [f"V_bus{i}" for i in range(1, 34)]
    + [f"Angle_bus{i}" for i in range(1, 34)]
    + [f"V0_bus{i}" for i in range(1, 34)]
    + [f"V1_bus{i}" for i in range(1, 34)]
    + [f"V2_bus{i}" for i in range(1, 34)]
)

MODEL_ACCURACY = {
    "rf": {
        "name": "Random Forest",
        "task_a_accuracy": 0.9948,
        "task_a_f1": 0.9948,
        "task_b_accuracy": 0.8875,
        "task_b_f1": 0.8886,
    },
    "xgb": {
        "name": "XGBoost",
        "task_a_accuracy": 0.9943,
        "task_a_f1": 0.9943,
        "task_b_accuracy": 0.8781,
        "task_b_f1": 0.8794,
    },
    "ann": {
        "name": "Artificial Neural Network",
        "task_a_accuracy": 0.9349,
        "task_a_f1": 0.9323,
        "task_b_accuracy": 0.9187,
        "task_b_f1": 0.9094,
    },
}


# ──────────────────────────────────────────────
#  Global model store (populated at startup)
# ──────────────────────────────────────────────
models = {}
label_encoders = {}
scalers = {}
shap_explainers = {}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models, encoders, and scalers at startup."""

    # Label encoders
    label_encoders["type"] = load_pickle(os.path.join(DATA_DIR, "le_type.pkl"))
    label_encoders["bus"] = load_pickle(os.path.join(DATA_DIR, "le_bus.pkl"))

    # Scalers
    scalers["A"] = load_pickle(os.path.join(DATA_DIR, "scaler_A.pkl"))
    scalers["B"] = load_pickle(os.path.join(DATA_DIR, "scaler_B.pkl"))

    # Random Forest
    models["rf_A"] = load_pickle(os.path.join(MODEL_DIR, "rf_taskA.pkl"))
    models["rf_B"] = load_pickle(os.path.join(MODEL_DIR, "rf_taskB.pkl"))

    # XGBoost
    models["xgb_A"] = load_pickle(os.path.join(MODEL_DIR, "xgb_taskA.pkl"))
    models["xgb_B"] = load_pickle(os.path.join(MODEL_DIR, "xgb_taskB.pkl"))

    # ANN
    try:
        from tensorflow import keras
        models["ann_A"] = keras.models.load_model(os.path.join(MODEL_DIR, "ann_taskA.keras"))
        models["ann_B"] = keras.models.load_model(os.path.join(MODEL_DIR, "ann_taskB.keras"))
    except Exception as e:
        print(f"Warning: Could not load ANN models: {e}")

    # SHAP explainers (tree-based only, ANN SHAP is too slow for API)
    shap_explainers["rf_A"] = shap.TreeExplainer(models["rf_A"])
    shap_explainers["rf_B"] = shap.TreeExplainer(models["rf_B"])
    shap_explainers["xgb_A"] = shap.TreeExplainer(models["xgb_A"])
    shap_explainers["xgb_B"] = shap.TreeExplainer(models["xgb_B"])

    print("All models loaded successfully.")
    yield
    print("Shutting down.")


# ──────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Fault Detection API",
    description="AI-Based Power System Fault Detection on IEEE 33-Bus Distribution Network",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
#  Request / Response schemas
# ──────────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=165,
        max_length=165,
        description="165 voltage features in order: V_bus1-33, Angle_bus1-33, V0_bus1-33, V1_bus1-33, V2_bus1-33",
    )
    model: Optional[Literal["rf", "xgb", "ann"]] = Field(
        default=None,
        description="Model to use. Default: RF for fault type, ANN for fault location.",
    )
    include_shap: Optional[bool] = Field(
        default=False,
        description="Include SHAP explanation in response (adds latency).",
    )


class BatchPredictionRequest(BaseModel):
    rows: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of feature rows, each with 165 values.",
    )
    model: Optional[Literal["rf", "xgb", "ann"]] = Field(default=None)


class FaultTypePrediction(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict[str, float]


class FaultLocationPrediction(BaseModel):
    predicted_bus: int
    confidence: float
    top_3_buses: list[dict]


class PredictionResponse(BaseModel):
    fault_type: FaultTypePrediction
    fault_location: FaultLocationPrediction
    models_used: dict[str, str]
    shap_explanation: Optional[dict] = None


class BatchPredictionResponse(BaseModel):
    predictions: list[dict]
    count: int
    models_used: dict[str, str]


# ──────────────────────────────────────────────
#  Prediction helpers
# ──────────────────────────────────────────────
def predict_fault_type(features_scaled: np.ndarray, model_key: str = "rf"):
    model = models[f"{model_key}_A"]
    le = label_encoders["type"]

    if model_key == "ann":
        probs = model.predict(features_scaled, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
    else:
        probs = model.predict_proba(features_scaled)[0]
        pred_idx = int(np.argmax(probs))

    pred_class = le.classes_[pred_idx]
    confidence = float(probs[pred_idx])
    all_probs = {le.classes_[i]: round(float(probs[i]), 4) for i in range(len(le.classes_))}

    return FaultTypePrediction(
        predicted_class=pred_class,
        confidence=round(confidence, 4),
        all_probabilities=all_probs,
    )


def predict_fault_location(features_scaled: np.ndarray, model_key: str = "ann"):
    model = models[f"{model_key}_B"]
    le = label_encoders["bus"]

    if model_key == "ann":
        probs = model.predict(features_scaled, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
    else:
        probs = model.predict_proba(features_scaled)[0]
        pred_idx = int(np.argmax(probs))

    pred_bus = int(le.classes_[pred_idx])
    confidence = float(probs[pred_idx])

    top_3_idx = np.argsort(probs)[::-1][:3]
    top_3 = [
        {"bus": int(le.classes_[i]), "confidence": round(float(probs[i]), 4)}
        for i in top_3_idx
    ]

    return FaultLocationPrediction(
        predicted_bus=pred_bus,
        confidence=round(confidence, 4),
        top_3_buses=top_3,
    )


def get_shap_explanation(features_raw: np.ndarray, model_key: str = "rf"):
    if model_key == "ann":
        return {"note": "SHAP not available for ANN in API (too slow)."}

    features_flat = features_raw.flatten().tolist()

    # Task A SHAP
    explainer_A = shap_explainers[f"{model_key}_A"]
    shap_vals_A = explainer_A.shap_values(features_raw)
    model_A = models[f"{model_key}_A"]
    pred_idx = int(model_A.predict(features_raw)[0])

    if isinstance(shap_vals_A, list):
        vals_A = np.array(shap_vals_A[pred_idx][0])
    else:
        vals_A = np.array(shap_vals_A[0])

    # Ensure exactly 165 values
    vals_A = vals_A.flatten()[:165]
    top_idx_A = np.argsort(np.abs(vals_A))[::-1][:10]
    type_explanation = [
        {
            "feature": FEATURE_NAMES[int(i)],
            "shap_value": round(float(vals_A[int(i)]), 4),
            "feature_value": round(float(features_flat[int(i)]), 4),
        }
        for i in top_idx_A
    ]

    # Task B SHAP
    explainer_B = shap_explainers[f"{model_key}_B"]
    features_B_scaled = scalers["B"].transform(features_raw)
    shap_vals_B = explainer_B.shap_values(features_B_scaled)
    model_B = models[f"{model_key}_B"]
    pred_idx_B = int(model_B.predict(features_B_scaled)[0])

    if isinstance(shap_vals_B, list):
        vals_B = np.array(shap_vals_B[pred_idx_B][0])
    else:
        vals_B = np.array(shap_vals_B[0])

    vals_B = vals_B.flatten()[:165]
    top_idx_B = np.argsort(np.abs(vals_B))[::-1][:10]
    location_explanation = [
        {
            "feature": FEATURE_NAMES[int(i)],
            "shap_value": round(float(vals_B[int(i)]), 4),
            "feature_value": round(float(features_flat[int(i)]), 4),
        }
        for i in top_idx_B
    ]

    return {
        "fault_type_explanation": type_explanation,
        "fault_location_explanation": location_explanation,
    }
# ──────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    html_path = os.path.join(BASE_DIR, "..", "dashboard", "index.html")
    return FileResponse(html_path, media_type="text/html")


@app.get("/health")
def health():
    loaded = list(models.keys())
    return {
        "status": "healthy",
        "models_loaded": len(loaded),
        "models": loaded,
    }


@app.get("/models")
def model_info():
    return {
        "available_models": MODEL_ACCURACY,
        "default_fault_type_model": "rf",
        "default_fault_location_model": "ann",
        "feature_count": 165,
        "fault_types": list(label_encoders["type"].classes_),
        "fault_buses": [int(b) for b in label_encoders["bus"].classes_],
    }


@app.get("/features")
def feature_list():
    groups = {
        "voltage_magnitude": [f"V_bus{i}" for i in range(1, 34)],
        "voltage_angle": [f"Angle_bus{i}" for i in range(1, 34)],
        "zero_sequence": [f"V0_bus{i}" for i in range(1, 34)],
        "positive_sequence": [f"V1_bus{i}" for i in range(1, 34)],
        "negative_sequence": [f"V2_bus{i}" for i in range(1, 34)],
    }
    return {
        "total_features": 165,
        "feature_groups": groups,
        "all_features": FEATURE_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    features_raw = np.array(req.features).reshape(1, -1)

    # Determine models
    type_model = req.model or "rf"
    loc_model = req.model or "ann"

    # Validate model availability
    if f"{type_model}_A" not in models:
        raise HTTPException(status_code=400, detail=f"Model '{type_model}' not loaded for Task A.")
    if f"{loc_model}_B" not in models:
        raise HTTPException(status_code=400, detail=f"Model '{loc_model}' not loaded for Task B.")

    # Scale features
    features_A_scaled = scalers["A"].transform(features_raw)
    features_B_scaled = scalers["B"].transform(features_raw)

    # Predict
    fault_type = predict_fault_type(features_A_scaled, type_model)
    fault_location = predict_fault_location(features_B_scaled, loc_model)

    # SHAP (optional)
    shap_explanation = None
    if req.include_shap:
        shap_model = req.model or "rf"
        if shap_model != "ann":
            shap_explanation = get_shap_explanation(features_raw, shap_model)
        else:
            shap_explanation = {"note": "SHAP not available for ANN in API."}

    return PredictionResponse(
        fault_type=fault_type,
        fault_location=fault_location,
        models_used={
            "fault_type": MODEL_ACCURACY[type_model]["name"],
            "fault_location": MODEL_ACCURACY[loc_model]["name"],
        },
        shap_explanation=shap_explanation,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictionRequest):
    if any(len(row) != 165 for row in req.rows):
        raise HTTPException(status_code=400, detail="Every row must have exactly 165 features.")

    type_model = req.model or "rf"
    loc_model = req.model or "ann"

    if f"{type_model}_A" not in models:
        raise HTTPException(status_code=400, detail=f"Model '{type_model}' not loaded.")
    if f"{loc_model}_B" not in models:
        raise HTTPException(status_code=400, detail=f"Model '{loc_model}' not loaded.")

    predictions = []
    for row in req.rows:
        features_raw = np.array(row).reshape(1, -1)
        features_A_scaled = scalers["A"].transform(features_raw)
        features_B_scaled = scalers["B"].transform(features_raw)

        ft = predict_fault_type(features_A_scaled, type_model)
        fl = predict_fault_location(features_B_scaled, loc_model)

        predictions.append({
            "fault_type": ft.predicted_class,
            "fault_type_confidence": ft.confidence,
            "fault_bus": fl.predicted_bus,
            "fault_bus_confidence": fl.confidence,
        })

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        models_used={
            "fault_type": MODEL_ACCURACY[type_model]["name"],
            "fault_location": MODEL_ACCURACY[loc_model]["name"],
        },
    )
