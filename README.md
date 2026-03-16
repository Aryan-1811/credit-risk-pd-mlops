# Credit Risk PD Modelling — MLOps Pipeline
**End-to-end Basel II/III-aligned Probability of Default model with full MLOps lifecycle**

## 🚀 Live Demo
**[https://credit-risk-pd-mlops.vercel.app](https://credit-risk-pd-mlops.vercel.app)**

FastAPI backend: [https://pd-scorer-api.onrender.com/docs](https://pd-scorer-api.onrender.com/docs)

## Architecture

```
New Data
   │
   ▼
Prefect Training Pipeline
   ├── Ingest & Validate
   ├── WoE Feature Engineering
   ├── Train: LogReg / XGBoost / LightGBM
   ├── Evaluate: AUC, KS, Gini, Brier
   └── Register to MLflow Model Registry
              │
              ▼
        MLflow Registry
        (Champion Model: Production)
              │
              ▼
      FastAPI Scoring Service          ← POST /predict
      (Dockerised, port 8000)          ← POST /predict/batch
              │
              ▼
   Prefect Monitoring Pipeline (weekly)
   ├── Load reference vs production data
   ├── Compute PSI (score + features)
   ├── Generate Evidently HTML report
   └── Trigger retraining if PSI > 0.25
              │
              ▼
   GitHub Actions CI/CD
   ├── ci.yml       — lint + test on every PR
   └── retrain.yml  — monthly scheduled retraining
```

## Tech Stack

| Layer | Tool |
|---|---|
| ML Models | Logistic Regression, XGBoost, LightGBM |
| Feature Engineering | WoE/IV (Basel-standard) |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Orchestration | Prefect |
| Drift Monitoring | Evidently AI + PSI |
| Serving | FastAPI + Uvicorn |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |

## Quant Finance Concepts

| Concept | Implementation |
|---|---|
| **EL = PD × LGD × EAD** | `src/models/expected_loss.py` |
| **KS Statistic** | `src/evaluation/metrics.py` |
| **Gini Coefficient** | `src/evaluation/metrics.py` |
| **PSI (drift metric)** | `monitoring/drift_detector.py` |
| **Vasicek Stressed PD** | `src/models/expected_loss.py` |
| **WoE / IV** | `src/features/woe_encoder.py` |
| **Model Calibration** | `src/models/train.py` |

## Quickstart

### Local Development
```bash
pip install -r requirements.txt

# Run full training pipeline
python pipelines/training_pipeline.py

# Start MLflow UI
mlflow ui  # → http://localhost:5000

# Start Prefect UI
prefect server start  # → http://localhost:4200
```

### Docker (Full Stack)
```bash
docker-compose up --build
# MLflow  → http://localhost:5000
# FastAPI → http://localhost:8000/docs
# Prefect → http://localhost:4200
```

### API Usage
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "revolving_utilization": 0.45,
    "age": 42,
    "num_30_59_days_late": 0,
    "debt_ratio": 0.3,
    "monthly_income": 5000,
    "num_open_credit_lines": 4,
    "num_90_days_late": 0,
    "num_real_estate_loans": 1,
    "num_60_89_days_late": 0,
    "num_dependents": 2
  }'
```

Response:
```json
{
  "pd_score": 0.047,
  "risk_band": "LOW",
  "expected_loss": 0.021,
  "model_version": "v3"
}
```

### Run Tests
```bash
pytest tests/ -v
```

## Dataset
Download **Give Me Some Credit** from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data)  
→ Place `cs-training.csv` in `data/raw/`

## PSI Thresholds (credit industry standard)
| PSI | Interpretation |
|-----|---------------|
| < 0.10 | Stable - no action |
| 0.10 – 0.25 | Moderate drift - investigate |
| > 0.25 | Major drift - retrain model |
