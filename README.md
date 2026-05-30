---
title: Fees Agent K12
emoji: 🐠
colorFrom: gray
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# K12 Fee Defaulter Detection — Production API

FastAPI-based ML system for school fee defaulter prediction.

## 🚀 First-Time Deployment (HF Spaces)

### Step 1 — Add Secrets
Go to **Settings → Variables and Secrets** and add:

| Key | Value |
|-----|-------|
| `API_SECRET_KEY` | `trizk12secure2026` (or any strong secret) |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Your Supabase service role key |

### Step 2 — Restart Space
After saving secrets: **Settings → Restart this Space**

### Step 3 — Models Auto-Train
On first startup, if `models/*.pkl` are empty/missing, the app **automatically trains all models** from `data/Fees Defaulter Report - Overall.xlsx`. This happens **once only**. After training, models are saved to disk and loaded on every subsequent restart — no retraining needed.

### Step 4 — Authorize in Swagger
Open `https://your-space-url.hf.space/` → click **Authorize** (top right) → enter your `API_SECRET_KEY` → click **Authorize** → **Close**.

> ⚠️ Do NOT prefix with "Bearer" — FastAPI adds it automatically in the new Swagger schema.

## 🔑 API Authentication

All `/api/v6/*` endpoints require:
```
Authorization: Bearer trizk12secure2026
```

## 📡 Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v6/batch-inference` | Run ML predictions for all students |
| GET | `/api/v6/early-warning` | Get High/Critical risk students |
| GET | `/health/models` | Check if models are loaded |
| GET | `/health` | Basic health check |

## Example: Run Batch Inference

```bash
curl -X POST 'https://trizk-12-fees-agent-k12.hf.space/api/v6/batch-inference' \
  -H 'Authorization: Bearer trizk12secure2026' \
  -H 'Content-Type: application/json' \
  -d '{
    "agent_id": "c40a3e05-0b64-4332-80c6-0fcbd3067766",
    "gr_number": "14135",
    "student_name": "ANANYA - RAI",
    "std_div": "IV/A",
    "academic_year": "2026",
    "data_path": "data/Fees Defaulter Report - Overall.xlsx",
    "mode": "annual",
    "persist": true,
    "threshold_days": 30
  }'
```

## Example: Get Early Warning Report

```bash
curl 'https://trizk-12-fees-agent-k12.hf.space/api/v6/early-warning?academic_year=2026&limit=5000' \
  -H 'Authorization: Bearer trizk12secure2026'
```

## 🏗️ Architecture

- **Deployment mode:** Inference Only — no runtime retraining
- **Human-in-the-loop:** All actions require school admin approval
- **Grade bands:** Nursery, Primary, Secondary (separate ML models per band)
- **Risk levels:** Low → Medium → High → Critical

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
