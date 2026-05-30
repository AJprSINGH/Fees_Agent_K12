# K12 Fee Defaulter API — Hugging Face Deployment & Authentication Guide

## Why 401 Unauthorized Happens

Your API is protected by Bearer Token authentication.

If you call the API **without** sending the Authorization header, FastAPI returns:
```json
{"detail": "Invalid or missing API key."}
```

This means the server is working correctly — it is simply blocking unauthenticated access.

---

## Full Authentication Flow

```
STEP 1: Backend stores the secret key
        HF Secrets → API_SECRET_KEY = trizk12secure2026

STEP 2: Client sends the same key in the HTTP header
        Authorization: Bearer trizk12secure2026

STEP 3: FastAPI compares both keys
        if token == API_SECRET_KEY → ✅ Access granted
        else                       → ❌ 401 Unauthorized
```

---

## Important: Key Goes in Header, NOT in JSON Payload

❌ **WRONG** — Do NOT put the key inside JSON:
```json
{
  "api_key": "trizk12secure2026",
  "student_name": "KARNAVI DINESHBHAI PATEL"
}
```

✅ **CORRECT** — Key goes in the HTTP Authorization header:
```
Authorization: Bearer trizk12secure2026
```

The JSON body contains only student data. Security token is always in the header.

This is the same concept as OpenRouter:
```python
# OpenRouter — same pattern
headers = {"Authorization": "Bearer sk-xxxx"}   # ← security in header
data    = {"model": "...", "messages": [...]}    # ← data in body
```

---

## Correct CURL Examples

### POST /api/v6/batch-inference
```bash
curl -X POST 'https://trizk-12-fees-agent-k12.hf.space/api/v6/batch-inference' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer trizk12secure2026' \
  -H 'Content-Type: application/json' \
  -d '{
    "agent_id": "c40a3e05-0b64-4332-80c6-0fcbd3067766",
    "gr_number": "13124",
    "student_name": "KARNAVI DINESHBHAI PATEL",
    "std_div": "V/F",
    "academic_year": "2026",
    "data_path": "data/Fees Defaulter Report - Overall.xlsx",
    "mode": "annual",
    "persist": true,
    "threshold_days": 30
  }'
```

### GET /api/v6/early-warning
```bash
curl -X GET \
  'https://trizk-12-fees-agent-k12.hf.space/api/v6/early-warning?academic_year=2026&limit=5000' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer trizk12secure2026'
```

---

## Swagger UI Instructions

1. Open: `https://trizk-12-fees-agent-k12.hf.space/`
2. Click the **Authorize** button (top right of the Swagger page)
3. Enter **ONLY**: `trizk12secure2026`
   > ⚠️ Do NOT type "Bearer" — Swagger UI adds it automatically
4. Click **Authorize** → **Close**
5. All protected APIs now work directly from Swagger

---

## Hugging Face Secret Setup

### Step 1
Open your HF Space → **Settings** → **Variables and Secrets**

### Step 2
Click **New Secret** and add:

| Key | Value |
|-----|-------|
| `API_SECRET_KEY` | `trizk12secure2026` |
| `SUPABASE_URL` | your Supabase project URL |
| `SUPABASE_KEY` | your Supabase service role key |

### Step 3
Click **Save**

### Step 4 — MANDATORY
**Settings → Restart this Space**

Without restart, `os.getenv("API_SECRET_KEY")` still returns empty.

---

## Backend Validation Explanation

Inside `middleware/auth.py`:

```python
key = os.getenv("API_SECRET_KEY", "")          # reads from HF Secrets

if credentials.credentials != key:             # compares with client token
    raise HTTPException(status_code=401, ...)  # blocks if mismatch
```

The client token comes from:
```
Authorization: Bearer trizk12secure2026
                       ^^^^^^^^^^^^^^^^
                       This part is credentials.credentials
```

---

## Model Auto-Training

On first startup, if `.pkl` model files are missing, the app automatically:
1. Reads `data/Fees Defaulter Report - Overall.xlsx`
2. Trains models for all 3 grade bands (nursery, primary, secondary)
3. Saves `.pkl` files to disk
4. Loads them into memory immediately

**This happens ONCE only.** Every subsequent restart loads saved models — no retraining.

Expected startup log after first boot:
```
🔄 AUTO-TRAINING: Missing models detected.
✅ AUTO-TRAINING COMPLETE — bands trained: ['nursery', 'primary', 'secondary']
✅ MODELS NOW LOADED: ['nursery', 'primary', 'secondary']
```

---

## APIs That Are Protected

| Method | Endpoint | Auth Required |
|--------|----------|---------------|
| POST | `/api/v6/batch-inference` | ✅ Bearer token |
| GET | `/api/v6/early-warning` | ✅ Bearer token |
| GET | `/api/v6/drift/status` | ✅ Bearer token |
| GET | `/api/v6/models/versions` | ✅ Bearer token |
| GET | `/api/v6/models/active` | ✅ Bearer token |
| POST | `/api/v6/models/promote` | ✅ Bearer token |
| POST | `/api/v6/retrain` | ✅ Bearer token |
| GET | `/health` | 🌐 Public |
| GET | `/health/models` | 🌐 Public |

---

## Final Result After Correct Setup

✅ Swagger Authorize button works  
✅ CURL requests work with `-H 'Authorization: Bearer trizk12secure2026'`  
✅ HF Secret loads via `os.getenv("API_SECRET_KEY")`  
✅ APIs are secured — unauthorized users get 401  
✅ Models auto-train on first boot and stay loaded permanently  
✅ `/api/v6/batch-inference` works correctly  
✅ `/api/v6/early-warning` works correctly  
