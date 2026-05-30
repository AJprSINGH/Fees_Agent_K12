# =============================================================
# middleware/auth.py — Bearer Token Authentication
# =============================================================
#
# STEP 1 — Backend stores the secret key securely:
#   • Set API_SECRET_KEY in Hugging Face → Settings → Variables and Secrets
#   • Example value: Triz@669Mar69
#   • FastAPI reads it via: os.getenv("API_SECRET_KEY")
#   • The key is NEVER visible publicly — it lives only in HF Secrets
#
# STEP 2 — Client sends the same key in the HTTP Authorization header:
#   • Correct format:   Authorization: Bearer Triz@669Mar69
#   • Wrong format:     Put key inside JSON payload ← DO NOT DO THIS
#
# STEP 3 — FastAPI compares both keys:
#   • If they match   → ✅ API access granted
#   • If they differ  → ❌ 401 Unauthorized returned
#
# ─────────────────────────────────────────────────────
# SWAGGER UI INSTRUCTIONS:
#   1. Open your HF Space URL
#   2. Click the "Authorize" button (top right)
#   3. Enter ONLY:  Triz@669Mar69
#      (Do NOT type "Bearer" — Swagger adds it automatically)
#   4. Click Authorize → Close
#   5. All protected APIs now work from Swagger
#
# ─────────────────────────────────────────────────────
# FIX NOTES:
#   auto_error=False  → Our function controls ALL errors (not FastAPI).
#                        Without this, FastAPI returns 403 "Not authenticated"
#                        before our code even runs — bypassing custom messages.
#   scheme_name="BearerAuth" → Matches the OpenAPI securitySchemes name in
#                        app.py's custom_openapi(). Without this, FastAPI
#                        registers as "HTTPBearer" — Swagger sees two schemes
#                        and doesn't send the token in requests.
# =============================================================

import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# CRITICAL:
#   auto_error=False  — disables FastAPI's own 403 "Not authenticated" so our
#                       function returns the proper 401 with a clear message.
#   scheme_name="BearerAuth" — must match the name in app.py custom_openapi()
#                       so Swagger sends the token in the Authorization header.
security = HTTPBearer(auto_error=False, scheme_name="BearerAuth")


def require_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """
    FastAPI dependency — inject into any route that needs protection.

    Usage in route decorator (recommended — Swagger shows lock icon):
        @router.post("/endpoint", dependencies=[Depends(require_api_key)])
        def my_endpoint(payload: MyModel):
            ...

    HOW IT WORKS:
    1. FastAPI extracts the token from: Authorization: Bearer <token>
    2. This function reads API_SECRET_KEY from environment at call time
    3. Compares the two — match → proceed, mismatch → 401 Unauthorized

    SETUP:
    • Add API_SECRET_KEY to HF Secrets (Settings → Variables and Secrets)
    • Restart the HF Space after adding the secret
    • Use the same value in your Authorization header when calling the API
    """
    # Re-read from env at every call — picks up new HF Secret after restart
    key = os.getenv("API_SECRET_KEY")

    if not key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "API_SECRET_KEY is not configured on the server. "
                "Add API_SECRET_KEY to HF Secrets → Settings → Variables and Secrets, "
                "then restart the Space."
            ),
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Missing Authorization header. "
                "Send your key as: Authorization: Bearer <API_SECRET_KEY>"
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )

    incoming_token = credentials.credentials

    if incoming_token != key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid API key. "
                "Send your key as: Authorization: Bearer <API_SECRET_KEY>"
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True
