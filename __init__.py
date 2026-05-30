# =============================================================
# middleware/auth.py — Bearer Token Authentication
#
# Protects admin/sensitive endpoints from unauthenticated access.
# Set API_SECRET_KEY in your .env before deploying.
# =============================================================

import os
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_bearer = HTTPBearer(auto_error=False)

API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")


def require_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> str:
    """
    Dependency — inject into any route that should be admin-only:

        @app.post("/api/v6/retrain")
        def retrain(payload: dict, _: str = Depends(require_api_key)):
            ...

    Set API_SECRET_KEY in .env.  If the env var is unset the server
    will raise on startup so you notice immediately.
    """
    if not API_SECRET_KEY:
        raise RuntimeError(
            "API_SECRET_KEY is not set.  "
            "Add it to .env before running the server."
        )

    if credentials is None or credentials.credentials != API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
