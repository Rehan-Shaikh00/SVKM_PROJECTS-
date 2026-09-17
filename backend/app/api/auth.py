"""Admin authentication.

Shared-secret HTTP Basic for the staff dashboard — deliberately simple, because
in a real deployment this sits behind the university's SSO (SAML/OIDC) or a
reverse proxy. Set ADMIN_AUTH_ENABLED=false for local development only.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from ..config import settings

logger = logging.getLogger("nims.auth")

basic_auth = HTTPBasic(auto_error=False)


async def require_admin(
    credentials: HTTPBasicCredentials | None = Depends(basic_auth),
) -> str:
    if not settings.admin_auth_enabled:
        return "dev"
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": "Basic realm=\"NMIMS Dhule Voice Assistant Admin\""},
        )
    username_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"), settings.admin_username.encode("utf-8")
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"), settings.admin_password.encode("utf-8")
    )
    if not (username_ok and password_ok):
        logger.warning("failed admin login attempt for user=%s", credentials.username[:40])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
            headers={"WWW-Authenticate": "Basic realm=\"NMIMS Dhule Voice Assistant Admin\""},
        )
    return credentials.username
