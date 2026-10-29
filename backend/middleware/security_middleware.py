import os
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

def _build_csp(report_only: bool = False) -> str:
    directives = [
        "default-src 'self'",
        "script-src 'self'",
        "frame-ancestors 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ]
    csp_string = "; ".join(directives)
    if report_only:
        csp_string += "; report-uri /csp-report"
    return csp_string

def _build_permissions_policy() -> str:
    denied = [
        "geolocation=()",
        "microphone=()",
        "camera=()",
        "payment=()",
        "usb=()"
    ]
    return ", ".join(denied)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._hsts_max_age = int(os.environ.get("HSTS_MAX_AGE", "31536000"))
        self._hsts_include_subdomains = os.environ.get("HSTS_INCLUDE_SUBDOMAINS", "true").lower() == "true"
        self._hsts_preload = os.environ.get("HSTS_PRELOAD", "false").lower() == "true"
        self._csp_report_only = os.environ.get("CSP_REPORT_ONLY", "false").lower() == "true"

    def _hsts_value(self) -> str:
        value = f"max-age={self._hsts_max_age}"
        if self._hsts_include_subdomains:
            value += "; includeSubDomains"
        if self._hsts_preload:
            value += "; preload"
        return value

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)

        csp_value = _build_csp(report_only=self._csp_report_only)
        if self._csp_report_only:
            response.headers["Content-Security-Policy-Report-Only"] = csp_value
        else:
            response.headers["Content-Security-Policy"] = csp_value

        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = _build_permissions_policy()
        response.headers["Strict-Transport-Security"] = self._hsts_value()
        response.headers["X-XSS-Protection"] = "0"

        return response