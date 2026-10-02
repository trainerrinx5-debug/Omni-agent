"""Minimal OpenAI-compatible Chat Completions client using only the stdlib."""

from __future__ import annotations

import ipaddress
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward API credentials to a redirect target.
        return None


_SAFE_OPENER = build_opener(_NoRedirectHandler)


class ProviderError(RuntimeError):
    """Raised when the configured model endpoint fails or returns bad data."""


def _validate_base_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    try:
        parts = urlsplit(value)
        host = parts.hostname
        port = parts.port  # also validates the port syntax
    except ValueError as exc:
        raise ValueError("OMNI_BASE_URL is not a valid URL") from exc
    if parts.scheme not in {"https", "http"} or not host:
        raise ValueError("OMNI_BASE_URL must be an http(s) URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("OMNI_BASE_URL must not contain credentials, a query, or a fragment")
    if parts.scheme == "http":
        is_loopback = host.casefold() == "localhost"
        if not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise ValueError("Use HTTPS for remote model endpoints; plain HTTP is allowed only on loopback")
    path = parts.path.rstrip("/") or "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 90.0,
    ) -> None:
        self.base_url = _validate_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with _SAFE_OPENER.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(4_000_001)
        except HTTPError as exc:
            body = exc.read(1_000).decode("utf-8", errors="replace").strip()
            if self.api_key:
                body = body.replace(self.api_key, "[redacted]")
            detail = f": {body}" if body else ""
            raise ProviderError(f"Model provider returned HTTP {exc.code}{detail}") from exc
        except URLError as exc:
            reason = getattr(exc, "reason", "connection failed")
            raise ProviderError(f"Could not reach model provider: {reason}") from exc
        except TimeoutError as exc:
            raise ProviderError("Model provider request timed out") from exc

        if len(raw) > 4_000_000:
            raise ProviderError("Model provider response exceeded the 4 MB safety limit")
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError("Model provider returned invalid JSON") from exc
        if not isinstance(result, dict) or not isinstance(result.get("choices"), list) or not result["choices"]:
            message = result.get("error") if isinstance(result, dict) else None
            if isinstance(message, dict):
                message = message.get("message")
            suffix = f": {message}" if isinstance(message, str) and message else ""
            raise ProviderError(f"Model provider response did not contain a choice{suffix}")
        choice = result["choices"][0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ProviderError("Model provider returned a malformed choice")
        return result
