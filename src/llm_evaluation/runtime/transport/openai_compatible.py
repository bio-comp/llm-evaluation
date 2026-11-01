"""Owned, no-retry HTTPX clients with exact-wire and bounded response semantics."""

import asyncio
import time
from collections.abc import Mapping
from urllib.parse import urlsplit
import httpx
from .base import BearerAuthorization
from .response_models import (
    ProviderChatProjection,
    ProviderResponsesProjection,
    ProviderParser,
)
from ..codec import parse_json_unique, sha256_hex
from ..contracts import OperationalEndpoint, HttpTimeouts
from ..evidence import RawResponseEvidence, GenerationExecution, GenerationObservation
from ..failures import RuntimeFailure, FailureKind, FailurePhase
from ..identity import EffectiveRequest
from ..timing import RequestTiming

_SENSITIVE = {
    "authorization",
    "proxy_authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "client_secret",
}


def reject_sensitive(value: object, secrets: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            reject_sensitive(str(k), secrets)
            if str(k).strip().lower().replace("-", "_") in _SENSITIVE:
                raise RuntimeFailure(FailureKind.CREDENTIAL_REFLECTION)
            reject_sensitive(v, secrets)
    elif isinstance(value, (tuple, list)):
        for item in value:
            reject_sensitive(item, secrets)
    elif isinstance(value, str) and any(s in value for s in secrets):
        raise RuntimeFailure(FailureKind.CREDENTIAL_REFLECTION)


class AsyncOpenAICompatibleTransport:
    """Client/transport injection is intentionally unsupported to enforce policy.

    Every call owns and closes its client in the caller's loop. A timeout or
    cancellation after possible dispatch leaves remote completion unknown.
    """

    def __init__(
        self,
        endpoint: OperationalEndpoint,
        *,
        authorization: BearerAuthorization | None = None,
        timeouts: HttpTimeouts | None = None,
        max_response_bytes: int = 2 * 1024 * 1024,
    ):
        self.endpoint = endpoint
        self.authorization = authorization
        self.timeouts = timeouts or HttpTimeouts()
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1
        ):
            raise ValueError("positive evidence ceiling required")
        self.max_response_bytes = max_response_bytes
        p = urlsplit(endpoint.base_url)
        if (
            authorization is not None
            and p.scheme != "https"
            and p.hostname not in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError("credential requires HTTPS except loopback")

    async def request_json(
        self, method: str, path: str, body: bytes = b""
    ) -> tuple[object, RawResponseEvidence, float]:
        if (
            not path.startswith("/")
            or path.startswith("//")
            or "?" in path
            or "#" in path
        ):
            raise ValueError("relative API path required")
        started = time.perf_counter()
        dispatched = False
        key = self.authorization.reveal() if self.authorization else None
        secrets = (key, "Bearer " + key) if key else ()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
        if key:
            headers["Authorization"] = "Bearer " + key
        failure: RuntimeFailure | asyncio.CancelledError | None = None
        result: tuple[object, RawResponseEvidence, float] | None = None
        timeout = httpx.Timeout(
            connect=self.timeouts.connect,
            read=self.timeouts.read,
            write=self.timeouts.write,
            pool=self.timeouts.pool,
        )
        try:
            async with asyncio.timeout(self.timeouts.total):
                async with httpx.AsyncClient(
                    timeout=timeout,
                    trust_env=False,
                    follow_redirects=False,
                    transport=httpx.AsyncHTTPTransport(retries=0, trust_env=False),
                ) as client:
                    dispatched = True
                    async with client.stream(
                        method,
                        self.endpoint.base_url + path,
                        content=body,
                        headers=headers,
                    ) as response:
                        if not 200 <= response.status_code < 300:
                            raise RuntimeFailure(
                                FailureKind.HTTP_STATUS,
                                status_code=response.status_code,
                            )
                        if (
                            response.headers.get("Content-Encoding", "identity").lower()
                            != "identity"
                        ):
                            raise RuntimeFailure(FailureKind.INVALID_RESPONSE)
                        request_id = response.headers.get("X-Request-ID")
                        if request_id is not None and len(request_id) > 4096:
                            raise RuntimeFailure(FailureKind.INVALID_RESPONSE)
                        reject_sensitive(request_id, secrets)
                        raw = bytearray()
                        async for chunk in response.aiter_raw():
                            if len(raw) + len(chunk) > self.max_response_bytes:
                                raise RuntimeFailure(FailureKind.RESPONSE_TOO_LARGE)
                            raw.extend(chunk)
                        raw_bytes = bytes(raw)
                        # Malformed bodies never leave this method, including any embedded key.
                        payload = parse_json_unique(raw_bytes)
                        reject_sensitive(payload, secrets)
                        result = (
                            payload,
                            RawResponseEvidence(
                                body=raw_bytes,
                                body_sha256=sha256_hex(raw_bytes),
                                request_id=request_id,
                            ),
                            time.perf_counter() - started,
                        )
        except asyncio.CancelledError:
            failure = asyncio.CancelledError(
                "request cancelled; remote completion may be unknown"
            )
            setattr(failure, "remote_completion_unknown", dispatched)
        except RuntimeFailure as exc:
            failure = RuntimeFailure(
                exc.kind,
                phase=exc.phase,
                status_code=exc.status_code,
                remote_completion_unknown=exc.remote_completion_unknown,
            )
        except (httpx.TimeoutException, TimeoutError) as exc:
            kinds: dict[type[BaseException], FailureKind] = {
                httpx.ConnectTimeout: FailureKind.CONNECT_TIMEOUT,
                httpx.ReadTimeout: FailureKind.READ_TIMEOUT,
                httpx.WriteTimeout: FailureKind.WRITE_TIMEOUT,
                httpx.PoolTimeout: FailureKind.POOL_TIMEOUT,
            }
            kind = kinds.get(type(exc), FailureKind.TOTAL_TIMEOUT)
            failure = RuntimeFailure(
                kind,
                phase=FailurePhase.POSSIBLE_DISPATCH,
                remote_completion_unknown=dispatched,
            )
        except httpx.HTTPError:
            failure = RuntimeFailure(
                FailureKind.NETWORK_ERROR,
                phase=FailurePhase.POSSIBLE_DISPATCH,
                remote_completion_unknown=dispatched,
            )
        except (ValueError, RecursionError):
            failure = RuntimeFailure(FailureKind.INVALID_JSON)
        # Raise outside handlers: no secret-bearing exception retained as __context__.
        if failure is not None:
            raise failure
        assert result is not None
        return result

    async def generate(self, request: EffectiveRequest) -> GenerationExecution:
        if request.timeouts != self.timeouts:
            raise ValueError(
                "transport timeout policy differs from prepared request identity"
            )
        payload, evidence, elapsed = await self.request_json(
            "POST", request.path, request.wire_bytes
        )
        failure = False
        observation = None
        try:
            if not isinstance(payload, dict):
                raise ValueError("object response required")
            if request.api_mode == "chat_completions":
                ProviderChatProjection.model_validate(payload)
                fields = ProviderParser._parse_chat(payload)
            else:
                ProviderResponsesProjection.model_validate(payload)
                fields = ProviderParser._parse_responses(payload)
            body_id = payload.get("request_id")
            fields["provider_request_id"] = (
                body_id if isinstance(body_id, str) and body_id else evidence.request_id
            )
            reported = fields["backend_model_id"]
            fields["reported_model_matches_condition"] = (
                None if reported is None else reported == request.endpoint_model_id
            )
            observation = GenerationObservation(**fields)
        except ValueError:
            failure = True
        if failure:
            raise RuntimeFailure(FailureKind.INVALID_RESPONSE)
        assert observation is not None
        return GenerationExecution(
            observation=observation,
            evidence=evidence,
            timing=RequestTiming(total_request_wall_seconds=elapsed),
            effective_request=request,
        )
