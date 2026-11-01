"""Distinct historical v1 and operational-address-independent v2 identities."""

from enum import StrEnum
from .codec import canonical_json_bytes, sha256_hex
from .contracts import ClosedRecord, ScientificEndpointIdentity, HttpTimeouts


class RequestIdentityVersion(StrEnum):
    MODEL_RUNNER_V1 = "model_runner_v1"
    RUNTIME_V2 = "runtime_v2"


class EffectiveRequest(ClosedRecord):
    identity_version: RequestIdentityVersion = RequestIdentityVersion.RUNTIME_V2
    path: str
    api_mode: str
    wire_bytes: bytes
    body_sha256: str
    prompt_packet_sha256: str
    condition_sha256: str
    request_identity_sha256: str
    backend_config_id: str
    condition_id: str
    endpoint_model_id: str
    timeouts: HttpTimeouts


def request_identity_v1(
    *,
    endpoint: str,
    api_mode: str,
    backend_config_id: str,
    condition_sha256: str,
    body_sha256: str,
    prompt_packet_sha256: str,
    timeout_seconds: float,
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            dict(
                http_method="POST",
                endpoint=endpoint,
                api_mode=api_mode,
                backend_config_id=backend_config_id,
                condition_sha256=condition_sha256,
                body_sha256=body_sha256,
                prompt_packet_sha256=prompt_packet_sha256,
                timeout_seconds=timeout_seconds,
            )
        )
    )


def request_identity_v2(
    *,
    endpoint: ScientificEndpointIdentity,
    api_mode: str,
    path: str,
    backend_config_id: str,
    condition_sha256: str,
    body_sha256: str,
    prompt_packet_sha256: str,
    timeouts: HttpTimeouts,
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            dict(
                identity_version="runtime_v2",
                http_method="POST",
                scientific_endpoint=endpoint.value,
                api_mode=api_mode,
                path=path,
                backend_config_id=backend_config_id,
                condition_sha256=condition_sha256,
                body_sha256=body_sha256,
                prompt_packet_sha256=prompt_packet_sha256,
                timeouts=timeouts.model_dump(),
            )
        )
    )
