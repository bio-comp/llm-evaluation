"""Safe technical failures, with no provider body/exception chain retention."""

from enum import StrEnum


class FailureKind(StrEnum):
    CONNECT_TIMEOUT = "connect_timeout"
    READ_TIMEOUT = "read_timeout"
    WRITE_TIMEOUT = "write_timeout"
    POOL_TIMEOUT = "pool_timeout"
    TOTAL_TIMEOUT = "total_timeout"
    NETWORK_ERROR = "network_error"
    HTTP_STATUS = "http_status"
    INVALID_JSON = "invalid_json"
    INVALID_RESPONSE = "invalid_response"
    RESPONSE_TOO_LARGE = "response_too_large"
    CREDENTIAL_REFLECTION = "credential_reflection"


class FailurePhase(StrEnum):
    BEFORE_DISPATCH = "before_dispatch"
    POSSIBLE_DISPATCH = "possible_dispatch"
    RESPONSE_RECEIVED = "response_received"


class RuntimeFailure(RuntimeError):
    def __init__(
        self,
        kind: FailureKind,
        *,
        phase: FailurePhase = FailurePhase.RESPONSE_RECEIVED,
        remote_completion_unknown: bool = False,
        status_code: int | None = None,
    ):
        self.kind = kind
        self.phase = phase
        self.remote_completion_unknown = remote_completion_unknown
        self.status_code = status_code
        super().__init__(
            kind.value + (f" HTTP {status_code}" if status_code is not None else "")
        )
