"""Operational secrets deliberately have no model serialization method."""

from typing import Protocol
from pydantic import SecretStr
from ..evidence import GenerationExecution
from ..identity import EffectiveRequest


class BearerAuthorization:
    __slots__ = ("_key",)

    def __init__(self, key: str):
        if not isinstance(key, str) or not key or "\r" in key or "\n" in key:
            raise ValueError("nonempty single-line credential required")
        self._key = SecretStr(key)

    def reveal(self) -> str:
        return self._key.get_secret_value()

    def __repr__(self) -> str:
        return "BearerAuthorization([REDACTED])"

    __str__ = __repr__


class AsyncGenerationTransport(Protocol):
    async def generate(self, request: EffectiveRequest) -> GenerationExecution: ...
