from __future__ import annotations

from typing import Protocol

from .domain import AuthorityObjectSpec, SourceContent, SourceObservation, TransportProvenance


class SourceTransport(Protocol):
    @property
    def provenance(self) -> TransportProvenance: ...

    def observe(self, spec: AuthorityObjectSpec) -> SourceObservation: ...

    def read(self, spec: AuthorityObjectSpec) -> SourceContent: ...

    def list_members(self, parent_id: str) -> tuple[SourceObservation, ...]: ...
