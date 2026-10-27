class CanonicalSyncError(RuntimeError):
    """Base exception for fail-closed canonical synchronization mechanics."""


class AuthorityMismatchError(CanonicalSyncError):
    """An observed source does not match the declared authority profile."""


class SourceMovementError(CanonicalSyncError):
    """A source or exact membership changed during an acquisition attempt."""


class SnapshotIntegrityError(CanonicalSyncError):
    """A staged or persisted snapshot failed integrity verification."""


class PublicationBoundaryError(CanonicalSyncError):
    """A proposed public projection violates the publication boundary."""
