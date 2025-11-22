class SearchError(Exception):
    """Raised when a search fails for all mirrors."""


class MirrorError(Exception):
    """Raised when a mirror is unreachable or invalid."""


class VerificationError(Exception):
    """Raised when PGP/onion verification fails."""
