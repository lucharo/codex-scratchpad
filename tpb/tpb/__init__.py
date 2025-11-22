"""Research-oriented terminal client for The Pirate Bay."""

from .client import TPBClient
from .exceptions import MirrorError, SearchError, VerificationError
from .models import Torrent

__all__ = [
    "TPBClient",
    "Torrent",
    "MirrorError",
    "SearchError",
    "VerificationError",
]
