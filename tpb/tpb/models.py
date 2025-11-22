from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Torrent:
    title: str
    magnet: str
    seeders: int
    leechers: int
    size: str
    uploader: str
