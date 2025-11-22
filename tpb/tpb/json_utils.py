from dataclasses import asdict
import json
from .models import Torrent


def torrents_to_json(torrents: list[Torrent]) -> str:
    return json.dumps([asdict(t) for t in torrents], indent=2)
