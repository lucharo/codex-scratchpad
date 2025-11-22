import json

from tpb.json_utils import torrents_to_json
from tpb.models import Torrent


def test_torrents_to_json():
    torrents = [
        Torrent(
            title="Example",
            magnet="magnet:?xt=urn:btih:example",
            seeders=10,
            leechers=2,
            size="700 MiB",
            uploader="user",
        )
    ]

    output = torrents_to_json(torrents)
    payload = json.loads(output)

    assert payload == [
        {
            "title": "Example",
            "magnet": "magnet:?xt=urn:btih:example",
            "seeders": 10,
            "leechers": 2,
            "size": "700 MiB",
            "uploader": "user",
        }
    ]
