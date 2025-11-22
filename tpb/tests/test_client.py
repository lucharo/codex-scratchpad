from urllib import error
from unittest import mock

import pytest

from tpb.client import TPBClient
from tpb.exceptions import MirrorError, SearchError, VerificationError
from tpb.models import Torrent


SAMPLE_HTML = """
<table id="searchResult">
<tr>
<td>
    <div class="detName"><a class="detLink" href="/torrent/1">Sample Torrent</a></div>
    <a href="magnet:?xt=urn:btih:example">Magnet</a>
    <font class="detDesc">Uploaded 01-01, Size 1.23 GiB, ULed by uploader1</font>
</td>
<td align="right">120</td>
<td align="right">55</td>
</tr>
</table>
"""


def build_mock_opener(html: str | Exception):
    opener = mock.Mock()
    if isinstance(html, Exception):
        opener.open.side_effect = html
    else:
        response = mock.Mock()
        response.read.return_value = html.encode()
        response.__enter__ = lambda s: s
        response.__exit__ = mock.Mock()
        opener.open.return_value = response
    return opener


def test_search_happy_path(monkeypatch):
    mock_opener = build_mock_opener(SAMPLE_HTML)
    monkeypatch.setattr("tpb.client.request.build_opener", lambda *_: mock_opener)

    client = TPBClient(mirrors=["https://mirror.example"], use_tor=False)
    results = client.search("ubuntu", limit=5)

    assert len(results) == 1
    assert results[0] == Torrent(
        title="Sample Torrent",
        magnet="magnet:?xt=urn:btih:example",
        seeders=120,
        leechers=55,
        size="1.23 GiB",
        uploader="uploader1",
    )


def test_all_mirrors_fail(monkeypatch):
    error_instance = error.URLError("timeout")
    mock_opener = build_mock_opener(error_instance)
    monkeypatch.setattr("tpb.client.request.build_opener", lambda *_: mock_opener)

    client = TPBClient(mirrors=["https://mirror.example"], use_tor=False)
    with pytest.raises(SearchError):
        client.search("ubuntu")


def test_onion_requires_tor():
    with pytest.raises(MirrorError):
        TPBClient(onion_mirror="http://example.onion", use_tor=False)


def test_pgp_strict_failure(monkeypatch):
    mock_gpg = mock.Mock()
    mock_gpg.verify_data.return_value = mock.Mock(valid=False)
    monkeypatch.setattr("tpb.client.gnupg.GPG", lambda: mock_gpg)

    monkeypatch.setattr("tpb.client.TPBClient._fetch", lambda *_, **__: "data")

    with pytest.raises(VerificationError):
        TPBClient(
            onion_mirror="http://example.onion",
            use_tor=True,
            pgp_check=True,
            pgp_public_key="KEY",
            allow_unverified_onion=False,
        )


def test_tor_enabled_by_default(monkeypatch):
    captured_proxies = {}

    def fake_proxy_handler(proxies):
        nonlocal captured_proxies
        captured_proxies = proxies
        return "proxy-handler"

    monkeypatch.setattr("tpb.client.request.ProxyHandler", fake_proxy_handler)
    monkeypatch.setattr("tpb.client.request.build_opener", lambda *handlers: handlers[0])

    client = TPBClient()
    assert client._opener == "proxy-handler"
    assert captured_proxies == {
        "http": "socks5h://127.0.0.1:9050",
        "https": "socks5h://127.0.0.1:9050",
    }
