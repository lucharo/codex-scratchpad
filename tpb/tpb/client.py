from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Iterable
from urllib import error, parse, request

import gnupg

from .exceptions import MirrorError, SearchError, VerificationError
from .models import Torrent


class _RowParser(HTMLParser):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit
        self.rows: list[Torrent] = []
        self._in_row = False
        self._in_title = False
        self._in_details = False
        self._title_parts: list[str] = []
        self._details_parts: list[str] = []
        self._current: dict[str, str | int] = {}
        self._align_right_index = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag == "tr":
            if len(self.rows) >= self.limit:
                return
            self._in_row = True
            self._current = {
                "title": "",
                "magnet": "",
                "seeders": 0,
                "leechers": 0,
                "size": "Unknown",
                "uploader": "Unknown",
            }
            self._title_parts = []
            self._details_parts = []
            self._align_right_index = 0
            return

        if not self._in_row:
            return

        attr_map = {key: value for key, value in attrs}

        if tag == "a":
            href = attr_map.get("href", "")
            if href.startswith("magnet:"):
                self._current["magnet"] = href
            class_attr = attr_map.get("class", "") or ""
            if "detLink" in class_attr.split():
                self._in_title = True
            return

        if tag == "td":
            align_value = (attr_map.get("align", "") or "").lower()
            if align_value == "right":
                self._align_right_index += 1
            class_attr = attr_map.get("class", "") or ""
            if "detDesc" in class_attr.split():
                self._in_details = True
            return

        if tag == "font":
            class_attr = attr_map.get("class", "") or ""
            if "detDesc" in class_attr.split():
                self._in_details = True

    def handle_endtag(self, tag: str):
        if tag == "a" and self._in_title:
            self._in_title = False
        if tag in {"td", "font"} and self._in_details:
            self._in_details = False
        if tag == "tr" and self._in_row:
            if len(self.rows) < self.limit and self._current.get("magnet"):
                title = " ".join(part for part in self._title_parts if part).strip()
                details_text = " ".join(part for part in self._details_parts if part)
                size, uploader = self._parse_details(details_text)
                self._current["title"] = title
                self._current["size"] = size
                self._current["uploader"] = uploader
                torrent = Torrent(
                    title=title or "",
                    magnet=str(self._current["magnet"]),
                    seeders=int(self._current.get("seeders", 0)),
                    leechers=int(self._current.get("leechers", 0)),
                    size=str(self._current.get("size", "Unknown")),
                    uploader=str(self._current.get("uploader", "Unknown")),
                )
                self.rows.append(torrent)
            self._in_row = False

    def handle_data(self, data: str):
        if not self._in_row:
            return
        text = data.strip()
        if not text:
            return

        if self._in_title:
            self._title_parts.append(text)
            return

        if self._in_details:
            self._details_parts.append(text)
            return

        if self._align_right_index == 1:
            self._current["seeders"] = int(text) if text.isdigit() else 0
        elif self._align_right_index == 2:
            self._current["leechers"] = int(text) if text.isdigit() else 0

    @staticmethod
    def _parse_details(details: str) -> tuple[str, str]:
        size_match = re.search(r"Size ([^,]+)", details)
        uploader_match = re.search(r"ULed by ([^,]+)", details)
        size = size_match.group(1).strip() if size_match else "Unknown"
        uploader = uploader_match.group(1).strip() if uploader_match else "Unknown"
        return size, uploader


def _build_url(mirror: str, query: str, page: int, sort: int) -> str:
    quoted_query = parse.quote(query)
    base = mirror if mirror.endswith("/") else f"{mirror}/"
    return parse.urljoin(base, f"search/{quoted_query}/{page}/{sort}/0")


class TPBClient:
    DEFAULT_MIRRORS = [
        "https://thepiratebay.org",
        "https://pirateproxy.live",
        "https://tpb.party",
    ]

    DEFAULT_ONION = "http://uj3wazyk5u4hnvtk.onion"

    def __init__(
        self,
        mirrors: Iterable[str] | None = None,
        use_tor: bool = True,
        timeout: int = 8,
        onion_mirror: str | None = None,
        pgp_check: bool = False,
        pgp_public_key: str | None = None,
        allow_unverified_onion: bool = False,
    ):
        if onion_mirror and not use_tor:
            raise MirrorError("Onion mirrors require Tor to be enabled.")

        self.use_tor = use_tor
        self.timeout = timeout
        self.onion_mirror = onion_mirror
        self.pgp_check = pgp_check
        self.pgp_public_key = pgp_public_key
        self.allow_unverified_onion = allow_unverified_onion
        self._onion_verified = True

        self.mirrors = list(mirrors) if mirrors is not None else list(self.DEFAULT_MIRRORS)
        if onion_mirror:
            self.mirrors = [onion_mirror] + self.mirrors

        self._opener = self._build_opener(use_tor)

        if pgp_check and onion_mirror and pgp_public_key:
            self._verify_onion()

    def _build_opener(self, use_tor: bool):
        proxies: dict[str, str] = {}
        if use_tor:
            proxies = {
                "http": "socks5h://127.0.0.1:9050",
                "https": "socks5h://127.0.0.1:9050",
            }
        proxy_handler = request.ProxyHandler(proxies)
        return request.build_opener(proxy_handler)

    def _fetch(self, url: str, *, head: bool = False) -> str:
        req = request.Request(url)
        if head:
            req.get_method = lambda: "HEAD"
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except error.URLError as exc:  # pragma: no cover - exercised via mocks
            raise MirrorError(str(exc)) from exc
        except OSError as exc:  # pragma: no cover - exercised via mocks
            raise MirrorError(str(exc)) from exc

    def _verify_onion(self):
        assert self.onion_mirror is not None
        assert self.pgp_public_key is not None

        gpg = gnupg.GPG()
        gpg.import_keys(self.pgp_public_key)

        resource_url = parse.urljoin(self.onion_mirror, "mirrors.txt")
        signature_url = f"{resource_url}.asc"

        resource = self._fetch(resource_url)
        signature = self._fetch(signature_url)

        verification = gpg.verify_data(signature, resource.encode("utf-8"))
        is_valid = bool(verification) and getattr(verification, "valid", True)
        if not is_valid:
            self._onion_verified = False
            if not self.allow_unverified_onion:
                raise VerificationError("Onion mirror could not be verified via PGP.")

    def search(
        self,
        query: str,
        limit: int = 10,
        sort: int = 99,
        page: int = 0,
    ) -> list[Torrent]:
        errors: list[Exception] = []
        for mirror in self.mirrors:
            try:
                url = _build_url(mirror, query, page, sort)
                html = self._fetch(url)
                rows = self._parse_rows(html, limit)
                return rows[:limit]
            except MirrorError as exc:
                errors.append(exc)
                continue
        raise SearchError(f"All mirrors failed: {[str(e) for e in errors]}")

    @staticmethod
    def _parse_rows(html: str, limit: int) -> list[Torrent]:
        parser = _RowParser(limit)
        parser.feed(html)
        return parser.rows
