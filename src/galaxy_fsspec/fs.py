"""The Galaxy fsspec filesystem."""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Iterable
from typing import Any, Literal

import fsspec
import requests
from fsspec.spec import AbstractFileSystem

from galaxy_fsspec.client import build_galaxy_instance, show_hid_in_names_from_env
from galaxy_fsspec.exceptions import NotFoundError, ReadOnlyError
from galaxy_fsspec.file import GalaxyFile
from galaxy_fsspec.paths import (
    dedupe_names,
    name_with_prefix,
    sanitize_segment,
)

ROOT = ""
HISTORIES_DIR = "histories"
LIBRARIES_DIR = "libraries"
_CACHE_TTL = 60.0  # seconds


class GalaxyFileSystem(AbstractFileSystem):
    """Read-only fsspec filesystem exposing a Galaxy account.

    Layout::

        galaxy://
        ├── histories/
        │   └── <history>/
        │       ├── <dataset>
        │       └── <collection>/   (list, paired, nested ...)
        │           └── ...
        └── libraries/
            └── <library>/
                ├── <dataset>
                └── <folder>/
                    └── ...

    History folders expose ``created`` and ``last_modified`` timestamps via
    :meth:`info`. Set ``show_hid_in_names=True`` (or ``GALAXY_FSSPEC_SHOW_HID_IN_NAMES=true``)
    to prefix every entry with its Galaxy ``hid`` in the style ``1-my-dataset``.
    """

    protocol = "galaxy"
    root_marker = ""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        show_hid_in_names: bool | None = None,
        cache_ttl: float = _CACHE_TTL,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.gi = build_galaxy_instance(url=url, api_key=api_key)
        self._url: str = str(url or self.gi.base_url)
        self._key: str = str(api_key or self.gi.key)
        self.show_hid_in_names: bool = (
            show_hid_in_names if show_hid_in_names is not None else show_hid_in_names_from_env()
        )
        self._cache_ttl = float(cache_ttl)
        # path -> (entries, timestamp)  for directory listings
        self._dir_cache: dict[str, tuple[list[dict], float]] = {}
        # path -> info dict
        self._info_cache: dict[str, dict] = {}
        # library_id -> (flat_contents, timestamp) — avoids re-fetching the
        # full library tree on every folder listing within the same library.
        self._library_flat_cache: dict[str, tuple[list[dict], float]] = {}

    # ------------------------------------------------------------------ #
    # Public fsspec API
    # ------------------------------------------------------------------ #

    def ls(self, path: str, detail: bool = False, **kwargs: Any) -> list:
        entries = self._ls(path)
        if detail:
            return entries
        return [e["name"] for e in entries]

    def _ls(self, path: str) -> list[dict]:
        path = self._strip(path)
        cached = self._cached_dir(path)
        if cached is not None:
            return cached
        entries = self._list(path)
        self._dir_cache[path] = (entries, time.time())
        return entries

    def info(self, path: str, **kwargs: Any) -> dict:
        return self._info(self._strip_protocol(path), **kwargs)

    def _info(self, path: str, **kwargs: Any) -> dict:
        path = self._strip(path)
        if path in self._info_cache:
            return self._info_cache[path]
        if path == ROOT:
            return {"name": ROOT, "size": 0, "type": "directory"}
        parent, _, leaf = path.rpartition("/")
        if leaf == HISTORIES_DIR and parent == ROOT:
            return {"name": HISTORIES_DIR, "size": 0, "type": "directory"}
        if leaf == LIBRARIES_DIR and parent == ROOT:
            return {"name": LIBRARIES_DIR, "size": 0, "type": "directory"}
        # Find this entry within its parent listing.
        try:
            parent_entries = self._ls(parent or ROOT)
        except NotFoundError as exc:
            raise NotFoundError(path) from exc
        for entry in parent_entries:
            if entry["name"] == path:
                # Enrich history folders with timestamps on demand.
                if parent == HISTORIES_DIR and entry.get("created") is None:
                    hist = self.gi.histories.show_history(entry["history_id"], contents=False)
                    entry["created"] = hist.get("create_time")
                    entry["last_modified"] = hist.get("update_time")
                return entry
        raise NotFoundError(path)

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,
        cache_type: str = "bytes",
        **kwargs: Any,
    ) -> GalaxyFile:
        if mode not in ("rb", "r"):
            raise ReadOnlyError(f"galaxy-fsspec is read-only; cannot open {mode!r}")
        # The size found below is cached under this path, so it must be the stripped path the
        # reads use, or a path starting with "/" (as Galaxy asks) reads as an empty file.
        path = self._strip(path)
        info = self._info(path)
        if info["type"] != "file":
            raise IsADirectoryError(path)
        # Dataset leaves inside collections and library folders are listed with
        # size 0 because the listing endpoints don't report file_size. Fetch the
        # real size here so AbstractBufferedFile.read() actually returns bytes.
        if info.get("size", 0) == 0:
            if "library_dataset_id" in info:
                ldda_id, size = self._library_dataset_details(
                    info["library_id"], info["library_dataset_id"]
                )
                info = {**info, "size": size, "ldda_id": ldda_id}
            else:
                size, dl_url = self._dataset_details(info.get("dataset_id"))
                info = {**info, "size": size, "download_url": dl_url}
            self._info_cache[path] = info
        return GalaxyFile(
            self,
            path,
            mode="rb",
            block_size=block_size or (8 << 20),
            cache_type=cache_type,
            **kwargs,
        )

    def _dataset_details(
        self, dataset_id: str | None, hda_ldda: Literal["hda", "ldda"] = "hda"
    ) -> tuple[int, str | None]:
        """Return ``(file_size, download_url)`` for a dataset via the datasets API."""
        if not dataset_id:
            return 0, None
        try:
            details = self.gi.datasets.show_dataset(dataset_id, hda_ldda=hda_ldda)
        except Exception:
            return 0, None
        if not isinstance(details, dict):
            return 0, None
        size = _to_int(details.get("file_size")) or 0
        dl_url = details.get("download_url")
        return size, dl_url

    def _library_dataset_details(self, library_id: str, dataset_id: str) -> tuple[str, int]:
        """Return ``(ldda_id, file_size)`` for a library dataset.

        A library listing gives LibraryDataset ids, but the bytes live under an LDDA. Decoding one
        as the other finds a different dataset instead of failing; only this endpoint maps them.
        """
        details = self.gi.libraries.show_dataset(library_id, dataset_id)
        return details["ldda_id"], _to_int(details.get("file_size")) or 0

    def _fetch_dataset_range(self, path: str, start: int, end: int) -> bytes:
        info = self._info(path)
        dl_url = info.get("download_url")
        if dl_url:
            return self._download_range_from_url(dl_url, start, end)
        if "ldda_id" in info:
            return self._download_range(
                info["ldda_id"], start, end, hda_ldda="ldda"
            )
        return self._download_range(info["dataset_id"], start, end)

    # Read-only enforcement -------------------------------------------------
    def _rm(self, path):
        raise ReadOnlyError("galaxy-fsspec is read-only")

    def rm_file(self, path):
        raise ReadOnlyError("galaxy-fsspec is read-only")

    def rm(self, path, recursive=False, maxdepth=None):
        raise ReadOnlyError("galaxy-fsspec is read-only")

    def mkdir(self, path, create_parents=True, **kwargs):
        raise ReadOnlyError("galaxy-fsspec is read-only")

    def makedirs(self, path, exist_ok=False):
        raise ReadOnlyError("galaxy-fsspec is read-only")

    def _mkdir(self, path, **kwargs):
        raise ReadOnlyError("galaxy-fsspec is read-only")

    def pipe_file(self, path, value, **kwargs):
        raise ReadOnlyError("galaxy-fsspec is read-only")

    def _pipe_file(self, path, value, **kwargs):
        raise ReadOnlyError("galaxy-fsspec is read-only")

    def touch(self, path, **kwargs):
        raise ReadOnlyError("galaxy-fsspec is read-only")

    # ------------------------------------------------------------------ #
    # Path resolution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip(path: str) -> str:
        if not path:
            return ROOT
        # Drop protocol prefix if present.
        if "://" in path:
            path = path.split("://", 1)[1]
        path = path.strip("/")
        return path

    def _cached_dir(self, path: str) -> list[dict] | None:
        if path in self._dir_cache:
            entries, ts = self._dir_cache[path]
            if time.time() - ts < self._cache_ttl:
                return entries
            del self._dir_cache[path]
        return None

    def _clear_cache(self) -> None:
        self._dir_cache.clear()
        self._info_cache.clear()
        self._library_flat_cache.clear()

    def _list(self, path: str) -> list[dict]:
        if path == ROOT:
            return [
                {
                    "name": HISTORIES_DIR,
                    "size": 0,
                    "type": "directory",
                },
                {
                    "name": LIBRARIES_DIR,
                    "size": 0,
                    "type": "directory",
                },
            ]
        segments = path.split("/")
        head = segments[0]
        if head == HISTORIES_DIR:
            rest = segments[1:]
            if not rest:
                return self._list_histories()
            history = self._resolve_history(rest[0])
            if len(rest) == 1:
                return self._list_history_contents(history, path)
            return self._list_collection_path(history, rest[1:], path)
        if head == LIBRARIES_DIR:
            rest = segments[1:]
            if not rest:
                return self._list_libraries()
            library = self._resolve_library(rest[0])
            if len(rest) == 1:
                return self._list_library_contents(library, path)
            return self._list_library_path(library, rest[1:], path)
        raise NotFoundError(path)

    # ------------------------------------------------------------------ #
    # Histories
    # ------------------------------------------------------------------ #

    def _list_histories(self) -> list[dict]:
        raw = self.gi.histories.get_histories()
        named = dedupe_names(raw, numbered=self.show_hid_in_names)
        entries: list[dict] = []
        for display, h in named:
            entries.append(
                {
                    "name": f"{HISTORIES_DIR}/{display}",
                    "size": 0,
                    "type": "directory",
                    "history_id": h["id"],
                    "hid": None,
                    "created": h.get("create_time"),
                    "last_modified": h.get("update_time"),
                }
            )
        return entries

    def _resolve_history(self, segment: str) -> dict:
        histories = self.gi.histories.get_histories()
        for h in histories:
            disp = name_with_prefix(None, h.get("name") or h["id"], self.show_hid_in_names)
            # Histories have no hid; numbered prefix not applied, so compare by full name.
            if segment == disp:
                return h
            # Fall back to matching by raw name or id.
            if segment == sanitize_segment(h.get("name") or "") or segment == h["id"]:
                return h
        # Allow lookup by raw id.
        for h in histories:
            if h["id"] == segment:
                return h
        raise NotFoundError(f"histories/{segment}")

    # ------------------------------------------------------------------ #
    # Libraries
    # ------------------------------------------------------------------ #

    def _list_libraries(self) -> list[dict]:
        raw = self.gi.libraries.get_libraries()
        named = dedupe_names(raw, numbered=False)
        entries: list[dict] = []
        for display, lib in named:
            entries.append(
                {
                    "name": f"{LIBRARIES_DIR}/{display}",
                    "size": 0,
                    "type": "directory",
                    "library_id": lib["id"],
                }
            )
        return entries

    def _resolve_library(self, segment: str) -> dict:
        libraries = self.gi.libraries.get_libraries()
        for lib in libraries:
            disp = name_with_prefix(None, lib.get("name") or lib["id"], False)
            if segment == disp:
                return lib
            if segment == sanitize_segment(lib.get("name") or "") or segment == lib["id"]:
                return lib
        for lib in libraries:
            if lib["id"] == segment:
                return lib
        raise NotFoundError(f"libraries/{segment}")

    def _list_library_contents(self, library: dict, path: str) -> list[dict]:
        """List the root folder of a library."""
        return self._list_library_path(library, [], path)

    def _library_flat_contents(self, library_id: str) -> list[dict]:
        """Return the flat contents of a library, cached with TTL."""
        cached = self._library_flat_cache.get(library_id)
        if cached is not None and time.time() - cached[1] < self._cache_ttl:
            return cached[0]
        flat = self.gi.libraries.show_library(library_id, contents=True)
        self._library_flat_cache[library_id] = (flat, time.time())
        return flat

    def _list_library_path(
        self, library: dict, segments: list[str], path: str
    ) -> list[dict]:
        """List contents at a path within a library.

        Galaxy's ``show_library(contents=True)`` returns a flat list of every
        item with full paths (e.g. ``/folder/dataset.ext``).  We cache that
        flat list per library so browsing nested folders doesn't re-fetch it.
        """
        flat = self._library_flat_contents(library["id"])
        # Build the Galaxy-side prefix.  Root is "/", a sub-folder is "/seg1/seg2".
        prefix = "/" + "/".join(segments) if segments else "/"
        entries: list[dict] = []
        for item in flat:
            name = item.get("name", "")
            if name == "/" or not name.startswith(prefix):
                continue
            # Remainder after the prefix.
            remainder = name[1:] if prefix == "/" else name[len(prefix) + 1 :]
            if not remainder or "/" in remainder:
                continue  # skip self and nested descendants
            is_folder = item.get("type") == "folder"
            entry: dict = {
                "name": f"{path}/{remainder}",
                "type": "directory" if is_folder else "file",
            }
            if is_folder:
                entry["size"] = 0
                entry["library_folder_id"] = item["id"]
            else:
                entry["size"] = 0
                entry["library_dataset_id"] = item["id"]
                entry["library_id"] = library["id"]
            entries.append(entry)
        return entries

    # ------------------------------------------------------------------ #
    # History contents
    # ------------------------------------------------------------------ #

    def _history_contents(self, history_id: str) -> list[dict]:
        # What the history panel shows. Without the filters Galaxy also returns deleted datasets and
        # the hidden copies it makes of every file put into a collection; details brings file_size.
        return self.gi.histories.show_history(
            history_id, contents=True, deleted=False, visible=True, details="all"
        )

    def _list_history_contents(self, history: dict, path: str) -> list[dict]:
        hid = history["id"]
        contents = self._history_contents(hid)
        return self._contents_to_entries(contents, path, history_id=hid)

    def _contents_to_entries(
        self, contents: Iterable[dict], parent_path: str, history_id: str
    ) -> list[dict]:
        named = dedupe_names(list(contents), numbered=self.show_hid_in_names)
        entries: list[dict] = []
        for display, item in named:
            is_collection = item.get("history_content_type") == "dataset_collection"
            entry: dict = {
                "name": f"{parent_path}/{display}",
                "type": "directory" if is_collection else "file",
                "hid": _to_int(item.get("hid")),
            }
            if is_collection:
                entry["size"] = 0
                entry["collection_id"] = item["id"]
                entry["collection_type"] = item.get("collection_type")
            else:
                entry["size"] = _to_int(item.get("file_size") or item.get("size")) or 0
                entry["dataset_id"] = item.get("id")
            entries.append(entry)
        return entries

    # ------------------------------------------------------------------ #
    # Collection navigation
    # ------------------------------------------------------------------ #

    def _list_collection_path(self, history: dict, segments: list[str], path: str) -> list[dict]:
        """List the directory at ``path``, which lies inside at least one collection.

        ``segments`` is everything below the history folder; ``segments[0]`` is a
        top-level collection, any later segments descend into nested collections.
        """
        contents = self._history_contents(history["id"])
        current = self._resolve_in_contents(contents, segments[0], history["id"])
        if not current.get("_is_collection"):
            # A top-level dataset has no children.
            raise NotFoundError(path)
        # Walk intermediate segments through nested collections.
        for seg in segments[1:]:
            elements = self._collection_elements(current["id"])
            current = self._resolve_in_elements(elements, seg, current["id"])
            if not current.get("_is_collection"):
                # Landed on a dataset leaf; no further descent is possible.
                raise NotFoundError(path)
        # ``current`` is the final collection; list its elements.
        elements = self._collection_elements(current["id"])
        return self._elements_to_entries(elements, path, history["id"])

    def _resolve_in_contents(self, contents: list[dict], segment: str, history_id: str) -> dict:
        named = dedupe_names(contents, numbered=self.show_hid_in_names)
        for display, item in named:
            disp_name = display.rsplit("/", 1)[-1]
            if segment == disp_name:
                if item.get("history_content_type") == "dataset_collection":
                    return {"id": item["id"], "_is_collection": True}
                return {"id": item["id"], "_is_collection": False}
        raise NotFoundError(segment)

    def _resolve_in_elements(
        self, elements: list[dict], segment: str, parent_collection_id: str
    ) -> dict:
        for display, original, _hid in self._name_elements(elements):
            disp_name = display.rsplit("/", 1)[-1]
            if segment != disp_name:
                continue
            inner = _element_inner(original)
            if original.get("element_type") == "dataset_collection":
                return {"id": inner.get("id"), "_is_collection": True}
            return {"id": inner.get("id"), "_is_collection": False}
        raise NotFoundError(segment)

    def _collection_elements(self, collection_id: str) -> list[dict]:
        details = self.gi.dataset_collections.show_dataset_collection(collection_id)
        return list(details.get("elements") or [])

    def _name_elements(self, elements: list[dict]) -> list[tuple[str, dict, int | None]]:
        """Return ``(display_name, original_element, hid)`` tuples.

        ``hid`` is the 1-based element index (when numbered naming is active),
        used purely so ``dedupe_names`` can build the ``<n>-`` prefix.
        """
        normalized: list[dict] = []
        for el in elements:
            inner = _element_inner(el)
            idx = el.get("element_index")
            normalized.append(
                {
                    "id": inner.get("id") or el.get("element_id"),
                    "name": el.get("element_identifier") or inner.get("name") or "",
                    "hid": (idx + 1) if isinstance(idx, int) else None,
                }
            )
        deduped = dedupe_names(normalized, numbered=self.show_hid_in_names)
        # Re-pair display names with the *original* elements (preserve order).
        return [
            (display, elements[i], normalized[i].get("hid"))
            for i, (display, _item) in enumerate(deduped)
        ]

    def _elements_to_entries(
        self, elements: list[dict], parent_path: str, history_id: str
    ) -> list[dict]:
        entries: list[dict] = []
        for display, original, hid in self._name_elements(elements):
            inner = _element_inner(original)
            is_collection = original.get("element_type") == "dataset_collection"
            entry: dict = {
                "name": f"{parent_path}/{display}",
                "type": "directory" if is_collection else "file",
                "hid": hid,
            }
            if is_collection:
                entry["size"] = 0
                entry["collection_id"] = inner.get("id")
                entry["collection_type"] = (inner.get("object") or {}).get("collection_type")
            else:
                entry["size"] = _to_int(inner.get("file_size") or inner.get("size")) or 0
                entry["dataset_id"] = inner.get("id")
            entries.append(entry)
        return entries

    # ------------------------------------------------------------------ #
    # Dataset download (range-aware, with full-download fallback)
    # ------------------------------------------------------------------ #

    def _download_range(
        self, dataset_id: str, start: int, end: int, hda_ldda: str = "hda"
    ) -> bytes:
        """Fetch bytes ``[start, end)`` from a dataset without buffering the
        full response in memory.

        Uses HTTP streaming: for partial-content responses (206) we read only
        the requested chunk; if the server ignores the Range header (200) we
        stream-discard the first ``start`` bytes then read the needed slice.
        """
        if end <= start:
            return b""
        # raw asks for the stored file; rendering it fails for a library dataset, which has no hid.
        url = f"{self._url}/api/datasets/{urllib.parse.quote(dataset_id)}/display?raw=true"
        if hda_ldda != "hda":
            url += f"&hda_ldda={hda_ldda}"
        return self._download_from_url(url, start, end, dataset_id)

    def _download_range_from_url(
        self, download_url: str, start: int, end: int
    ) -> bytes:
        """Download a range using a ``download_url`` from the datasets API.

        The ``download_url`` is a relative path like
        ``/api/datasets/{id}/display?to_ext=txt``.  We strip the ``to_ext``
        query param (which forces a full download with content-type headers)
        and use the base URL with Range headers instead.
        """
        if end <= start:
            return b""
        url = download_url
        if url.startswith("/"):
            url = f"{self._url}{url}"
        # Remove to_ext param so we get raw bytes, not a forced download.
        if "?to_ext=" in url:
            url = url.split("?to_ext=")[0]
        return self._download_from_url(url, start, end, url)

    def _download_from_url(
        self, url: str, start: int, end: int, label: str
    ) -> bytes:
        length = end - start
        headers = {"x-api-key": self._key, "Range": f"bytes={start}-{end - 1}"}
        resp = requests.get(url, headers=headers, timeout=60, stream=True)
        if resp.status_code == 206:
            return _read_stream(resp, length)
        if resp.status_code == 200:
            return _skip_then_read_stream(resp, start, length)
        if resp.status_code in (401, 403):
            raise ReadOnlyError(f"Galaxy refused dataset access: {resp.status_code}")
        raise NotFoundError(f"dataset {label} (HTTP {resp.status_code})")


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _element_inner(element: dict) -> dict:
    """Return the inner object describing a collection element.

    Galaxy's ``show_dataset_collection`` returns each element with the nested
    dataset/collection under the ``"object"`` key (see bioblend's own tests:
    ``element["object"]["id"]``). Some older payloads used ``"element"``;
    accept both for robustness.
    """
    inner = element.get("object")
    if inner is None:
        inner = element.get("element") or {}
    return inner if isinstance(inner, dict) else {}


def _read_stream(resp: requests.Response, length: int) -> bytes:
    """Read exactly ``length`` bytes from a streaming response in chunks."""
    chunks: list[bytes] = []
    remaining = length
    for chunk in resp.iter_content(chunk_size=64 << 10):
        if not chunk:
            continue
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _skip_then_read_stream(resp: requests.Response, skip: int, length: int) -> bytes:
    """Stream-discard ``skip`` bytes, then read ``length`` bytes."""
    chunks: list[bytes] = []
    remaining_skip = skip
    remaining_read = length
    for chunk in resp.iter_content(chunk_size=64 << 10):
        if not chunk:
            continue
        offset = 0
        if remaining_skip > 0:
            if len(chunk) <= remaining_skip:
                remaining_skip -= len(chunk)
                continue
            offset = remaining_skip
            remaining_skip = 0
        if remaining_read <= 0:
            break
        take = min(len(chunk) - offset, remaining_read)
        chunks.append(chunk[offset : offset + take])
        remaining_read -= take
    return b"".join(chunks)


# Register the protocol entry point at import time as well, so direct imports
# work even when the package metadata has not been loaded.
fsspec.register_implementation("galaxy", GalaxyFileSystem, clobber=True)
