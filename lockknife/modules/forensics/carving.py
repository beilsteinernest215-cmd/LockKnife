from __future__ import annotations

import pathlib
from collections.abc import Generator, Iterator
from typing import Any, TypedDict

from lockknife.modules.forensics.recovery import _sqlite_page_size

_CHUNK_SIZE = 8 * 1024 * 1024


class _CarveSignature(TypedDict):
    kind: str
    header: bytes
    footer: bytes
    extension: str
    max_size: int


_SIGNATURES: list[_CarveSignature] = [
    {
        "kind": "jpg",
        "header": b"\xff\xd8\xff",
        "footer": b"\xff\xd9",
        "extension": ".jpg",
        "max_size": 25 * 1024 * 1024,
    },
    {
        "kind": "png",
        "header": b"\x89PNG\r\n\x1a\n",
        "footer": b"IEND\xaeB`\x82",
        "extension": ".png",
        "max_size": 25 * 1024 * 1024,
    },
    {
        "kind": "pdf",
        "header": b"%PDF-",
        "footer": b"%%EOF",
        "extension": ".pdf",
        "max_size": 50 * 1024 * 1024,
    },
    {
        "kind": "zip",
        "header": b"PK\x03\x04",
        "footer": b"PK\x05\x06",
        "extension": ".zip",
        "max_size": 50 * 1024 * 1024,
    },
]


def carve_deleted_files(
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    source: str = "auto",
    max_matches: int = 50,
) -> dict[str, Any]:
    """Carve complete signature candidates within their configured size limits.

    ``sources`` summarizes visited physical files by origin and source kind, not
    every scan window or SQLite page. Match offsets remain physical file offsets.
    Oversized candidates are skipped, never emitted as truncated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        return {
            "input": str(input_path),
            "source": source,
            "output_dir": str(output_dir),
            "carved_count": 0,
            "sources": [],
            "carved": [],
            "error": f"Input path does not exist: {input_path}",
        }

    carved: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, str]] = set()
    counter = 0
    seen_offsets: set[tuple[str, int]] = set()
    scan_sources = _scan_sources(input_path, source=source)
    try:
        for source_entry in scan_sources if max_matches > 0 else ():
            source_key = (source_entry["origin"], source_entry["source_kind"])
            if source_key not in seen_sources:
                sources.append({"origin": source_key[0], "source_kind": source_key[1]})
                seen_sources.add(source_key)
            batch, counter = _carve_from_blob(
                source_entry,
                output_dir=output_dir,
                max_matches=max_matches - len(carved),
                counter=counter,
                seen_offsets=seen_offsets,
            )
            carved.extend(batch)
            if len(carved) >= max_matches:
                break
    finally:
        scan_sources.close()
    return {
        "input": str(input_path),
        "source": source,
        "output_dir": str(output_dir),
        "carved_count": len(carved),
        "sources": sources,
        "carved": carved,
    }


def _scan_sources(
    input_path: pathlib.Path, *, source: str
) -> Generator[dict[str, Any], None, None]:
    if not input_path.exists():
        return
    with input_path.open("rb") as f:
        header = f.read(100)

    is_sqlite = header.startswith(b"SQLite format 3\x00")
    if source == "sqlite" or (source == "auto" and is_sqlite):
        # The main scan already covers freelist and overflow pages; rescanning them
        # would duplicate physical evidence and require retaining the database.
        yield from _file_windows(input_path, "main-db")
        journal = input_path.with_name(input_path.name + "-journal")
        if journal.exists():
            yield from _file_windows(journal, "rollback-journal")
        wal = input_path.with_name(input_path.name + "-wal")
        if wal.exists():
            with wal.open("rb") as handle:
                wal_header = handle.read(32)
                if len(wal_header) < 32:
                    return
                page_size = _sqlite_page_size(header) or int.from_bytes(wal_header[8:12], "big")
                if not 512 <= page_size <= 65536 or page_size & (page_size - 1):
                    return
                frame_index = 0
                while frame_header := handle.read(24):
                    if len(frame_header) < 24:
                        break
                    offset = handle.tell()
                    blob = handle.read(page_size)
                    if len(blob) < page_size:
                        break
                    yield {
                        "source_kind": "wal-frame",
                        "origin": str(wal),
                        "page_number": int.from_bytes(frame_header[:4], "big"),
                        "frame_index": frame_index,
                        "offset": offset,
                        "blob": blob,
                    }
                    frame_index += 1
        return
    yield from _file_windows(input_path, "raw-image")


def _file_windows(path: pathlib.Path, source_kind: str) -> Iterator[dict[str, Any]]:
    # Every header in the primary chunk has enough lookahead for a complete
    # maximum-sized object. No evidence bytes survive beyond the current window.
    overlap = max(signature["max_size"] for signature in _SIGNATURES)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        offset = 0
        while offset < file_size:
            f.seek(offset)
            data = f.read(_CHUNK_SIZE + overlap)
            if not data:
                break
            yield {
                "source_kind": source_kind,
                "origin": str(path),
                "offset": offset,
                "scan_length": min(_CHUNK_SIZE, len(data)),
                "blob": data,
            }
            offset += _CHUNK_SIZE


def _carve_from_blob(
    source_entry: dict[str, Any],
    *,
    output_dir: pathlib.Path,
    max_matches: int,
    counter: int,
    seen_offsets: set[tuple[str, int]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    blob = bytes(source_entry.get("blob") or b"")
    if max_matches <= 0:
        return [], counter
    if seen_offsets is None:
        seen_offsets = set()
    out: list[dict[str, Any]] = []
    for signature in _SIGNATURES:
        start = 0
        while len(out) < max_matches:
            index = blob.find(signature["header"], start)
            if index < 0 or index >= source_entry.get("scan_length", len(blob)):
                break
            end_index = blob.find(
                signature["footer"],
                index + len(signature["header"]),
                index + signature["max_size"],
            )
            if end_index < 0:
                start = index + len(signature["header"])
                continue
            end = end_index + len(signature["footer"])
            abs_offset = int(source_entry.get("offset") or 0) + index
            key = (str(source_entry.get("origin") or ""), abs_offset)
            if key in seen_offsets:
                start = end
                continue
            seen_offsets.add(key)
            carved_bytes = blob[index:end]
            file_name = f"carved_{counter:03d}_{signature['kind']}{signature['extension']}"
            path = output_dir / file_name
            path.write_bytes(carved_bytes)
            out.append(
                {
                    "kind": signature["kind"],
                    "path": str(path),
                    "size_bytes": len(carved_bytes),
                    "source_kind": source_entry.get("source_kind"),
                    "origin": source_entry.get("origin"),
                    "offset": abs_offset,
                    "page_number": source_entry.get("page_number"),
                }
            )
            start = end
            counter += 1
    return out, counter
