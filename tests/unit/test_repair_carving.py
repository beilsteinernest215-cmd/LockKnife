from __future__ import annotations

import pathlib

import pytest

from lockknife.modules.forensics import carving


@pytest.fixture
def small_windows(monkeypatch):
    monkeypatch.setattr(carving, "_CHUNK_SIZE", 16)
    monkeypatch.setattr(
        carving,
        "_SIGNATURES",
        [
            {
                "kind": "pdf",
                "header": b"%PDF-",
                "footer": b"%%EOF",
                "extension": ".pdf",
                "max_size": 64,
            }
        ],
    )


@pytest.mark.parametrize("offset", [12, 14, 15, 16, 29])
def test_cross_boundary_header_body_footer(tmp_path, small_windows, offset):
    payload = b"%PDF-" + b"x" * 39 + b"%%EOF"
    image = tmp_path / "image"
    image.write_bytes(b"\0" * offset + payload + b"\0" * 96)
    result = carving.carve_deleted_files(image, tmp_path / "out")
    assert result["carved_count"] == 1
    assert result["carved"][0]["offset"] == offset
    assert pathlib.Path(result["carved"][0]["path"]).read_bytes() == payload
    assert len(result["sources"]) == 1


def test_lazy_bounded_reads_and_early_limit(tmp_path, small_windows, monkeypatch):
    image = tmp_path / "image"
    image.write_bytes(b"%PDF-x%%EOF" + b"\0" * 1000)
    reads = []
    original_open = pathlib.Path.open

    class Reader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def seek(self, offset):
            return self.handle.seek(offset)

        def read(self, size=-1):
            assert size > 0
            reads.append(size)
            return self.handle.read(size)

    def tracked_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return Reader(handle) if path == image else handle

    monkeypatch.setattr(pathlib.Path, "open", tracked_open)
    result = carving.carve_deleted_files(image, tmp_path / "out", max_matches=1)
    assert result["carved_count"] == 1
    assert reads == [100, 80]


def test_zero_limit_never_opens_input(tmp_path, monkeypatch):
    image = tmp_path / "image"
    image.touch()

    def fail_open(*args, **kwargs):
        pytest.fail("zero matches should not open evidence")

    monkeypatch.setattr(pathlib.Path, "open", fail_open)
    assert carving.carve_deleted_files(image, tmp_path / "out", max_matches=0)["carved"] == []


def test_same_offsets_in_main_wal_journal_are_distinct(tmp_path, small_windows):
    db = tmp_path / "db"
    payload = b"%PDF-x%%EOF"
    header = bytearray(100)
    header[:16] = b"SQLite format 3\0"
    header[16:18] = (512).to_bytes(2, "big")
    offset = 120
    db.write_bytes(header + b"\0" * (offset - 100) + payload)
    journal = db.with_name("db-journal")
    journal.write_bytes(b"\0" * offset + payload)
    wal_header = bytearray(32)
    wal_header[8:12] = (512).to_bytes(4, "big")
    frame = bytearray(24 + 512)
    frame[:4] = (3).to_bytes(4, "big")
    frame[offset - 32 : offset - 32 + len(payload)] = payload
    db.with_name("db-wal").write_bytes(wal_header + frame)
    result = carving.carve_deleted_files(db, tmp_path / "out")
    assert result["carved_count"] == 3
    assert {item["offset"] for item in result["carved"]} == {offset}
    assert {item["source_kind"] for item in result["carved"]} == {
        "main-db",
        "rollback-journal",
        "wal-frame",
    }
    assert result["carved"][-1]["page_number"] == 3


def test_oversize_candidate_is_not_truncated_into_success(tmp_path, small_windows):
    image = tmp_path / "image"
    image.write_bytes(b"%PDF-" + b"x" * 60 + b"%%EOF")
    assert carving.carve_deleted_files(image, tmp_path / "out")["carved"] == []


def test_source_generator_closed_on_output_failure(tmp_path, monkeypatch):
    image = tmp_path / "image"
    image.touch()
    closed = []

    def sources(*args, **kwargs):
        try:
            yield {"origin": str(image), "source_kind": "raw-image", "blob": b"%PDF-x%%EOF"}
            pytest.fail("unexpected second source")
        finally:
            closed.append(True)

    def fail_write(*args, **kwargs):
        raise OSError("full")

    monkeypatch.setattr(carving, "_scan_sources", sources)
    monkeypatch.setattr(pathlib.Path, "write_bytes", fail_write)
    with pytest.raises(OSError, match="full"):
        carving.carve_deleted_files(image, tmp_path / "out")
    assert closed == [True]


@pytest.mark.parametrize("oversize", [0, 1])
def test_large_image_boundary_at_signature_size_limit(tmp_path, oversize):
    limit = 50 * 1024 * 1024
    offset = carving._CHUNK_SIZE * 7 - 2
    image = tmp_path / "large-image"
    # A sparse image over 100 MiB, with a header straddling the 56 MiB boundary.
    with image.open("wb") as handle:
        handle.seek(offset)
        handle.write(b"%PDF-")
        handle.seek(offset + limit + oversize - 5)
        handle.write(b"%%EOF")

    result = carving.carve_deleted_files(image, tmp_path / "out", max_matches=1)

    assert result["sources"] == [{"origin": str(image), "source_kind": "raw-image"}]
    if oversize:
        assert result["carved"] == []
        assert list((tmp_path / "out").iterdir()) == []
    else:
        assert result["carved_count"] == 1
        item = result["carved"][0]
        assert item["offset"] == offset
        assert item["size_bytes"] == limit
        with pathlib.Path(item["path"]).open("rb") as handle:
            assert handle.read(5) == b"%PDF-"
            handle.seek(-5, 2)
            assert handle.read() == b"%%EOF"


def test_no_match_windows_and_wal_frames_do_not_grow_bookkeeping(
    tmp_path, small_windows, monkeypatch
):
    db = tmp_path / "db"
    header = bytearray(100)
    header[:16] = b"SQLite format 3\0"
    header[16:18] = (512).to_bytes(2, "big")
    db.write_bytes(header + b"\0" * 4096)
    db.with_name("db-journal").write_bytes(b"\0" * 4096)
    wal_header = bytearray(32)
    wal_header[8:12] = (512).to_bytes(4, "big")
    db.with_name("db-wal").write_bytes(wal_header + (b"\0" * (24 + 512)) * 1000)
    seen_sizes = []
    original = carving._carve_from_blob

    def tracked_carve(*args, **kwargs):
        result = original(*args, **kwargs)
        seen_sizes.append(len(kwargs["seen_offsets"]))
        return result

    monkeypatch.setattr(carving, "_carve_from_blob", tracked_carve)
    result = carving.carve_deleted_files(db, tmp_path / "out")
    assert result["carved_count"] == 0
    assert len(seen_sizes) > 1500
    assert max(seen_sizes) == 0
    assert len(result["sources"]) == 3
