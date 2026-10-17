# Extracted from scripts/flintstones_klein/archive_io.py; historical orchestration removed.
import json
import os
from pathlib import Path

def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)

import struct
import zipfile
import zlib
CHUNK = 1024 * 1024

def zip_payload(archive: Path, entry: zipfile.ZipInfo):
    if entry.flag_bits & 1:
        raise ValueError("Encrypted entries are unsupported")
    if entry.compress_type not in (0, 8):
        raise ValueError(f"Unsupported compression method: {entry.compress_type}")
    with archive.open("rb") as source:
        source.seek(entry.header_offset)
        local = source.read(30)
        if len(local) != 30 or local[:4] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        name_size, extra_size = struct.unpack_from("<HH", local, 26)
        source.seek(name_size + extra_size, 1)
        remaining = entry.compress_size
        total = 0
        decoder = zlib.decompressobj(-15) if entry.compress_type == 8 else None
        while remaining:
            data = source.read(min(remaining, CHUNK))
            if not data:
                raise EOFError("Truncated ZIP payload")
            remaining -= len(data)
            output = decoder.decompress(data) if decoder else data
            total += len(output)
            if total > entry.file_size:
                raise ValueError("ZIP payload exceeds declared size")
            if output:
                yield output
        if decoder:
            output = decoder.flush()
            total += len(output)
            if output:
                yield output
            if not decoder.eof or decoder.unused_data:
                raise ValueError("Invalid DEFLATE stream boundary")
        if total != entry.file_size:
            raise ValueError(f"ZIP size mismatch for {entry.filename}: {total}")
