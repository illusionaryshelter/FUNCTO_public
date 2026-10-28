"""Lossless JSON files for shared completed evidence.

The ordinary report remains JSON. Each CompletedEvidence object is written
once to evidence/<record_id>.json; occurrences use a relative JSON Reference.
Reading reconstructs the shared immutable record, including diagnostic values.
Different records are never merged by content or by a numeric tolerance.
"""

import json
import os
from pathlib import Path

from .completed_evidence import CompletedEvidence


def externalize_evidence(document, directory, *, write_record):
    """Replace shared immutable records with local file references.

    ``write_record(path, record)`` is the caller's existing JSON value writer;
    it defines diagnostic NaN/Inf encoding and complete-file replacement.
    Ordinary containers are traversed, not deeply copied before serialization.
    """
    directory = Path(directory)

    def visit(value):
        if isinstance(value, CompletedEvidence):
            # BEGIN one record remains one file across planning and execution archives.
            path = getattr(value, "archive_path", None)
            if path is None:
                path = (directory / "evidence" / f"{value.record_id}.json").resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                write_record(path, value)
                value.archive_path = path
            return {"$ref": Path(os.path.relpath(path, directory)).as_posix()}
            # END changing a report header never duplicates completed numerical evidence.
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [visit(item) for item in value]
        return value

    return visit(document)


def read_evidence_json(path, *, max_bytes=None):
    """Read one document and each referenced evidence file exactly once.

    The optional existing archive limit counts the main file and every unique
    referenced file before decoding. References are local relative files and
    carry a complete record, so no network or fragment-resolution API is used.
    """
    path = Path(path)
    records, total = {}, 0

    def read_file(source):
        nonlocal total
        total += source.stat().st_size
        if max_bytes is not None and total > max_bytes:
            raise ValueError("Complete evidence archive exceeds its declared byte limit")
        return json.loads(source.read_text(encoding="utf-8"))

    def visit(value, directory):
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                source = (directory / value["$ref"]).resolve()
                if source not in records:
                    records[source] = CompletedEvidence(read_file(source), record_id=source.stem)
                    records[source].archive_path = source
                return records[source]
            return {key: visit(item, directory) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, directory) for item in value]
        return value

    return visit(read_file(path), path.parent)


def write_evidence_json(path, document, *, convert):
    """Keep the caller's JSON numeric policy and share its completed records."""
    path = Path(path)

    def write_record(target, value):
        target.write_text(json.dumps(convert(value), indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")

    write_record(path, externalize_evidence(document, path.parent, write_record=write_record))


def referenced_evidence_bytes(document, directory):
    """Count each complete referenced file once for an existing archive limit."""
    paths, pending = set(), [document]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                paths.add((Path(directory) / value["$ref"]).resolve())
            else:
                pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
    return sum(path.stat().st_size for path in paths)
