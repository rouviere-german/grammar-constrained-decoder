"""Registered schemas with a hash-chained audit trail.

Registration (admin scope only, enforced by the API) compiles the schema first; only a schema that
compiles within the limits is stored. Each registration, each rejected registration and each
integrity failure is appended to a hash-chained audit log (``core/audit.py``) with the actor, the
schema id and the SHA-256 of the canonical schema, never the schema body.

A stored schema file is trusted only if its content hash equals the hash recorded by its audited
registration entry, so a file edited on disk after registration is refused. Registrations are
immutable: registering an existing id with different content is a conflict. Builtin schemas (the
bundled ones) are read-only and cannot be shadowed.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grammar_constrained_decoder.core.audit import AuditLog, iter_entries
from grammar_constrained_decoder.core.canonical import canonical_json, sha256_hex
from grammar_constrained_decoder.core.fileio import atomic_write_text, file_lock

__all__ = [
    "SCHEMA_ID",
    "SchemaConflictError",
    "SchemaIntegrityError",
    "SchemaNotFoundError",
    "SchemaRecord",
    "SchemaRegistry",
]

SCHEMA_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class SchemaNotFoundError(KeyError):
    pass


class SchemaConflictError(ValueError):
    pass


class SchemaIntegrityError(RuntimeError):
    """A stored schema no longer matches its audited registration."""


@dataclass(frozen=True, slots=True)
class SchemaRecord:
    schema_id: str
    schema: dict[str, Any]
    sha256: str
    builtin: bool
    registered_by: str | None = None


class SchemaRegistry:
    def __init__(self, root: Path, *, builtins: Mapping[str, dict[str, Any]] | None = None) -> None:
        self.root = root
        self.schemas_dir = root / "schemas"
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(root / "registry_audit.jsonl", redact=False)
        self._builtins = {
            name: SchemaRecord(name, dict(schema), sha256_hex(canonical_json(schema)), True)
            for name, schema in (builtins or {}).items()
        }
        self._lock = threading.Lock()
        self._registered: dict[str, tuple[str, str]] = {}
        self._reload_index()

    def _reload_index(self) -> None:
        index: dict[str, tuple[str, str]] = {}
        for entry in iter_entries(self.audit.path):
            if entry.action == "schema.register":
                schema_id = str(entry.data.get("schema_id"))
                index[schema_id] = (str(entry.data.get("sha256")), entry.actor)
        self._registered = index

    def verify(self) -> bool:
        """True when the audit chain verifies and every stored schema matches its entry."""
        try:
            self._reload_index()
            for schema_id in self._registered:
                self.get(schema_id)
        except (SchemaIntegrityError, SchemaNotFoundError, RuntimeError, ValueError):
            return False
        return True

    def _path(self, schema_id: str) -> Path:
        if not SCHEMA_ID.fullmatch(schema_id):
            raise SchemaNotFoundError(schema_id)
        return self.schemas_dir / f"{schema_id}.json"

    def get(self, schema_id: str) -> SchemaRecord:
        builtin = self._builtins.get(schema_id)
        if builtin is not None:
            return builtin
        path = self._path(schema_id)
        with self._lock:
            registered = self._registered.get(schema_id)
        if registered is None:
            raise SchemaNotFoundError(schema_id)
        expected, actor = registered
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaIntegrityError(f"schema {schema_id} is unreadable") from exc
        digest = sha256_hex(canonical_json(schema))
        if digest != expected:
            raise SchemaIntegrityError(f"schema {schema_id} does not match its audited hash")
        return SchemaRecord(schema_id, schema, digest, False, actor)

    def list(self) -> list[SchemaRecord]:
        records = list(self._builtins.values())
        with self._lock:
            names = sorted(self._registered)
        for name in names:
            try:
                records.append(self.get(name))
            except SchemaIntegrityError:
                continue
        return sorted(records, key=lambda record: record.schema_id)

    def register(
        self, schema_id: str, schema: dict[str, Any], *, actor: str, stats: Mapping[str, int]
    ) -> SchemaRecord:
        """Store a compiled schema and audit it. Idempotent for identical content."""
        if not SCHEMA_ID.fullmatch(schema_id):
            raise ValueError("invalid schema id")
        if schema_id in self._builtins:
            raise SchemaConflictError("schema id is reserved by a builtin schema")
        digest = sha256_hex(canonical_json(schema))
        path = self._path(schema_id)
        with self._lock, file_lock(path):
            self._reload_index()
            existing = self._registered.get(schema_id)
            if existing is not None:
                if existing[0] != digest:
                    raise SchemaConflictError("schema id is already registered with other content")
                return SchemaRecord(schema_id, schema, digest, False, existing[1])
            atomic_write_text(path, canonical_json(schema) + "\n")
            self.audit.append(
                actor,
                "schema.register",
                {"schema_id": schema_id, "sha256": digest, **{k: int(v) for k, v in stats.items()}},
            )
            self._registered[schema_id] = (digest, actor)
        return SchemaRecord(schema_id, schema, digest, False, actor)

    def record_rejection(
        self, *, actor: str, schema_id: str | None, digest: str, reason: str
    ) -> None:
        self.audit.append(
            actor,
            "schema.reject",
            {
                "schema_id": schema_id if schema_id and SCHEMA_ID.fullmatch(schema_id) else None,
                "sha256": digest,
                "reason": reason,
            },
        )
