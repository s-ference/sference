from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema


def _repo_root() -> Path:
    # .../sdk-python/tests/test_contract.py -> workspace root at parents[2]
    return Path(__file__).resolve().parents[2]


def _load_openapi() -> dict[str, Any]:
    path = _repo_root() / "contract" / "openapi.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _build_operation_index(openapi: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paths = openapi.get("paths") or {}
    out: dict[str, dict[str, Any]] = {}
    for _path, spec in paths.items():
        if not isinstance(spec, dict):
            continue
        for _method, op in spec.items():
            if not isinstance(op, dict):
                continue
            operation_id = op.get("operationId")
            if isinstance(operation_id, str):
                out[operation_id] = op
    return out


def _deref_schema(openapi: dict[str, Any], node: Any) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            name = ref.split("/")[-1]
            target = (openapi.get("components") or {}).get("schemas", {}).get(name)
            if target is None:
                raise KeyError(f"Unknown schema ref: {ref}")
            return _deref_schema(openapi, target)
        return {k: _deref_schema(openapi, v) for k, v in node.items()}
    if isinstance(node, list):
        return [_deref_schema(openapi, v) for v in node]
    return node


def _schema_for(openapi: dict[str, Any], operation_id: str, status_code: str) -> dict[str, Any]:
    op_index = _build_operation_index(openapi)
    op = op_index.get(operation_id)
    if op is None:
        raise KeyError(f"Unknown operationId: {operation_id}")
    responses = op.get("responses") or {}
    resp = responses.get(status_code)
    if resp is None:
        raise KeyError(f"operationId {operation_id} has no response {status_code}")
    content = (resp.get("content") or {}).get("application/json") or {}
    schema = content.get("schema")
    if schema is None:
        raise KeyError(f"operationId {operation_id} response {status_code} has no application/json schema")
    resolved = _deref_schema(openapi, schema)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved schema is not an object")
    return resolved


def test_mock_fixtures_validate_against_openapi_contract() -> None:
    openapi = _load_openapi()
    fixtures_root = Path(__file__).resolve().parent / "fixtures"
    if not fixtures_root.exists():
        # No API-mocking fixtures owned by this suite yet.
        return

    fixture_files = sorted(fixtures_root.rglob("*.json"))
    for f in fixture_files:
        rel = f.relative_to(fixtures_root)
        if len(rel.parts) != 2:
            raise AssertionError(f"Fixture must be fixtures/{{operationId}}/{{statusCode}}.json, got {rel.as_posix()}")
        operation_id = rel.parts[0]
        status_code = rel.stem
        instance = json.loads(f.read_text(encoding="utf-8"))
        schema = _schema_for(openapi, operation_id, status_code)
        jsonschema.validate(instance=instance, schema=schema)

