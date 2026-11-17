from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def _relative_label(path: Path, root: Path | None) -> str:
    resolved = path.resolve()
    if root is None:
        return str(resolved)
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): _normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(v) for v in value]
    return value


def build_logic_fingerprint(
    files: Sequence[Path],
    *,
    extra: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    file_hashes: dict[str, str] = {}
    for path in files:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Fingerprint input file does not exist: {resolved}")
        file_hashes[_relative_label(resolved, root)] = hashlib.sha256(resolved.read_bytes()).hexdigest()

    payload: dict[str, Any] = {
        "version": 1,
        "files": file_hashes,
    }
    if extra:
        payload["extra"] = _normalize_jsonable(extra)

    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["digest"] = hashlib.sha256(canonical).hexdigest()
    return payload


def resolve_latest_run(output_root: Path, required_relpaths: Sequence[str]) -> Path:
    if not output_root.exists():
        raise FileNotFoundError(f"Run root does not exist: {output_root}")

    candidates: list[tuple[float, Path]] = []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        run_id = child.name
        missing = []
        for rel in required_relpaths:
            target = child / rel.format(run_id=run_id)
            if not target.exists():
                missing.append(str(target))
        if missing:
            continue
        candidates.append((child.stat().st_mtime, child))

    if not candidates:
        required_text = ", ".join(required_relpaths)
        raise FileNotFoundError(
            f"No valid run found under {output_root} with required paths: {required_text}"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
