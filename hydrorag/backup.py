from __future__ import annotations

import hashlib
import json
import sqlite3
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings


BACKUP_VERSION = 1
_BACKED_UP_PATHS = (
    Path("data/documents"),
    Path("data/chroma"),
    Path("data/memory.sqlite3"),
    Path("data/jobs.sqlite3"),
    Path("data/logs"),
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_to_backup(settings: Settings, stage: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for relative in _BACKED_UP_PATHS:
        path = settings.root_dir / relative
        if path.is_file():
            if path.suffix == ".sqlite3":
                snapshot = stage / relative
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                source = sqlite3.connect(path)
                destination = sqlite3.connect(snapshot)
                try:
                    source.backup(destination)
                finally:
                    destination.close()
                    source.close()
                files.append((snapshot, relative.as_posix()))
            else:
                files.append((path, relative.as_posix()))
        elif path.is_dir():
            files.extend(
                (child, child.relative_to(settings.root_dir).as_posix())
                for child in sorted(path.rglob("*"))
                if child.is_file()
            )
    return files


def create_backup(settings: Settings, destination: Path | None = None) -> Path:
    settings.ensure_dirs()
    backup_dir = settings.root_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = Path(destination) if destination else backup_dir / (
        "hydrorag-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".zip"
    )
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hydrorag-backup-") as temporary:
        entries = _files_to_backup(settings, Path(temporary))
        manifest = {
            "format_version": BACKUP_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": [
                {"path": relative, "sha256": _digest(path), "bytes": path.stat().st_size}
                for path, relative in entries
            ],
        }
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for path, relative in entries:
                archive.write(path, relative)
    return target


def restore_backup(settings: Settings, archive_path: Path, confirm: bool = False) -> Path:
    if not confirm:
        raise ValueError("restore requires explicit confirm=True")
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format_version") != BACKUP_VERSION:
            raise ValueError("unsupported HydroRAG backup format")
        members = {str(item["path"]) for item in manifest.get("files", [])}
        for member in members:
            path = Path(member)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe backup path: {member}")
        stage = Path(tempfile.mkdtemp(prefix="hydrorag-restore-", dir=settings.root_dir))
        try:
            archive.extractall(stage, members=members)
            for item in manifest.get("files", []):
                path = stage / item["path"]
                if not path.is_file() or _digest(path) != item["sha256"]:
                    raise ValueError(f"backup checksum failed: {item['path']}")
            for relative in _BACKED_UP_PATHS:
                staged = stage / relative
                if staged.is_dir():
                    for path in staged.rglob("*"):
                        if path.is_file() and path.suffix == ".sqlite3":
                            with sqlite3.connect(path) as connection:
                                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                                    raise ValueError(f"invalid SQLite backup: {path.name}")
                elif staged.is_file() and staged.suffix == ".sqlite3":
                    with sqlite3.connect(staged) as connection:
                        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                            raise ValueError(f"invalid SQLite backup: {relative}")
            safety_backup = create_backup(settings)
            for relative in _BACKED_UP_PATHS:
                target = settings.root_dir / relative
                staged = stage / relative
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                if staged.is_dir():
                    shutil.copytree(staged, target)
                elif staged.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(staged, target)
            return safety_backup
        finally:
            shutil.rmtree(stage, ignore_errors=True)
