import json
import zipfile

from hydrorag.backup import create_backup, restore_backup
from hydrorag.config import Settings
from hydrorag.jobs import IndexJobStore
from hydrorag.metrics import Metrics
from hydrorag.rate_limit import RateLimiter


def test_index_jobs_persist_lifecycle(tmp_path):
    jobs = IndexJobStore(tmp_path / "jobs.sqlite3")
    source = tmp_path / "report.pdf"
    source.write_bytes(b"pdf")
    job_id = jobs.create(str(source), source.name)
    assert jobs.get(job_id)["status"] == "pending"
    assert jobs.active_for_path(str(source))
    jobs.mark_running(job_id)
    jobs.complete(job_id, "document-1", 3)
    job = jobs.get(job_id)
    assert job["status"] == "succeeded"
    assert job["chunk_count"] == 3
    assert jobs.counts()["succeeded"] == 1


def test_rate_limiter_and_metrics():
    limiter = RateLimiter()
    assert limiter.allow("user", 1)[0]
    allowed, retry_after = limiter.allow("user", 1)
    assert not allowed and retry_after >= 1

    metrics = Metrics()
    metrics.increment("jobs")
    metrics.observe("latency", 12)
    snapshot = metrics.snapshot()
    assert snapshot["counters"]["jobs"] == 1
    assert snapshot["timers"]["latency"]["average"] == 12


def test_backup_restore_checks_manifest_and_restores_data(tmp_path):
    settings = Settings.from_env(tmp_path)
    settings.ensure_dirs()
    (settings.documents_dir / "note.txt").write_text("内部资料", encoding="utf-8")
    backup_path = create_backup(settings, tmp_path / "snapshot.zip")
    (settings.documents_dir / "note.txt").write_text("changed", encoding="utf-8")

    safety_backup = restore_backup(settings, backup_path, confirm=True)

    assert (settings.documents_dir / "note.txt").read_text(encoding="utf-8") == "内部资料"
    assert safety_backup.is_file()
    with zipfile.ZipFile(backup_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["format_version"] == 1
