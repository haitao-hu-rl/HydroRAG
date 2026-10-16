import pytest

from hydrorag.config import Settings


def test_required_api_token_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRORAG_REQUIRE_API_TOKEN", "true")
    monkeypatch.delenv("HYDRORAG_API_TOKEN", raising=False)

    settings = Settings.from_env(tmp_path)

    with pytest.raises(ValueError, match="requires HYDRORAG_API_TOKEN"):
        settings.validate_auth()
