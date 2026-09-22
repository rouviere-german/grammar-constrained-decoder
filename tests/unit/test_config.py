from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from grammar_constrained_decoder.core.config import Settings, get_settings

pytestmark = pytest.mark.unit


def test_defaults_fail_closed() -> None:
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == []
    assert settings.cors_allow_origins == []
    assert settings.enable_docs is False
    assert settings.metrics_require_auth is True
    assert settings.host == "127.0.0.1"
    assert settings.jwt_secret is None
    assert settings.max_tokens_cap == 256
    assert settings.compile_timeout_seconds == 5.0


def test_environment_variables_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_API_KEY_SHA256", f"{'a' * 64}, {'b' * 64}")
    monkeypatch.setenv("APP_CORS_ALLOW_ORIGINS", "https://app.example.com,http://localhost:3000")
    monkeypatch.setenv("APP_JWT_SECRET", "not-a-real-secret-but-long-enough-000000")
    monkeypatch.setenv("APP_MAX_TOKENS_CAP", "64")
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == ["a" * 64, "b" * 64]
    assert settings.cors_allow_origins == ["https://app.example.com", "http://localhost:3000"]
    assert settings.jwt_secret is not None
    assert settings.jwt_secret.get_secret_value().startswith("not-a-real-secret")
    assert "not-a-real-secret" not in repr(settings)
    assert settings.max_tokens_cap == 64


def test_empty_environment_values_count_as_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # docker compose and a copied .env.example both produce ``KEY=`` for unset settings.
    monkeypatch.setenv("APP_JWT_SECRET", "")
    monkeypatch.setenv("APP_API_KEY_SHA256", "")
    env_file = tmp_path / ".env"
    env_file.write_text("APP_CORS_ALLOW_ORIGINS=\nAPP_MAX_TOKENS_CAP=\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)
    assert settings.jwt_secret is None
    assert settings.api_key_sha256 == []
    assert settings.cors_allow_origins == []
    assert settings.max_tokens_cap == 256


def test_short_jwt_secrets_are_rejected() -> None:
    with pytest.raises(ValidationError, match="32 bytes"):
        Settings(_env_file=None, jwt_secret="short")


@pytest.mark.parametrize("origins", [["*"], ["https://example.com/"], ["example.com"]])
def test_wildcard_or_malformed_origins_are_rejected(origins: list[str]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_allow_origins=origins)


@pytest.mark.parametrize("digest", ["A" * 64, "abc", "g" * 64])
def test_non_sha256_key_hashes_are_rejected(digest: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_key_sha256=[digest])


def test_env_example_documents_every_setting(repo_root: Path) -> None:
    lines = (repo_root / ".env.example").read_text(encoding="utf-8").splitlines()
    documented = {
        line.split("=", 1)[0].strip() for line in lines if line.strip() and not line.startswith("#")
    }
    expected = {f"APP_{name.upper()}" for name in Settings.model_fields}
    assert documented == expected


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
