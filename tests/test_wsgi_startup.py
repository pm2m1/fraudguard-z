import pytest
from src.serve import check_deployment_mode


def test_local_mode_passes_without_secrets(monkeypatch):
    monkeypatch.delenv("FGZ_DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("FGZ_AUTH_PASSWORD", raising=False)
    monkeypatch.delenv("FGZ_PRIVACY_KEY", raising=False)
    assert check_deployment_mode(production=False) is None


def test_dev_server_refuses_production(monkeypatch):
    monkeypatch.setenv("FGZ_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("FGZ_AUTH_PASSWORD", "A" * 20)
    monkeypatch.setenv("FGZ_PRIVACY_KEY", "B" * 40)
    with pytest.raises(RuntimeError, match="development server"):
        check_deployment_mode(production=False)


def test_wsgi_fails_without_secrets(monkeypatch):
    monkeypatch.setenv("FGZ_DEPLOYMENT_MODE", "production")
    monkeypatch.delenv("FGZ_AUTH_PASSWORD", raising=False)
    monkeypatch.delenv("FGZ_PRIVACY_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FGZ_AUTH_PASSWORD"):
        check_deployment_mode(production=True)


def test_wsgi_requires_explicit_database(monkeypatch):
    monkeypatch.setenv("FGZ_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("FGZ_AUTH_PASSWORD", "A" * 20)
    monkeypatch.setenv("FGZ_PRIVACY_KEY", "B" * 40)
    monkeypatch.delenv("FGZ_DB_PATH", raising=False)
    with pytest.raises(RuntimeError, match="FGZ_DB_PATH"):
        check_deployment_mode(production=True)


def test_wsgi_accepts_complete_configuration(monkeypatch):
    monkeypatch.setenv("FGZ_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("FGZ_AUTH_PASSWORD", "A" * 20)
    monkeypatch.setenv("FGZ_PRIVACY_KEY", "B" * 40)
    monkeypatch.setenv("FGZ_DB_PATH", "/tmp/fraudguard-test.sqlite3")
    assert check_deployment_mode(production=True) == "B" * 40


def test_wsgi_rejects_weak_privacy_key(monkeypatch):
    monkeypatch.setenv("FGZ_DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("FGZ_AUTH_PASSWORD", "A" * 20)
    monkeypatch.setenv("FGZ_PRIVACY_KEY", "short")
    monkeypatch.setenv("FGZ_DB_PATH", "/tmp/fraudguard-test.sqlite3")
    with pytest.raises(RuntimeError, match="FGZ_PRIVACY_KEY"):
        check_deployment_mode(production=True)
