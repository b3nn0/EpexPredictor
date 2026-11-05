"""Shared API and configuration names."""

DOMAIN = "epexpredictor"
PUBLIC_URL = "https://epexpredictor.batzill.com"
REGIONS = ["DE", "AT", "BE", "FR", "NL", "SE1", "SE2", "SE3", "SE4", "DK1", "DK2", "ES", "PT"]
DEFAULTS = {"region": "DE", "surcharge": 0.0, "tax_percent": 0.0}


def settings(entry):
    return entry.data | entry.options


def normalize_url(value):
    from urllib.parse import urlsplit

    value = value.strip().rstrip("/")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Expected an HTTP(S) base URL without credentials or query parameters")
    return value
