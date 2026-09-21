from types import SimpleNamespace

import triton.backends as triton_backends


class _EntryPoints:

    def __init__(self, entries):
        self.entries = entries

    def select(self, group):
        assert group == "triton.backends"
        return self.entries


def test_discover_backends_skips_missing_backend_package(monkeypatch):
    missing_backend = SimpleNamespace(name="missing", value="missing_backend")

    monkeypatch.setattr(
        triton_backends,
        "entry_points",
        lambda: _EntryPoints([missing_backend]),
    )

    assert triton_backends._discover_backends() == {}
