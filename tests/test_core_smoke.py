import modelmeld


def test_package_imports() -> None:
    """Smoke test — package imports cleanly + reports a non-empty version."""
    assert isinstance(modelmeld.__version__, str)
    assert modelmeld.__version__.count(".") >= 2  # major.minor.patch shape
