import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/component/" in path:
            item.add_marker(pytest.mark.component)
        elif "/e2e/" in path:
            item.add_marker(pytest.mark.e2e)
