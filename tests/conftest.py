import os
import sys
import tempfile
from pathlib import Path

# Configure before evalsvc is imported: throwaway SQLite DB, inline task execution, offline judge.
_tmp = tempfile.mkdtemp()
os.environ["EVALSVC_DATABASE_URL"] = f"sqlite:///{Path(_tmp) / 'test.db'}"
os.environ["EVALSVC_SYNC_TASKS"] = "true"
os.environ["EVALSVC_DEFAULT_JUDGE"] = "heuristic"
os.environ.pop("EVALSVC_REDIS_URL", None)
os.environ.pop("EVALSVC_API_KEY", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from evalsvc.api import app

    with TestClient(app) as c:
        yield c
