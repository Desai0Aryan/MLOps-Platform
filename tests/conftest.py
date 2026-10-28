import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlops.config import load_config


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    c = load_config()
    c["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"
    c["data"]["n_train"] = 4000
    return c
