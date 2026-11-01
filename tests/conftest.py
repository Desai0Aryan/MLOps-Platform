import sys
from pathlib import Path
import os
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlops.config import load_config


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    c = load_config()
    c["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"
    c["data"]["n_train"] = 4000
    return c


@pytest.fixture(scope="session", autouse=True)
def _isolated_cwd(tmp_path_factory):
    """MLflow's sqlite store puts artifacts under ./mlruns relative to the CWD. Run the whole suite from a
    temp directory so tests never litter (or depend on) the repository checkout."""
    old = os.getcwd(); os.chdir(tmp_path_factory.mktemp("cwd"))
    yield
    os.chdir(old)
