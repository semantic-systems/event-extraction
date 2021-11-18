import pytest
from hydra import initialize


@pytest.fixture()
def initialize_hydra():
    initialize(config_path="./configs", job_name="test")