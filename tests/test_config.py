from tinytransformer.config.cli import parse_and_apply
from tinytransformer.config import config as C

def test_cli_config_override(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--num_steps", "123"])
    parse_and_apply()
    assert C.NUM_STEPS == 123
