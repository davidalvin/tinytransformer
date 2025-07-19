# tests/test_build_dataset.py

import pytest
from unittest.mock import patch, MagicMock
from tinytransformer.data import build_dataset


def test_run_success():
    """Test that run() does not raise when subprocess returns success."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        build_dataset.run("dummy.module")  # Should not raise


def test_run_failure():
    """Test that run() raises RuntimeError on subprocess failure."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        with pytest.raises(RuntimeError, match="❌ Script failed: dummy.module"):
            build_dataset.run("dummy.module")


def test_main_all_success():
    """Test main() calls run() for each module without errors."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        build_dataset.main()  # Should run all modules without raising
