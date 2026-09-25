import pytest
import subprocess
import sys
from pathlib import Path


def test_main_script_runs():
    """Verifies main.py runs without error in sample mode."""
    res = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "PIPELINE COMPLETED SUCCESSFULLY & VALIDATION PASSED!" in res.stdout
    assert Path("output/matching_results.tsv").exists()
    assert Path("output/candidate_pairs.tsv").exists()
