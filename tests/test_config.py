"""
Basic sanity test for configuration and project setup.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


def test_config_classes():
    """Verify 7 HAM10000 classes and mappings."""
    assert len(config.CLASS_NAMES) == 7
    assert config.NUM_CLASSES == 7
    expected = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    assert set(config.CLASS_NAMES) == expected
    for name in config.CLASS_NAMES:
        assert name in config.CLASS_LABELS
        assert config.LABEL_TO_IDX[name] in range(7)


def test_ensemble_weights():
    """Verify ensemble weights sum to 1.0."""
    total_weight = sum(config.ENSEMBLE_WEIGHTS.values())
    assert abs(total_weight - 1.0) < 1e-5


def test_directory_paths():
    """Verify directory paths are Path objects."""
    assert isinstance(config.DATA_DIR, Path)
    assert isinstance(config.MODELS_DIR, Path)
    assert isinstance(config.BASE_DIR, Path)
