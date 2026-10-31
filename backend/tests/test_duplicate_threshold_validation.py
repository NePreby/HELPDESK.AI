import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

import pytest


for mod_name in ["sentence_transformers", "torch"]:
    sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

sys.modules["sentence_transformers"].SentenceTransformer = MagicMock()
sys.modules["sentence_transformers"].util = MagicMock()

sys.modules.pop("backend.services.duplicate_service", None)
_spec = importlib.util.spec_from_file_location(
    "duplicate_service_real_threshold_validation",
    os.path.join(os.path.dirname(__file__), "..", "services", "duplicate_service.py"),
)
_duplicate_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_duplicate_module)

DuplicateService = _duplicate_module.DuplicateService


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_check_duplicate_rejects_out_of_range_threshold_before_availability_check(threshold):
    service = DuplicateService()
    service._load_failed = True

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        service.check_duplicate("same issue", threshold=threshold)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_check_duplicate_allows_threshold_boundaries_in_degraded_mode(threshold):
    service = DuplicateService()
    service._load_failed = True

    result = service.check_duplicate("same issue", threshold=threshold)

    assert result == {
        "is_duplicate": False,
        "duplicate_ticket_id": None,
        "similarity": 0.0,
    }


@pytest.mark.parametrize("invalid_threshold", [-0.5, -100.0, 1.01, 1.5, 99.0, "invalid", True, False])
def test_check_duplicate_rejects_various_invalid_thresholds(invalid_threshold):
    service = DuplicateService()
    service._load_failed = True

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        service.check_duplicate("ticket text", threshold=invalid_threshold)


@pytest.mark.parametrize("valid_threshold", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_check_duplicate_accepts_valid_threshold_range(valid_threshold):
    service = DuplicateService()
    service._load_failed = True

    result = service.check_duplicate("ticket text", threshold=valid_threshold)
    assert result["is_duplicate"] is False
    assert result["similarity"] == 0.0
