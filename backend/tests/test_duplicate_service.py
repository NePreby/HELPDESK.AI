"""
Unit tests for DuplicateService (backend/services/duplicate_service.py).
Covers initialization, model loading, degraded mode, thread-safety,
and check_duplicate method edge cases (Issue #4078).
"""

import sys
import os

# Clean sys.path to avoid importing the root dummy numpy directory
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _repo_root and p != ""]
_backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import json
import pytest
from unittest.mock import MagicMock, patch
import threading
import time

from services.duplicate_service import DuplicateService, SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_env():
    """Ensure clean environment for each test."""
    env_vars = [
        "SENTENCE_TRANSFORMER_MODEL_PATH",
        "ALLOW_DEGRADED_STARTUP",
        "DUPLICATE_CACHE_MAX",
    ]
    for var in env_vars:
        os.environ.pop(var, None)
    yield
    for var in env_vars:
        os.environ.pop(var, None)


@pytest.fixture
def mock_sentence_transformers():
    """Mock the sentence_transformers module."""
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_st.SentenceTransformer.return_value = mock_model
    return mock_st, mock_model


# ---------------------------------------------------------------------------
# load() — model loading
# ---------------------------------------------------------------------------

class TestDuplicateServiceLoad:
    """Tests for DuplicateService.load()."""

    @patch("services.duplicate_service._HAS_SENTENCE", True)
    @patch("services.duplicate_service.SentenceTransformer")
    def test_loads_from_local_path_when_set(self, mock_ST):
        os.environ["SENTENCE_TRANSFORMER_MODEL_PATH"] = "/tmp/my-model"
        with patch("os.path.exists", return_value=True):
            svc = DuplicateService()
            svc.load()
            mock_ST.assert_called_with("/tmp/my-model")
            assert svc._loaded is True

    @patch("services.duplicate_service._HAS_SENTENCE", True)
    @patch("services.duplicate_service.SentenceTransformer")
    def test_loads_from_huggingface_when_no_local_path(self, mock_ST):
        svc = DuplicateService()
        svc.load()
        mock_ST.assert_called_with("all-MiniLM-L6-v2")
        assert svc._loaded is True

    @patch("services.duplicate_service._HAS_SENTENCE", True)
    @patch("services.duplicate_service.SentenceTransformer")
    def test_load_is_idempotent(self, mock_ST):
        svc = DuplicateService()
        svc.load()
        svc.load()  # Second call should be a no-op
        assert mock_ST.call_count == 1

    @patch("services.duplicate_service._HAS_SENTENCE", True)
    @patch("services.duplicate_service.SentenceTransformer")
    def test_load_reads_storage_file(self, mock_ST, tmp_path):
        storage_file = tmp_path / "case_history_cache.json"
        storage_file.write_text(json.dumps([
            {"ticket_id": "t1", "text": "hello world"},
            {"ticket_id": "t2", "text": "foo bar"},
        ]))

        svc = DuplicateService()
        svc.storage_file = str(storage_file)
        svc.load()

        assert svc._loaded is True
        assert len(svc._tickets) == 2


# ---------------------------------------------------------------------------
# load() — degraded startup
# ---------------------------------------------------------------------------

class TestDuplicateServiceDegraded:
    """Tests for ALLOW_DEGRADED_STARTUP handling."""

    @patch("services.duplicate_service._HAS_SENTENCE", False)
    def test_degraded_mode_when_no_sentence_transformers(self):
        os.environ["ALLOW_DEGRADED_STARTUP"] = "1"
        svc = DuplicateService()
        svc.load()
        assert svc._loaded is False
        assert svc.model is None
        assert svc.is_available() is False

    @patch("services.duplicate_service._HAS_SENTENCE", False)
    def test_raises_when_no_sentence_transformers_and_no_degraded(self):
        svc = DuplicateService()
        with pytest.raises(ImportError, match="sentence-transformers"):
            svc.load()


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------

class TestDuplicateServiceIsAvailable:
    """Tests for is_available()."""

    @patch("services.duplicate_service._HAS_SENTENCE", True)
    @patch("services.duplicate_service.SentenceTransformer")
    def test_available_after_successful_load(self, mock_ST):
        svc = DuplicateService()
        svc.load()
        assert svc.is_available() is True

    @patch("services.duplicate_service._HAS_SENTENCE", False)
    def test_not_available_when_load_failed(self):
        os.environ["ALLOW_DEGRADED_STARTUP"] = "1"
        svc = DuplicateService()
        svc.load()
        assert svc.is_available() is False


# ---------------------------------------------------------------------------
# save_to_disk & add_ticket thread safety & empty files
# ---------------------------------------------------------------------------

class TestDuplicateServiceThreadSafety:
    """Tests for DuplicateService thread-safety and empty file handling."""

    @patch("services.duplicate_service._HAS_SENTENCE", True)
    @patch("services.duplicate_service.SentenceTransformer")
    def test_empty_json_file_does_not_crash(self, mock_ST, tmp_path):
        storage_file = tmp_path / "case_history_cache.json"
        storage_file.write_text("")

        svc = DuplicateService()
        svc.storage_file = str(storage_file)
        svc.load()
        assert len(svc._tickets) == 0

    @patch("services.duplicate_service._HAS_SENTENCE", True)
    @patch("services.duplicate_service.SentenceTransformer")
    def test_corrupt_json_file_handles_gracefully(self, mock_ST, tmp_path):
        storage_file = tmp_path / "case_history_cache.json"
        storage_file.write_text("invalid json content")

        svc = DuplicateService()
        svc.storage_file = str(storage_file)
        svc.load()
        assert len(svc._tickets) == 0

    @patch("services.duplicate_service._HAS_SENTENCE", True)
    @patch("services.duplicate_service.SentenceTransformer")
    def test_save_to_disk_acquires_lock(self, mock_ST, tmp_path):
        storage_file = tmp_path / "case_history_cache.json"
        svc = DuplicateService()
        svc.storage_file = str(storage_file)

        svc._lock.acquire()
        
        blocker_ran = False
        def blocker():
            nonlocal blocker_ran
            svc.save_to_disk("t1", "hello")
            blocker_ran = True

        t = threading.Thread(target=blocker)
        t.start()
        
        time.sleep(0.1)
        assert blocker_ran is False
        
        svc._lock.release()
        t.join()
        assert blocker_ran is True

    @patch("services.duplicate_service._HAS_SENTENCE", True)
    @patch("services.duplicate_service.SentenceTransformer")
    def test_concurrent_adds_and_checks(self, mock_ST, tmp_path):
        storage_file = tmp_path / "case_history_cache.json"
        svc = DuplicateService()
        svc.storage_file = str(storage_file)
        
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        svc.model = mock_model
        svc._loaded = True

        threads = []
        for i in range(10):
            t = threading.Thread(target=svc.add_ticket, args=(f"t-{i}", f"text-{i}"))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(svc._tickets) == 10


# ---------------------------------------------------------------------------
# check_duplicate() — comprehensive tests (fixes #4078)
# ---------------------------------------------------------------------------

class TestDuplicateServiceCheckDuplicate:
    """Dedicated tests for DuplicateService.check_duplicate edge cases."""

    def test_check_duplicate_empty_ticket_store(self):
        """When no tickets are stored, returns is_duplicate=False with 0.0 similarity."""
        svc = DuplicateService()
        svc._loaded = True
        svc.model = MagicMock()
        svc._tickets = []

        result = svc.check_duplicate("Any query text")
        assert result == {
            "is_duplicate": False,
            "duplicate_ticket_id": None,
            "similarity": 0.0,
        }

    def test_check_duplicate_empty_or_whitespace_text(self):
        """Empty or whitespace-only query text returns non-duplicate immediately."""
        svc = DuplicateService()
        svc._loaded = True
        svc.model = MagicMock()
        svc._tickets = [("ticket-1", MagicMock(), "Existing issue")]

        for empty_input in ["", "   ", "\t\n", None]:
            result = svc.check_duplicate(empty_input)
            assert result == {
                "is_duplicate": False,
                "duplicate_ticket_id": None,
                "similarity": 0.0,
            }
        # Model encode should never be called for empty strings
        assert svc.model.encode.call_count == 0

    def test_check_duplicate_degraded_mode_behavior(self):
        """In degraded startup mode, check_duplicate fails safely without raising exceptions."""
        os.environ["ALLOW_DEGRADED_STARTUP"] = "1"
        svc = DuplicateService()
        svc.model = None
        svc._loaded = False
        svc._load_failed = True

        result = svc.check_duplicate("Need password reset")
        assert result == {
            "is_duplicate": False,
            "duplicate_ticket_id": None,
            "similarity": 0.0,
        }

    @patch("services.duplicate_service.util")
    def test_check_duplicate_default_threshold(self, mock_util):
        """Uses default SIMILARITY_THRESHOLD when no threshold argument is provided."""
        svc = DuplicateService()
        svc._loaded = True
        mock_model = MagicMock()
        mock_model.encode.return_value = "query_emb"
        svc.model = mock_model

        mock_cos_sim = MagicMock()
        mock_cos_sim.item.return_value = 0.72
        mock_util.cos_sim.return_value = mock_cos_sim

        svc._tickets = [("T-100", "stored_emb_1", "Cannot login to portal")]

        # Default threshold is 0.70; score 0.72 >= 0.70 -> duplicate
        result = svc.check_duplicate("Cannot login to portal")
        assert result["is_duplicate"] is True
        assert result["duplicate_ticket_id"] == "T-100"
        assert result["similarity"] == 0.72

    @patch("services.duplicate_service.util")
    def test_check_duplicate_threshold_override(self, mock_util):
        """Threshold override changes whether a match is considered a duplicate."""
        svc = DuplicateService()
        svc._loaded = True
        mock_model = MagicMock()
        mock_model.encode.return_value = "query_emb"
        svc.model = mock_model

        mock_cos_sim = MagicMock()
        mock_cos_sim.item.return_value = 0.85
        mock_util.cos_sim.return_value = mock_cos_sim

        svc._tickets = [("T-200", "stored_emb_2", "Database connection timeout")]

        # Score is 0.85:
        # 1. Overridden to strict threshold 0.90 -> NOT duplicate
        res_strict = svc.check_duplicate("Database connection timeout", threshold=0.90)
        assert res_strict["is_duplicate"] is False
        assert res_strict["duplicate_ticket_id"] is None
        assert res_strict["similarity"] == 0.85

        # 2. Overridden to loose threshold 0.80 -> IS duplicate
        res_loose = svc.check_duplicate("Database connection timeout", threshold=0.80)
        assert res_loose["is_duplicate"] is True
        assert res_loose["duplicate_ticket_id"] == "T-200"
        assert res_loose["similarity"] == 0.85

    @patch("services.duplicate_service.util")
    def test_check_duplicate_selects_highest_similarity_candidate(self, mock_util):
        """Iterates through multiple tickets and selects the candidate with the highest similarity."""
        svc = DuplicateService()
        svc._loaded = True
        mock_model = MagicMock()
        mock_model.encode.return_value = "query_emb"
        svc.model = mock_model

        # Mock cos_sim returning different values for 3 stored tickets
        score_t1 = MagicMock()
        score_t1.item.return_value = 0.65
        score_t2 = MagicMock()
        score_t2.item.return_value = 0.94
        score_t3 = MagicMock()
        score_t3.item.return_value = 0.82

        mock_util.cos_sim.side_effect = [score_t1, score_t2, score_t3]

        svc._tickets = [
            ("T-1", "emb_1", "VPN issue"),
            ("T-2", "emb_2", "Network VPN disconnected unexpectedly"),
            ("T-3", "emb_3", "Wifi connection drop"),
        ]

        result = svc.check_duplicate("VPN disconnected unexpectedly", threshold=0.70)
        assert result["is_duplicate"] is True
        assert result["duplicate_ticket_id"] == "T-2"
        assert result["similarity"] == 0.94

    def test_check_duplicate_threshold_validation_errors(self):
        """Invalid threshold types or ranges raise ValueError."""
        svc = DuplicateService()
        svc._loaded = True
        svc.model = MagicMock()
        svc._tickets = [("T-1", MagicMock(), "Ticket")]

        invalid_thresholds = [-0.01, 1.01, -5.0, 10.0, True, False, "invalid_str", None]
        for t in invalid_thresholds:
            if t is None:
                continue
            with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
                svc.check_duplicate("Test text", threshold=t)

    @patch("services.duplicate_service.util")
    def test_check_duplicate_with_mocked_supabase_sync(self, mock_util):
        """Integration test with mocked Supabase client fetching ticket history."""
        mock_supabase = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            {"id": "supa-1", "title": "Printer paper jam", "description": "Tray 2 jammed"},
            {"id": "supa-2", "title": "Outlook crash", "description": "Crash on launch"},
        ]
        mock_supabase.table().select().execute.return_value = mock_response

        svc = DuplicateService()
        svc._loaded = True
        mock_model = MagicMock()
        mock_model.encode.return_value = "query_vector"
        svc.model = mock_model

        # Simulate loading Supabase tickets into service memory
        tickets = mock_supabase.table("tickets").select("id, title, description").execute()
        for t in tickets.data:
            text = f"{t['title']}: {t['description']}"
            svc._tickets.append((t["id"], "emb_" + t["id"], text))

        assert len(svc._tickets) == 2

        score_1 = MagicMock()
        score_1.item.return_value = 0.96
        score_2 = MagicMock()
        score_2.item.return_value = 0.40
        mock_util.cos_sim.side_effect = [score_1, score_2]

        result = svc.check_duplicate("Printer paper jam in tray 2", threshold=0.85)
        assert result["is_duplicate"] is True
        assert result["duplicate_ticket_id"] == "supa-1"
        assert result["similarity"] == 0.96
