"""
Duplicate Detection Service
Uses sentence-transformers all-MiniLM-L6-v2 to detect similar tickets.
"""

import json
import logging
import os
import threading
import uuid

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_SENTENCE = True
except ImportError:
    SentenceTransformer = None
    util = None
    _HAS_SENTENCE = False

SIMILARITY_THRESHOLD = 0.70


class DuplicateService:
    def __init__(self):
        self.model = None
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()
        # In-memory store: list of (ticket_id, embedding, text)
        self._tickets: list[tuple[str, object, str]] = []
        self.storage_file = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")
        try:
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        except Exception:
            pass

    def is_available(self) -> bool:
        """Check if service is loaded and ready for inference."""
        return self._loaded and self.model is not None and not self._load_failed

    def load(self):
        """Load the sentence-transformer model and saved tickets."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            if not _HAS_SENTENCE or SentenceTransformer is None:
                if os.getenv("ALLOW_DEGRADED_STARTUP") == "1":
                    logger.warning("[DuplicateService] Running in degraded mode: sentence-transformers not installed")
                    self.model = None
                    self._loaded = False
                    self._load_failed = True
                    return
                raise ImportError("sentence-transformers is required but not installed.")

            model_path = os.getenv("SENTENCE_TRANSFORMER_MODEL_PATH")
            if model_path and os.path.exists(model_path):
                target_model = model_path
            else:
                target_model = "all-MiniLM-L6-v2"

            try:
                print(f"[DuplicateService] Loading model: {target_model}...")
                self.model = SentenceTransformer(target_model)
                self._loaded = True
                self._load_failed = False
            except Exception as e:
                if os.getenv("ALLOW_DEGRADED_STARTUP") == "1":
                    logger.warning(f"[DuplicateService] Running in degraded mode: {e}")
                    self.model = None
                    self._loaded = False
                    self._load_failed = True
                    return
                raise

            if self.storage_file and os.path.exists(self.storage_file):
                print(f"[DuplicateService] Loading knowledge base from {self.storage_file}...")
                try:
                    with open(self.storage_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            data = json.loads(content)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and "text" in item and "ticket_id" in item:
                                        text = item["text"]
                                        embedding = self.model.encode(text, convert_to_tensor=True) if self.model else None
                                        self._tickets.append((item["ticket_id"], embedding, text))
                                print(f"[DuplicateService] Loaded {len(self._tickets)} tickets.")
                except Exception as e:
                    print(f"[DuplicateService] Error loading storage: {e}")

    def save_to_disk(self, ticket_id: str, text: str):
        """Append a new ticket to the JSON storage."""
        with self._lock:
            data = []
            try:
                if self.storage_file:
                    os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
                    if os.path.exists(self.storage_file):
                        with open(self.storage_file, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                try:
                                    loaded = json.loads(content)
                                    if isinstance(loaded, list):
                                        data = loaded
                                except Exception:
                                    data = []

                    data.append({"ticket_id": ticket_id, "text": text})
                    with open(self.storage_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    print(f"[DuplicateService] Saved ticket {ticket_id} to knowledge base.")
            except Exception as e:
                print(f"[DuplicateService] Failed to save to disk: {e}")

    def add_ticket(self, ticket_id: str, text: str):
        """Add a ticket to the in-memory store and persist to disk."""
        self.load()
        with self._lock:
            embedding = self.model.encode(text, convert_to_tensor=True) if self.model else None
            self._tickets.append((ticket_id, embedding, text))
        self.save_to_disk(ticket_id, text)

    def check_duplicate(self, text: str, threshold: float = None) -> dict:
        """
        Check if a ticket is a duplicate of any stored ticket.

        Args:
            text: The ticket text to check.
            threshold: Optional override for the similarity threshold.

        Returns:
            {
                "is_duplicate": bool,
                "duplicate_ticket_id": str | None,
                "similarity": float
            }
        """
        if threshold is not None:
            try:
                val = float(threshold)
                if val < 0.0 or val > 1.0 or isinstance(threshold, bool):
                    raise ValueError("Threshold must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                raise ValueError("Threshold must be between 0.0 and 1.0")

        # Early return for empty or whitespace-only text
        if not text or not str(text).strip():
            return {
                "is_duplicate": False,
                "duplicate_ticket_id": None,
                "similarity": 0.0,
            }

        self.load()

        if not self.is_available() or self.model is None or not self._tickets:
            return {
                "is_duplicate": False,
                "duplicate_ticket_id": None,
                "similarity": 0.0,
            }

        active_threshold = float(threshold) if threshold is not None else SIMILARITY_THRESHOLD

        query_embedding = self.model.encode(text, convert_to_tensor=True)

        best_score = 0.0
        best_id = None

        for ticket_id, stored_emb, _ in self._tickets:
            if stored_emb is not None and util is not None:
                score = util.cos_sim(query_embedding, stored_emb).item()
                if score > best_score:
                    best_score = score
                    best_id = ticket_id

        is_dup = best_score >= active_threshold

        return {
            "is_duplicate": is_dup,
            "duplicate_ticket_id": best_id if is_dup else None,
            "similarity": round(best_score, 4),
        }
