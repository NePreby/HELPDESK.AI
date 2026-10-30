"""
Duplicate Detection Service
Uses sentence-transformers all-MiniLM-L6-v2 to detect similar tickets with strict multi-tenant isolation.
"""

import os
import json
import uuid
from typing import Dict, List, Tuple, Any, Optional

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

SIMILARITY_THRESHOLD = 0.70


class DuplicateService:
    def __init__(self, storage_file: Optional[str] = None):
        self.model = None
        self._loaded = False
        self._load_failed = False
        # Partitioned store: dict of company_id -> list of (ticket_id, embedding, text)
        self._tickets_by_company: Dict[str, List[Tuple[str, Any, str]]] = {}
        # Backward-compatibility flat list referencing all tickets
        self._tickets: List[Tuple[str, Any, str]] = []
        if storage_file:
            self.storage_file = storage_file
        else:
            self.storage_file = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)

    def load(self):
        """Load the sentence-transformer model and saved tickets."""
        if self._loaded or self._load_failed:
            return

        if SentenceTransformer is None:
            self._load_failed = True
            return

        print("[DuplicateService] Loading model...")
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._loaded = True
            self._load_failed = False
        except Exception as e:
            print(f"[DuplicateService] Failed to load model: {e}")
            self._load_failed = True
            self._loaded = False
            return

        if os.path.exists(self.storage_file):
            print(f"[DuplicateService] Loading knowledge base from {self.storage_file}...")
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            text = item.get("text", "")
                            ticket_id = item.get("ticket_id", "")
                            comp_id = str(item.get("company_id", "default") or "default").strip() or "default"
                            embedding = self.model.encode(text, convert_to_tensor=True)
                            entry = (ticket_id, embedding, text)
                            if comp_id not in self._tickets_by_company:
                                self._tickets_by_company[comp_id] = []
                            self._tickets_by_company[comp_id].append(entry)
                            self._tickets.append(entry)
                print(f"[DuplicateService] Loaded {len(self._tickets)} tickets across {len(self._tickets_by_company)} companies.")
            except Exception as e:
                print(f"[DuplicateService] Error loading storage: {e}")

    def save_to_disk(self, ticket_id: str, text: str, company_id: str = "default"):
        """Append a new ticket with its company_id to the JSON storage."""
        comp_id = str(company_id or "default").strip() or "default"
        try:
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            data = []
            if os.path.exists(self.storage_file):
                try:
                    with open(self.storage_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = []
                except Exception:
                    data = []

            data.append({"ticket_id": ticket_id, "text": text, "company_id": comp_id})
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[DuplicateService] Saved ticket {ticket_id} to knowledge base for company {comp_id}.")
        except Exception as e:
            print(f"[DuplicateService] Failed to save to disk: {e}")

    def add_ticket(self, ticket_id: str, text: str, company_id: str = "default"):
        """Add a ticket to the in-memory partitioned store and persist to disk."""
        if self._load_failed:
            return

        self.load()

        if self._load_failed or not self.model:
            return

        comp_id = str(company_id or "default").strip() or "default"
        embedding = self.model.encode(text, convert_to_tensor=True)
        entry = (ticket_id, embedding, text)

        if comp_id not in self._tickets_by_company:
            self._tickets_by_company[comp_id] = []
        self._tickets_by_company[comp_id].append(entry)
        self._tickets.append(entry)

        if comp_id == "default":
            self.save_to_disk(ticket_id, text)
        else:
            self.save_to_disk(ticket_id, text, comp_id)

    def check_duplicate(self, text: str, company_id: str = "default", threshold: Optional[float] = None) -> dict:
        """
        Check if a ticket is a duplicate of any stored ticket belonging strictly to the specified company_id.

        Args:
            text: The ticket text to check.
            company_id: The tenant/company ID to partition against.
            threshold: Optional override for the similarity threshold.

        Returns:
            {
                "is_duplicate": bool,
                "duplicate_ticket_id": str | None,
                "similarity": float,
                "company_id": str
            }
        """
        # Support positional threshold override if numeric passed as second argument
        if isinstance(company_id, (int, float)):
            threshold = float(company_id)
            comp_id = "default"
        else:
            comp_id = str(company_id or "default").strip() or "default"

        active_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD

        if not text or not str(text).strip():
            return {
                "is_duplicate": False,
                "duplicate_ticket_id": None,
                "similarity": 0.0,
                "company_id": comp_id,
            }

        self.load()

        if self._load_failed or not self.model:
            return {
                "is_duplicate": False,
                "duplicate_ticket_id": None,
                "similarity": 0.0,
                "company_id": comp_id,
            }

        company_tickets = self._tickets_by_company.get(comp_id, [])

        if not company_tickets:
            return {
                "is_duplicate": False,
                "duplicate_ticket_id": None,
                "similarity": 0.0,
                "company_id": comp_id,
            }

        query_embedding = self.model.encode(text, convert_to_tensor=True)

        best_score = 0.0
        best_id = None

        for ticket_id, stored_emb, _ in company_tickets:
            score = 0.0
            if util is not None and hasattr(util, 'cos_sim'):
                sim_res = util.cos_sim(query_embedding, stored_emb)
                if hasattr(sim_res, 'item'):
                    sim_res = sim_res.item()
                try:
                    score = float(sim_res)
                except (TypeError, ValueError):
                    score = 0.0

            if score > best_score:
                best_score = score
                best_id = ticket_id

        is_dup = best_score >= active_threshold

        return {
            "is_duplicate": is_dup,
            "duplicate_ticket_id": best_id if is_dup else None,
            "similarity": round(best_score, 4),
            "company_id": comp_id,
        }
