import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))
import services.duplicate_service as ds_module
from services.duplicate_service import DuplicateService, SIMILARITY_THRESHOLD


class TestDuplicateServiceMultiTenant:
    """15 Unit tests proving strict multi-tenant company_id partitioning and zero cross-tenant leakage."""

    @pytest.fixture
    def service_with_mock_model(self, tmp_path):
        storage_file = str(tmp_path / "test_kb.json")
        svc = DuplicateService(storage_file=storage_file)
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda text, convert_to_tensor=True: f"emb_{text}"
        svc.model = mock_model
        svc._loaded = True
        svc._load_failed = False
        return svc

    def test_01_add_ticket_partitions_by_company_id(self, service_with_mock_model):
        """1. Verify tickets are stored under separate company_id keys in in-memory dict."""
        svc = service_with_mock_model
        svc.add_ticket("T1", "Printer issue", company_id="company_A")
        svc.add_ticket("T2", "Printer issue", company_id="company_B")

        assert "company_A" in svc._tickets_by_company
        assert "company_B" in svc._tickets_by_company
        assert len(svc._tickets_by_company["company_A"]) == 1
        assert len(svc._tickets_by_company["company_B"]) == 1
        assert svc._tickets_by_company["company_A"][0][0] == "T1"
        assert svc._tickets_by_company["company_B"][0][0] == "T2"

    def test_02_add_ticket_default_company_id_fallback(self, service_with_mock_model):
        """2. When company_id is None or empty, fall back cleanly to 'default'."""
        svc = service_with_mock_model
        svc.add_ticket("T3", "Login issue", company_id=None)
        svc.add_ticket("T4", "Password reset", company_id="   ")

        assert "default" in svc._tickets_by_company
        assert len(svc._tickets_by_company["default"]) == 2

    def test_03_add_ticket_persists_company_id_to_disk(self, service_with_mock_model):
        """3. Ensure company_id is persisted accurately inside the JSON disk storage."""
        svc = service_with_mock_model
        svc.add_ticket("T_DISK", "Disk full on server", company_id="acme_corp")

        assert os.path.exists(svc.storage_file)
        with open(svc.storage_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["ticket_id"] == "T_DISK"
        assert data[0]["company_id"] == "acme_corp"

    def test_04_zero_cross_company_collision(self, service_with_mock_model):
        """4. Crucial: Identical ticket text in Company A must NEVER be flagged as duplicate for Company B."""
        svc = service_with_mock_model
        svc.add_ticket("T_COMP_A", "Database server down with error 500", company_id="tenant_alpha")

        mock_sim = MagicMock()
        mock_sim.item.return_value = 0.99
        with patch.object(ds_module.util, 'cos_sim', return_value=mock_sim):
            # Company B queries with identical text
            res_b = svc.check_duplicate("Database server down with error 500", company_id="tenant_beta")
            assert res_b["is_duplicate"] is False
            assert res_b["duplicate_ticket_id"] is None
            assert res_b["similarity"] == 0.0

    def test_05_detects_duplicate_within_same_company(self, service_with_mock_model):
        """5. Within the same company, high similarity triggers duplicate detection correctly."""
        svc = service_with_mock_model
        svc.add_ticket("T_COMP_A", "Database server down with error 500", company_id="tenant_alpha")

        mock_sim = MagicMock()
        mock_sim.item.return_value = 0.95
        with patch.object(ds_module.util, 'cos_sim', return_value=mock_sim):
            res_a = svc.check_duplicate("Database server down with error 500", company_id="tenant_alpha")
            assert res_a["is_duplicate"] is True
            assert res_a["duplicate_ticket_id"] == "T_COMP_A"
            assert res_a["similarity"] == 0.95

    def test_06_check_duplicate_respects_custom_threshold(self, service_with_mock_model):
        """6. Custom threshold overrides global default within company scope."""
        svc = service_with_mock_model
        svc.add_ticket("T10", "Network outage in HQ", company_id="tenant_alpha")

        mock_sim = MagicMock()
        mock_sim.item.return_value = 0.75
        with patch.object(ds_module.util, 'cos_sim', return_value=mock_sim):
            res_default = svc.check_duplicate("Network outage in HQ", company_id="tenant_alpha", threshold=0.70)
            assert res_default["is_duplicate"] is True

            res_strict = svc.check_duplicate("Network outage in HQ", company_id="tenant_alpha", threshold=0.85)
            assert res_strict["is_duplicate"] is False

    def test_07_empty_or_whitespace_text_returns_safe_dict(self, service_with_mock_model):
        """7. Empty and whitespace queries return early with safe zeroed response."""
        svc = service_with_mock_model
        assert svc.check_duplicate("", company_id="c1")["is_duplicate"] is False
        assert svc.check_duplicate("   ", company_id="c1")["is_duplicate"] is False
        assert svc.check_duplicate(None, company_id="c1")["is_duplicate"] is False

    def test_08_querying_empty_company_returns_not_duplicate(self, service_with_mock_model):
        """8. If a company has zero tickets stored, duplicate check immediately returns False."""
        svc = service_with_mock_model
        svc.add_ticket("T_OTHER", "Some ticket", company_id="existing_corp")

        res = svc.check_duplicate("Some ticket", company_id="empty_corp")
        assert res["is_duplicate"] is False
        assert res["duplicate_ticket_id"] is None

    def test_09_numeric_positional_threshold_compatibility(self, service_with_mock_model):
        """9. Backward compatibility: check_duplicate(text, 0.85) properly interprets numeric as threshold."""
        svc = service_with_mock_model
        svc.add_ticket("T_DEF", "Email delivery delay", company_id="default")

        mock_sim = MagicMock()
        mock_sim.item.return_value = 0.80
        with patch.object(ds_module.util, 'cos_sim', return_value=mock_sim):
            # Pass numeric 0.85 as second parameter (legacy signature without company_id)
            res = svc.check_duplicate("Email delivery delay", 0.85)
            assert res["is_duplicate"] is False
            assert res["company_id"] == "default"

    def test_10_load_from_disk_reconstructs_all_company_partitions(self, tmp_path):
        """10. Loading pre-existing JSON partitions all tickets by company into in-memory structure."""
        storage_file = str(tmp_path / "multi_comp.json")
        sample_data = [
            {"ticket_id": "T1", "text": "Issue 1", "company_id": "corp_X"},
            {"ticket_id": "T2", "text": "Issue 2", "company_id": "corp_X"},
            {"ticket_id": "T3", "text": "Issue 3", "company_id": "corp_Y"},
        ]
        with open(storage_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        svc = DuplicateService(storage_file=storage_file)
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda text, convert_to_tensor=True: f"emb_{text}"
        svc.model = mock_model
        svc.load()

        assert len(svc._tickets_by_company["corp_X"]) == 2
        assert len(svc._tickets_by_company["corp_Y"]) == 1
        assert len(svc._tickets) == 3

    def test_11_load_legacy_disk_records_without_company_id(self, tmp_path):
        """11. Legacy JSON records without company_id attribute default gracefully to 'default'."""
        storage_file = str(tmp_path / "legacy.json")
        legacy_data = [
            {"ticket_id": "LEGACY_1", "text": "Old ticket before multitenancy"}
        ]
        with open(storage_file, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f)

        svc = DuplicateService(storage_file=storage_file)
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda text, convert_to_tensor=True: "emb"
        svc.model = mock_model
        svc.load()

        assert "default" in svc._tickets_by_company
        assert svc._tickets_by_company["default"][0][0] == "LEGACY_1"

    def test_12_isolation_across_multiple_independent_companies(self, service_with_mock_model):
        """12. Interleaving additions across 5 companies maintains strict separation."""
        svc = service_with_mock_model
        companies = ["comp_1", "comp_2", "comp_3", "comp_4", "comp_5"]
        for idx, comp in enumerate(companies):
            for t_num in range(3):
                svc.add_ticket(f"{comp}_T{t_num}", f"Content for {comp}", company_id=comp)

        for comp in companies:
            assert len(svc._tickets_by_company[comp]) == 3

    def test_13_degraded_mode_on_load_failure(self, tmp_path):
        """13. When model loading fails, add and check operations degrade safely without throwing."""
        svc = DuplicateService(storage_file=str(tmp_path / "degraded.json"))
        svc._loaded = False
        svc._load_failed = True

        svc.add_ticket("T_FAIL", "Some text", company_id="comp_fail")
        assert len(svc._tickets_by_company) == 0

        res = svc.check_duplicate("Some text", company_id="comp_fail")
        assert res["is_duplicate"] is False
        assert res["duplicate_ticket_id"] is None

    def test_14_sanitizes_company_id_whitespace_and_casing(self, service_with_mock_model):
        """14. Whitespace in company_id is stripped consistently to prevent subtle key mismatch."""
        svc = service_with_mock_model
        svc.add_ticket("T_WS", "Vpn disconnects", company_id="  tenant_spc  ")
        assert "tenant_spc" in svc._tickets_by_company

        mock_sim = MagicMock()
        mock_sim.item.return_value = 0.90
        with patch.object(ds_module.util, 'cos_sim', return_value=mock_sim):
            res = svc.check_duplicate("Vpn disconnects", company_id="tenant_spc")
            assert res["is_duplicate"] is True
            assert res["duplicate_ticket_id"] == "T_WS"

    def test_15_returns_highest_scoring_ticket_within_tenant(self, service_with_mock_model):
        """15. When multiple tickets exist in the tenant, returns the one with the maximum similarity."""
        svc = service_with_mock_model
        svc.add_ticket("T_LOW", "Billing invoice error", company_id="corp_fin")
        svc.add_ticket("T_HIGH", "Billing receipt discrepancy", company_id="corp_fin")

        scores = {"emb_Billing invoice error": 0.72, "emb_Billing receipt discrepancy": 0.94}
        
        def fake_cos_sim(q, stored):
            res = MagicMock()
            res.item.return_value = scores.get(stored, 0.0)
            return res

        with patch.object(ds_module.util, 'cos_sim', side_effect=fake_cos_sim):
            res = svc.check_duplicate("Billing receipt discrepancy", company_id="corp_fin")
            assert res["is_duplicate"] is True
            assert res["duplicate_ticket_id"] == "T_HIGH"
            assert res["similarity"] == 0.94
