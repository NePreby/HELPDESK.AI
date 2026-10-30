"""
Regression test for Issue #4137:
Ensure backend.main has all standard typing constructs imported (Optional, Dict, List, Any, Union)
to prevent NameError during FastAPI endpoint route definition and startup.
"""

import sys
import os
from unittest.mock import MagicMock

# Mock heavy ML modules to allow standalone test execution
for module in ["torch", "torch.nn", "torch.nn.functional", "transformers", "sentence_transformers"]:
    sys.modules.setdefault(module, MagicMock())

os.environ["ALLOW_DEGRADED_STARTUP"] = "1"
os.environ["SUPABASE_URL"] = "https://placeholder.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "placeholder-key"

import backend.main as main_mod


def test_main_typing_imports_available():
    """Verify that backend.main exports standard typing constructs."""
    assert hasattr(main_mod, "Optional"), "backend.main must import Optional from typing"
    assert hasattr(main_mod, "Dict"), "backend.main must import Dict from typing"
    assert hasattr(main_mod, "List"), "backend.main must import List from typing"
    assert hasattr(main_mod, "Any"), "backend.main must import Any from typing"
    assert hasattr(main_mod, "Union"), "backend.main must import Union from typing"


def test_optional_annotation_evaluates():
    """Verify that Optional[str] and Dict[str, int] can be used without NameError."""
    def sample_route(company_id: main_mod.Optional[str] = None) -> main_mod.Dict[str, main_mod.Any]:
        return {"company_id": company_id}

    assert sample_route(company_id="acme") == {"company_id": "acme"}
    assert sample_route() == {"company_id": None}
