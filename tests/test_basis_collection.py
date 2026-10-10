import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.basis_history import collect_basis, basis_statistics
from src.research_storage import ResearchStore


class BasisCollectionTests(unittest.TestCase):
    @patch("src.basis_history.bounded_ak_frame")
    def test_observed_prices_determine_sign_and_state_is_incremental(self, fetch):
        fetch.return_value = pd.DataFrame([{"date": "20260930", "var": "SF", "sp": 5682.86,
                                           "dom_price": 5994, "dom_symbol": "SF611"}])
        with tempfile.TemporaryDirectory() as root:
            store = ResearchStore(Path(root), bucket="")
            result = collect_basis(store, "2026-09-30")
            self.assertAlmostEqual(result["SF"][0]["basis"], -311.14)
            self.assertLess(result["SF"][0]["basis_rate_pct"], 0)
            stats = basis_statistics(result["SF"], "2026-09-30", "SF2611")
            self.assertEqual(stats["status"], "insufficient_history")
            self.assertEqual(stats["contract"], "SF2611")
            collect_basis(store, "2026-09-30")
            self.assertEqual(len(store.read_json("state/basis/history-v1.json")["rows"]), 1)
