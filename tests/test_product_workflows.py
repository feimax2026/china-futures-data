from __future__ import annotations

import unittest

from src.build_weighted_index import product_output_path
from src.download_contracts import contract_symbols
from src.train_xgboost_compare import product_configs


class ProductWorkflowTests(unittest.TestCase):
    def test_contract_symbols_are_parameterized_by_product(self) -> None:
        self.assertEqual(contract_symbols("I", 2026, 2026)[:3], ["I2601", "I2602", "I2603"])
        self.assertEqual(contract_symbols("JM", 2026, 2026)[-1], "JM2612")

    def test_weighted_index_output_path_uses_lowercase_product_code(self) -> None:
        self.assertTrue(str(product_output_path("I")).endswith("data/processed/i_weighted_index.csv"))

    def test_iron_ore_compare_configs_use_i0_and_i_weighted_index(self) -> None:
        configs = product_configs("I")

        self.assertEqual(configs[0].output_prefix, "i_main")
        self.assertEqual(configs[1].output_prefix, "i_weighted")
        self.assertTrue(str(configs[0].path).endswith("iron_ore_I0.csv"))
        self.assertTrue(str(configs[1].path).endswith("i_weighted_index.csv"))

    def test_manganese_silicon_compare_configs_use_sm0_and_sm_weighted_index(self) -> None:
        configs = product_configs("SM")

        self.assertEqual(contract_symbols("SM", 2026, 2026)[:2], ["SM2601", "SM2602"])
        self.assertEqual(configs[0].output_prefix, "sm_main")
        self.assertEqual(configs[1].output_prefix, "sm_weighted")
        self.assertTrue(str(configs[0].path).endswith("manganese_silicon_SM0.csv"))
        self.assertTrue(str(configs[1].path).endswith("sm_weighted_index.csv"))

    def test_copper_compare_configs_use_cu0_and_cu_weighted_index(self) -> None:
        configs = product_configs("CU")

        self.assertEqual(contract_symbols("CU", 2026, 2026)[:2], ["CU2601", "CU2602"])
        self.assertEqual(configs[0].output_prefix, "cu_main")
        self.assertEqual(configs[1].output_prefix, "cu_weighted")
        self.assertTrue(str(configs[0].path).endswith("copper_CU0.csv"))
        self.assertTrue(str(configs[1].path).endswith("cu_weighted_index.csv"))


if __name__ == "__main__":
    unittest.main()
