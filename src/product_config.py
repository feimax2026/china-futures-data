from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProductConfig:
    code: str
    name_en: str
    name_zh: str
    main_csv: str
    default_start_year: int = 2019

    @property
    def lower_code(self) -> str:
        return self.code.lower()


PRODUCTS: dict[str, ProductConfig] = {
    "JM": ProductConfig(
        code="JM",
        name_en="Coking Coal",
        name_zh="焦煤",
        main_csv="coking_coal_JM0.csv",
        default_start_year=2019,
    ),
    "I": ProductConfig(
        code="I",
        name_en="Iron Ore",
        name_zh="铁矿石",
        main_csv="iron_ore_I0.csv",
        default_start_year=2019,
    ),
    "SM": ProductConfig(
        code="SM",
        name_en="Manganese Silicon",
        name_zh="锰硅",
        main_csv="manganese_silicon_SM0.csv",
        default_start_year=2019,
    ),
    "CU": ProductConfig(
        code="CU",
        name_en="Copper",
        name_zh="沪铜",
        main_csv="copper_CU0.csv",
        default_start_year=2019,
    ),
}


def get_product(code: str) -> ProductConfig:
    normalized = code.upper()
    if normalized not in PRODUCTS:
        raise ValueError(f"Unsupported product {code!r}. Supported: {', '.join(sorted(PRODUCTS))}")
    return PRODUCTS[normalized]
