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
    exchange: str = "DCE"
    sector: str = "industrial"
    power_exposure: str = "indirect"

    @property
    def continuous_symbol(self) -> str:
        return f"{self.code}0"

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
        exchange="CZCE",
        power_exposure="smelting",
    ),
    "CU": ProductConfig(
        code="CU",
        name_en="Copper",
        name_zh="沪铜",
        main_csv="copper_CU0.csv",
        default_start_year=2019,
        exchange="SHFE",
    ),
    "AU": ProductConfig("AU", "Gold", "黄金", "gold_AU0.csv", 2008, "SHFE", "precious_metals"),
    "AG": ProductConfig("AG", "Silver", "白银", "silver_AG0.csv", 2012, "SHFE", "precious_metals"),
    "SC": ProductConfig("SC", "Crude Oil", "原油", "crude_oil_SC0.csv", 2018, "INE", "energy"),
    "AL": ProductConfig("AL", "Aluminium", "沪铝", "aluminium_AL0.csv", 2010, "SHFE", "nonferrous", "smelting"),
    "SI": ProductConfig("SI", "Industrial Silicon", "工业硅", "industrial_silicon_SI0.csv", 2022, "GFEX", "silicon", "smelting"),
    "SF": ProductConfig("SF", "Ferrosilicon", "硅铁", "ferrosilicon_SF0.csv", 2014, "CZCE", "ferroalloys", "smelting"),
}

# Keep the existing morning-brief contract stable while expanding research.
LEGACY_PRODUCTS = ("JM", "I", "SM", "CU")
SENTINEL_PRODUCTS = ("JM", "I", "SF", "SM", "AU", "AG", "CU")


def get_product(code: str) -> ProductConfig:
    normalized = code.upper()
    if normalized not in PRODUCTS:
        raise ValueError(f"Unsupported product {code!r}. Supported: {', '.join(sorted(PRODUCTS))}")
    return PRODUCTS[normalized]
