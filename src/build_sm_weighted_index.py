from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.build_weighted_index import build_product_index


def main() -> None:
    build_product_index("SM")


if __name__ == "__main__":
    main()
