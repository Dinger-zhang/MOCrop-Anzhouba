from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .models import Crop, Homestay, LaborAvailability, MarketPrice, Parcel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DataBundle:
    parcels: list[Parcel]
    crops: list[Crop]
    market_prices: list[MarketPrice]
    labor: list[LaborAvailability]
    homestays: list[Homestay]


def _read_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"演示数据文件不存在: {path}")
    return pd.read_csv(path)


def load_parcels() -> list[Parcel]:
    return [Parcel(**row) for row in _read_csv("parcels.csv").to_dict("records")]


def load_crops() -> list[Crop]:
    return [Crop(**row) for row in _read_csv("crops.csv").to_dict("records")]


def load_market_prices() -> list[MarketPrice]:
    return [MarketPrice(**row) for row in _read_csv("market_prices.csv").to_dict("records")]


def load_labor() -> list[LaborAvailability]:
    return [LaborAvailability(**row) for row in _read_csv("labor.csv").to_dict("records")]


def load_homestays() -> list[Homestay]:
    return [Homestay(**row) for row in _read_csv("homestays.csv").to_dict("records")]


def load_data_bundle() -> DataBundle:
    return DataBundle(
        parcels=load_parcels(),
        crops=load_crops(),
        market_prices=load_market_prices(),
        labor=load_labor(),
        homestays=load_homestays(),
    )


def recommendations_to_dataframe(recommendations) -> pd.DataFrame:
    rows = []
    for rec in recommendations:
        item = rec.model_dump()
        item["推荐理由"] = "；".join(item.pop("reasons"))
        rows.append(item)
    return pd.DataFrame(rows)
