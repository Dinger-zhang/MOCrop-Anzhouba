from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["低", "中", "高"]
ModeName = Literal["生态优先模式", "经济优先模式", "农旅融合模式"]
LandType = Literal["slope", "forest_understory", "abandoned", "roadside", "homestay_nearby"]
ErosionRisk = Literal["low", "medium", "high"]
NeedLevel = Literal["low", "medium", "high"]


class Parcel(BaseModel):
    parcel_id: str
    name: str
    area_mu: float = Field(gt=0)
    latitude: float
    longitude: float
    slope_degree: float = Field(ge=0, le=60)
    soil_ph: float = Field(ge=3.5, le=10)
    organic_matter: float = Field(ge=0, le=80)
    sunlight_hours: float = Field(ge=0, le=14)
    water_distance_m: float = Field(ge=0)
    land_type: LandType
    erosion_risk: ErosionRisk
    labor_available: int = Field(ge=0)
    idle_years: int = Field(ge=0)
    current_status: str

    @property
    def slope_deg(self) -> float:
        return self.slope_degree

    @property
    def water_access(self) -> float:
        return max(0.0, min(1.0, 1 - self.water_distance_m / 900))

    @property
    def erosion_risk_score(self) -> float:
        return {"low": 25.0, "medium": 60.0, "high": 90.0}[self.erosion_risk]


class Crop(BaseModel):
    crop_id: str
    crop_name: str
    crop_category: str
    suitable_land_types: str
    ph_min: float = Field(ge=3.5, le=10)
    ph_max: float = Field(ge=3.5, le=10)
    slope_tolerance: float = Field(ge=0)
    water_need_level: NeedLevel
    labor_intensity: NeedLevel
    expected_yield_per_mu: float = Field(ge=0)
    market_price: float = Field(ge=0)
    input_cost_per_mu: float = Field(ge=0)
    soil_conservation_score: float = Field(ge=0, le=100)
    tourism_score: float = Field(ge=0, le=100)
    risk_level: NeedLevel
    cycle_years: int = Field(ge=1)
    reason_tags: str

    @property
    def name(self) -> str:
        return self.crop_name

    @property
    def category(self) -> str:
        return self.crop_category

    @property
    def suitable_land_type_set(self) -> set[str]:
        return {item.strip() for item in self.suitable_land_types.split(";") if item.strip()}


class MarketPrice(BaseModel):
    crop_id: str
    price_yuan_per_kg: float = Field(ge=0)
    demand_score: float = Field(ge=0, le=100)
    price_volatility: float = Field(ge=0, le=100)
    source_note: str


class LaborAvailability(BaseModel):
    season: str
    available_workers: int = Field(ge=0)
    available_days: int = Field(ge=0)
    average_wage_yuan_per_day: float = Field(ge=0)
    notes: str


class Homestay(BaseModel):
    house_id: str
    name: str
    rooms: int = Field(gt=0)
    renovation_cost_yuan: float = Field(ge=0)
    base_occupancy_rate: float = Field(ge=0, le=1)
    avg_price_yuan: float = Field(ge=0)
    operating_cost_rate: float = Field(ge=0, le=1)
    village_share_rate: float = Field(ge=0, le=1)
    tourism_score: float = Field(ge=0, le=100)
    status: str


class Recommendation(BaseModel):
    parcel_id: str
    parcel_name: str
    mode: ModeName
    crop_id: str
    crop_name: str
    crop_category: str
    score: float = Field(ge=0, le=120)
    reasons: list[str]
    expected_investment_yuan: float
    expected_revenue_yuan: float
    expected_profit_yuan: float
    labor_days: float
    ecological_score: float = Field(ge=0, le=100)
    economic_return_score: float = Field(ge=0, le=100)
    tourism_score: float = Field(ge=0, le=100)
    labor_adaptation_score: float = Field(ge=0, le=100)
    risk_penalty_score: float = Field(ge=0, le=100)
    score_breakdown: dict[str, float]
    risk_level: RiskLevel


class HomestayProjection(BaseModel):
    house_id: str
    name: str
    rooms: int
    renovation_cost_yuan: float
    occupancy_rate: float = Field(ge=0, le=1)
    avg_price_yuan: float
    annual_revenue_yuan: float
    operating_cost_yuan: float
    net_income_yuan: float
    village_share_yuan: float
    payback_years: float
