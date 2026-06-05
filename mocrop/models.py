from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["低", "中", "高"]
ModeName = Literal["生态优先模式", "经济优先模式", "农旅融合模式"]


class Parcel(BaseModel):
    parcel_id: str
    name: str
    area_mu: float = Field(gt=0)
    latitude: float
    longitude: float
    slope_deg: float = Field(ge=0, le=60)
    altitude_m: float = Field(ge=0)
    soil_depth_cm: float = Field(ge=0)
    water_access: float = Field(ge=0, le=1)
    sunlight: float = Field(ge=0, le=1)
    distance_to_road_km: float = Field(ge=0)
    erosion_risk: float = Field(ge=0, le=100)
    soil_fertility: float = Field(ge=0, le=100)
    tourism_accessibility: float = Field(ge=0, le=100)
    idle_years: int = Field(ge=0)
    current_status: str


class Crop(BaseModel):
    crop_id: str
    name: str
    category: str
    max_slope_deg: float = Field(ge=0)
    min_soil_depth_cm: float = Field(ge=0)
    water_need: float = Field(ge=0, le=100)
    sunlight_need: float = Field(ge=0, le=100)
    altitude_min_m: float = Field(ge=0)
    altitude_max_m: float = Field(ge=0)
    input_cost_per_mu: float = Field(ge=0)
    expected_yield_kg_per_mu: float = Field(ge=0)
    labor_days_per_mu: float = Field(ge=0)
    ecological_score: float = Field(ge=0, le=100)
    tourism_score: float = Field(ge=0, le=100)
    market_preference: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    cycle_years: int = Field(ge=1)
    reason_tags: str


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
    tourism_score: float = Field(ge=0, le=100)
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
