from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Crop, LaborAvailability, MarketPrice, ModeName, Parcel, Recommendation, RiskLevel


MODES: tuple[ModeName, ...] = ("生态优先模式", "经济优先模式", "农旅融合模式")

MODE_CATEGORY_BONUS: dict[ModeName, set[str]] = {
    "生态优先模式": {"生态修复"},
    "经济优先模式": {"中药材", "食用菌", "坚果油料"},
    "农旅融合模式": {"农旅体验", "中药材"},
}

RISK_BASE = {"低": 18, "中": 42, "高": 68}


@dataclass(frozen=True)
class CropEvaluation:
    crop: Crop
    suitability_score: float
    profit_per_mu: float
    expected_investment_yuan: float
    expected_revenue_yuan: float
    expected_profit_yuan: float
    labor_days: float
    ecological_score: float
    tourism_score: float
    demand_score: float
    risk_points: float
    risk_level: RiskLevel
    road_score: float


class RecommendationEngine:
    """规则评分 + 加权推荐引擎。

    输入地块条件、作物知识库、市场价格与劳动力数据，输出三类策略推荐。
    """

    def __init__(
        self,
        crops: list[Crop],
        market_prices: list[MarketPrice],
        labor: list[LaborAvailability],
    ) -> None:
        self.crops = crops
        self.price_by_crop = {price.crop_id: price for price in market_prices}
        self.labor = labor
        self.average_wage = self._average_wage()
        self.annual_labor_capacity_days = sum(
            item.available_workers * item.available_days for item in labor
        )

    def recommend_for_parcel(self, parcel: Parcel) -> dict[ModeName, Recommendation]:
        evaluations = [self.evaluate_crop(parcel, crop) for crop in self.crops]
        profit_values = [item.profit_per_mu for item in evaluations]
        min_profit = min(profit_values)
        max_profit = max(profit_values)

        results: dict[ModeName, Recommendation] = {}
        for mode in MODES:
            ranked = sorted(
                evaluations,
                key=lambda item: self._mode_score(
                    mode=mode,
                    evaluation=item,
                    profit_norm=self._normalize(item.profit_per_mu, min_profit, max_profit),
                ),
                reverse=True,
            )
            best = ranked[0]
            score = self._mode_score(
                mode=mode,
                evaluation=best,
                profit_norm=self._normalize(best.profit_per_mu, min_profit, max_profit),
            )
            results[mode] = Recommendation(
                parcel_id=parcel.parcel_id,
                parcel_name=parcel.name,
                mode=mode,
                crop_id=best.crop.crop_id,
                crop_name=best.crop.name,
                crop_category=best.crop.category,
                score=round(score, 2),
                reasons=self._build_reasons(parcel, best, mode),
                expected_investment_yuan=round(best.expected_investment_yuan, 2),
                expected_revenue_yuan=round(best.expected_revenue_yuan, 2),
                expected_profit_yuan=round(best.expected_profit_yuan, 2),
                labor_days=round(best.labor_days, 1),
                ecological_score=round(best.ecological_score, 1),
                tourism_score=round(best.tourism_score, 1),
                risk_level=best.risk_level,
            )
        return results

    def recommend_all(self, parcels: list[Parcel]) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        for parcel in parcels:
            recommendations.extend(self.recommend_for_parcel(parcel).values())
        return recommendations

    def evaluate_crop(self, parcel: Parcel, crop: Crop) -> CropEvaluation:
        price = self.price_by_crop[crop.crop_id]
        area = parcel.area_mu

        slope_score = self._slope_score(parcel.slope_deg, crop.max_slope_deg)
        soil_score = min(100.0, parcel.soil_depth_cm / crop.min_soil_depth_cm * 100)
        water_score = self._need_score(parcel.water_access * 100, crop.water_need)
        sunlight_score = self._need_score(parcel.sunlight * 100, crop.sunlight_need)
        altitude_score = self._altitude_score(
            parcel.altitude_m,
            crop.altitude_min_m,
            crop.altitude_max_m,
        )
        road_score = max(0.0, 100 - parcel.distance_to_road_km * 18)

        suitability_score = (
            slope_score * 0.20
            + soil_score * 0.20
            + water_score * 0.20
            + sunlight_score * 0.15
            + altitude_score * 0.15
            + road_score * 0.10
        )

        labor_days = crop.labor_days_per_mu * area
        labor_cost = labor_days * self.average_wage
        input_cost = crop.input_cost_per_mu * area
        expected_investment = input_cost + labor_cost
        expected_revenue = crop.expected_yield_kg_per_mu * price.price_yuan_per_kg * area
        expected_profit = expected_revenue - expected_investment

        ecological_score = np.clip(
            crop.ecological_score * 0.70
            + parcel.erosion_risk * 0.20
            + (100 - parcel.soil_fertility) * 0.10,
            0,
            100,
        )
        tourism_score = np.clip(
            crop.tourism_score * 0.60
            + parcel.tourism_accessibility * 0.30
            + road_score * 0.10,
            0,
            100,
        )
        water_deficit = max(0.0, crop.water_need - parcel.water_access * 100)
        slope_excess = max(0.0, parcel.slope_deg - crop.max_slope_deg)
        risk_points = np.clip(
            RISK_BASE[crop.risk_level]
            + water_deficit * 0.25
            + slope_excess * 2.0
            + price.price_volatility * 0.18,
            0,
            100,
        )

        return CropEvaluation(
            crop=crop,
            suitability_score=float(np.clip(suitability_score, 0, 100)),
            profit_per_mu=expected_profit / area,
            expected_investment_yuan=expected_investment,
            expected_revenue_yuan=expected_revenue,
            expected_profit_yuan=expected_profit,
            labor_days=labor_days,
            ecological_score=float(ecological_score),
            tourism_score=float(tourism_score),
            demand_score=price.demand_score,
            risk_points=float(risk_points),
            risk_level=self._risk_level(float(risk_points)),
            road_score=road_score,
        )

    def _mode_score(
        self,
        mode: ModeName,
        evaluation: CropEvaluation,
        profit_norm: float,
    ) -> float:
        labor_fit = self._labor_fit(evaluation.labor_days)
        risk_inverse = 100 - evaluation.risk_points
        category_bonus = (
            8.0 if evaluation.crop.category in MODE_CATEGORY_BONUS[mode] else 0.0
        )

        if mode == "生态优先模式":
            score = (
                evaluation.suitability_score * 0.34
                + evaluation.ecological_score * 0.38
                + profit_norm * 0.08
                + risk_inverse * 0.10
                + labor_fit * 0.05
                + evaluation.demand_score * 0.05
            )
        elif mode == "经济优先模式":
            score = (
                evaluation.suitability_score * 0.24
                + profit_norm * 0.38
                + evaluation.demand_score * 0.18
                + risk_inverse * 0.10
                + labor_fit * 0.10
            )
        else:
            score = (
                evaluation.suitability_score * 0.23
                + evaluation.tourism_score * 0.36
                + evaluation.ecological_score * 0.14
                + profit_norm * 0.15
                + evaluation.road_score * 0.07
                + risk_inverse * 0.05
            )
        return float(np.clip(score + category_bonus, 0, 120))

    def _build_reasons(
        self,
        parcel: Parcel,
        evaluation: CropEvaluation,
        mode: ModeName,
    ) -> list[str]:
        crop = evaluation.crop
        reasons = [
            f"{mode}权重下综合得分 {round(self._safe_score(evaluation), 1)}，适配{parcel.name}的{parcel.current_status}特征",
        ]
        if evaluation.suitability_score >= 80:
            reasons.append("坡度、水分、光照和海拔匹配度高")
        elif evaluation.suitability_score >= 65:
            reasons.append("基础地力条件可满足种植，需配套轻量管护")
        else:
            reasons.append("存在一定地块约束，建议小规模试种后扩面")

        if evaluation.ecological_score >= 80:
            reasons.append("固土保水和生态修复价值突出")
        if evaluation.tourism_score >= 78:
            reasons.append("具备观赏、采摘或研学体验转化空间")
        if evaluation.expected_profit_yuan > 0:
            reasons.append(f"预计年净收益约 {evaluation.expected_profit_yuan:,.0f} 元")
        reasons.append(f"作物特征：{crop.reason_tags.replace(';', '、')}")
        reasons.append(f"风险等级为{evaluation.risk_level}，需关注价格波动与季节性管护")
        return reasons

    def _safe_score(self, evaluation: CropEvaluation) -> float:
        return (
            evaluation.suitability_score * 0.4
            + evaluation.ecological_score * 0.25
            + evaluation.tourism_score * 0.2
            + (100 - evaluation.risk_points) * 0.15
        )

    def _average_wage(self) -> float:
        total_days = sum(item.available_workers * item.available_days for item in self.labor)
        if total_days <= 0:
            return 0.0
        weighted_wage = sum(
            item.available_workers * item.available_days * item.average_wage_yuan_per_day
            for item in self.labor
        )
        return weighted_wage / total_days

    def _labor_fit(self, labor_days: float) -> float:
        if labor_days <= 0:
            return 100.0
        if self.annual_labor_capacity_days <= 0:
            return 0.0
        ratio = labor_days / self.annual_labor_capacity_days
        return float(np.clip(100 - ratio * 180, 0, 100))

    @staticmethod
    def _normalize(value: float, min_value: float, max_value: float) -> float:
        if max_value == min_value:
            return 50.0
        return float((value - min_value) / (max_value - min_value) * 100)

    @staticmethod
    def _need_score(available: float, required: float) -> float:
        if required <= 0:
            return 100.0
        if available >= required:
            surplus_penalty = min(15.0, (available - required) * 0.15)
            return 100.0 - surplus_penalty
        return max(0.0, available / required * 100)

    @staticmethod
    def _slope_score(slope: float, max_slope: float) -> float:
        if slope <= max_slope:
            return max(75.0, 100 - slope / max(max_slope, 1) * 15)
        return max(0.0, 75 - (slope - max_slope) * 8)

    @staticmethod
    def _altitude_score(altitude: float, min_altitude: float, max_altitude: float) -> float:
        if min_altitude <= altitude <= max_altitude:
            return 100.0
        if altitude < min_altitude:
            return max(0.0, 100 - (min_altitude - altitude) * 0.25)
        return max(0.0, 100 - (altitude - max_altitude) * 0.25)

    @staticmethod
    def _risk_level(points: float) -> RiskLevel:
        if points < 38:
            return "低"
        if points < 62:
            return "中"
        return "高"
