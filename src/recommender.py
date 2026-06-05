from __future__ import annotations

from dataclasses import dataclass

from mocrop.models import Crop, LaborAvailability, ModeName, Parcel, Recommendation, RiskLevel


MODES: tuple[ModeName, ...] = ("生态优先模式", "经济优先模式", "农旅融合模式")
SCORE_KEYS: tuple[str, ...] = ("生态适配分", "经济收益分", "农旅体验分", "劳动力适配分")

MODE_WEIGHTS: dict[ModeName, dict[str, float]] = {
    "生态优先模式": {
        "生态适配分": 0.45,
        "经济收益分": 0.20,
        "农旅体验分": 0.15,
        "劳动力适配分": 0.20,
    },
    "经济优先模式": {
        "生态适配分": 0.20,
        "经济收益分": 0.45,
        "农旅体验分": 0.15,
        "劳动力适配分": 0.20,
    },
    "农旅融合模式": {
        "生态适配分": 0.20,
        "经济收益分": 0.20,
        "农旅体验分": 0.45,
        "劳动力适配分": 0.15,
    },
}

EROSION_RISK_SCORE = {"low": 20.0, "medium": 60.0, "high": 90.0}
RISK_PENALTY_BASE = {"low": 3.0, "medium": 8.0, "high": 14.0}
LABOR_DAYS_PER_MU = {"low": 5.0, "medium": 12.0, "high": 20.0}
LABOR_WORKERS_PER_10_MU = {"low": 2.0, "medium": 4.0, "high": 7.0}
WATER_DISTANCE_TOLERANCE = {"low": 800.0, "medium": 450.0, "high": 220.0}
LAND_TYPE_TOURISM_BASE = {
    "homestay_nearby": 92.0,
    "roadside": 84.0,
    "abandoned": 64.0,
    "forest_understory": 58.0,
    "slope": 48.0,
}


@dataclass(frozen=True)
class CropEvaluation:
    crop: Crop
    ecological_adaptation_score: float
    economic_return_score: float
    tourism_experience_score: float
    labor_adaptation_score: float
    risk_penalty_score: float
    expected_investment_yuan: float
    expected_revenue_yuan: float
    expected_profit_yuan: float
    profit_per_mu: float
    labor_days: float
    risk_level: RiskLevel


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """将评分限制在指定区间内，避免异常输入导致图表或模型输出越界。"""

    return max(low, min(high, value))


def score_ecological_adaptation(parcel: Parcel, crop: Crop) -> float:
    """计算生态适配分，综合地类匹配、坡度承载、土壤 pH、有机质、水源距离和水土保持能力。

    分值范围为 0-100。高坡度、高水土流失风险地块会更偏向根系固土能力强、
    坡度耐受高且水分需求较低的作物。
    """

    land_type_score = 100.0 if parcel.land_type in crop.suitable_land_type_set else 45.0
    slope_score = 100.0 if parcel.slope_degree <= crop.slope_tolerance else max(
        0.0,
        100 - (parcel.slope_degree - crop.slope_tolerance) * 6,
    )
    ph_score = _score_ph(parcel.soil_ph, crop.ph_min, crop.ph_max)
    organic_score = clamp(parcel.organic_matter / 35 * 100)
    water_score = _score_water_distance(parcel.water_distance_m, crop.water_need_level)
    sunlight_score = _score_sunlight(parcel.sunlight_hours, parcel.land_type)
    conservation_need = EROSION_RISK_SCORE[parcel.erosion_risk]
    conservation_score = crop.soil_conservation_score * 0.65 + conservation_need * 0.35

    return round(
        clamp(
            land_type_score * 0.18
            + slope_score * 0.18
            + ph_score * 0.16
            + organic_score * 0.12
            + water_score * 0.12
            + sunlight_score * 0.08
            + conservation_score * 0.16
        ),
        2,
    )


def score_economic_return(profit_per_mu: float, min_profit: float, max_profit: float) -> float:
    """计算经济收益分，将亩均净收益在候选作物中归一化到 0-100。

    归一化使不同量级的作物收益可与生态、农旅和劳动力分在同一尺度下比较。
    如果所有候选作物收益相同，则返回中性分 50。
    """

    if max_profit == min_profit:
        return 50.0
    return round(clamp((profit_per_mu - min_profit) / (max_profit - min_profit) * 100), 2)


def score_tourism_experience(parcel: Parcel, crop: Crop) -> float:
    """计算农旅体验分，综合作物观赏/采摘/研学价值和地块旅居可达性。

    靠近民宿、村道或可形成打卡景观的地块会获得更高地块端分值；
    作物端以 tourism_score 表示体验产品转化潜力。
    """

    land_base = LAND_TYPE_TOURISM_BASE[parcel.land_type]
    sunlight_bonus = 8.0 if parcel.sunlight_hours >= 7.0 else 0.0
    water_bonus = 6.0 if parcel.water_distance_m <= 250 else 0.0
    land_experience_score = clamp(land_base + sunlight_bonus + water_bonus)
    return round(clamp(crop.tourism_score * 0.68 + land_experience_score * 0.32), 2)


def score_labor_adaptation(parcel: Parcel, crop: Crop) -> float:
    """计算劳动力适配分，比较地块可用劳动力与作物管护强度。

    分值越高表示村内劳动力越能覆盖整地、栽植、管护、采收和体验接待等需求。
    高劳动强度作物在劳动力不足地块上会显著降分。
    """

    required_workers = LABOR_WORKERS_PER_10_MU[crop.labor_intensity] * parcel.area_mu / 10
    if required_workers <= 0:
        return 100.0
    coverage = parcel.labor_available / required_workers
    if coverage >= 1:
        surplus_bonus = min(8.0, (coverage - 1) * 8)
        return round(clamp(92 + surplus_bonus), 2)
    return round(clamp(coverage * 92), 2)


def score_risk_penalty(parcel: Parcel, crop: Crop) -> float:
    """计算风险惩罚分，覆盖作物风险、水源约束、坡度超限、pH 偏离和生态脆弱性。

    风险惩罚分直接从加权总分中扣除，不再作为正向指标参与加权。
    分值越高代表越需要降低推荐优先级。
    """

    penalty = RISK_PENALTY_BASE[crop.risk_level]
    if parcel.water_distance_m > WATER_DISTANCE_TOLERANCE[crop.water_need_level]:
        penalty += min(
            8.0,
            (parcel.water_distance_m - WATER_DISTANCE_TOLERANCE[crop.water_need_level]) / 90,
        )
    if parcel.slope_degree > crop.slope_tolerance:
        penalty += min(10.0, (parcel.slope_degree - crop.slope_tolerance) * 0.9)
    if not (crop.ph_min <= parcel.soil_ph <= crop.ph_max):
        penalty += min(6.0, _ph_gap(parcel.soil_ph, crop.ph_min, crop.ph_max) * 2.2)
    if parcel.erosion_risk == "high" and crop.soil_conservation_score < 70:
        penalty += 6.0
    if parcel.labor_available < 4 and crop.labor_intensity == "high":
        penalty += 5.0
    return round(clamp(penalty, 0, 35), 2)


def calculate_total_score(
    mode: ModeName,
    ecological_adaptation_score: float,
    economic_return_score: float,
    tourism_experience_score: float,
    labor_adaptation_score: float,
    risk_penalty_score: float,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """按模式权重计算多目标总分，并返回可展示的评分拆解。

    总分 = 生态适配分 * 生态权重 + 经济收益分 * 经济权重
    + 农旅体验分 * 农旅权重 + 劳动力适配分 * 劳动力权重 - 风险惩罚分。
    """

    weights = normalize_weights(weights or MODE_WEIGHTS[mode])
    ecological_weighted = ecological_adaptation_score * weights["生态适配分"]
    economic_weighted = economic_return_score * weights["经济收益分"]
    tourism_weighted = tourism_experience_score * weights["农旅体验分"]
    labor_weighted = labor_adaptation_score * weights["劳动力适配分"]
    raw_total = (
        ecological_weighted
        + economic_weighted
        + tourism_weighted
        + labor_weighted
        - risk_penalty_score
    )
    total = round(clamp(raw_total), 2)
    return total, {
        "生态适配分": round(ecological_adaptation_score, 2),
        "生态权重": weights["生态适配分"],
        "生态加权分": round(ecological_weighted, 2),
        "经济收益分": round(economic_return_score, 2),
        "经济权重": weights["经济收益分"],
        "经济加权分": round(economic_weighted, 2),
        "农旅体验分": round(tourism_experience_score, 2),
        "农旅权重": weights["农旅体验分"],
        "农旅加权分": round(tourism_weighted, 2),
        "劳动力适配分": round(labor_adaptation_score, 2),
        "劳动力权重": weights["劳动力适配分"],
        "劳动力加权分": round(labor_weighted, 2),
        "风险惩罚分": round(risk_penalty_score, 2),
        "总分": total,
    }


class RecommendationEngine:
    """可解释多目标推荐引擎。"""

    def __init__(
        self,
        crops: list[Crop],
        market_prices: object | None = None,
        labor: list[LaborAvailability] | None = None,
    ) -> None:
        self.crops = crops
        self.labor = labor or []
        self.average_wage = self._average_wage()

    def recommend_for_parcel(
        self,
        parcel: Parcel,
        mode_weight_overrides: dict[ModeName, dict[str, float]] | None = None,
    ) -> dict[ModeName, Recommendation]:
        evaluations = self._evaluate_candidates(parcel)
        results: dict[ModeName, Recommendation] = {}
        weight_overrides = mode_weight_overrides or {}
        for mode in MODES:
            best_evaluation, score, breakdown = max(
                (
                    (
                        evaluation,
                        *calculate_total_score(
                            mode,
                            evaluation.ecological_adaptation_score,
                            evaluation.economic_return_score,
                            evaluation.tourism_experience_score,
                            evaluation.labor_adaptation_score,
                            evaluation.risk_penalty_score,
                            weight_overrides.get(mode),
                        ),
                    )
                    for evaluation in evaluations
                ),
                key=lambda item: item[1],
            )
            results[mode] = self._to_recommendation(parcel, best_evaluation, mode, score, breakdown)
        return results

    def recommend_all(
        self,
        parcels: list[Parcel],
        mode_weight_overrides: dict[ModeName, dict[str, float]] | None = None,
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        for parcel in parcels:
            recommendations.extend(
                self.recommend_for_parcel(parcel, mode_weight_overrides).values()
            )
        return recommendations

    def _evaluate_candidates(self, parcel: Parcel) -> list[CropEvaluation]:
        raw = [self._evaluate_raw(parcel, crop) for crop in self.crops]
        min_profit = min(item.profit_per_mu for item in raw)
        max_profit = max(item.profit_per_mu for item in raw)
        return [
            CropEvaluation(
                crop=item.crop,
                ecological_adaptation_score=item.ecological_adaptation_score,
                economic_return_score=score_economic_return(
                    item.profit_per_mu,
                    min_profit,
                    max_profit,
                ),
                tourism_experience_score=item.tourism_experience_score,
                labor_adaptation_score=item.labor_adaptation_score,
                risk_penalty_score=item.risk_penalty_score,
                expected_investment_yuan=item.expected_investment_yuan,
                expected_revenue_yuan=item.expected_revenue_yuan,
                expected_profit_yuan=item.expected_profit_yuan,
                profit_per_mu=item.profit_per_mu,
                labor_days=item.labor_days,
                risk_level=item.risk_level,
            )
            for item in raw
        ]

    def _evaluate_raw(self, parcel: Parcel, crop: Crop) -> CropEvaluation:
        labor_days = LABOR_DAYS_PER_MU[crop.labor_intensity] * parcel.area_mu
        labor_cost = labor_days * self.average_wage
        input_cost = crop.input_cost_per_mu * parcel.area_mu
        expected_investment = input_cost + labor_cost
        expected_revenue = crop.expected_yield_per_mu * crop.market_price * parcel.area_mu
        expected_profit = expected_revenue - expected_investment

        return CropEvaluation(
            crop=crop,
            ecological_adaptation_score=score_ecological_adaptation(parcel, crop),
            economic_return_score=0,
            tourism_experience_score=score_tourism_experience(parcel, crop),
            labor_adaptation_score=score_labor_adaptation(parcel, crop),
            risk_penalty_score=score_risk_penalty(parcel, crop),
            expected_investment_yuan=expected_investment,
            expected_revenue_yuan=expected_revenue,
            expected_profit_yuan=expected_profit,
            profit_per_mu=expected_profit / parcel.area_mu,
            labor_days=labor_days,
            risk_level=self._risk_level(score_risk_penalty(parcel, crop)),
        )

    def _to_recommendation(
        self,
        parcel: Parcel,
        evaluation: CropEvaluation,
        mode: ModeName,
        score: float,
        breakdown: dict[str, float],
    ) -> Recommendation:
        crop = evaluation.crop
        return Recommendation(
            parcel_id=parcel.parcel_id,
            parcel_name=parcel.name,
            mode=mode,
            crop_id=crop.crop_id,
            crop_name=crop.crop_name,
            crop_category=crop.crop_category,
            score=score,
            reasons=self._build_reasons(parcel, evaluation, mode, score),
            expected_investment_yuan=round(evaluation.expected_investment_yuan, 2),
            expected_revenue_yuan=round(evaluation.expected_revenue_yuan, 2),
            expected_profit_yuan=round(evaluation.expected_profit_yuan, 2),
            labor_days=round(evaluation.labor_days, 1),
            ecological_score=round(evaluation.ecological_adaptation_score, 2),
            economic_return_score=round(evaluation.economic_return_score, 2),
            tourism_score=round(evaluation.tourism_experience_score, 2),
            labor_adaptation_score=round(evaluation.labor_adaptation_score, 2),
            risk_penalty_score=round(evaluation.risk_penalty_score, 2),
            score_breakdown=breakdown,
            risk_level=evaluation.risk_level,
        )

    def _build_reasons(
        self,
        parcel: Parcel,
        evaluation: CropEvaluation,
        mode: ModeName,
        score: float,
    ) -> list[str]:
        crop = evaluation.crop
        reasons = [
            f"{mode}采用多目标权重，当前作物总分 {score:.1f}。",
        ]
        if parcel.slope_degree >= 18 and parcel.erosion_risk in {"medium", "high"}:
            if crop.soil_conservation_score >= 75:
                reasons.append(
                    "该地块坡度较高且水土流失风险较大，因此优先推荐根系固土能力强的作物。"
                )
            else:
                reasons.append("该地块坡度和水土流失风险偏高，建议加强覆盖、截排水和管护。")
        if parcel.land_type in crop.suitable_land_type_set:
            reasons.append(f"地块类型 {land_type_label(parcel.land_type)} 与作物适宜地类匹配。")
        if crop.ph_min <= parcel.soil_ph <= crop.ph_max:
            reasons.append(f"土壤 pH {parcel.soil_ph:.1f} 位于作物适宜区间。")
        if evaluation.expected_profit_yuan > 0:
            reasons.append(f"预计净收益约 {evaluation.expected_profit_yuan:,.0f} 元，具备经营转化空间。")
        if evaluation.tourism_experience_score >= 78:
            reasons.append("农旅体验分较高，可结合采摘、研学、花境打卡或民宿套餐设计。")
        if evaluation.labor_adaptation_score >= 80:
            reasons.append("可用劳动力基本覆盖该作物管护强度。")
        if evaluation.risk_penalty_score >= 16:
            reasons.append("风险惩罚偏高，落地时应先小规模试种并锁定订单或技术服务。")
        reasons.append(f"作物特征：{crop.reason_tags.replace(';', '、')}")
        return reasons

    def _average_wage(self) -> float:
        total_days = sum(item.available_workers * item.available_days for item in self.labor)
        if total_days <= 0:
            return 165.0
        weighted_wage = sum(
            item.available_workers * item.available_days * item.average_wage_yuan_per_day
            for item in self.labor
        )
        return weighted_wage / total_days

    @staticmethod
    def _risk_level(risk_penalty_score: float) -> RiskLevel:
        if risk_penalty_score < 8:
            return "低"
        if risk_penalty_score < 16:
            return "中"
        return "高"


def evaluation_score_text(evaluation: CropEvaluation, mode: ModeName) -> str:
    """生成用于解释文本的总分字符串，避免理由中出现与推荐排序不一致的临时分值。"""

    score, _ = calculate_total_score(
        mode,
        evaluation.ecological_adaptation_score,
        evaluation.economic_return_score,
        evaluation.tourism_experience_score,
        evaluation.labor_adaptation_score,
        evaluation.risk_penalty_score,
    )
    return f"{score:.1f}"


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """归一化四项目标权重，使滑块输入始终能按同一尺度参与总分计算。"""

    total = sum(max(0.0, weights.get(key, 0.0)) for key in SCORE_KEYS)
    if total <= 0:
        return MODE_WEIGHTS["生态优先模式"].copy()
    return {key: max(0.0, weights.get(key, 0.0)) / total for key in SCORE_KEYS}


def land_type_label(land_type: str) -> str:
    """将地块类型编码转换为中文标签，便于页面和报告解释。"""

    return {
        "slope": "坡地",
        "forest_understory": "林下地",
        "abandoned": "闲置撂荒地",
        "roadside": "道路沿线地",
        "homestay_nearby": "民宿周边地",
    }.get(land_type, land_type)


def erosion_risk_label(erosion_risk: str) -> str:
    """将水土流失风险编码转换为中文标签。"""

    return {"low": "低", "medium": "中", "high": "高"}.get(erosion_risk, erosion_risk)


def _score_ph(soil_ph: float, ph_min: float, ph_max: float) -> float:
    if ph_min <= soil_ph <= ph_max:
        return 100.0
    return clamp(100 - _ph_gap(soil_ph, ph_min, ph_max) * 28)


def _ph_gap(soil_ph: float, ph_min: float, ph_max: float) -> float:
    if soil_ph < ph_min:
        return ph_min - soil_ph
    if soil_ph > ph_max:
        return soil_ph - ph_max
    return 0.0


def _score_water_distance(water_distance_m: float, water_need_level: str) -> float:
    tolerance = WATER_DISTANCE_TOLERANCE[water_need_level]
    if water_distance_m <= tolerance:
        return 100.0
    return clamp(100 - (water_distance_m - tolerance) / tolerance * 80)


def _score_sunlight(sunlight_hours: float, land_type: str) -> float:
    if land_type == "forest_understory":
        target = 4.5
        return clamp(100 - abs(sunlight_hours - target) * 15)
    target = 7.5
    return clamp(100 - abs(sunlight_hours - target) * 12)
