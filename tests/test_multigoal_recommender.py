import pytest

from mocrop.data_loader import load_data_bundle
from src.recommender import (
    calculate_total_score,
    normalize_weights,
    score_ecological_adaptation,
    score_economic_return,
    score_labor_adaptation,
    score_risk_penalty,
    score_tourism_experience,
    RecommendationEngine,
)


def test_total_score_uses_requested_weights():
    total, breakdown = calculate_total_score(
        "生态优先模式",
        ecological_adaptation_score=80,
        economic_return_score=40,
        tourism_experience_score=60,
        labor_adaptation_score=70,
        risk_penalty_score=10,
    )

    assert total == 57
    assert breakdown["生态权重"] == 0.45
    assert breakdown["经济权重"] == 0.20
    assert breakdown["农旅权重"] == 0.15
    assert breakdown["劳动力权重"] == 0.20


def test_scoring_functions_have_docstrings():
    scoring_functions = [
        score_ecological_adaptation,
        score_economic_return,
        score_tourism_experience,
        score_labor_adaptation,
        score_risk_penalty,
    ]

    assert all(func.__doc__ for func in scoring_functions)


def test_recommendation_contains_score_breakdown():
    data = load_data_bundle()
    engine = RecommendationEngine(data.crops, data.market_prices, data.labor)

    recommendation = engine.recommend_for_parcel(data.parcels[0])["生态优先模式"]

    assert recommendation.score_breakdown["生态适配分"] == recommendation.ecological_score
    assert "经济收益分" in recommendation.score_breakdown
    assert "农旅体验分" in recommendation.score_breakdown
    assert "劳动力适配分" in recommendation.score_breakdown
    assert "风险惩罚分" in recommendation.score_breakdown
    assert recommendation.score_breakdown["总分"] == recommendation.score


def test_high_slope_high_erosion_reason_is_explainable():
    data = load_data_bundle()
    engine = RecommendationEngine(data.crops, data.market_prices, data.labor)

    recommendation = engine.recommend_for_parcel(data.parcels[0])["生态优先模式"]
    reasons = "；".join(recommendation.reasons)

    assert "坡度较高且水土流失风险较大" in reasons
    assert "根系固土能力强" in reasons


def test_custom_weight_override_is_reflected_in_breakdown():
    data = load_data_bundle()
    engine = RecommendationEngine(data.crops, data.market_prices, data.labor)
    custom_weights = normalize_weights(
        {
            "生态适配分": 0.10,
            "经济收益分": 0.10,
            "农旅体验分": 0.70,
            "劳动力适配分": 0.10,
        }
    )

    recommendation = engine.recommend_for_parcel(
        data.parcels[4],
        mode_weight_overrides={"农旅融合模式": custom_weights},
    )["农旅融合模式"]

    assert recommendation.score_breakdown["农旅权重"] == pytest.approx(0.70)
    assert recommendation.score_breakdown["生态权重"] == pytest.approx(0.10)
