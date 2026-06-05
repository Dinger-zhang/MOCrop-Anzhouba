from mocrop.data_loader import load_data_bundle
from mocrop.recommender import MODES, RecommendationEngine


def test_recommend_for_parcel_returns_three_modes():
    data = load_data_bundle()
    engine = RecommendationEngine(data.crops, data.market_prices, data.labor)

    result = engine.recommend_for_parcel(data.parcels[0])

    assert set(result) == set(MODES)
    for recommendation in result.values():
        assert recommendation.crop_name
        assert recommendation.expected_revenue_yuan >= 0
        assert recommendation.labor_days >= 0
        assert 0 <= recommendation.ecological_score <= 100
        assert 0 <= recommendation.tourism_score <= 100


def test_recommend_all_has_one_plan_per_mode_per_parcel():
    data = load_data_bundle()
    engine = RecommendationEngine(data.crops, data.market_prices, data.labor)

    recommendations = engine.recommend_all(data.parcels)

    assert len(recommendations) == len(data.parcels) * len(MODES)
    assert all(item.risk_level in {"低", "中", "高"} for item in recommendations)
