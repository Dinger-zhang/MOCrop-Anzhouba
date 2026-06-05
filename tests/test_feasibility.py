from datetime import datetime

from mocrop.data_loader import load_data_bundle
from mocrop.homestay import calculate_portfolio_projection
from src.feasibility import (
    build_feasibility_result,
    save_feasibility_report,
    simulate_anzhouba_parcels,
)
from src.recommender import RecommendationEngine


def test_simulate_anzhouba_parcels_returns_ten_samples():
    parcels = simulate_anzhouba_parcels()

    assert len(parcels) == 10
    assert all(parcel.name for parcel in parcels)


def test_build_feasibility_result_contains_summary_and_report():
    data = load_data_bundle()
    engine = RecommendationEngine(data.crops, data.market_prices, data.labor)
    homestays = calculate_portfolio_projection(data.homestays)

    result = build_feasibility_result(engine, homestays)

    assert len(result.parcels) == 10
    assert len(result.recommendations) == 30
    assert result.summary["总种植面积"] > 0
    assert result.summary["总收入"] > 0
    assert result.summary["参与就业人数"] > 0
    assert result.summary["女性/中老年岗位数量"] > 0
    assert "项目可行性证明报告" in result.markdown
    assert "数据来源说明" in result.markdown
    assert "风险与应对" in result.markdown


def test_save_feasibility_report_filename(tmp_path):
    path = save_feasibility_report("# test", output_dir=tmp_path)
    today = datetime.now().strftime("%Y%m%d")

    assert path.name == f"mocrop_feasibility_report_{today}.md"
    assert path.read_text(encoding="utf-8") == "# test"
