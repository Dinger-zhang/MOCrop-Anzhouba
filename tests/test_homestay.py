from mocrop.data_loader import load_homestays
from mocrop.homestay import calculate_homestay_projection, calculate_portfolio_projection


def test_calculate_homestay_projection_has_positive_income():
    homestay = load_homestays()[0]

    projection = calculate_homestay_projection(homestay)

    assert projection.annual_revenue_yuan > 0
    assert projection.net_income_yuan > 0
    assert projection.village_share_yuan > 0
    assert projection.payback_years > 0


def test_calculate_portfolio_projection_keeps_occupancy_bounds():
    homestays = load_homestays()

    projections = calculate_portfolio_projection(homestays, occupancy_delta=0.8)

    assert len(projections) == len(homestays)
    assert all(0 <= item.occupancy_rate <= 0.95 for item in projections)
