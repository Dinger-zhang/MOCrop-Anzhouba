from __future__ import annotations

from .models import Homestay, HomestayProjection


def calculate_homestay_projection(
    homestay: Homestay,
    occupancy_rate: float | None = None,
    avg_price_yuan: float | None = None,
) -> HomestayProjection:
    """测算单个闲置农宅的微旅居年化收益。"""

    occupancy = homestay.base_occupancy_rate if occupancy_rate is None else occupancy_rate
    price = homestay.avg_price_yuan if avg_price_yuan is None else avg_price_yuan
    annual_revenue = homestay.rooms * 365 * occupancy * price
    operating_cost = annual_revenue * homestay.operating_cost_rate
    net_income = annual_revenue - operating_cost
    village_share = annual_revenue * homestay.village_share_rate
    payback_years = homestay.renovation_cost_yuan / net_income if net_income > 0 else 999.0

    return HomestayProjection(
        house_id=homestay.house_id,
        name=homestay.name,
        rooms=homestay.rooms,
        renovation_cost_yuan=round(homestay.renovation_cost_yuan, 2),
        occupancy_rate=round(occupancy, 4),
        avg_price_yuan=round(price, 2),
        annual_revenue_yuan=round(annual_revenue, 2),
        operating_cost_yuan=round(operating_cost, 2),
        net_income_yuan=round(net_income, 2),
        village_share_yuan=round(village_share, 2),
        payback_years=round(payback_years, 2),
    )


def calculate_portfolio_projection(
    homestays: list[Homestay],
    occupancy_delta: float = 0.0,
    price_delta: float = 0.0,
) -> list[HomestayProjection]:
    """按入住率和客单价调整参数，批量测算微旅居收益。"""

    projections: list[HomestayProjection] = []
    for item in homestays:
        adjusted_occupancy = min(max(item.base_occupancy_rate + occupancy_delta, 0), 0.95)
        adjusted_price = max(item.avg_price_yuan * (1 + price_delta), 0)
        projections.append(
            calculate_homestay_projection(
                item,
                occupancy_rate=adjusted_occupancy,
                avg_price_yuan=adjusted_price,
            )
        )
    return projections
