from __future__ import annotations

import streamlit as st

from mocrop.data_loader import load_data_bundle
from mocrop.homestay import calculate_portfolio_projection
from src.feasibility import build_feasibility_result
from src.recommender import RecommendationEngine
from src.ui_components import (
    inject_pitch_style,
    render_core_metrics,
    render_ecology_radar,
    render_feasibility_section,
    render_homestay_section,
    render_income_chart,
    render_parcel_info_card,
    render_pitch_header,
    render_reason_panel,
    render_recommendation_cards,
    render_report_actions,
    render_score_breakdown,
    render_sidebar_controls,
    render_timeline,
    selected_mode_recommendation,
)


st.set_page_config(
    page_title="MOCrop 因地智宜决策系统",
    page_icon="M",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_demo_data():
    return load_data_bundle()


def main() -> None:
    inject_pitch_style()
    data = get_demo_data()

    selected_parcel, selected_mode, custom_weights, occupancy_delta, price_delta = (
        render_sidebar_controls(data.parcels)
    )
    mode_weight_overrides = {selected_mode: custom_weights}

    engine = RecommendationEngine(data.crops, data.market_prices, data.labor)
    all_recommendations = engine.recommend_all(
        data.parcels,
        mode_weight_overrides=mode_weight_overrides,
    )
    parcel_recommendations = list(
        engine.recommend_for_parcel(
            selected_parcel,
            mode_weight_overrides=mode_weight_overrides,
        ).values()
    )
    selected_recommendation = selected_mode_recommendation(parcel_recommendations, selected_mode)
    homestay_projections = calculate_portfolio_projection(
        data.homestays,
        occupancy_delta=occupancy_delta,
        price_delta=price_delta,
    )
    feasibility_result = build_feasibility_result(engine, homestay_projections)

    render_pitch_header()
    render_core_metrics(data.parcels, all_recommendations, homestay_projections, selected_mode)

    st.divider()
    st.markdown("## 一地一策决策演示")
    render_parcel_info_card(selected_parcel)
    render_recommendation_cards(parcel_recommendations, selected_mode)

    reason_col, score_col = st.columns([1.15, 0.85])
    with reason_col:
        render_reason_panel(selected_recommendation)
    with score_col:
        with st.container(border=True):
            st.subheader("评分拆解")
            render_score_breakdown(selected_recommendation)

    chart_col, radar_col = st.columns([1.15, 0.85])
    with chart_col:
        with st.container(border=True):
            render_income_chart(parcel_recommendations, selected_parcel.name)
    with radar_col:
        with st.container(border=True):
            render_ecology_radar(parcel_recommendations)

    st.divider()
    st.markdown("## 微旅居与年度落地路径")
    bottom_left, bottom_right = st.columns([1.15, 0.85])
    with bottom_left:
        render_homestay_section(homestay_projections)
    with bottom_right:
        render_timeline()
        render_report_actions(all_recommendations, homestay_projections)

    st.divider()
    render_feasibility_section(feasibility_result)


if __name__ == "__main__":
    main()
