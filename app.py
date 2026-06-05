from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mocrop.data_loader import load_data_bundle
from mocrop.homestay import calculate_portfolio_projection
from mocrop.models import HomestayProjection, Parcel, Recommendation
from mocrop.recommender import MODES, RecommendationEngine, erosion_risk_label, land_type_label
from mocrop.report import render_markdown_report, save_markdown_report

try:
    import folium
    from streamlit_folium import st_folium

    MAP_AVAILABLE = True
except Exception:
    MAP_AVAILABLE = False


st.set_page_config(
    page_title="MOCrop 因地智宜决策系统",
    page_icon="M",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_demo_data():
    return load_data_bundle()


@st.cache_data(show_spinner=False)
def get_recommendations() -> list[Recommendation]:
    data = get_demo_data()
    engine = RecommendationEngine(data.crops, data.market_prices, data.labor)
    return engine.recommend_all(data.parcels)


def money(value: float) -> str:
    return f"{value:,.0f} 元"


def pct(value: float) -> str:
    return f"{value:.0%}"


def risk_badge(risk_level: str) -> str:
    color = {
        "低": "#2f6f3f",
        "中": "#9a6b16",
        "高": "#a63b32",
    }.get(risk_level, "#657267")
    return (
        f"<span style='display:inline-block;padding:3px 10px;border-radius:999px;"
        f"background:{color}1f;color:{color};font-weight:700;'>{risk_level}风险</span>"
    )


def score_breakdown_dataframe(recommendation: Recommendation) -> pd.DataFrame:
    breakdown = recommendation.score_breakdown
    rows = [
        [
            "生态适配",
            breakdown["生态适配分"],
            f"{breakdown['生态权重']:.0%}",
            breakdown["生态加权分"],
        ],
        [
            "经济收益",
            breakdown["经济收益分"],
            f"{breakdown['经济权重']:.0%}",
            breakdown["经济加权分"],
        ],
        [
            "农旅体验",
            breakdown["农旅体验分"],
            f"{breakdown['农旅权重']:.0%}",
            breakdown["农旅加权分"],
        ],
        [
            "劳动力适配",
            breakdown["劳动力适配分"],
            f"{breakdown['劳动力权重']:.0%}",
            breakdown["劳动力加权分"],
        ],
        ["风险惩罚", breakdown["风险惩罚分"], "扣减", -breakdown["风险惩罚分"]],
        ["总分", breakdown["总分"], "-", breakdown["总分"]],
    ]
    return pd.DataFrame(rows, columns=["指标", "原始分", "权重", "加权贡献"])


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #17211b;
            --muted: #657267;
            --paper: #f8f3e7;
            --field: #dfe8c8;
            --moss: #426b46;
            --earth: #a0653a;
            --gold: #d59c42;
        }
        .stApp {
            background:
                radial-gradient(circle at 14% 12%, rgba(213, 156, 66, 0.18), transparent 32%),
                radial-gradient(circle at 85% 6%, rgba(66, 107, 70, 0.18), transparent 26%),
                linear-gradient(135deg, #fbf7ef 0%, #eef4df 44%, #f7efe2 100%);
            color: var(--ink);
        }
        .hero {
            border: 1px solid rgba(66, 107, 70, 0.18);
            border-radius: 28px;
            padding: 30px 34px;
            background:
                linear-gradient(120deg, rgba(255,255,255,0.78), rgba(246,239,221,0.78)),
                repeating-linear-gradient(45deg, rgba(66,107,70,0.05) 0 8px, transparent 8px 16px);
            box-shadow: 0 22px 60px rgba(39, 58, 43, 0.12);
            margin-bottom: 18px;
        }
        .hero h1 {
            font-family: 'Noto Serif SC', 'Source Han Serif SC', 'STSong', 'SimSun', serif;
            font-size: 2.75rem;
            line-height: 1.12;
            margin: 0 0 12px 0;
            color: #203c27;
        }
        .hero p {
            font-size: 1.03rem;
            color: var(--muted);
            max-width: 920px;
            margin: 0;
        }
        .tag {
            display: inline-block;
            border: 1px solid rgba(66, 107, 70, 0.28);
            background: rgba(255, 255, 255, 0.72);
            color: #365a38;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.86rem;
            margin: 5px 8px 0 0;
        }
        .plan-card {
            border-radius: 20px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(160,101,58,0.18);
            padding: 16px;
            min-height: 280px;
        }
        .plan-card h3 {
            margin: 0 0 8px 0;
            color: #203c27;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,0.72);
            border-color: rgba(160,101,58,0.18);
            box-shadow: 0 10px 30px rgba(43, 69, 46, 0.07);
        }
        .risk-low { color: #2f6f3f; font-weight: 700; }
        .risk-mid { color: #9a6b16; font-weight: 700; }
        .risk-high { color: #a63b32; font-weight: 700; }
        @media (max-width: 1100px) {
            .hero h1 { font-size: 2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>MOCrop 因地智宜决策系统</h1>
          <p>
          面向北京市怀柔区琉璃庙镇安洲坝村的可演示 MVP。
          系统以“数智种养 + 微旅居”双循环生态产业模式为主线，
          结合地块条件、作物知识库、市场价格、劳动力供给、生态修复需求和旅居潜力，
          为边际土地生成生态优先、经济优先、农旅融合三类方案。
          </p>
          <div style="margin-top:14px;">
            <span class="tag">规则评分</span>
            <span class="tag">一地一策</span>
            <span class="tag">生态修复</span>
            <span class="tag">林下经济</span>
            <span class="tag">微旅居测算</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(
    parcels: list[Parcel],
    recommendations: list[Recommendation],
    homestays: list[HomestayProjection],
) -> None:
    economic_profit = sum(
        item.expected_profit_yuan for item in recommendations if item.mode == "经济优先模式"
    )
    eco_scores = [
        item.ecological_score for item in recommendations if item.mode == "生态优先模式"
    ]
    tourism_scores = [
        item.tourism_score for item in recommendations if item.mode == "农旅融合模式"
    ]
    homestay_revenue = sum(item.annual_revenue_yuan for item in homestays)
    village_share = sum(item.village_share_yuan for item in homestays)

    cards = [
        ("边际地块", f"{len(parcels)} 块", f"合计 {sum(p.area_mu for p in parcels):.1f} 亩"),
        ("推荐方案", f"{len(recommendations)} 个", "每块地输出三类策略"),
        ("经济方案净收益", money(economic_profit), "按经济优先方案合计"),
        ("生态修复均分", f"{pd.Series(eco_scores).mean():.1f}", "生态优先方案平均分"),
        ("村集体分成", money(village_share), f"微旅居年收入 {money(homestay_revenue)}"),
    ]
    columns = st.columns(len(cards))
    for column, (label, value, note) in zip(columns, cards):
        with column:
            with st.container(border=True):
                st.metric(label, value)
                st.caption(note)


def render_map(
    parcels: list[Parcel],
    recommendations: list[Recommendation],
    selected_mode: str,
) -> None:
    st.subheader("地块地图与一地一策结果")
    rec_by_parcel = {
        item.parcel_id: item for item in recommendations if item.mode == selected_mode
    }
    if MAP_AVAILABLE:
        center = [
            sum(item.latitude for item in parcels) / len(parcels),
            sum(item.longitude for item in parcels) / len(parcels),
        ]
        fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
        risk_color = {"低": "#3f7d45", "中": "#d59c42", "高": "#b9473d"}
        for parcel in parcels:
            rec = rec_by_parcel[parcel.parcel_id]
            popup = f"""
            <b>{parcel.name}</b><br>
            面积：{parcel.area_mu:.1f} 亩<br>
            当前状态：{parcel.current_status}<br>
            {selected_mode}：{rec.crop_name}<br>
            预计净收益：{money(rec.expected_profit_yuan)}<br>
            风险等级：{rec.risk_level}
            """
            folium.CircleMarker(
                location=[parcel.latitude, parcel.longitude],
                radius=max(7, min(18, parcel.area_mu / 1.3)),
                color=risk_color[rec.risk_level],
                fill=True,
                fill_color=risk_color[rec.risk_level],
                fill_opacity=0.74,
                tooltip=f"{parcel.name}：{rec.crop_name}",
                popup=folium.Popup(popup, max_width=280),
            ).add_to(fmap)
        st_folium(fmap, height=430, use_container_width=True)
    else:
        st.info("未检测到 folium 组件，使用 Streamlit 基础地图展示。")
        st.map(
            pd.DataFrame(
                [
                    {
                        "lat": item.latitude,
                        "lon": item.longitude,
                        "name": item.name,
                    }
                    for item in parcels
                ]
            ),
            latitude="lat",
            longitude="lon",
        )


def render_recommendation_cards(
    parcels: list[Parcel],
    recommendations: list[Recommendation],
) -> None:
    st.subheader("地块推荐结果")
    parcel_names = [item.name for item in parcels]
    selected_name = st.selectbox("选择要查看的地块", parcel_names)
    selected = next(item for item in parcels if item.name == selected_name)
    parcel_recs = [item for item in recommendations if item.parcel_id == selected.parcel_id]

    st.caption(
        f"{selected.name} | {selected.area_mu:.1f} 亩 | 坡度 {selected.slope_degree:.0f}° | "
        f"土壤 pH {selected.soil_ph:.1f} | 有机质 {selected.organic_matter:.0f} g/kg | "
        f"日照 {selected.sunlight_hours:.1f} 小时 | 距水源 {selected.water_distance_m:.0f} 米 | "
        f"地块类型 {land_type_label(selected.land_type)} | "
        f"水土流失风险 {erosion_risk_label(selected.erosion_risk)} | "
        f"可用劳动力 {selected.labor_available} 人"
    )

    columns = st.columns(3)
    for col, rec in zip(columns, parcel_recs):
        with col:
            with st.container(border=True):
                st.markdown(f"### {rec.mode}")
                st.markdown(f"**推荐作物：{rec.crop_name}**")
                st.caption(f"作物类型：{rec.crop_category}")

                m1, m2 = st.columns(2)
                m1.metric("预计投入", money(rec.expected_investment_yuan))
                m2.metric("预计收入", money(rec.expected_revenue_yuan))
                st.metric("预计净收益", money(rec.expected_profit_yuan))

                st.markdown(f"劳动力需求：**{rec.labor_days:.1f} 工日**")
                st.markdown(f"生态适配评分：**{rec.ecological_score:.1f}/100**")
                st.progress(rec.ecological_score / 100)
                st.markdown(f"经济收益评分：**{rec.economic_return_score:.1f}/100**")
                st.progress(rec.economic_return_score / 100)
                st.markdown(f"农旅体验评分：**{rec.tourism_score:.1f}/100**")
                st.progress(rec.tourism_score / 100)
                st.markdown(f"劳动力适配评分：**{rec.labor_adaptation_score:.1f}/100**")
                st.progress(rec.labor_adaptation_score / 100)
                st.markdown(f"风险等级：{risk_badge(rec.risk_level)}", unsafe_allow_html=True)
                st.markdown(f"风险惩罚：**-{rec.risk_penalty_score:.1f} 分**")

                st.markdown("**评分拆解**")
                st.dataframe(
                    score_breakdown_dataframe(rec),
                    width="stretch",
                    hide_index=True,
                )

                with st.expander("推荐理由"):
                    for reason in rec.reasons:
                        st.markdown(f"- {reason}")

    chart_df = pd.DataFrame([item.model_dump() for item in parcel_recs])
    chart_df = chart_df.rename(
        columns={
            "mode": "模式",
            "expected_investment_yuan": "预计投入",
            "expected_revenue_yuan": "预计收入",
            "expected_profit_yuan": "预计净收益",
        }
    )
    fig = px.bar(
        chart_df,
        x="模式",
        y=["预计投入", "预计收入", "预计净收益"],
        barmode="group",
        color_discrete_sequence=["#a0653a", "#426b46", "#d59c42"],
        title=f"{selected.name} 三类方案收益测算",
    )
    fig.update_layout(legend_title_text="", yaxis_title="金额（元）", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_income_charts(recommendations: list[Recommendation]) -> None:
    st.subheader("全村种植方案收益对比")
    df = pd.DataFrame([item.model_dump() for item in recommendations])
    fig = px.bar(
        df,
        x="parcel_name",
        y="expected_profit_yuan",
        color="mode",
        barmode="group",
        color_discrete_map={
            "生态优先模式": "#426b46",
            "经济优先模式": "#d59c42",
            "农旅融合模式": "#a0653a",
        },
        labels={
            "parcel_name": "地块",
            "expected_profit_yuan": "预计净收益（元）",
            "mode": "推荐模式",
        },
    )
    fig.update_layout(legend_title_text="", xaxis_title="", yaxis_title="预计净收益（元）")
    st.plotly_chart(fig, use_container_width=True)

    df["收益气泡大小"] = df["expected_profit_yuan"].clip(lower=0) + 5_000
    score_fig = px.scatter(
        df,
        x="ecological_score",
        y="tourism_score",
        size="收益气泡大小",
        color="mode",
        hover_name="crop_name",
        hover_data={
            "parcel_name": True,
            "risk_level": True,
            "expected_profit_yuan": ":,.0f",
            "收益气泡大小": False,
        },
        color_discrete_map={
            "生态优先模式": "#426b46",
            "经济优先模式": "#d59c42",
            "农旅融合模式": "#a0653a",
        },
        labels={
            "ecological_score": "生态修复评分",
            "tourism_score": "农旅体验评分",
            "mode": "推荐模式",
        },
        title="生态修复、农旅体验与收益的综合分布",
    )
    score_fig.update_layout(legend_title_text="")
    st.plotly_chart(score_fig, use_container_width=True)


def render_homestay_module(projections: list[HomestayProjection]) -> None:
    st.subheader("微旅居收益测算")
    df = pd.DataFrame([item.model_dump() for item in projections])
    display = df.rename(
        columns={
            "name": "民宿资源",
            "rooms": "房间数",
            "renovation_cost_yuan": "改造成本",
            "occupancy_rate": "入住率",
            "avg_price_yuan": "客单价",
            "annual_revenue_yuan": "年收入",
            "net_income_yuan": "净收入",
            "village_share_yuan": "村集体分成",
            "payback_years": "回收期",
        }
    )
    display_table = display[
        [
            "民宿资源",
            "房间数",
            "改造成本",
            "入住率",
            "客单价",
            "年收入",
            "净收入",
            "村集体分成",
            "回收期",
        ]
    ].copy()
    for column in ["改造成本", "客单价", "年收入", "净收入", "村集体分成"]:
        display_table[column] = display_table[column].map(money)
    display_table["入住率"] = display_table["入住率"].map(pct)
    display_table["回收期"] = display_table["回收期"].map(lambda value: f"{value:.1f} 年")

    st.dataframe(
        display_table,
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns([1.2, 1])
    with left:
        bar_df = display.melt(
            id_vars="民宿资源",
            value_vars=["年收入", "净收入", "村集体分成"],
            var_name="指标",
            value_name="金额",
        )
        fig = px.bar(
            bar_df,
            x="民宿资源",
            y="金额",
            color="指标",
            barmode="group",
            color_discrete_sequence=["#426b46", "#d59c42", "#a0653a"],
            title="微旅居收入与分成",
        )
        fig.update_layout(xaxis_title="", yaxis_title="金额（元）", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_bar(
            x=display["民宿资源"],
            y=display["改造成本"],
            name="改造成本",
            marker_color="#a0653a",
        )
        fig.add_scatter(
            x=display["民宿资源"],
            y=display["回收期"],
            name="回收期（年）",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#426b46", width=3),
        )
        fig.update_layout(
            title="改造成本与投资回收期",
            xaxis_title="",
            yaxis=dict(title="改造成本（元）"),
            yaxis2=dict(title="回收期（年）", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_annual_plan() -> None:
    st.subheader("年度实施计划")
    plan = pd.DataFrame(
        [
            ["2026 年 Q3", "地块建档、土壤快检、劳动力摸底", "形成一地一策数据底座"],
            ["2026 年 Q4", "冰草、披碱草和文冠果生态修复试种", "建成生态优先示范片"],
            ["2027 年 H1", "黄花菜、药草花境、林下菌菇体验点", "推出采摘研学线路"],
            ["2027 年 H2", "样板民宿改造和村集体分成机制", "形成微旅居运营闭环"],
            ["2028 年", "合作社订单、品牌包装、跨地块扩面复制", "沉淀双循环产业模型"],
        ],
        columns=["阶段", "重点任务", "路演产出"],
    )
    st.dataframe(plan, width="stretch", hide_index=True)


def render_report_export(
    recommendations: list[Recommendation],
    projections: list[HomestayProjection],
) -> None:
    st.subheader("报告导出")
    markdown = render_markdown_report(recommendations, projections)
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("生成 Markdown 报告", type="primary"):
            path = save_markdown_report(markdown)
            try:
                shown_path = path.relative_to(Path.cwd())
            except ValueError:
                shown_path = path
            st.success(f"已导出到 {shown_path}")
    with col2:
        st.download_button(
            label="下载当前报告内容",
            data=markdown,
            file_name="mocrop_report.md",
            mime="text/markdown",
            width="stretch",
        )
    with st.expander("预览 Markdown 报告"):
        st.markdown(markdown)


def main() -> None:
    inject_style()
    data = get_demo_data()
    recommendations = get_recommendations()

    st.sidebar.title("演示参数")
    selected_mode = st.sidebar.selectbox("地图展示模式", MODES, index=1)
    occupancy_delta = st.sidebar.slider("入住率情景调整", -0.15, 0.25, 0.05, 0.01)
    price_delta = st.sidebar.slider("客单价情景调整", -0.20, 0.30, 0.00, 0.05)
    st.sidebar.caption("所有数据均为本地演示数据，不依赖外部 API。")

    homestay_projections = calculate_portfolio_projection(
        data.homestays,
        occupancy_delta=occupancy_delta,
        price_delta=price_delta,
    )

    render_hero()
    render_metric_cards(data.parcels, recommendations, homestay_projections)

    tabs = st.tabs(["总览地图", "地块推荐", "收益图表", "微旅居", "年度计划与导出"])
    with tabs[0]:
        render_map(data.parcels, recommendations, selected_mode)
        st.info(
            "颜色代表当前地图模式下的风险等级：绿色为低风险，黄色为中风险，红色为高风险。"
        )
    with tabs[1]:
        render_recommendation_cards(data.parcels, recommendations)
    with tabs[2]:
        render_income_charts(recommendations)
    with tabs[3]:
        render_homestay_module(homestay_projections)
    with tabs[4]:
        render_annual_plan()
        render_report_export(recommendations, homestay_projections)


if __name__ == "__main__":
    main()
