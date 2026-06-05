from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mocrop.models import HomestayProjection, ModeName, Parcel, Recommendation
from mocrop.report import render_markdown_report, save_markdown_report
from src.feasibility import VillageFeasibilityResult, save_feasibility_report
from src.recommender import MODE_WEIGHTS, MODES, erosion_risk_label, land_type_label, normalize_weights


MODE_LABELS: dict[str, ModeName] = {
    "生态优先": "生态优先模式",
    "经济优先": "经济优先模式",
    "农旅融合": "农旅融合模式",
}


def money(value: float) -> str:
    """将金额格式化为适合路演展示的中文文本。"""

    if abs(value) >= 10_000:
        return f"{value / 10_000:,.1f} 万元"
    return f"{value:,.0f} 元"


def pct(value: float) -> str:
    """将小数格式化为百分比文本。"""

    return f"{value:.0%}"


def inject_pitch_style() -> None:
    """注入本地 CSS，让页面更像路演 Demo，而不是普通后台界面。"""

    st.markdown(
        """
        <style>
        :root {
            --mocrop-ink: #17211b;
            --mocrop-muted: #667267;
            --mocrop-moss: #426b46;
            --mocrop-earth: #a0653a;
            --mocrop-gold: #d59c42;
            --mocrop-paper: #fbf7ef;
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(213, 156, 66, 0.20), transparent 30%),
                radial-gradient(circle at 82% 6%, rgba(66, 107, 70, 0.18), transparent 26%),
                linear-gradient(135deg, #fbf7ef 0%, #eef4df 46%, #f7efe2 100%);
            color: var(--mocrop-ink);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #203c27 0%, #344f32 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #f8f3e7;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] div[data-baseweb="select"] *,
        div[role="listbox"] * {
            color: #17211b !important;
            -webkit-text-fill-color: #17211b !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"],
        section[data-testid="stSidebar"] div[data-baseweb="input"] {
            background: #fffaf0 !important;
            border-color: rgba(213, 156, 66, 0.45) !important;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label,
        section[data-testid="stSidebar"] [role="radiogroup"] label * {
            color: #f8f3e7 !important;
            -webkit-text-fill-color: #f8f3e7 !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.76);
            border-color: rgba(66, 107, 70, 0.16);
            box-shadow: 0 14px 40px rgba(43, 69, 46, 0.09);
        }
        h1, h2, h3 {
            color: #203c27;
        }
        .stMetric {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_pitch_header() -> None:
    """渲染路演顶部标题和副标题。"""

    st.title("MOCrop 因地智宜决策系统")
    st.markdown("### 面向生态涵养区的数智种养 + 微旅居双循环产业决策平台")
    st.caption("安洲坝村边际土地盘活、生态修复、特色种养与微旅居收益测算一体化 Demo")


def render_core_metrics(
    parcels: list[Parcel],
    recommendations: list[Recommendation],
    homestay_projections: list[HomestayProjection],
    selected_mode: ModeName,
) -> None:
    """渲染顶部四个核心指标卡。"""

    selected_recommendations = [item for item in recommendations if item.mode == selected_mode]
    income = sum(max(item.expected_profit_yuan, 0) for item in selected_recommendations)
    restored_area = sum(
        parcel.area_mu
        for parcel in parcels
        if any(
            item.parcel_id == parcel.parcel_id and item.ecological_score >= 70
            for item in selected_recommendations
        )
    )
    homestay_revenue = sum(item.annual_revenue_yuan for item in homestay_projections)

    cards = [
        ("推荐地块数量", f"{len(parcels)} 块", f"纳入边际地 {sum(p.area_mu for p in parcels):.1f} 亩"),
        ("预计村民增收", money(income), f"按当前{selected_mode.replace('模式', '')}方案测算"),
        ("预计生态修复面积", f"{restored_area:.1f} 亩", "生态适配分不低于 70 的地块"),
        ("预计旅居年收入", money(homestay_revenue), "闲置农宅微旅居情景测算"),
    ]

    columns = st.columns(4)
    for column, (label, value, note) in zip(columns, cards):
        with column:
            with st.container(border=True):
                st.metric(label, value)
                st.caption(note)


def render_sidebar_controls(parcels: list[Parcel]) -> tuple[Parcel, ModeName, dict[str, float], float, float]:
    """渲染左侧筛选栏，并返回当前地块、模式、权重和民宿情景参数。"""

    st.sidebar.title("路演控制台")
    parcel_name = st.sidebar.radio("选择地块", [parcel.name for parcel in parcels])
    selected_parcel = next(parcel for parcel in parcels if parcel.name == parcel_name)

    mode_label = st.sidebar.radio("选择模式", list(MODE_LABELS.keys()), horizontal=False)
    selected_mode = MODE_LABELS[mode_label]
    defaults = MODE_WEIGHTS[selected_mode]

    st.sidebar.markdown("#### 调整权重滑块")
    raw_weights = {
        "生态适配分": st.sidebar.slider(
            "生态适配权重",
            0.0,
            1.0,
            float(defaults["生态适配分"]),
            0.05,
            key=f"eco_weight_{selected_mode}",
        ),
        "经济收益分": st.sidebar.slider(
            "经济收益权重",
            0.0,
            1.0,
            float(defaults["经济收益分"]),
            0.05,
            key=f"economic_weight_{selected_mode}",
        ),
        "农旅体验分": st.sidebar.slider(
            "农旅体验权重",
            0.0,
            1.0,
            float(defaults["农旅体验分"]),
            0.05,
            key=f"tourism_weight_{selected_mode}",
        ),
        "劳动力适配分": st.sidebar.slider(
            "劳动力适配权重",
            0.0,
            1.0,
            float(defaults["劳动力适配分"]),
            0.05,
            key=f"labor_weight_{selected_mode}",
        ),
    }
    weights = normalize_weights(raw_weights)
    st.sidebar.caption(
        "系统会自动归一化权重："
        f"生态 {weights['生态适配分']:.0%}，"
        f"经济 {weights['经济收益分']:.0%}，"
        f"农旅 {weights['农旅体验分']:.0%}，"
        f"劳动力 {weights['劳动力适配分']:.0%}"
    )

    st.sidebar.markdown("#### 微旅居情景")
    occupancy_delta = st.sidebar.slider("入住率调整", -0.15, 0.25, 0.05, 0.01)
    price_delta = st.sidebar.slider("客单价调整", -0.20, 0.30, 0.00, 0.05)
    st.sidebar.caption("所有数据均为本地演示数据，不依赖外部 API。")
    return selected_parcel, selected_mode, weights, occupancy_delta, price_delta


def render_parcel_info_card(parcel: Parcel) -> None:
    """渲染当前地块基础信息卡片。"""

    with st.container(border=True):
        st.subheader("地块基础信息")
        st.markdown(f"**{parcel.name}** | 当前状态：{parcel.current_status}")
        cols = st.columns(4)
        cols[0].metric("面积", f"{parcel.area_mu:.1f} 亩")
        cols[1].metric("坡度", f"{parcel.slope_degree:.0f}°")
        cols[2].metric("水源距离", f"{parcel.water_distance_m:.0f} 米")
        cols[3].metric("可用劳动力", f"{parcel.labor_available} 人")

        cols = st.columns(4)
        cols[0].metric("土壤 pH", f"{parcel.soil_ph:.1f}")
        cols[1].metric("有机质", f"{parcel.organic_matter:.0f} g/kg")
        cols[2].metric("日照", f"{parcel.sunlight_hours:.1f} 小时")
        cols[3].metric("流失风险", erosion_risk_label(parcel.erosion_risk))
        st.caption(f"地块类型：{land_type_label(parcel.land_type)}")


def render_recommendation_cards(
    recommendations: list[Recommendation],
    selected_mode: ModeName,
) -> None:
    """渲染三种推荐方案卡片。"""

    st.subheader("三种推荐方案")
    columns = st.columns(3)
    for column, recommendation in zip(columns, _sort_recommendations(recommendations)):
        with column:
            with st.container(border=True):
                if recommendation.mode == selected_mode:
                    st.success("当前演示模式")
                st.markdown(f"### {recommendation.mode.replace('模式', '')}")
                st.markdown(f"**{recommendation.crop_name}**")
                st.caption(f"{recommendation.crop_category} | 总分 {recommendation.score:.1f}")
                st.metric("预计净收益", money(recommendation.expected_profit_yuan))
                st.metric("预计收入", money(recommendation.expected_revenue_yuan))
                st.caption(
                    f"投入 {money(recommendation.expected_investment_yuan)} | "
                    f"用工 {recommendation.labor_days:.1f} 工日 | "
                    f"风险 {recommendation.risk_level}"
                )
                st.progress(recommendation.score / 100)


def render_reason_panel(recommendation: Recommendation) -> None:
    """突出展示当前模式的推荐理由。"""

    with st.container(border=True):
        st.subheader("推荐理由")
        st.markdown(
            f"当前选择 **{recommendation.mode.replace('模式', '')}**，"
            f"系统推荐 **{recommendation.crop_name}**，综合得分 **{recommendation.score:.1f}**。"
        )
        for reason in recommendation.reasons:
            st.markdown(f"- {reason}")


def render_score_breakdown(recommendation: Recommendation) -> None:
    """渲染当前推荐结果的评分拆解表。"""

    st.dataframe(score_breakdown_dataframe(recommendation), width="stretch", hide_index=True)


def render_income_chart(recommendations: list[Recommendation], parcel_name: str) -> None:
    """渲染当前地块三种方案的投入、收入和净收益对比图。"""

    df = pd.DataFrame([item.model_dump() for item in _sort_recommendations(recommendations)])
    chart_df = df.rename(
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
        title=f"{parcel_name} 收益测算",
    )
    fig.update_layout(legend_title_text="", yaxis_title="金额（元）", xaxis_title="")
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def render_ecology_radar(recommendations: list[Recommendation]) -> None:
    """渲染生态、经济、农旅、劳动力和风险控制的雷达图。"""

    categories = ["生态适配", "经济收益", "农旅体验", "劳动力适配", "风险控制"]
    fig = go.Figure()
    for recommendation in _sort_recommendations(recommendations):
        risk_control = max(0.0, min(100.0, 100 - recommendation.risk_penalty_score * 3))
        values = [
            recommendation.ecological_score,
            recommendation.economic_return_score,
            recommendation.tourism_score,
            recommendation.labor_adaptation_score,
            risk_control,
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=recommendation.mode.replace("模式", ""),
            )
        )
    fig.update_layout(
        title="生态评分雷达图",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def render_homestay_section(projections: list[HomestayProjection]) -> None:
    """渲染微旅居收益测算模块。"""

    with st.container(border=True):
        st.subheader("微旅居测算模块")
        total_revenue = sum(item.annual_revenue_yuan for item in projections)
        village_share = sum(item.village_share_yuan for item in projections)
        cols = st.columns(3)
        cols[0].metric("旅居年收入", money(total_revenue))
        cols[1].metric("村集体分成", money(village_share))
        cols[2].metric("盘活农宅", f"{len(projections)} 处")

        df = pd.DataFrame([item.model_dump() for item in projections])
        chart_df = df.rename(
            columns={
                "name": "民宿资源",
                "annual_revenue_yuan": "年收入",
                "village_share_yuan": "村集体分成",
            }
        )
        fig = px.bar(
            chart_df,
            x="民宿资源",
            y=["年收入", "村集体分成"],
            barmode="group",
            color_discrete_sequence=["#426b46", "#d59c42"],
        )
        fig.update_layout(legend_title_text="", xaxis_title="", yaxis_title="金额（元）")
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )


def render_timeline() -> None:
    """渲染年度实施计划时间线。"""

    with st.container(border=True):
        st.subheader("年度实施计划时间线")
        steps = [
            ("2026 Q3", "建档评估", "地块建档、土壤快检、劳动力摸底"),
            ("2026 Q4", "生态试种", "固土草本、药材和生态经济林小规模试种"),
            ("2027 H1", "农旅成型", "黄花菜采摘、药草花境、林下菌菇研学点"),
            ("2027 H2", "旅居闭环", "样板民宿改造、村集体分成和套餐化运营"),
            ("2028", "复制扩面", "合作社订单、品牌包装和跨地块复制"),
        ]
        for index, (period, title, detail) in enumerate(steps, start=1):
            st.markdown(f"**{index}. {period} | {title}**")
            st.caption(detail)


def render_report_actions(
    recommendations: list[Recommendation],
    projections: list[HomestayProjection],
) -> None:
    """渲染一键导出 Markdown 报告按钮。"""

    with st.container(border=True):
        st.subheader("一键导出 Markdown 报告")
        markdown = render_markdown_report(recommendations, projections)
        left, right = st.columns([1, 1])
        with left:
            if st.button("生成本地报告", type="primary", use_container_width=True):
                path = save_markdown_report(markdown)
                try:
                    shown_path = path.relative_to(Path.cwd())
                except ValueError:
                    shown_path = path
                st.success(f"已导出：{shown_path}")
        with right:
            st.download_button(
                "下载 Markdown",
                data=markdown,
                file_name="mocrop_report.md",
                mime="text/markdown",
                use_container_width=True,
            )


def render_feasibility_section(result: VillageFeasibilityResult) -> None:
    """渲染项目作证模块，证明方案具备落地可行性。"""

    with st.container(border=True):
        st.subheader("项目作证：10 个地块样本可行性验证")
        st.caption("该模块用于路演时回答“能不能落地、怎么算出来、村里能得到什么”的问题。")

        summary = result.summary
        cols = st.columns(4)
        cols[0].metric("总种植面积", f"{summary['总种植面积']:.1f} 亩")
        cols[1].metric("总投入", money(summary["总投入"]))
        cols[2].metric("总收入", money(summary["总收入"]))
        cols[3].metric("总净收益", money(summary["总净收益"]))

        cols = st.columns(4)
        cols[0].metric("参与就业人数", f"{summary['参与就业人数']:.0f} 人")
        cols[1].metric("女性/中老年岗位", f"{summary['女性/中老年岗位数量']:.0f} 个")
        cols[2].metric("生态修复均分", f"{summary['生态修复评分平均值']:.1f}")
        cols[3].metric("农旅体验点", f"{summary['农旅体验点数量']:.0f} 个")

        with st.expander("查看 10 个模拟地块与优选方案", expanded=True):
            st.dataframe(
                feasibility_dataframe(result),
                width="stretch",
                hide_index=True,
            )

        left, right = st.columns([1, 1])
        with left:
            if st.button("导出项目可行性证明报告", type="primary", use_container_width=True):
                path = save_feasibility_report(result.markdown)
                try:
                    shown_path = path.relative_to(Path.cwd())
                except ValueError:
                    shown_path = path
                st.success(f"已导出：{shown_path}")
        with right:
            st.download_button(
                "下载可行性报告 Markdown",
                data=result.markdown,
                file_name="mocrop_feasibility_report.md",
                mime="text/markdown",
                use_container_width=True,
            )


def feasibility_dataframe(result: VillageFeasibilityResult) -> pd.DataFrame:
    """将项目作证结果转换为页面表格。"""

    parcels_by_id = {parcel.parcel_id: parcel for parcel in result.parcels}
    rows = []
    for recommendation in result.preferred_recommendations:
        parcel = parcels_by_id[recommendation.parcel_id]
        rows.append(
            {
                "地块": parcel.name,
                "面积": f"{parcel.area_mu:.1f} 亩",
                "地块类型": land_type_label(parcel.land_type),
                "风险": erosion_risk_label(parcel.erosion_risk),
                "优选模式": recommendation.mode.replace("模式", ""),
                "推荐作物": recommendation.crop_name,
                "投入": money(recommendation.expected_investment_yuan),
                "收入": money(recommendation.expected_revenue_yuan),
                "净收益": money(recommendation.expected_profit_yuan),
                "生态分": f"{recommendation.ecological_score:.1f}",
                "农旅分": f"{recommendation.tourism_score:.1f}",
                "用工": f"{recommendation.labor_days:.1f} 工日",
            }
        )
    return pd.DataFrame(rows)


def score_breakdown_dataframe(recommendation: Recommendation) -> pd.DataFrame:
    """将推荐结果的评分拆解转换为表格。"""

    breakdown = recommendation.score_breakdown
    rows = [
        ["生态适配", breakdown["生态适配分"], f"{breakdown['生态权重']:.0%}", breakdown["生态加权分"]],
        ["经济收益", breakdown["经济收益分"], f"{breakdown['经济权重']:.0%}", breakdown["经济加权分"]],
        ["农旅体验", breakdown["农旅体验分"], f"{breakdown['农旅权重']:.0%}", breakdown["农旅加权分"]],
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


def selected_mode_recommendation(
    recommendations: list[Recommendation],
    selected_mode: ModeName,
) -> Recommendation:
    """从当前地块三类方案中取出选中模式的推荐结果。"""

    return next(item for item in recommendations if item.mode == selected_mode)


def _sort_recommendations(recommendations: list[Recommendation]) -> list[Recommendation]:
    return sorted(recommendations, key=lambda item: MODES.index(item.mode))
