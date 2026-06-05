from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mocrop.models import HomestayProjection, Parcel, Recommendation
from src.recommender import MODES, RecommendationEngine, erosion_risk_label, land_type_label


REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


@dataclass(frozen=True)
class VillageFeasibilityResult:
    parcels: list[Parcel]
    recommendations: list[Recommendation]
    preferred_recommendations: list[Recommendation]
    summary: dict[str, float]
    markdown: str


def simulate_anzhouba_parcels() -> list[Parcel]:
    """模拟生成 10 个安洲坝村地块样本，用于比赛路演作证。"""

    samples = [
        ("S001", "北坡撂荒梯田", 14.8, 40.6941, 116.6639, 24, 6.7, 22, 7.5, 560, "slope", "high", 6, 5, "多年撂荒坡地"),
        ("S002", "村口缓坡菜园", 9.6, 40.6902, 116.6685, 7, 6.8, 34, 8.4, 160, "abandoned", "medium", 9, 2, "闲置菜园"),
        ("S003", "林下阴坡空地", 13.2, 40.6974, 116.6615, 16, 6.1, 38, 4.4, 390, "forest_understory", "medium", 6, 4, "疏林空地"),
        ("S004", "民宿小院周边", 6.4, 40.6924, 116.6724, 5, 6.6, 28, 8.1, 210, "homestay_nearby", "low", 5, 1, "农宅周边闲地"),
        ("S005", "河滩边际地", 8.5, 40.6888, 116.6718, 3, 7.2, 26, 8.8, 80, "abandoned", "low", 7, 1, "河滩边际地"),
        ("S006", "村道观景坡台", 7.8, 40.6920, 116.6728, 9, 6.5, 25, 8.2, 260, "roadside", "medium", 5, 3, "道路沿线坡台"),
        ("S007", "高海拔薄土地", 18.9, 40.7001, 116.6582, 29, 6.3, 17, 7.2, 720, "slope", "high", 5, 7, "石质化坡地"),
        ("S008", "老栗树林空隙地", 11.4, 40.6961, 116.6609, 13, 6.0, 41, 4.9, 330, "forest_understory", "medium", 6, 3, "林下空隙地"),
        ("S009", "南向闲置坡地", 10.7, 40.6914, 116.6668, 18, 6.9, 24, 8.6, 420, "slope", "high", 6, 4, "南向闲置坡地"),
        ("S010", "民宿连片花境地", 5.9, 40.6928, 116.6731, 4, 6.7, 30, 8.0, 180, "homestay_nearby", "low", 4, 1, "旅居动线节点"),
    ]
    return [
        Parcel(
            parcel_id=parcel_id,
            name=name,
            area_mu=area_mu,
            latitude=latitude,
            longitude=longitude,
            slope_degree=slope_degree,
            soil_ph=soil_ph,
            organic_matter=organic_matter,
            sunlight_hours=sunlight_hours,
            water_distance_m=water_distance_m,
            land_type=land_type,
            erosion_risk=erosion_risk,
            labor_available=labor_available,
            idle_years=idle_years,
            current_status=current_status,
        )
        for (
            parcel_id,
            name,
            area_mu,
            latitude,
            longitude,
            slope_degree,
            soil_ph,
            organic_matter,
            sunlight_hours,
            water_distance_m,
            land_type,
            erosion_risk,
            labor_available,
            idle_years,
            current_status,
        ) in samples
    ]


def build_feasibility_result(
    engine: RecommendationEngine,
    homestay_projections: list[HomestayProjection],
) -> VillageFeasibilityResult:
    """生成项目作证结果，包括模拟地块、三类方案、汇总指标和 Markdown 报告。"""

    parcels = simulate_anzhouba_parcels()
    recommendations = engine.recommend_all(parcels)
    preferred = [_preferred_recommendation(parcel, recommendations) for parcel in parcels]
    summary = calculate_village_summary(parcels, preferred, homestay_projections)
    markdown = render_feasibility_report(
        parcels=parcels,
        recommendations=recommendations,
        preferred_recommendations=preferred,
        summary=summary,
        homestay_projections=homestay_projections,
    )
    return VillageFeasibilityResult(
        parcels=parcels,
        recommendations=recommendations,
        preferred_recommendations=preferred,
        summary=summary,
        markdown=markdown,
    )


def calculate_village_summary(
    parcels: list[Parcel],
    preferred_recommendations: list[Recommendation],
    homestay_projections: list[HomestayProjection],
) -> dict[str, float]:
    """计算全村级汇总指标。"""

    total_area = sum(parcel.area_mu for parcel in parcels)
    total_investment = sum(item.expected_investment_yuan for item in preferred_recommendations)
    total_revenue = sum(item.expected_revenue_yuan for item in preferred_recommendations)
    total_profit = sum(item.expected_profit_yuan for item in preferred_recommendations)
    total_labor_days = sum(item.labor_days for item in preferred_recommendations)
    participating_jobs = max(1, round(total_labor_days / 80))
    inclusive_jobs = max(1, round(participating_jobs * 0.58))
    ecological_average = (
        sum(item.ecological_score for item in preferred_recommendations)
        / len(preferred_recommendations)
    )
    tourism_points = sum(1 for item in preferred_recommendations if item.tourism_score >= 78)
    homestay_revenue = sum(item.annual_revenue_yuan for item in homestay_projections)
    village_share = sum(item.village_share_yuan for item in homestay_projections)

    return {
        "总种植面积": round(total_area, 2),
        "总投入": round(total_investment, 2),
        "总收入": round(total_revenue, 2),
        "总净收益": round(total_profit, 2),
        "参与就业人数": float(participating_jobs),
        "女性/中老年岗位数量": float(inclusive_jobs),
        "生态修复评分平均值": round(ecological_average, 2),
        "农旅体验点数量": float(tourism_points),
        "微旅居年收入": round(homestay_revenue, 2),
        "村集体分成": round(village_share, 2),
    }


def render_feasibility_report(
    parcels: list[Parcel],
    recommendations: list[Recommendation],
    preferred_recommendations: list[Recommendation],
    summary: dict[str, float],
    homestay_projections: list[HomestayProjection],
) -> str:
    """渲染适合评委阅读的项目可行性证明报告。"""

    generated_at = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# MOCrop 项目可行性证明报告",
        "",
        f"- 生成日期：{generated_at}",
        "- 服务对象：北京市怀柔区琉璃庙镇安洲坝村",
        "- 报告用途：挑战杯/乡村振兴路演支撑材料",
        "",
        "## 一、核心结论",
        "",
        f"本系统基于 10 个安洲坝村模拟地块样本，形成 {len(recommendations)} 个可比较推荐方案，并为每个地块给出优选落地路径。测算结果显示：",
        f"- 可纳入统筹种植面积：{summary['总种植面积']:.1f} 亩",
        f"- 预计总投入：{summary['总投入']:,.0f} 元",
        f"- 预计总收入：{summary['总收入']:,.0f} 元",
        f"- 预计总净收益：{summary['总净收益']:,.0f} 元",
        f"- 可带动参与就业：{summary['参与就业人数']:.0f} 人，其中女性/中老年友好岗位约 {summary['女性/中老年岗位数量']:.0f} 个",
        f"- 生态修复评分平均值：{summary['生态修复评分平均值']:.1f}/100",
        f"- 可形成农旅体验点：{summary['农旅体验点数量']:.0f} 个",
        "",
        "以上结果说明，MOCrop 不是单纯展示页面，而是能把地块条件、作物知识库、劳动力供给和旅居资源转化为可执行的产业组合建议。",
        "",
        "## 二、数据来源说明",
        "",
        "- 地块样本：围绕安洲坝村山地、林下、道路沿线、民宿周边和撂荒边际地特征，模拟生成 10 个地块样本。",
        "- 作物知识库：来自项目本地作物库，包含中药材、生态修复草本、食用菌、坚果油料和农旅体验作物。",
        "- 市场与投入：使用本地演示价格、亩产、投入成本和劳动强度参数，用于路演测算，不作为实际经营承诺。",
        "- 微旅居数据：使用本地闲置农宅样本，测算改造后入住率、客单价、年收入和村集体分成。",
        "",
        "## 三、推荐逻辑说明",
        "",
        "系统采用可解释多目标评分模型，不调用外部大模型，也不依赖真实 API。每个作物对每个地块计算四类正向评分和一类风险扣分：",
        "- 生态适配分：看地块类型、坡度、pH、有机质、水源距离、日照和作物固土能力。",
        "- 经济收益分：看亩均净收益，并在候选作物中归一化比较。",
        "- 农旅体验分：看作物观赏、采摘、研学价值和地块旅居可达性。",
        "- 劳动力适配分：看可用劳动力是否能覆盖作物管护强度。",
        "- 风险惩罚分：看作物风险、水源约束、坡度超限、pH 偏离和生态脆弱性。",
        "",
        "三类方案分别强调生态修复、经济增收和农旅融合，因此同一地块可以给村集体提供不同发展取向下的可比方案。",
        "",
        "## 四、10 个地块推荐结果",
        "",
        _recommendation_table(parcels, recommendations),
        "",
        "## 五、优选方案经济收益测算",
        "",
        _preferred_table(preferred_recommendations),
        "",
        "从测算结果看，优选组合既保留了生态修复型作物，也纳入了黄精、黄花菜、药草花境等具备经济和体验转化潜力的方向。这样可以避免单一作物押注，降低市场和管护风险。",
        "",
        "## 六、生态修复测算",
        "",
        f"10 个样本地块的优选方案生态修复评分平均值为 {summary['生态修复评分平均值']:.1f}/100。对坡度较高、水土流失风险较大的地块，系统更倾向推荐披碱草、冰草、文冠果等根系固土能力强、维护强度较低的作物；对林下和民宿周边地块，则优先考虑兼具生态覆盖和体验价值的药材、花境或菌菇方案。",
        "",
        "生态修复的意义不是简单绿化，而是把撂荒地、薄土地、坡地转化为可管护、可收益、可展示的生态产业空间。",
        "",
        "## 七、微旅居收益测算",
        "",
        f"本轮测算纳入 {len(homestay_projections)} 处闲置农宅样本，预计旅居年收入 {summary['微旅居年收入']:,.0f} 元，村集体分成 {summary['村集体分成']:,.0f} 元。",
        "",
        _homestay_table(homestay_projections),
        "",
        "微旅居模块的作用，是把种植端的花期、采摘、研学、林下体验与村内闲置房屋连接起来，形成“种养内容供给 + 旅居消费场景”的双循环。",
        "",
        "## 八、风险与应对",
        "",
        "| 风险 | 可能影响 | 应对策略 |",
        "|---|---|---|",
        "| 价格波动 | 中药材、菌菇类收益不稳定 | 先小规模试种，优先锁定订单和合作社收购渠道 |",
        "| 管护不足 | 高劳动强度作物影响产量 | 优先匹配村内可用劳动力，设置女性/中老年友好岗位 |",
        "| 水源距离 | 高需水作物在坡地落地难度较高 | 高需水作物优先放在林下、河滩或民宿周边可达水源地块 |",
        "| 生态约束 | 生态涵养区不能走高扰动开发路线 | 优先低扰动种植、林下经济、花境和轻量旅居体验 |",
        "| 运营断点 | 种植和旅居两端脱节 | 用年度计划串联试种、采摘、研学、民宿套餐和村集体分成 |",
        "",
        "## 九、路演判断",
        "",
        "MOCrop 的可行性来自三点：第一，数据结构能落到真实地块；第二，推荐逻辑可解释、可调权重、可复核；第三，收益测算同时覆盖种植端、就业端、生态端和微旅居端。该系统适合作为安洲坝村先导试点工具，后续可接入真实测绘、土壤检测和运营数据持续校准。",
    ]
    return "\n".join(lines)


def save_feasibility_report(markdown: str, output_dir: Path | None = None) -> Path:
    """按指定文件名格式保存项目可行性证明报告。"""

    directory = output_dir or REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"mocrop_feasibility_report_{datetime.now().strftime('%Y%m%d')}.md"
    path = directory / filename
    path.write_text(markdown, encoding="utf-8")
    return path


def _preferred_recommendation(
    parcel: Parcel,
    recommendations: list[Recommendation],
) -> Recommendation:
    parcel_recommendations = [item for item in recommendations if item.parcel_id == parcel.parcel_id]
    if parcel.erosion_risk == "high":
        return next(item for item in parcel_recommendations if item.mode == "生态优先模式")
    if parcel.land_type in {"homestay_nearby", "roadside"}:
        return next(item for item in parcel_recommendations if item.mode == "农旅融合模式")
    return next(item for item in parcel_recommendations if item.mode == "经济优先模式")


def _recommendation_table(
    parcels: list[Parcel],
    recommendations: list[Recommendation],
) -> str:
    rows = [
        "| 地块 | 地块类型 | 风险 | 生态优先 | 经济优先 | 农旅融合 |",
        "|---|---|---|---|---|---|",
    ]
    for parcel in parcels:
        parcel_recommendations = {
            item.mode: item for item in recommendations if item.parcel_id == parcel.parcel_id
        }
        rows.append(
            "| "
            + " | ".join(
                [
                    f"{parcel.name}（{parcel.area_mu:.1f}亩）",
                    land_type_label(parcel.land_type),
                    erosion_risk_label(parcel.erosion_risk),
                    _plan_cell(parcel_recommendations["生态优先模式"]),
                    _plan_cell(parcel_recommendations["经济优先模式"]),
                    _plan_cell(parcel_recommendations["农旅融合模式"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _preferred_table(recommendations: list[Recommendation]) -> str:
    rows = [
        "| 地块 | 优选模式 | 推荐作物 | 投入 | 收入 | 净收益 | 生态分 | 农旅分 | 用工 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in recommendations:
        rows.append(
            "| "
            + " | ".join(
                [
                    item.parcel_name,
                    item.mode,
                    item.crop_name,
                    f"{item.expected_investment_yuan:,.0f}",
                    f"{item.expected_revenue_yuan:,.0f}",
                    f"{item.expected_profit_yuan:,.0f}",
                    f"{item.ecological_score:.1f}",
                    f"{item.tourism_score:.1f}",
                    f"{item.labor_days:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _homestay_table(projections: list[HomestayProjection]) -> str:
    rows = [
        "| 闲置资源 | 房间数 | 入住率 | 客单价 | 年收入 | 村集体分成 | 回收期 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in projections:
        rows.append(
            "| "
            + " | ".join(
                [
                    item.name,
                    str(item.rooms),
                    f"{item.occupancy_rate:.0%}",
                    f"{item.avg_price_yuan:,.0f}",
                    f"{item.annual_revenue_yuan:,.0f}",
                    f"{item.village_share_yuan:,.0f}",
                    f"{item.payback_years:.1f} 年",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _plan_cell(recommendation: Recommendation) -> str:
    return (
        f"{recommendation.crop_name}"
        f" / 净收益 {recommendation.expected_profit_yuan:,.0f} 元"
        f" / 总分 {recommendation.score:.1f}"
    )
