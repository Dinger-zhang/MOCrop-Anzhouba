from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .models import HomestayProjection, Recommendation


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def render_markdown_report(
    recommendations: list[Recommendation],
    homestay_projections: list[HomestayProjection],
    title: str = "MOCrop 因地智宜决策报告",
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_profit = sum(item.expected_profit_yuan for item in recommendations)
    total_revenue = sum(item.annual_revenue_yuan for item in homestay_projections)
    village_share = sum(item.village_share_yuan for item in homestay_projections)

    lines = [
        f"# {title}",
        "",
        f"- 生成时间：{generated_at}",
        f"- 推荐方案数：{len(recommendations)}",
        f"- 种植方案预计净收益合计：{total_profit:,.0f} 元",
        f"- 微旅居预计年收入合计：{total_revenue:,.0f} 元",
        f"- 村集体预计分成合计：{village_share:,.0f} 元",
        "",
        "## 地块推荐结果",
        "",
        _recommendation_table(recommendations),
        "",
        "## 微旅居收益测算",
        "",
        _homestay_table(homestay_projections),
        "",
        "## 年度实施计划",
        "",
        "| 阶段 | 重点任务 | 产出 |",
        "|---|---|---|",
        "| 2026 年 Q3 | 完成地块建档、土壤快检、村民劳动力摸底 | 一地一策数据库 |",
        "| 2026 年 Q4 | 启动生态修复试种和林下经济样板 | 生态优先示范片 |",
        "| 2027 年 H1 | 建设药草花境、黄花菜采摘和研学动线 | 农旅融合体验产品 |",
        "| 2027 年 H2 | 改造样板民宿并联动采摘、研学、康养套餐 | 微旅居运营闭环 |",
        "| 2028 年 | 扩面复制并接入合作社订单和村集体分成机制 | 双循环产业模型 |",
    ]
    return "\n".join(lines)


def save_markdown_report(markdown: str, output_dir: Path | None = None) -> Path:
    directory = output_dir or REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"mocrop_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path = directory / filename
    path.write_text(markdown, encoding="utf-8")
    return path


def _recommendation_table(recommendations: list[Recommendation]) -> str:
    rows = [
        "| 地块 | 模式 | 推荐作物 | 投入 | 收入 | 净收益 | 用工 | 生态分 | 农旅分 | 风险 | 推荐理由 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
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
                    f"{item.labor_days:.1f}",
                    f"{item.ecological_score:.1f}",
                    f"{item.tourism_score:.1f}",
                    item.risk_level,
                    "；".join(item.reasons),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _homestay_table(projections: list[HomestayProjection]) -> str:
    rows = [
        "| 民宿资源 | 房间 | 改造成本 | 入住率 | 客单价 | 年收入 | 净收入 | 村集体分成 | 回收期 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in projections:
        rows.append(
            "| "
            + " | ".join(
                [
                    item.name,
                    str(item.rooms),
                    f"{item.renovation_cost_yuan:,.0f}",
                    f"{item.occupancy_rate:.0%}",
                    f"{item.avg_price_yuan:,.0f}",
                    f"{item.annual_revenue_yuan:,.0f}",
                    f"{item.net_income_yuan:,.0f}",
                    f"{item.village_share_yuan:,.0f}",
                    f"{item.payback_years:.1f} 年",
                ]
            )
            + " |"
        )
    return "\n".join(rows)
