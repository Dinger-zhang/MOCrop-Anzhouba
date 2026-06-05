from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import pandas as pd

from .models import Crop, Homestay, LaborAvailability, MarketPrice, Parcel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEMO_DATA_FILES: dict[str, str] = {
    "parcels.csv": """\
parcel_id,name,area_mu,latitude,longitude,slope_degree,soil_ph,organic_matter,sunlight_hours,water_distance_m,land_type,erosion_risk,labor_available,idle_years,current_status
P001,老梯田北坡,18.5,40.6938,116.6643,22,6.8,24,7.8,520,slope,high,7,5,撂荒梯田
P002,沟口缓坡地,12.0,40.6905,116.6681,8,6.6,31,8.2,180,abandoned,medium,10,2,闲置菜地
P003,林下阴坡地,16.2,40.6972,116.6617,15,6.2,36,4.6,360,forest_understory,medium,6,4,疏林空地
P004,河滩边际地,9.8,40.6889,116.6716,3,7.1,29,8.8,90,abandoned,low,8,1,河滩边际地
P005,村道观景台周边,7.5,40.6921,116.6727,6,6.7,27,8.5,240,homestay_nearby,medium,6,3,闲置坡台
P006,高海拔薄土地,21.0,40.7004,116.6579,28,6.4,18,7.4,680,slope,high,5,7,石质化坡地
""",
    "crops.csv": """\
crop_id,crop_name,crop_category,suitable_land_types,ph_min,ph_max,slope_tolerance,water_need_level,labor_intensity,expected_yield_per_mu,market_price,input_cost_per_mu,soil_conservation_score,tourism_score,risk_level,cycle_years,reason_tags
C001,冰草,生态修复,slope;abandoned;roadside,6.0,8.2,35,low,low,450,1.8,620,92,38,low,1,耐旱固土;适合坡地;养护简单
C002,披碱草,生态修复,slope;abandoned,5.8,8.3,38,low,low,420,1.6,580,95,32,low,1,根系发达;抑制水土流失;适合薄土
C003,黄精,中药材,forest_understory;homestay_nearby;abandoned,5.5,7.2,18,medium,high,320,72,4200,65,72,medium,3,林下经济;药食同源;研学价值
C004,苍术,中药材,slope;abandoned;forest_understory,5.8,7.5,25,medium,medium,280,48,3000,68,60,medium,2,耐旱药材;市场需求稳定;适合山地
C005,柴胡,中药材,slope;abandoned;roadside,6.0,8.0,30,low,medium,220,42,2200,72,55,medium,2,坡地药材;管理强度适中;根系护坡
C006,赤松茸,食用菌,forest_understory;homestay_nearby,5.5,7.0,12,high,high,900,16,5200,58,76,high,1,林下菌菇;采摘体验;收益弹性高
C007,木耳,食用菌,forest_understory;homestay_nearby,5.5,7.2,15,high,high,650,28,4800,54,70,high,1,设施化栽培;可做体验课程;需水较高
C008,榛子,坚果油料,slope;abandoned;roadside,6.0,7.8,22,medium,medium,180,42,3600,78,68,medium,5,生态经济林;长期收益;采摘属性
C009,文冠果,坚果油料,slope;abandoned;roadside,6.2,8.5,32,low,low,160,36,2800,86,66,medium,5,耐旱木本油料;固碳护坡;景观花期
C010,黄花菜,农旅体验,abandoned;roadside;homestay_nearby,6.0,7.8,18,medium,medium,520,14,2600,70,90,medium,2,花期观赏;采摘加工;适合研学
C011,观赏谷子,农旅体验,roadside;homestay_nearby;abandoned,6.0,8.0,16,low,medium,380,8,1800,62,88,low,1,色彩景观;节庆打卡;管理周期短
C012,药草花境,农旅体验,homestay_nearby;roadside;abandoned,6.0,7.8,20,medium,medium,260,36,2400,76,94,low,1,香草体验;低影响景观;可连接民宿
""",
    "market_prices.csv": """\
crop_id,price_yuan_per_kg,demand_score,price_volatility,source_note
C001,1.8,45,20,演示估算-饲草收储价
C002,1.6,42,18,演示估算-饲草收储价
C003,72,88,35,演示估算-中药材干品折算价
C004,48,84,32,演示估算-中药材干品折算价
C005,42,78,30,演示估算-中药材干品折算价
C006,16,86,48,演示估算-菌菇订单价
C007,28,82,45,演示估算-菌菇订单价
C008,42,76,34,演示估算-坚果初产折算价
C009,36,70,38,演示估算-木本油料折算价
C010,14,80,28,演示估算-鲜菜与加工综合价
C011,8,66,22,演示估算-景观谷物综合价
C012,36,62,18,演示估算-香草花境体验折算价
""",
    "labor.csv": """\
season,available_workers,available_days,average_wage_yuan_per_day,notes
春季,26,45,160,整地栽植和生态修复用工较集中
夏季,31,60,170,管护采摘和研学接待用工增加
秋季,34,50,165,采收加工和民宿运营用工增加
冬季,18,30,155,设施维护和农宅改造用工为主
""",
    "homestays.csv": """\
house_id,name,rooms,renovation_cost_yuan,base_occupancy_rate,avg_price_yuan,operating_cost_rate,village_share_rate,tourism_score,status
H001,石墙小院样板间,4,180000,0.36,420,0.32,0.12,86,可改造
H002,山景闲置农宅,6,260000,0.32,380,0.34,0.10,78,需整修
H003,研学接待小院,8,360000,0.42,460,0.35,0.15,90,优先盘活
H004,林下露营配套屋,3,120000,0.28,320,0.30,0.08,72,轻改造
""",
}


@dataclass(frozen=True)
class DataBundle:
    parcels: list[Parcel]
    crops: list[Crop]
    market_prices: list[MarketPrice]
    labor: list[LaborAvailability]
    homestays: list[Homestay]


def ensure_demo_data(data_dir: Path = DATA_DIR) -> None:
    """确保本地演示 CSV 存在；缺失时自动生成可运行 demo 数据。"""

    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in DEMO_DATA_FILES.items():
        path = data_dir / filename
        if not path.exists():
            path.write_text(dedent(content), encoding="utf-8")


def _read_csv(filename: str) -> pd.DataFrame:
    ensure_demo_data(DATA_DIR)
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"演示数据文件不存在: {path}")
    return pd.read_csv(path)


def load_parcels() -> list[Parcel]:
    """加载地块演示数据。"""

    return [Parcel(**row) for row in _read_csv("parcels.csv").to_dict("records")]


def load_crops() -> list[Crop]:
    """加载作物知识库演示数据。"""

    return [Crop(**row) for row in _read_csv("crops.csv").to_dict("records")]


def load_market_prices() -> list[MarketPrice]:
    """加载市场价格演示数据。"""

    return [MarketPrice(**row) for row in _read_csv("market_prices.csv").to_dict("records")]


def load_labor() -> list[LaborAvailability]:
    """加载劳动力供给演示数据。"""

    return [LaborAvailability(**row) for row in _read_csv("labor.csv").to_dict("records")]


def load_homestays() -> list[Homestay]:
    """加载闲置农宅和微旅居演示数据。"""

    return [Homestay(**row) for row in _read_csv("homestays.csv").to_dict("records")]


def load_data_bundle() -> DataBundle:
    """加载系统运行所需的全部本地演示数据。"""

    return DataBundle(
        parcels=load_parcels(),
        crops=load_crops(),
        market_prices=load_market_prices(),
        labor=load_labor(),
        homestays=load_homestays(),
    )
