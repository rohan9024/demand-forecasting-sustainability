from typing import List, Optional
from pydantic import BaseModel


class BatteryParams(BaseModel):
    capacity_kwh: float
    soc_init_kwh: float
    p_charge_max_kw: float
    p_discharge_max_kw: float
    eff_c: float = 0.95
    eff_d: float = 0.95


class GridParams(BaseModel):
    allow_export: bool = True
    export_limit_kw: float = 100.0


class TrainResponse(BaseModel):
    models: dict


class ForecastResponse(BaseModel):
    times: List[str]
    demand_kw: List[float]
    renewable_kw: List[float]
    ci_kg_per_kwh: List[float]


class OptimizeRequest(BaseModel):
    horizon_hours: Optional[int] = None
    battery: Optional[BatteryParams] = None
    grid: Optional[GridParams] = None


class SchedulePoint(BaseModel):
    time: str
    demand_kw: float
    renewable_kw: float
    ci_kg_per_kwh: float
    charge_kw: float
    discharge_kw: float
    grid_import_kw: float
    grid_export_kw: float
    soc_kwh: float
    renewable_used_kw: float
    curtailment_kw: float
    emissions_kg: float


class OptimizeSummary(BaseModel):
    baseline_emissions_kg: float
    optimized_emissions_kg: float
    reduction_kg: float
    reduction_pct: float
    renewable_utilization_pct: float
    curtailment_pct: float


class OptimizeResponse(BaseModel):
    schedule: List[SchedulePoint]
    summary: OptimizeSummary


class HistoryResponse(BaseModel):
    times: List[str]
    demand_kw: List[float]
    renewable_kw: List[float]
    ci_kg_per_kwh: List[float]