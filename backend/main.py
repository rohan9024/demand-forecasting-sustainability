from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    BatteryParams,
    ForecastResponse,
    GridParams,
    HistoryResponse,
    OptimizeRequest,
    OptimizeResponse,
    OptimizeSummary,
    SchedulePoint,
    TrainResponse,
)
from backend.data.simulator import MicrogridSimulator, SimConfig
from backend.models.forecaster import Forecaster
from backend.models.recommender import EnergyRecommender

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("microgrid")

# Load config
ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "config.yaml"
with open(CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

time_cfg = CFG["time"]
sim_cfg = CFG["simulation"]
bat_cfg = CFG["battery"]
grid_cfg = CFG["grid"]

# Core components (lightweight)
sim = MicrogridSimulator(
    SimConfig(
        freq_minutes=time_cfg["freq_minutes"],
        history_hours=time_cfg["history_hours"],
        tz=sim_cfg.get("timezone", "UTC"),
        seed=sim_cfg["seed"],
        base_demand_kw=sim_cfg["base_demand_kw"],
        demand_amp_kw=sim_cfg["demand_amp_kw"],
        solar_kw_peak=sim_cfg["solar_kw_peak"],
        wind_kw_peak=sim_cfg["wind_kw_peak"],
        ci_base=sim_cfg["ci_base"],
        ci_amp=sim_cfg["ci_amp"],
    )
)

forecaster = Forecaster(freq_minutes=time_cfg["freq_minutes"])
recommender = EnergyRecommender(dt_hours=time_cfg["freq_minutes"] / 60.0)

app = FastAPI(title="Microgrid AI Backend", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# App state flags
app.state.model_ready = False
app.state.training = False
app.state.train_metrics = None
app.state.last_error = None


def _train_models_blocking():
    try:
        app.state.training = True
        log.info("Training models...")
        hist = sim.all_history()[["demand_kw", "renewable_kw", "ci_kg_per_kwh"]]
        metrics = forecaster.fit(hist)
        app.state.train_metrics = metrics
        app.state.model_ready = True
        app.state.last_error = None
        log.info(f"Training done. Metrics: {metrics}")
    except Exception as e:
        app.state.last_error = str(e)
        app.state.model_ready = False
        log.exception("Training failed")
    finally:
        app.state.training = False


def ensure_models_ready():
    if not app.state.model_ready and not app.state.training:
        # Train synchronously on-demand (so endpoints work even if startup thread failed)
        _train_models_blocking()


@app.on_event("startup")
def on_startup():
    # Train models in the background so /health returns immediately
    t = threading.Thread(target=_train_models_blocking, daemon=True)
    t.start()


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_ready": bool(app.state.model_ready),
        "training": bool(app.state.training),
        "last_error": app.state.last_error,
        "version": app.version if hasattr(app, "version") else "1.1.0",
    }


@app.post("/train", response_model=TrainResponse)
def train():
    _train_models_blocking()
    return {"models": app.state.train_metrics or {}}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(horizon_hours: int = Query(default=time_cfg["forecast_horizon_hours"], ge=1, le=168)):
    ensure_models_ready()
    if not app.state.model_ready:
        raise HTTPException(status_code=503, detail=f"Models not ready: {app.state.last_error}")

    fdf = forecaster.forecast(horizon_hours)
    return {
        "times": [t.isoformat() for t in fdf.index],
        "demand_kw": [float(x) for x in fdf["demand_kw"].values],
        "renewable_kw": [float(x) for x in fdf["renewable_kw"].values],
        "ci_kg_per_kwh": [float(x) for x in fdf["ci_kg_per_kwh"].values],
    }


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    ensure_models_ready()
    if not app.state.model_ready:
        raise HTTPException(status_code=503, detail=f"Models not ready: {app.state.last_error}")

    horizon_hours = req.horizon_hours or time_cfg["forecast_horizon_hours"]
    fdf = forecaster.forecast(horizon_hours)

    battery = {
        "capacity_kwh": req.battery.capacity_kwh if req.battery else bat_cfg["capacity_kwh"],
        "soc_init_kwh": req.battery.soc_init_kwh if req.battery else bat_cfg["soc_init_kwh"],
        "p_charge_max_kw": req.battery.p_charge_max_kw if req.battery else bat_cfg["p_charge_max_kw"],
        "p_discharge_max_kw": req.battery.p_discharge_max_kw if req.battery else bat_cfg["p_discharge_max_kw"],
        "eff_c": req.battery.eff_c if req.battery else bat_cfg["eff_c"],
        "eff_d": req.battery.eff_d if req.battery else bat_cfg["eff_d"],
    }
    grid = {
        "allow_export": req.grid.allow_export if req.grid else grid_cfg["allow_export"],
        "export_limit_kw": req.grid.export_limit_kw if req.grid else grid_cfg["export_limit_kw"],
    }

    try:
        sched = recommender.optimize(fdf, battery, grid)
    except Exception as e:
        log.exception("Optimization failed")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

    kpis = sched.attrs["kpis"]

    points: List[SchedulePoint] = []
    for t, row in sched.reset_index().iterrows():
        points.append(
            SchedulePoint(
                time=row["time"].isoformat(),
                demand_kw=float(row["demand_kw"]),
                renewable_kw=float(row["renewable_kw"]),
                ci_kg_per_kwh=float(row["ci_kg_per_kwh"]),
                charge_kw=float(row["charge_kw"]),
                discharge_kw=float(row["discharge_kw"]),
                grid_import_kw=float(row["grid_import_kw"]),
                grid_export_kw=float(row["grid_export_kw"]),
                soc_kwh=float(row["soc_kwh"]),
                renewable_used_kw=float(row["renewable_used_kw"]),
                curtailment_kw=float(row["curtailment_kw"]),
                emissions_kg=float(row["emissions_kg"]),
            )
        )

    summary = OptimizeSummary(
        baseline_emissions_kg=float(kpis["baseline_emissions_kg"]),
        optimized_emissions_kg=float(kpis["optimized_emissions_kg"]),
        reduction_kg=float(kpis["reduction_kg"]),
        reduction_pct=float(kpis["reduction_pct"]),
        renewable_utilization_pct=float(kpis["renewable_utilization_pct"]),
        curtailment_pct=float(kpis["curtailment_pct"]),
    )
    return OptimizeResponse(schedule=points, summary=summary)


HISTORY_MAX = time_cfg["history_hours"]


@app.get("/history", response_model=HistoryResponse)
def history(hours: int = Query(default=48, ge=1, le=HISTORY_MAX)):
    h = sim.get_history(hours)[["demand_kw", "renewable_kw", "ci_kg_per_kwh"]]
    return {
        "times": [t.isoformat() for t in h.index],
        "demand_kw": [float(x) for x in h["demand_kw"].values],
        "renewable_kw": [float(x) for x in h["renewable_kw"].values],
        "ci_kg_per_kwh": [float(x) for x in h["ci_kg_per_kwh"].values],
    }


@app.post("/simulate/advance")
def simulate_advance(steps: int = 1):
    sim.step(steps=steps)
    # Keep forecaster history fresh (no re-train on every step; training is explicit)
    forecaster.history = sim.all_history()[["demand_kw", "renewable_kw", "ci_kg_per_kwh"]]
    return {"ok": True, "advanced_steps": steps}