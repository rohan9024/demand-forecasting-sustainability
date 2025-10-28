import os
import sys
import time
import atexit
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# Prefer 127.0.0.1 on Windows to avoid localhost edge cases
# BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
BACKEND_URL = os.getenv("BACKEND_URL", "https://backend-rxf7.onrender.com")

PARSED = urlparse(BACKEND_URL)
BACKEND_HOST = PARSED.hostname or "127.0.0.1"
BACKEND_PORT = PARSED.port or 8000

SESSION = requests.Session()


def is_backend_up() -> bool:
    try:
        r = SESSION.get(f"{BACKEND_URL}/health", timeout=2)
        return r.ok
    except requests.RequestException:
        return False


def start_backend_if_needed() -> subprocess.Popen | None:
    """
    Try to start FastAPI backend with uvicorn if not running.
    Returns a Popen handle if we started it, else None.
    """
    if is_backend_up():
        return None

    repo_root = Path(__file__).resolve().parents[1]
    uvicorn_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        BACKEND_HOST,
        "--port",
        str(BACKEND_PORT),
        "--log-level",
        "warning",
        "--no-access-log",
    ]

    # Start backend as a background process
    try:
        proc = subprocess.Popen(
            uvicorn_cmd,
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        st.error(f"Failed to start backend automatically: {e}")
        return None

    # Wait for readiness
    for _ in range(60):  # up to ~30s
        if is_backend_up():
            return proc
        time.sleep(0.5)

    # Didn't come up; clean up
    try:
        proc.terminate()
    except Exception:
        pass
    return None


def api_get(path: str, params: dict | None = None) -> dict:
    resp = SESSION.get(f"{BACKEND_URL}{path}", params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json: dict | None = None, params: dict | None = None) -> dict:
    resp = SESSION.post(f"{BACKEND_URL}{path}", json=json, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# Streamlit page setup
st.set_page_config(page_title="Microgrid AI Optimizer", layout="wide", page_icon="⚡")
st.title("⚡ Microgrid AI: Forecasting + Emissions-Optimized Dispatch")

# Ensure backend is running (auto-start if needed)
if "backend_proc" not in st.session_state:
    st.session_state.backend_proc = None

if not is_backend_up():
    with st.spinner("Starting backend server..."):
        proc = start_backend_if_needed()
        if proc:
            st.session_state.backend_proc = proc
            # Kill backend when Streamlit session ends
            def _cleanup():
                try:
                    st.session_state.backend_proc.terminate()
                except Exception:
                    pass

            atexit.register(_cleanup)

# If still not up, show friendly help and stop
if not is_backend_up():
    st.error(
        "Could not connect to the backend API. Please start it manually in another terminal:\n\n"
        f"uvicorn backend.main:app --host {BACKEND_HOST} --port {BACKEND_PORT}\n\n"
        "Then refresh this page."
    )
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
horizon = st.sidebar.slider("Forecast horizon (hours)", min_value=4, max_value=72, value=24, step=2)
autorefresh = st.sidebar.checkbox("Auto-refresh simulation (every 10s)", value=False)
advance_steps = st.sidebar.number_input("Advance steps per refresh", min_value=1, max_value=6, value=1, step=1)

st.sidebar.subheader("Battery")
cap = st.sidebar.number_input("Capacity (kWh)", min_value=10.0, max_value=2000.0, value=100.0, step=10.0)
soc0 = st.sidebar.number_input("Initial SoC (kWh)", min_value=0.0, max_value=2000.0, value=50.0, step=10.0)
pc = st.sidebar.number_input("Max charge (kW)", min_value=1.0, max_value=2000.0, value=50.0, step=5.0)
pdw = st.sidebar.number_input("Max discharge (kW)", min_value=1.0, max_value=2000.0, value=50.0, step=5.0)
eff_c = st.sidebar.slider("Charge efficiency", min_value=0.7, max_value=1.0, value=0.95, step=0.01)
eff_d = st.sidebar.slider("Discharge efficiency", min_value=0.7, max_value=1.0, value=0.95, step=0.01)

st.sidebar.subheader("Grid")
allow_export = st.sidebar.checkbox("Allow export to grid", value=True)
export_limit = st.sidebar.number_input("Export limit (kW)", min_value=0.0, max_value=5000.0, value=100.0, step=10.0)

# Layout columns
colL, colR = st.columns([1.2, 1])

# Recent history
with colL:
    try:
        hist = api_get("/history", params={"hours": 48})
        hdf = pd.DataFrame(
            {
                "time": pd.to_datetime(hist["times"]),
                "demand_kw": hist["demand_kw"],
                "renewable_kw": hist["renewable_kw"],
                "ci": hist["ci_kg_per_kwh"],
            }
        ).set_index("time")

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=hdf.index, y=hdf["demand_kw"], name="Demand (kW)", line=dict(color="#1f77b4")))
        fig_hist.add_trace(
            go.Scatter(x=hdf.index, y=hdf["renewable_kw"], name="Renewables (kW)", line=dict(color="#2ca02c"))
        )
        fig_hist.add_trace(
            go.Scatter(
                x=hdf.index,
                y=hdf["ci"] * 1000,
                name="Carbon Intensity (gCO2/kWh)",
                yaxis="y2",
                line=dict(color="#ff7f0e", dash="dot"),
            )
        )
        fig_hist.update_layout(
            title="Recent History",
            xaxis_title="Time",
            yaxis_title="kW",
            yaxis2=dict(title="gCO2/kWh", overlaying="y", side="right"),
            legend=dict(orientation="h"),
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    except requests.RequestException as e:
        st.error(f"Failed to load history from backend: {e}")
        st.stop()

with colR:
    if st.button("Train/Re-train models"):
        try:
            res = api_post("/train")
            st.success("Models trained")
            st.json(res["models"])
        except requests.RequestException as e:
            st.error(f"Training failed: {e}")

# Optional button to trigger rerun
if st.button("Run Forecast + Optimize"):
    pass  # a rerun is enough because the section below always re-requests

# Always compute optimize with current parameters
opt_req = {
    "horizon_hours": horizon,
    "battery": {
        "capacity_kwh": cap,
        "soc_init_kwh": soc0,
        "p_charge_max_kw": pc,
        "p_discharge_max_kw": pdw,
        "eff_c": eff_c,
        "eff_d": eff_d,
    },
    "grid": {"allow_export": allow_export, "export_limit_kw": export_limit},
}

try:
    opt = api_post("/optimize", json=opt_req)
except requests.RequestException as e:
    st.error(f"Optimization failed: {e}")
    st.stop()

# Build schedule dataframe
sched = pd.DataFrame(opt["schedule"])
sched["time"] = pd.to_datetime(sched["time"])
sched = sched.set_index("time")
kpis = opt["summary"]

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Baseline Emissions (kg CO₂)", f"{kpis['baseline_emissions_kg']:.1f}")
k2.metric("Optimized Emissions (kg CO₂)", f"{kpis['optimized_emissions_kg']:.1f}")
k3.metric("Reduction (kg)", f"{kpis['reduction_kg']:.1f}", f"{kpis['reduction_pct']:.1f}%")
k4.metric("Renewable Utilization", f"{kpis['renewable_utilization_pct']:.1f}%")

# Plots: demand/renewables + grid import/export + charge/discharge
fig = go.Figure()
fig.add_trace(go.Scatter(x=sched.index, y=sched["demand_kw"], name="Demand (kW)", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=sched.index, y=sched["renewable_kw"], name="Renewables (kW)", line=dict(color="#2ca02c")))
fig.add_trace(go.Bar(x=sched.index, y=sched["grid_import_kw"], name="Grid Import (kW)", marker_color="#9467bd", opacity=0.6))
if allow_export:
    fig.add_trace(
        go.Bar(x=sched.index, y=-sched["grid_export_kw"], name="Grid Export (kW)", marker_color="#8c564b", opacity=0.6)
    )
fig.add_trace(go.Bar(x=sched.index, y=sched["charge_kw"], name="Charge (kW)", marker_color="#17becf", opacity=0.5))
fig.add_trace(
    go.Bar(x=sched.index, y=-sched["discharge_kw"], name="Discharge (kW)", marker_color="#d62728", opacity=0.5)
)

fig.update_layout(
    title="Forecast + Optimized Dispatch",
    barmode="relative",
    xaxis_title="Time",
    yaxis_title="kW (export/discharge negative)",
    legend=dict(orientation="h"),
    height=440,
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# SoC + Carbon intensity
c1, c2 = st.columns(2)
with c1:
    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(x=sched.index, y=sched["soc_kwh"], name="SoC (kWh)", line=dict(color="#636efa")))
    fig_soc.update_layout(title="Battery State of Charge", xaxis_title="Time", yaxis_title="kWh", height=320)
    st.plotly_chart(fig_soc, use_container_width=True)
with c2:
    fig_ci = go.Figure()
    fig_ci.add_trace(
        go.Scatter(x=sched.index, y=sched["ci_kg_per_kwh"] * 1000, name="gCO2/kWh", line=dict(color="#ff7f0e"))
    )
    fig_ci.update_layout(title="Forecast Grid Carbon Intensity", xaxis_title="Time", yaxis_title="gCO₂/kWh", height=320)
    st.plotly_chart(fig_ci, use_container_width=True)

# Action hints for next 4 intervals
st.subheader("Next Actions")
next_rows = sched.iloc[:4]
actions = []
for t, r in next_rows.iterrows():
    if r["charge_kw"] > 1:
        act = f"{t}: Charge {r['charge_kw']:.1f} kW"
    elif r["discharge_kw"] > 1:
        act = f"{t}: Discharge {r['discharge_kw']:.1f} kW"
    else:
        act = f"{t}: Idle"
    actions.append(act)
st.write("- " + "\n- ".join(actions))

# Download schedule
st.download_button(
    "Download schedule CSV",
    data=sched.reset_index().to_csv(index=False),
    file_name="optimized_schedule.csv",
    mime="text/csv",
)

# Auto-refresh simulation
if autorefresh:
    try:
        api_post("/simulate/advance", params={"steps": int(advance_steps)})
    except requests.RequestException:
        pass
    # Streamlit 1.27+: st.rerun(); older versions: st.experimental_rerun()
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()