from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
import random


@dataclass
class SimConfig:
    freq_minutes: int
    history_hours: int
    tz: str
    seed: int
    base_demand_kw: float
    demand_amp_kw: float
    solar_kw_peak: float
    wind_kw_peak: float
    ci_base: float
    ci_amp: float


class MicrogridSimulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.freq = f"{cfg.freq_minutes}min"
        self.df = self._generate_history()

    def _now_local(self) -> pd.Timestamp:
        """
        Get current time as timezone-aware Timestamp in configured tz.
        Compatible with pandas 2.x where Timestamp.utcnow() is tz-aware.
        """
        # Start in UTC as tz-aware, then convert to target tz
        now_utc = pd.Timestamp.now(tz="UTC")
        try:
            return now_utc.tz_convert(self.cfg.tz)
        except Exception:
            # Fallback to UTC if provided tz is not recognized
            return now_utc

    def _generate_history(self) -> pd.DataFrame:
        now = self._now_local()
        # Align start and end to the simulator frequency
        end = now.floor(self.freq)
        start = (now - pd.Timedelta(hours=self.cfg.history_hours)).floor(self.freq)
        idx = pd.date_range(start=start, end=end, freq=self.freq)
        return self._simulate_on_index(idx)

    def _simulate_on_index(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        # Time features
        hour = idx.hour.values
        dow = idx.dayofweek.values
        doy = idx.dayofyear.values

        # Demand: base + daily sine + weekly modulation + noise
        demand_daily = np.sin((hour - 14) / 24 * 2 * np.pi)  # peak late afternoon
        demand_weekly = 0.1 * np.sin((dow) / 7 * 2 * np.pi)
        demand_kw = (
            self.cfg.base_demand_kw
            + self.cfg.demand_amp_kw * demand_daily
            + self.cfg.demand_amp_kw * demand_weekly
            + np.random.normal(0, 5, size=len(idx))
        )
        demand_kw = np.clip(demand_kw, 10, None)

        # Solar irradiance-like curve (very simplified)
        solar_shape = np.sin((hour - 6) / 12 * np.pi)
        solar_shape = np.clip(solar_shape, 0, None)
        # Seasonality by day-of-year
        seasonal = 0.6 + 0.4 * np.sin((doy - 172) / 365 * 2 * np.pi)
        solar_kw = self.cfg.solar_kw_peak * solar_shape * seasonal
        # Cloud noise (smoothed random)
        clouds = np.maximum(0, np.random.normal(0.9, 0.15, size=len(idx)))
        solar_kw = solar_kw * clouds

        # Wind: mean + noise + low-frequency variation
        wind_base = 0.5 * self.cfg.wind_kw_peak
        wind = wind_base + 0.5 * self.cfg.wind_kw_peak * (
            0.5 * np.sin((hour) / 24 * 2 * np.pi + 1.5) + 0.5 * np.sin((doy) / 14 * 2 * np.pi)
        )
        wind += np.random.normal(0, 5, size=len(idx))
        wind_kw = np.clip(wind, 0, None)

        renewable_kw = solar_kw + wind_kw

        # Carbon intensity: lower when renewable share is higher, plus daily cycle + noise
        ren_norm = renewable_kw / (demand_kw + 1e-6)
        ci = (
            self.cfg.ci_base
            + self.cfg.ci_amp * np.sin((hour - 20) / 24 * 2 * np.pi)  # higher evenings
            - 0.08 * ren_norm
            + np.random.normal(0, 0.01, size=len(idx))
        )
        ci = np.clip(ci, 0.05, 0.8)

        temp_c = 18 + 7 * np.sin((hour - 13) / 24 * 2 * np.pi) + np.random.normal(0, 1, size=len(idx))

        df = pd.DataFrame(
            {
                "time": idx,
                "demand_kw": demand_kw,
                "solar_kw": solar_kw,
                "wind_kw": wind_kw,
                "renewable_kw": renewable_kw,
                "ci_kg_per_kwh": ci,
                "temp_c": temp_c,
            }
        ).set_index("time")
        return df

    def step(self, steps: int = 1):
        # advance simulator by N steps and append
        last_ts = self.df.index[-1]
        new_idx = pd.date_range(
            start=last_ts + pd.Timedelta(minutes=self.cfg.freq_minutes),
            periods=steps,
            freq=self.freq,
        )
        new_df = self._simulate_on_index(new_idx)
        self.df = pd.concat([self.df, new_df])

    def get_history(self, hours: int) -> pd.DataFrame:
        cutoff = self.df.index[-1] - pd.Timedelta(hours=hours)
        return self.df[self.df.index >= cutoff].copy()

    def all_history(self) -> pd.DataFrame:
        return self.df.copy()