#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show & Well Synthetic Data Generator (CLEAN)
- Generates dimensions and fact tables as CSVs (Power BI / SQL ready)
- Fully synthetic; reproducible via --seed
- Includes membership tiers with daily proration
"""

import argparse
from datetime import datetime, timedelta
import calendar
import random
import string

import numpy as np
import pandas as pd

# ------------------------------
# Defaults
# ------------------------------
DEFAULT_CITIES = [
    "Minneapolis", "St Paul", "Eden Prairie", "Bloomington",
    "Hopkins", "Plymouth", "St Louis Park", "Edina"
]

DEFAULT_PROGRAMS = ["Movement", "Nutrition", "Stress", "Weight Loss Goal"]

PRICE_BY_PROGRAM = {
    "Movement": 75,
    "Nutrition": 95,
    "Stress": 100,
    "Weight Loss Goal": 85,
}

CREDENTIAL_OPTIONS = [
    ("Coach", 14, 28),
    ("LPC", 12, 22),
    ("LMFT", 12, 22),
    ("RD", 10, 18),  # Registered Dietitian
]

NOSHOW_RATES = {
    "Movement": (0.06, 0.14),
    "Nutrition": (0.05, 0.12),
    "Stress": (0.08, 0.16),
    "Weight Loss Goal": (0.06, 0.14),
}

SEASONALITY = {
    12: 1.15, 1: 1.30, 2: 1.10, 3: 1.05, 4: 1.00,
    5: 0.95, 6: 0.90, 7: 0.92, 8: 0.95, 9: 1.00
}

CITY_FACTOR = {
    "Minneapolis": 1.00, "St Paul": 0.85, "Eden Prairie": 0.45, "Bloomington": 0.55,
    "Hopkins": 0.35, "Plymouth": 0.50, "St Louis Park": 0.40, "Edina": 0.48,
}

SOURCES = ["Web", "Referral", "Event"]
SOURCE_WEIGHTS = {"Web": 0.60, "Referral": 0.27, "Event": 0.13}

WANTS_TRAINER_RATE = 0.35
BASE_WAITLIST_RATE = 0.20
WAITLIST_CONV_RANGE = (0.55, 0.85)

# Membership tiers
MEMBERSHIP_TIERS = [
    {"tier_key": 1, "tier_name": "Basic", "monthly_fee_usd": 50.0},
    {"tier_key": 2, "tier_name": "Premium", "monthly_fee_usd": 150.0},
    {"tier_key": 3, "tier_name": "Elite", "monthly_fee_usd": 300.0},
]
INIT_TIER_PROBS = np.array([0.5, 0.35, 0.15])
TIER_TRANSITION = np.array([
    [0.88, 0.10, 0.02],  # Basic -> Basic/Premium/Elite
    [0.06, 0.88, 0.06],  # Premium -> Basic/Premium/Elite
    [0.02, 0.12, 0.86],  # Elite -> Basic/Premium/Elite
])

# ------------------------------
# Helpers
# ------------------------------
def days_in_month(dt: pd.Timestamp) -> int:
    return calendar.monthrange(dt.year, dt.month)[1]

def random_name() -> str:
    first = random.choice(["Alex","Sam","Jordan","Taylor","Chris","Morgan","Jamie","Riley","Cameron","Avery","Skyler","Casey"])
    last  = "".join(random.choices(string.ascii_uppercase, k=1)) + "".join(random.choices(string.ascii_lowercase, k=6))
    return f"{first} {last}"

def make_date_dim(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": dates})
    df["date_key"] = df["date"].dt.strftime("%Y%m%d").astype(int)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["quarter"] = df["date"].dt.quarter
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["dow"] = df["date"].dt.weekday  # 0=Mon
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    return df[["date_key","date","year","quarter","month","day","weekofyear","dow","is_weekend"]]

def make_geo_dim(cities: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"city": cities})
    df["state"] = "MN"
    df["region"] = "Upper Midwest"
    # simple population buckets for variety
    pop_buckets = ["S", "M", "M", "M", "S", "M", "S", "M"]
    df["population_bucket"] = pop_buckets[:len(df)]
    df["geo_key"] = np.arange(1, len(df)+1, dtype=int)
    return df[["geo_key","city","state","region","population_bucket"]]

def make_program_dim(programs: list[str]) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(programs, start=1):
        rows.append({
            "program_key": i,
            "program_name": p,
            "price_per_session": PRICE_BY_PROGRAM.get(p, 85),
            "focus_area": p,
        })
    return pd.DataFrame(rows)

def make_provider_dim(n: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    total_days = (end - start).days
    for i in range(1, n+1):
        cred, cap_min, cap_max = random.choice(CREDENTIAL_OPTIONS)
        weekly_capacity = random.randint(cap_min, cap_max)
        ob_offset = random.randint(0, min(total_days, 90))
        onboarding_date = (start + timedelta(days=ob_offset)).date()
        rows.append({
            "provider_key": i,
            "provider_name": random_name(),
            "credential_level": cred,
            "onboarding_date": onboarding_date.isoformat(),
            "capacity_slots_week": weekly_capacity,
            "avg_session_length_min": random.choice([45, 50, 55, 60])
        })
    return pd.DataFrame(rows)

# ------------------------------
# Generation steps
# ------------------------------
def generate_signups(dim_date, dim_geo) -> pd.DataFrame:
    rows = []
    for _, drow in dim_date.iterrows():
        month = int(drow["month"])
        date_key = int(drow["date_key"])
        dow = int(drow["dow"])
        weekday_multiplier = 1.0 if dow < 5 else 0.85
        for _, grow in dim_geo.iterrows():
            city = grow["city"]
            city_mult = CITY_FACTOR.get(city, 0.5)
            base_city = 10 * city_mult
            seasonal = SEASONALITY.get(month, 1.0)
            lam = max(base_city * seasonal * weekday_multiplier, 0.3)
            total = np.random.poisson(lam=lam)
            weights = np.array([SOURCE_WEIGHTS[s] for s in SOURCES], dtype=float); weights /= weights.sum()
            parts = np.random.multinomial(total, weights) if total>0 else np.array([0,0,0])
            for src, cnt in zip(SOURCES, parts):
                if cnt == 0: continue
                rows.append({
                    "date_key": date_key,
                    "geo_key": int(grow["geo_key"]),
                    "source": src,
                    "signups": int(cnt),
                    "wants_trainer_count": int(np.random.binomial(cnt, WANTS_TRAINER_RATE)),
                })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date_key","geo_key","source","signups","wants_trainer_count"])
    return df

def generate_waitlist(fact_signups, dim_geo, dim_program, dim_provider):
    # provider capacity per geo (assign providers to a home geo)
    provider_geo_map = {pk: random.choice(dim_geo["geo_key"].tolist()) for pk in dim_provider["provider_key"]}
    prov = dim_provider.copy()
    prov["daily_capacity"] = (prov["capacity_slots_week"] / 7.0).round(2)
    prov["geo_key"] = prov["provider_key"].map(provider_geo_map)

    cap_by_geo = prov.groupby("geo_key", as_index=False)["daily_capacity"].sum().rename(columns={"daily_capacity":"geo_daily_capacity"})
    geo_daily_capacity = dim_geo[["geo_key"]].merge(cap_by_geo, on="geo_key", how="left").fillna({"geo_daily_capacity": 0.0})

    signups_by = fact_signups.groupby(["date_key","geo_key"], as_index=False)["signups"].sum()
    rows = []
    for _, row in signups_by.iterrows():
        date_key = int(row["date_key"]); geo_key = int(row["geo_key"])
        total_signups = int(row["signups"])
        cap = float(geo_daily_capacity.loc[geo_daily_capacity["geo_key"] == geo_key, "geo_daily_capacity"].fillna(0).values[0])
        ratio = (total_signups / (cap + 1e-6)) if cap > 0 else 2.0
        wl_rate = BASE_WAITLIST_RATE * min(2.0, 0.6 + ratio)
        wl_rate = float(np.clip(wl_rate, 0.05, 0.65))
        waitlist_adds_total = int(np.random.binomial(total_signups, wl_rate))
        prog_weights = np.array([0.28, 0.26, 0.24, 0.22]); prog_weights /= prog_weights.sum()
        assigns = np.random.multinomial(waitlist_adds_total, prog_weights) if waitlist_adds_total>0 else np.array([0,0,0,0])
        for pkey, adds in zip(dim_program["program_key"], assigns):
            conv_rate = np.random.uniform(WAITLIST_CONV_RANGE[0], WAITLIST_CONV_RANGE[1])
            conversions = int(np.random.binomial(adds, conv_rate))
            rows.append({
                "date_key": date_key,
                "geo_key": geo_key,
                "program_key": int(pkey),
                "waitlist_adds": int(adds),
                "waitlist_conversions": int(conversions),
            })
    fact_waitlist = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date_key","geo_key","program_key","waitlist_adds","waitlist_conversions"])
    provider_geo_df = pd.DataFrame({"provider_key": list(provider_geo_map.keys()), "geo_key": list(provider_geo_map.values())})
    return fact_waitlist, provider_geo_df

def generate_matches(fact_waitlist, fact_signups, dim_provider, dim_program, start_date, end_date, provider_geo_map):
    # Daily provider availability after onboarding
    prov_dates = []
    for _, prow in dim_provider.iterrows():
        ob_date = pd.to_datetime(prow["onboarding_date"])
        prov_active_dates = pd.date_range(max(ob_date, start_date), end_date, freq="D")
        for d in prov_active_dates:
            prov_dates.append({
                "date_key": int(d.strftime("%Y%m%d")),
                "provider_key": int(prow["provider_key"]),
                "base_daily_capacity": float(prow["capacity_slots_week"] / 7.0)
            })
    prov_dates_df = pd.DataFrame(prov_dates)
    if prov_dates_df.empty:
        return pd.DataFrame(columns=["date_key","geo_key","program_key","provider_key","match_requests","matches_approved","time_to_match_hours","cancellations"]), prov_dates_df

    # Program affinity per provider
    program_affinity = {int(pk): int(random.choice(dim_program["program_key"].tolist())) for pk in dim_provider["provider_key"]}

    # Demand: waitlist conversions + ~40% of same-day signups
    demand_gp = fact_waitlist.groupby(["date_key","geo_key","program_key"], as_index=False)[["waitlist_adds","waitlist_conversions"]].sum()
    signups_by_gp = fact_signups.groupby(["date_key","geo_key"], as_index=False)["signups"].sum()
    signups_by_gp["base_requests"] = (signups_by_gp["signups"] * 0.4).round().astype(int)
    base_req_rows = []
    for _, r in signups_by_gp.iterrows():
        weights = np.array([0.30,0.25,0.25,0.20])
        parts = np.random.multinomial(int(r["base_requests"]), weights)
        for pkey, cnt in zip(dim_program["program_key"], parts):
            base_req_rows.append({
                "date_key": int(r["date_key"]),
                "geo_key": int(r["geo_key"]),
                "program_key": int(pkey),
                "base_requests": int(cnt)
            })
    base_req_df = pd.DataFrame(base_req_rows)

    demand_full = pd.merge(demand_gp, base_req_df, on=["date_key","geo_key","program_key"], how="outer").fillna(0)
    demand_full["match_requests"] = demand_full["base_requests"] + demand_full["waitlist_conversions"]

    prov_dates_df = prov_dates_df.merge(provider_geo_map, on="provider_key", how="left")
    prov_dates_df["program_affinity"] = prov_dates_df["provider_key"].map(program_affinity).astype(int)
    prov_dates_df["util_cap"] = np.random.uniform(0.80, 0.95, size=len(prov_dates_df))
    prov_dates_df["supply_slots"] = (prov_dates_df["base_daily_capacity"] * prov_dates_df["util_cap"]).clip(lower=0).round(0).astype(int)

    match_rows = []
    for (date_key, geo_key, program_key), dsub in demand_full.groupby(["date_key","geo_key","program_key"]):
        total_req = int(dsub["match_requests"].sum())
        if total_req <= 0:
            continue
        active_prov = prov_dates_df[(prov_dates_df["date_key"]==date_key) & (prov_dates_df["geo_key"]==geo_key)].copy()
        if active_prov.empty:
            continue
        active_prov["weight"] = np.where(active_prov["program_affinity"]==program_key, 1.2, 1.0)
        active_prov["supply_slots"] = active_prov["supply_slots"].clip(lower=0)
        total_supply = int(active_prov["supply_slots"].sum())
        matches_possible = min(total_req, total_supply)
        if matches_possible <= 0:
            continue
        active_prov["alloc_base"] = active_prov["weight"] * (active_prov["supply_slots"] + 1e-6)
        alloc_weights = (active_prov["alloc_base"] / active_prov["alloc_base"].sum()).values
        alloc_counts = np.random.multinomial(matches_possible, alloc_weights)
        canc_rate = np.random.uniform(0.02, 0.06)
        pressure = (total_req / (total_supply + 1e-6))
        ttm_hours = float(np.clip(np.random.normal(36 * pressure, 6), 6, 168))
        for pk, cnt in zip(active_prov["provider_key"].tolist(), alloc_counts.tolist()):
            if cnt <= 0: 
                continue
            match_rows.append({
                "date_key": int(date_key),
                "geo_key": int(geo_key),
                "program_key": int(program_key),
                "provider_key": int(pk),
                "match_requests": int(np.round(total_req * (cnt / max(matches_possible,1)))),
                "matches_approved": int(cnt),
                "time_to_match_hours": float(round(ttm_hours,2)),
                "cancellations": int(np.random.binomial(cnt, canc_rate))
            })
    fact_matches = pd.DataFrame(match_rows) if match_rows else pd.DataFrame(columns=["date_key","geo_key","program_key","provider_key","match_requests","matches_approved","time_to_match_hours","cancellations"])
    return fact_matches, prov_dates_df

def generate_sessions(fact_matches, prov_dates_df, dim_program, start_date, end_date) -> pd.DataFrame:
    matches_by_pk_day = fact_matches.groupby(["provider_key","date_key"], as_index=False)["matches_approved"].sum()
    if prov_dates_df.empty or matches_by_pk_day.empty:
        return pd.DataFrame(columns=["date_key","program_key","provider_key","sessions_completed","no_shows","avg_csatscore","outcome_delta_index"])

    supply_lookup = prov_dates_df.set_index(["provider_key","date_key"])["supply_slots"].to_dict()
    # Each provider's program affinity approximated by most matches historically
    affinity = fact_matches.groupby(["provider_key","program_key"])["matches_approved"].sum().reset_index()
    provider_program = affinity.sort_values(["provider_key","matches_approved"], ascending=[True, False]).drop_duplicates("provider_key").set_index("provider_key")["program_key"].to_dict()

    session_rows = []
    for (pk, dkey), sub in matches_by_pk_day.groupby(["provider_key","date_key"]):
        approved_today = int(sub["matches_approved"].sum())
        if approved_today <= 0: continue
        spread_days = np.random.choice([0,1,2,3,4,5], size=approved_today, p=[0.25,0.30,0.20,0.15,0.07,0.03])
        add_counts = pd.Series(spread_days).value_counts().to_dict()
        for delta, cnt in add_counts.items():
            d = pd.to_datetime(str(dkey), format="%Y%m%d") + timedelta(days=int(delta))
            if d > end_date: continue
            dkey2 = int(d.strftime("%Y%m%d"))
            supply = int(supply_lookup.get((int(pk), dkey2), 0))
            if supply <= 0: continue
            scheduled = int(min(cnt, supply))
            # infer program from affinity (fallback to most common program overall)
            prog_key = int(provider_program.get(int(pk), int(dim_program["program_key"].mode().iloc[0])))
            prog_name = dim_program.loc[dim_program["program_key"]==prog_key, "program_name"].values[0]
            ns_min, ns_max = NOSHOW_RATES.get(prog_name, (0.05, 0.15))
            ns_rate = float(np.random.uniform(ns_min, ns_max))
            no_shows = int(np.random.binomial(scheduled, ns_rate))
            completed = int(max(0, scheduled - no_shows))
            csat = float(np.clip(np.random.normal(4.3, 0.25), 3.2, 5.0))
            outcome_delta = float(np.clip(0.08 * completed + np.random.normal(0.0, 0.05), 0.0, 1.0))
            session_rows.append({
                "date_key": dkey2,
                "program_key": prog_key,
                "provider_key": int(pk),
                "sessions_completed": int(completed),
                "no_shows": int(no_shows),
                "avg_csatscore": round(csat,2),
                "outcome_delta_index": round(outcome_delta,3)
            })
    fact_sessions = pd.DataFrame(session_rows) if session_rows else pd.DataFrame(columns=["date_key","program_key","provider_key","sessions_completed","no_shows","avg_csatscore","outcome_delta_index"])
    return fact_sessions

def generate_finance(fact_sessions, dim_program, fact_signups) -> pd.DataFrame:
    sess_prog = fact_sessions.groupby(["date_key","program_key"], as_index=False)["sessions_completed"].sum()
    signups_by_day = fact_signups.groupby(["date_key"], as_index=False)["signups"].sum()
    rows = []
    for _, row in sess_prog.iterrows():
        date_key = int(row["date_key"]); prog_key = int(row["program_key"])
        sessions_completed = int(row["sessions_completed"])
        price = float(dim_program.loc[dim_program["program_key"]==prog_key, "price_per_session"].values[0])
        revenue = sessions_completed * price
        refunds = float(np.round(revenue * np.random.uniform(0.00, 0.03), 2))
        payout_pct = np.random.uniform(0.55, 0.70)
        cost_provider = float(np.round(revenue * payout_pct, 2))
        daily_signups = int(signups_by_day.loc[signups_by_day["date_key"]==date_key, "signups"].sum()) if not signups_by_day.empty else 0
        cost_marketing = float(np.round(daily_signups * np.random.uniform(2.0, 5.0), 2))
        cost_ops = float(np.round(40 + sessions_completed * np.random.uniform(0.3, 0.8), 2))
        rows.append({
            "date_key": date_key,
            "program_key": prog_key,
            "revenue": float(np.round(revenue,2)),
            "refunds": refunds,
            "cost_provider": cost_provider,
            "cost_marketing": cost_marketing,
            "cost_ops": cost_ops
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date_key","program_key","revenue","refunds","cost_provider","cost_marketing","cost_ops"])

def generate_membership(dim_date, dim_provider, seed: int):
    dim_membership_tier = pd.DataFrame(MEMBERSHIP_TIERS).sort_values("tier_key").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    records = []
    start_date = dim_date["date"].min()
    end_date = dim_date["date"].max()
    for _, prow in dim_provider.iterrows():
        pk = int(prow["provider_key"])
        onboard = pd.to_datetime(prow["onboarding_date"])
        start = max(onboard, start_date)
        if start > end_date:
            continue
        current_tier = rng.choice([1,2,3], p=INIT_TIER_PROBS)
        month_cursor = pd.Timestamp(start.year, start.month, 1)
        tier_by_month = {}
        while month_cursor <= end_date:
            tier_by_month[(month_cursor.year, month_cursor.month)] = current_tier
            probs = TIER_TRANSITION[current_tier-1]
            current_tier = rng.choice([1,2,3], p=probs)
            # next month
            if month_cursor.month == 12:
                month_cursor = pd.Timestamp(month_cursor.year+1, 1, 1)
            else:
                month_cursor = pd.Timestamp(month_cursor.year, month_cursor.month+1, 1)
        provider_dates = dim_date[(dim_date["date"] >= start) & (dim_date["date"] <= end_date)].copy()
        for _, drow in provider_dates.iterrows():
            d = drow["date"]; date_key = int(drow["date_key"])
            tier = tier_by_month[(d.year, d.month)]
            fee = float(dim_membership_tier.loc[dim_membership_tier["tier_key"]==tier, "monthly_fee_usd"].iloc[0])
            revenue_daily = round(fee / days_in_month(d), 2)
            records.append({
                "date_key": date_key,
                "provider_key": pk,
                "tier_key": int(tier),
                "memberships_active": 1,
                "membership_revenue_usd": revenue_daily
            })
    fact_membership_revenue = pd.DataFrame(records) if records else pd.DataFrame(columns=["date_key","provider_key","tier_key","memberships_active","membership_revenue_usd"])
    return dim_membership_tier, fact_membership_revenue

# ------------------------------
# Orchestrator
# ------------------------------
def run_generator(start: str, end: str, out_dir: str, seed: int, providers: int, cities: list[str], programs: list[str]):
    # seeding
    random.seed(seed)
    np.random.seed(seed)

    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    # Dimensions
    dim_date = make_date_dim(start_date, end_date)
    dim_geo  = make_geo_dim(cities)
    dim_program = make_program_dim(programs)
    dim_provider = make_provider_dim(providers, start_date, end_date)

    # Facts
    fact_signups = generate_signups(dim_date, dim_geo)
    fact_waitlist, provider_geo_map = generate_waitlist(fact_signups, dim_geo, dim_program, dim_provider)

    fact_matches, prov_dates_df = generate_matches(
        fact_waitlist, fact_signups, dim_provider, dim_program, start_date, end_date, provider_geo_map
    )

    fact_sessions = generate_sessions(fact_matches, prov_dates_df, dim_program, start_date, end_date)
    fact_finance = generate_finance(fact_sessions, dim_program, fact_signups)
    dim_membership_tier, fact_membership_revenue = generate_membership(dim_date, dim_provider, seed)

    # Save
    import pathlib
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {
        "dim_date": out/"dim_date.csv",
        "dim_geo": out/"dim_geo.csv",
        "dim_program": out/"dim_program.csv",
        "dim_provider": out/"dim_provider.csv",
        "dim_membership_tier": out/"dim_membership_tier.csv",
        "fact_signups": out/"fact_signups.csv",
        "fact_waitlist": out/"fact_waitlist.csv",
        "fact_matches": out/"fact_matches.csv",
        "fact_sessions": out/"fact_sessions.csv",
        "fact_finance": out/"fact_finance.csv",
        "fact_membership_revenue": out/"fact_membership_revenue.csv",
        "provider_geo_map": out/"provider_geo_map.csv",
    }

    dim_date.to_csv(paths["dim_date"], index=False)
    dim_geo.to_csv(paths["dim_geo"], index=False)
    dim_program.to_csv(paths["dim_program"], index=False)
    dim_provider.to_csv(paths["dim_provider"], index=False)
    dim_membership_tier.to_csv(paths["dim_membership_tier"], index=False)
    fact_signups.to_csv(paths["fact_signups"], index=False)
    fact_waitlist.to_csv(paths["fact_waitlist"], index=False)
    fact_matches.to_csv(paths["fact_matches"], index=False)
    fact_sessions.to_csv(paths["fact_sessions"], index=False)
    fact_finance.to_csv(paths["fact_finance"], index=False)
    fact_membership_revenue.to_csv(paths["fact_membership_revenue"], index=False)
    provider_geo_map.to_csv(paths["provider_geo_map"], index=False)

    summary = {
        "dates": (str(dim_date["date"].min().date()), str(dim_date["date"].max().date())),
        "providers": int(len(dim_provider)),
        "signups_total": int(fact_signups["signups"].sum()) if not fact_signups.empty else 0,
        "sessions_total": int(fact_sessions["sessions_completed"].sum()) if not fact_sessions.empty else 0,
        "avg_ttm_hours": round(float(fact_matches["time_to_match_hours"].mean()), 2) if not fact_matches.empty else None,
        "membership_daily_avg_total": round(fact_membership_revenue.groupby("date_key")["membership_revenue_usd"].sum().mean(), 2) if not fact_membership_revenue.empty else 0.0,
        "output_dir": str(out.resolve())
    }
    return {"paths": {k: str(v) for k,v in paths.items()}, "summary": summary}

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Show & Well Synthetic Data Generator")
    ap.add_argument("--start", default="2025-12-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end",   default="2026-09-01", help="End date (YYYY-MM-DD)")
    ap.add_argument("--out",   default="./output", help="Output directory for CSVs")
    ap.add_argument("--seed",  type=int, default=3102, help="Random seed for reproducibility")
    ap.add_argument("--providers", type=int, default=28, help="Number of providers to simulate (25–30 typical)")
    ap.add_argument("--cities", nargs="*", default=DEFAULT_CITIES, help="Override city list")
    ap.add_argument("--programs", nargs="*", default=DEFAULT_PROGRAMS, help="Override program list")
    return ap.parse_args()

def main():
    args = parse_args()
    res = run_generator(args.start, args.end, args.out, args.seed, args.providers, args.cities, args.programs)
    print("=== Generation complete ===")
    print("Output directory:", res["summary"]["output_dir"])
    print("Date range:", res["summary"]["dates"])
    print("Providers:", res["summary"]["providers"])
    print("Total signups:", res["summary"]["signups_total"])
    print("Sessions completed:", res["summary"]["sessions_total"])
    print("Avg TTM (hrs):", res["summary"]["avg_ttm_hours"])
    print("Avg daily membership revenue (total):", res["summary"]["membership_daily_avg_total"])
    print("\nFiles:")
    for k, p in res["paths"].items():
        print(f"  {k}: {p}")

if __name__ == "__main__":
    main()
