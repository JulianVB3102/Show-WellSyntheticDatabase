# Show & Well Synthetic Data Generator — README

## What this does
Creates a small analytics sandbox. It generates dimensions and facts. It writes CSVs you can load into Power BI, SQL, or Tableau. The data is synthetic;  but it is reproducible with a seed. Membership revenue is grabbed daily from monthly tiers.

## Quick start
```bash
# Run with defaults
python show_and_well_generator_commented.py

# Set your dates, output folder, and seed
python show_and_well_generator_commented.py --start 2025-12-01 --end 2026-09-01 --out ./output --seed 3102 --providers 28
```
Outputs are written to `--out`. The script prints a short summary at the end.

## Inputs and knobs (CLI)
- `--start` YYYY-MM-DD; start of date range; default `2025-12-01`
- `--end` YYYY-MM-DD; end of date range; default `2026-09-01`
- `--out` folder; where CSVs go; default `./output`
- `--seed` int; sets `random` and `numpy` seeds; default `3102`
- `--providers` int; number of providers; default `28`
- `--cities` list; override default city list
- `--programs` list; override default program list

## Outputs
Star schema style with a few helper maps.

Dimensions
- `dim_date.csv`: calendar rows with `date_key`, year, month, quarter, week, DOW, weekend flag
- `dim_geo.csv`: city rows with `geo_key`, state, region, population bucket
- `dim_program.csv`: program rows with `program_key`, `program_name`, `price_per_session`, `focus_area`
- `dim_provider.csv`: provider rows with `provider_key`, name, credential, onboarding, weekly capacity, avg session minutes
- `dim_membership_tier.csv`: tier rows with `tier_key`, `tier_name`, `monthly_fee_usd`

Facts
- `fact_signups.csv`: day, geo, source; counts of signups and wants_trainer
- `fact_waitlist.csv`: day, geo, program; waitlist adds and conversions
- `fact_matches.csv`: day, geo, program, provider; requests, approved matches, cancellations, time to match hours
- `fact_sessions.csv`: day, program, provider; sessions completed, no-shows, CSAT, outcome index
- `fact_finance.csv`: day, program; revenue and costs
- `fact_membership_revenue.csv`: day, provider, tier; memberships_active, daily membership revenue

Helper
- `provider_geo_map.csv`: provider to home `geo_key` mapping

## Model shape
- Grain by table:
  - `fact_signups`: day-geo-source
  - `fact_waitlist`: day-geo-program
  - `fact_matches`: day-geo-program-provider
  - `fact_sessions`: day-program-provider
  - `fact_finance`: day-program
  - `fact_membership_revenue`: day-provider-tier
- Join keys:
  - Dates: `date_key`
  - Geo: `geo_key`
  - Program: `program_key`
  - Provider: `provider_key`
  - Membership tier: `tier_key`

## Assumptions, weights, and random draws
Global
- Seeds: `random.seed(seed)`, `np.random.seed(seed)`; memberships use `np.random.default_rng(seed)`
- Date key: `YYYYMMDD` integer; weekend flag when DOW ∈ {Sat, Sun}

Demand (signups)
- Poisson mean per city-day: `λ = max(10 * CITY_FACTOR[city] * SEASONALITY[month] * WEEKDAY_MULT, 0.3)`
- WEEKDAY_MULT = 1.0 Mon–Fri; 0.85 Sat–Sun
- CITY_FACTOR: Minneapolis 1.00; St Paul 0.85; Eden Prairie 0.45; Bloomington 0.55; Hopkins 0.35; Plymouth 0.50; St Louis Park 0.40; Edina 0.48
- SEASONALITY: Dec 1.15; Jan 1.30; Feb 1.10; Mar 1.05; Apr 1.00; May 0.95; Jun 0.90; Jul 0.92; Aug 0.95; Sep 1.00
- Source split: Web 0.60; Referral 0.27; Event 0.13
- Wants-trainer rate: 0.35

Supply (providers)
- Credentials and weekly capacity: Coach 14–28; LPC 12–22; LMFT 12–22; RD 10–18
- Daily capacity: weekly capacity / 7
- Onboarding offset from start: Uniform 0 to 90 days or end of range, whichever is sooner
- Home geo: each provider gets a random city
- Avg session length: one of 45, 50, 55, 60 minutes

Waitlist
- Geo daily supply: sum of provider daily capacity in that geo
- Waitlist rate: `wl_rate = clip(BASE_WAITLIST_RATE * min(2.0, 0.6 + ratio), 0.05, 0.65)`; `ratio = demand/supply` (`2.0` if `supply == 0`); `BASE_WAITLIST_RATE = 0.20`
- Program split for adds: [0.28, 0.26, 0.24, 0.22]
- Conversion to matches: per program `p ~ Uniform(0.55, 0.85)`

Matching
- Requests matched: `waitlist_conversions + base_requests`; `base_requests = round(0.40 * same-day signups)` split across programs [0.30, 0.25, 0.25, 0.20]
- Utilization factor for supply slots: `Uniform(0.80, 0.95)`
- Program affinity weight: 1.2 if provider affinity equals program; else 1.0
- Allocation: multinomial by `weight * supply_slots`
- Cancellations: `p ~ Uniform(0.02, 0.06)`
- Time-to-match hours: `Normal(36 * pressure, 6)` clipped to [6, 168]; `pressure = demand/supply`

Sessions
- Scheduling lag over {0,1,2,3,4,5} days with probs [0.25, 0.30, 0.20, 0.15, 0.07, 0.03]
- Program choice: provider’s top historical program; fallback to program mode
- No-show rate ranges: Movement 0.06–0.14; Nutrition 0.05–0.12; Stress 0.08–0.16; Weight Loss Goal 0.06–0.14
- CSAT: `Normal(4.3, 0.25)` clipped to [3.2, 5.0]
- Outcome delta index: `clip(0.08 * sessions_completed + Normal(0, 0.05), 0, 1)`

Finance
- Price per session: Movement 75; Nutrition 95; Stress 100; Weight Loss Goal 85
- Revenue: `sessions_completed * price`
- Refunds: `Uniform(0.00, 0.03) * revenue`
- Provider payout: `Uniform(0.55, 0.70) * revenue`
- Marketing cost: `Uniform(2.0, 5.0) * total_signups_that_day`
- Ops cost: `40 + sessions_completed * Uniform(0.3, 0.8)`

Memberships
- Tiers and monthly fees: Basic 50; Premium 150; Elite 300
- Initial tier probabilities: [0.50, 0.35, 0.15]
- Monthly transition matrix:
  - Basic → [0.88, 0.10, 0.02]
  - Premium → [0.06, 0.88, 0.06]
  - Elite → [0.02, 0.12, 0.86]
- Daily revenue = monthly fee / days in month
- Each active provider counts as one membership per day

## Field glossaries (short)
**dim_date**: `date_key` int; `date` ISO; `year`, `quarter`, `month`, `day`, `weekofyear`, `dow` 0=Mon; `is_weekend` 0/1  
**dim_geo**: `geo_key` int; `city`; `state`; `region`; `population_bucket` S/M  
**dim_program**: `program_key` int; `program_name`; `price_per_session` float; `focus_area`  
**dim_provider**: `provider_key` int; `provider_name`; `credential_level`; `onboarding_date` ISO; `capacity_slots_week` int; `avg_session_length_min` int  
**dim_membership_tier**: `tier_key`; `tier_name`; `monthly_fee_usd`  

**fact_signups**: `date_key`; `geo_key`; `source` in {Web, Referral, Event}; `signups` int; `wants_trainer_count` int  
**fact_waitlist**: `date_key`; `geo_key`; `program_key`; `waitlist_adds` int; `waitlist_conversions` int  
**fact_matches**: `date_key`; `geo_key`; `program_key`; `provider_key`; `match_requests` int; `matches_approved` int; `time_to_match_hours` float; `cancellations` int  
**fact_sessions**: `date_key`; `program_key`; `provider_key`; `sessions_completed` int; `no_shows` int; `avg_csatscore` float 3.2–5.0; `outcome_delta_index` 0–1  
**fact_finance**: `date_key`; `program_key`; `revenue` float; `refunds` float; `cost_provider` float; `cost_marketing` float; `cost_ops` float  
**fact_membership_revenue**: `date_key`; `provider_key`; `tier_key`; `memberships_active` int; `membership_revenue_usd` float  
**provider_geo_map**: `provider_key`; `geo_key`

## Reproducibility
Set `--seed` to fix stochastic draws across `random`, `numpy`, and the membership transition RNG. You will get identical tables for the same inputs.

## Scaling and edits
- Increase `--providers` for more supply.  
- Edit `CITY_FACTOR`, `SEASONALITY`, and `SOURCE_WEIGHTS` to reshape demand.  
- Tweak `WAITLIST_CONV_RANGE`, `util_cap`, and `Program affinity weight` to change match dynamics.  
- Prices and costs live in one place; easy to tune.

## Disclaimer
This is fake data for demos and training. Do not use it to make real decisions.
