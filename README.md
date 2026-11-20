# Week-One Trophy Forecast — GTA VI rarity, five quantiles per trophy

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ariffforai/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ariffforai/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every PlayStation 5 trophy of Grand Theft Auto VI's earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy.
Built for the
[Forecast how many GTA VI players earn each trophy in week one](https://make.fun/c/gta6-trophy-rarity-forecast-01)
bounty on make.fun.

## The model

Exophase does not print live rates: it refreshes its counts on its own
schedule, hours apart, so a case's history is a step function — flat
stretches, then jumps at refreshes. At a refresh the tracked-player count
jumps (launch night: 638 → 3,868) and almost every rate falls, because the
newly tracked players have earned far fewer trophies than the early cohort.
A flat forecast therefore loses: on the practice cases the house baseline
scores a pinball loss of about 1.51.

This entry is a refresh-aware forecaster, Python 3.13 standard library only:

- **Distinct reads only.** Consecutive identical reads carry no new
  information; the trend is read off the steps, not the frozen stretches.
- **Saturating trend.** Per trophy, `logit(rate) = L − A·exp(−k·x)` with
  `x = ln(1 + hours since launch)`, fitted by a grid search over `k`. The
  curve rises fast early and flattens toward a ceiling, which is how trophy
  rates actually behave — a straight line in log-time keeps climbing and
  overshoots a week out. Only the recent window (last 72 h, or the last 8
  distinct reads) is fitted, so launch-night dynamics cannot drive the
  forecast. The ceiling is shrunk toward the tier median (platinum, gold,
  silver, bronze behave differently). Series that are falling instead get a
  recency-weighted straight line in logit space.
- **Refresh-dip prior.** When the history shows no refresh at all, the
  trend is flat and the answer is almost certainly the refreshed page, so
  the median is pulled down by a dilution factor calibrated on launch
  night (rates fell to roughly a third). The pull shrinks with one or two
  refreshes and vanishes at three or more, when the trend already prices
  refreshes in, and it fades with horizon.
- **Honest spread.** The logit spread combines the fit's residual scatter,
  parameter uncertainty over the plausible saturation rates, refresh
  uncertainty (how much a refresh moves rates, times how many are expected
  before the target) and a model-form term growing with the horizon. The
  trophy sitting at 100% is forecast near 100% with a narrow band.

## Validation

The practice answers are not in the kit (the house publishes them with the
results), so validation is scenario-based plus synthetic cases with known
ground truth:

| Test | This entry | House baseline (starter) |
| --- | --- | --- |
| Practice, answers at 0.2× last rates | 0.499 | 1.491 |
| Practice, answers at 0.3× last rates | 0.452 | 1.092 |
| Practice, answers at 0.4× last rates | 0.464 | 0.810 |
| Synthetic final-shaped case, rising saturating trend | 0.107 | 0.288 |
| Synthetic dilution case, spike then flat/declining | 0.209 | 0.281 |

Losses are mean pinball loss on the logit scale (lower is better). The
practice answers sit at roughly 0.2–0.4× the last rates — the baseline's
typical miss of ~0.68 points against ~0.9% medians says so — which is
exactly where this entry wins by the widest margin.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `<id>.out` per
case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.
Deterministic — two runs write byte-identical output — standard library
only, no network, well under the 8-second limit.
