"""Forecast GTA VI trophy earn rates on Exophase, in five quantiles per trophy.

Model: a refresh-aware saturating trend with tier shrinkage and an honest spread.

What the data is
----------------
Exophase tracks a set of players and prints, per trophy, the percent of them
who have earned it. The page is NOT live: Exophase refreshes its counts on
its own schedule (hours apart), so consecutive reads are often identical and
a read shows the rates as of the last refresh before it. At a refresh the
tracked-player count jumps (launch night: 638 -> 3,868) and almost every rate
falls, because the newly tracked players have earned far fewer trophies than
the early cohort. Between refreshes the true rates keep moving (players keep
playing) but the page stays frozen.

So a case's history is a step function: flat stretches, then jumps at
refreshes. The forecast has to (a) read the trend off the steps, not the
frozen stretches, (b) expect the next refresh to dilute rates when the
history shows no refresh yet, and (c) widen honestly with the horizon,
because pinball loss on the logit scale rewards the quantiles you really
believe and punishes both narrow and wide misses.

Method, per trophy
------------------
1. Collapse the history to distinct reads (consecutive reads with the same
   players-tracked count and the same rates carry no new information).
2. Fit a saturating curve in logit space: logit(rate) = L - A * exp(-k * x)
   with x = ln(1 + hours since launch), by a grid search over k with a
   weighted linear fit for L and A at each k. The curve rises fast early
   and flattens toward a ceiling L, which is how trophy rates actually
   behave: a straight line in log-time would keep climbing forever and
   overshoot a week out (verified on a synthetic final-shaped case, where
   the straight line's median sat above the answer on all 55 trophies).
   The ceiling is shrunk toward the tier median (platinum, gold, silver,
   bronze behave differently). When the series is falling instead (a
   trophy being diluted faster than players earn it), the saturating fit
   cannot represent it, so a recency-weighted straight line in logit space
   is used for those trophies instead.
3. Refresh-dip prior. Count the refresh events in the history (a distinct
   read whose players-tracked count changed). When the history shows no
   refresh at all, the trend is flat and the answer is almost certainly the
   refreshed page, so the median is pulled down by a dilution factor
   (calibrated on launch night, where rates fell to roughly a third). With
   one or two refreshes the pull is smaller; with three or more the trend
   already prices refreshes in and no pull is applied. The pull also fades
   with horizon, since over a long horizon the trend dominates.
4. Spread. The logit spread combines the fit's residual scatter, the
   parameter uncertainty (how far the curve moves over the plausible k
   values), refresh uncertainty (how much a refresh moves rates, times how
   many are expected before the target) and a model-form term that grows
   with the horizon. It is floored and capped so no trophy is ever forecast
   with false precision.
5. Quantiles come from a logit-normal with that median and spread. The
   trophy sitting at 100% is handled separately: it stays near 100% (every
   player who plays completes the opening mission fast enough that a
   refresh did not move it on launch night either).

Determinism: no randomness anywhere (the only "simulation" is closed-form
arithmetic), so two runs write byte-identical output. Standard library only.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch of GTA VI: the clock the trend is measured on.
LAUNCH = "2026-11-19T00:00:00Z"

# Median dilution a refresh applies when the history shows no refresh yet,
# by how many refreshes the history does show. Launch night's first refresh
# took the tracked count 638 -> 3,868 and most rates to roughly a third of
# where they stood; later refreshes, on a larger base, dilute far less.
DIP_BY_REFRESHES = {0: 0.32, 1: 0.65, 2: 0.80}
# How fast the dip prior fades with horizon: full strength at 12 h or less.
DIP_HORIZON_MINUTES = 720.0
# A refresh is expected about this often when the history shows none.
DEFAULT_REFRESH_MINUTES = 240.0
# Log-ratio scatter of one refresh when the history shows none.
DEFAULT_REFRESH_SCATTER = 0.8

# Grid for the saturation rate k in logit(rate) = L - A * exp(-k * x).
K_GRID = [0.05 * (1.3 ** j) for j in range(26)]


def clamp_rate(pct):
    """A rate into the logit-safe range the scoring clamps to."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def sigmoid(logit_value):
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds (no dateutil in stdlib)."""
    from datetime import datetime

    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def distinct_reads(history):
    """Collapse consecutive identical reads to the reads that moved.

    Returns a list of (epoch seconds, players_tracked, rates) with no two
    adjacent entries identical, oldest first.
    """
    out = []
    for read in history:
        rates = read.get("rates") or []
        entry = (parse_time(read["at"]), read.get("players_tracked"), rates)
        if out and out[-1][1] == entry[1] and out[-1][2] == entry[2]:
            continue
        out.append(entry)
    return out


def refresh_events(reads):
    """Indices (into reads) where the tracked-player count changed."""
    return [j for j in range(1, len(reads)) if reads[j][1] != reads[j - 1][1]]


def weighted_line(xs, ys, weights):
    """Weighted least squares y = a + b*x. Returns (a, b, residual_sd, b_se).

    Falls back to a flat line through the weighted mean when the data
    cannot support a slope.
    """
    n = len(xs)
    sw = sum(weights)
    if n == 0 or sw <= 0:
        return 0.0, 0.0, 0.6, 1.0
    mx = sum(w * x for x, w in zip(xs, weights)) / sw
    my = sum(w * y for y, w in zip(ys, weights)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, weights))
    if n < 2 or sxx <= 1e-12:
        resid = [y - my for y in ys]
        sd = math.sqrt(sum(r * r for r in resid) / max(1, n)) if n else 0.6
        return my, 0.0, max(0.15, sd), 1.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, weights))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, n - 2)
    sd = math.sqrt(sum(r * r for r in resid) / dof)
    b_se = sd / math.sqrt(sxx) if sxx > 0 else 1.0
    return a, b, sd, b_se


def fit_saturating(xs, ys, weights):
    """Fit logit(rate) = L - A * exp(-k * x) by gridding k.

    For fixed k the model is linear in (L, A), so each grid point is a
    weighted linear fit. Returns (sse, L, A, k, resid_sd) of the best k,
    or None when the data cannot support the fit.
    """
    if len(xs) < 3:
        return None
    best = None
    for k in K_GRID:
        vs = [-math.exp(-k * x) for x in xs]
        a, b, sd, _se = weighted_line(vs, ys, weights)
        # y = a + b*v with v = -exp(-k*x), i.e. y = L - A*exp(-k*x)
        # with L = a and A = b (b > 0 for a rising series).
        sse = sd * sd * max(1, len(xs) - 2)
        if best is None or sse < best[0]:
            best = (sse, a, b, k, sd)
    return best


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    release = str(case.get("game", {}).get("release_date", "2026-11-19"))
    launch = parse_time(release + "T00:00:00Z" if len(release) == 10 else LAUNCH)
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_minutes = case["horizon_minutes"]
    horizon_days = horizon_minutes / 1440.0

    reads = distinct_reads(history)
    refresh_idx = refresh_events(reads)
    n_refreshes = len(refresh_idx)

    # Typical gap between refreshes, from the history; a flat history
    # falls back to the launch-night cadence.
    if len(refresh_idx) >= 2:
        gaps = [(reads[refresh_idx[j]][0] - reads[refresh_idx[j - 1]][0]) / 60.0
                for j in range(1, len(refresh_idx))]
        refresh_gap = median(gaps) or DEFAULT_REFRESH_MINUTES
    elif len(refresh_idx) == 1:
        refresh_gap = (reads[refresh_idx[0]][0] - reads[0][0]) / 60.0 or DEFAULT_REFRESH_MINUTES
    else:
        refresh_gap = DEFAULT_REFRESH_MINUTES
    refresh_gap = min(max(refresh_gap, 30.0), 1440.0)
    expected_refreshes = horizon_minutes / refresh_gap

    # How much one refresh moves rates, from the history's own refreshes:
    # the median log-ratio across trophies at each refresh, then the scatter
    # of those medians. A history with no refresh falls back to the
    # launch-night scatter.
    ratios = []
    for j in refresh_idx:
        before, after = reads[j - 1][2], reads[j][2]
        logs = []
        for rb, ra in zip(before, after):
            if rb is None or ra is None:
                continue
            rb, ra = clamp_rate(rb), clamp_rate(ra)
            if rb >= 99.99 or ra >= 99.99 or rb <= 0.01:
                continue
            logs.append(math.log(ra / rb))
        if logs:
            ratios.append(median(logs))
    if ratios:
        refresh_scatter = math.sqrt(sum(r * r for r in ratios) / len(ratios)) or 0.3
        refresh_scatter = min(max(refresh_scatter, 0.15), 1.2)
    else:
        refresh_scatter = DEFAULT_REFRESH_SCATTER

    # Dip prior: how much the median is pulled down for a refresh the
    # history has not shown yet. Fades with horizon and vanishes once the
    # history holds three or more refreshes (the trend prices them in).
    dip = DIP_BY_REFRESHES.get(n_refreshes, 1.0)
    fade = min(1.0, DIP_HORIZON_MINUTES / max(1.0, horizon_minutes))
    dip_factor = dip ** fade if dip < 1.0 else 1.0
    dip_log = math.log(dip_factor) if dip_factor < 1.0 else 0.0

    # Per-trophy series, in (x, logit(rate)) with recency weights.
    # Only the recent window is fitted: launch dynamics (an early spike
    # while the tracked base is tiny) must not drive a forecast a week
    # out. Keep the last 72 h, or the last 8 distinct reads if that is
    # fewer, so a sparse history still has points to fit.
    def series_for(i):
        points = []
        for (t, _n, rates) in reads:
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            hours = max(0.0, (t - launch) / 3600.0)
            age_hours = max(0.0, (at - t) / 3600.0)
            points.append((math.log1p(hours), logit(rate),
                           0.5 ** (age_hours / 24.0), age_hours))
        recent = [p for p in points if p[3] <= 72.0]
        use = recent if len(recent) >= 8 else points[-8:]
        return ([p[0] for p in use], [p[1] for p in use], [p[2] for p in use])

    series = [series_for(i) for i in range(len(trophies))]
    # Straight-line fits (for falling series) and saturating fits.
    lines = [weighted_line(*s) for s in series]
    sats = [fit_saturating(*s) for s in series]

    # Tier medians, for shrinkage: ceilings of the saturating fits, slopes
    # of the straight fits, residual sds.
    def tier_median(key, need_points=3):
        out = {}
        for tier in ("platinum", "gold", "silver", "bronze"):
            vals = []
            for i, t in enumerate(trophies):
                if t["tier"] != tier or len(series[i][0]) < need_points:
                    continue
                v = key(i)
                if v is not None:
                    vals.append(v)
            out[tier] = median(vals) if vals else None
        return out

    tier_ceiling = tier_median(lambda i: sats[i][1] if sats[i] else None)
    tier_slope = tier_median(lambda i: lines[i][1])
    tier_resid = tier_median(lambda i: sats[i][4] if sats[i] else lines[i][2])

    x_target = math.log1p(max(0.0, (target - launch) / 3600.0))

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = reads[-1][2][i] if i < len(reads[-1][2]) else None
        if last_rate is None:
            last_rate = 1.0
        last_rate = clamp_rate(last_rate)

        # A trophy everyone holds stays near everyone holding it.
        if last_rate >= 99.99:
            med_log = logit(99.8)
            sigma = 0.4
            rows.append({"id": trophy["id"], "q": [
                round4(sigmoid(med_log + z * sigma)) for z in Z]})
            continue

        xs, ys, ws = series[i]
        n_points = len(xs)
        sat = sats[i]
        _a, b, _sd, b_se = lines[i]

        if sat is not None and sat[2] > 0:
            # Rising (or flat) series: saturating curve, ceiling shrunk
            # toward the tier median.
            _sse, ceiling, amp, k, resid_sd = sat
            tier_c = tier_ceiling[trophy["tier"]]
            if tier_c is not None:
                ceiling = (n_points * ceiling + 3.0 * tier_c) / (n_points + 3.0)
            ceiling = max(ceiling, max(ys))
            trend_log = ceiling - amp * math.exp(-k * x_target)
            # Parameter uncertainty: how far the forecast moves over the
            # k values whose fit is nearly as good as the best.
            plausible = []
            for kk in K_GRID:
                vs = [-math.exp(-kk * x) for x in xs]
                a2, b2, _sd2, _se2 = weighted_line(vs, ys, ws)
                sse2 = _sd2 * _sd2 * max(1, len(xs) - 2)
                if sse2 <= _sse * 1.25 + 1e-9:
                    c2 = max(a2, max(ys))
                    plausible.append(c2 + b2 * (-math.exp(-kk * x_target)))
            if len(plausible) >= 2:
                pm = sum(plausible) / len(plausible)
                sd_param = math.sqrt(sum((p - pm) ** 2 for p in plausible) / len(plausible))
            else:
                sd_param = 0.15
            sd_resid = resid_sd if n_points >= 3 else (tier_resid[trophy["tier"]] or 0.6)
            sd_slope = 0.0
        else:
            # Falling series, or too few points for a curve: straight line
            # in logit space, slope shrunk toward the tier median, anchored
            # at the last read.
            tier_b = tier_slope[trophy["tier"]]
            if tier_b is None:
                tier_b = 0.0
            b_shrunk = (n_points * b + 3.0 * tier_b) / (n_points + 3.0)
            x_last = math.log1p(max(0.0, (reads[-1][0] - launch) / 3600.0))
            trend_log = logit(last_rate) + b_shrunk * (x_target - x_last)
            sd_resid = lines[i][2] if n_points >= 3 else (tier_resid[trophy["tier"]] or 0.6)
            sd_param = b_se * abs(x_target - x_last)
            sd_slope = 0.0

        med_log = trend_log + dip_log

        # Guardrail: no trophy's median may run away from its last read.
        # A week of play moves a rate a lot, but not from 24% to 100%.
        med_rate = sigmoid(med_log)
        cap = min(99.0, max(5.0 * last_rate, last_rate + 30.0))
        if med_rate > cap:
            med_log = logit(cap)

        # Spread: fit scatter, parameter uncertainty, refresh uncertainty,
        # and a model-form term growing with the horizon.
        sd_refresh = refresh_scatter * math.sqrt(min(expected_refreshes, 6.0))
        if n_refreshes >= 3:
            sd_refresh *= 0.3
        sd_form = 0.15 * math.sqrt(horizon_days)
        sigma = math.sqrt(sd_resid ** 2 + sd_param ** 2 + sd_refresh ** 2 + sd_form ** 2)
        sigma = min(max(sigma, 0.85 if n_refreshes == 0 else 0.35), 1.8)

        quantiles = []
        prev = None
        for z in Z:
            q = round4(sigmoid(med_log + z * sigma))
            q = min(99.99, max(0.01, q))
            if prev is not None and q < prev:
                q = prev
            quantiles.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
