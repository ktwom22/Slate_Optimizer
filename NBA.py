# nba_optimizer.py
# NBA DraftKings Classic — FAST DIVERSIFIED GPP BUILDER (ONE FILE)
# ✅ Always fetches a fresh CSV on every run (cache-bust + no-cache + md5 fingerprint)
# ✅ Projects ownership from YOUR stats (salary/proj/usage/value/tm_points/ou_total if present)
# ✅ Adds Chalk/Sneaky tiers + per-slot own% + lineup chalk/sneaky counts
# ✅ Supports contest_type profiles: cash | gpp_small | gpp_medium | gpp_large
# ✅ Better diversity: exposure cap + stronger uniqueness (true player overlap) + jitter
#
# Install:
#   pip install pandas requests pulp numpy
#
# CLI:
#   python -u nba_optimizer.py --num_lineups 20 --contest_type gpp_large
#
# Flask:
#   import nba_optimizer as NBA
#   df = NBA.generate_nba_df(num_lineups=20, contest_type="gpp_large")

import argparse
import hashlib
import math
import os
import random
import re
import time
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import numpy as np
import pandas as pd
import pulp
import requests

# -----------------------------
# DEFAULTS / ENV
# -----------------------------
CSV_URL_DEFAULT = os.environ.get(
    "NBA_CSV_URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vTF0d2pT0myrD7vjzsB2IrEzMa3o1lylX5_GYyas_5UISsgOud7WffGDxSVq6tJhS45UaxFOX_FolyT/pub?gid=2055904356&single=true&output=csv"
)

DK_SALARY_CAP = 50000
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
LINEUP_SIZE = len(DK_SLOTS)

MIN_SALARY_SPEND_DEFAULT = 49500
UTIL_SALARY_CAP_DEFAULT: Optional[int] = 5500

# Solver knobs
CBC_TIME_LIMIT_SEC = float(os.environ.get("NBA_CBC_TIME_LIMIT", "1.8"))
CBC_GAP_REL = float(os.environ.get("NBA_CBC_GAP_REL", "0.10"))
CBC_THREADS = int(os.environ.get("NBA_CBC_THREADS", "1"))
CBC_MSG = bool(int(os.environ.get("NBA_CBC_MSG", "0")))

# Attempts per lineup
RETRIES_PER_LINEUP = int(os.environ.get("NBA_RETRIES", "2"))

# Diversity knobs
MAX_PLAYER_EXPOSURE_DEFAULT = 0.45  # hard cap share of lineups
SOFT_EXPOSURE_PENALTY_DEFAULT = 0.0  # set >0 to softly discourage high-exposure players
MIN_UNIQUE_DEFAULT = 2

# Chalk/sneaky percentiles
CHALK_PCTILE_DEFAULT = 85
SNEAKY_PCTILE_DEFAULT = 25

# Relax ladder
RELAX_LADDER = [
    # (relax_util_cap, uniq_relax, min_salary_relax)
    (False, 0, 0),
    (True,  0, 0),
    (True,  1, 0),
    (True,  1, 800),
    (True,  2, 1500),
]

# -----------------------------
# DATA MODEL
# -----------------------------
@dataclass(frozen=True)
class Player:
    name: str
    team: str
    positions: Tuple[str, ...]  # ("PG","SG") etc
    salary: int
    proj: float

    usage: float
    value: float
    tm_points: float
    ou_total: float

    pred_own: float      # 0..1
    is_chalk: int        # 0/1
    is_sneaky: int       # 0/1


# -----------------------------
# HELPERS
# -----------------------------
_num_re = re.compile(r"[-+]?\d*\.?\d+")

def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    return c.strip("_")

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = { _norm_col(c): c for c in df.columns }
    for cand in candidates:
        k = _norm_col(cand)
        if k in norm_map:
            return norm_map[k]
    return None

def _clean(s) -> str:
    s = re.sub(r"&nbsp;?", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s

def _to_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        s = str(x).replace(",", "").replace("%", "").replace("$", "").strip()
        if s.lower() in ("", "nan", "none"):
            return default
        m = _num_re.search(s)
        return float(m.group(0)) if m else default
    except Exception:
        return default

def _to_int(x, default=0) -> int:
    v = _to_float(x, default=float(default))
    if v <= 0:
        return default
    # support 6.2 => 6200
    if 0 < v <= 100:
        return int(round(v * 1000))
    return int(round(v))

def zscore(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=0))
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / sd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def eligible_slots(p: Player) -> Set[str]:
    base = set(p.positions)
    slots = set(base)
    if "PG" in base or "SG" in base:
        slots.add("G")
    if "SF" in base or "PF" in base:
        slots.add("F")
    slots.add("UTIL")
    return slots

def _cache_bust(url: str) -> str:
    u = urlparse(url)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q["_cb"] = str(int(time.time() * 1000))
    return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(q), u.fragment))

def fetch_csv_to_df(url: str) -> pd.DataFrame:
    """
    Always fetch a fresh CSV (avoid caches) + print md5 fingerprint for proof.
    """
    bust_url = _cache_bust(url)
    headers = {"Cache-Control": "no-cache", "Pragma": "no-cache"}
    r = requests.get(bust_url, timeout=30, headers=headers)
    r.raise_for_status()
    txt = r.text
    sig = hashlib.md5(txt.encode("utf-8")).hexdigest()[:10]
    print(f"[NBA DEBUG] fetched bytes={len(txt)} md5={sig}", flush=True)
    return pd.read_csv(StringIO(txt))


# -----------------------------
# OWNERSHIP MODEL (from your stats)
# -----------------------------
def _target_total_own_mass(num_games: int) -> float:
    g = max(2, min(12, int(num_games)))
    return 8.0 + 0.35 * (g - 1)

def project_ownership(
    df: pd.DataFrame,
    salary_col: str,
    proj_col: str,
    usage_col: Optional[str],
    value_col: Optional[str],
    tmpts_col: Optional[str],
    ou_col: Optional[str],
    games: int
) -> pd.Series:
    sal = df[salary_col].apply(lambda v: _to_float(v, 0.0))
    proj = df[proj_col].apply(lambda v: _to_float(v, 0.0))

    usage = df[usage_col].apply(lambda v: _to_float(v, 0.0)) if usage_col else pd.Series(np.zeros(len(df)), index=df.index)
    val   = df[value_col].apply(lambda v: _to_float(v, 0.0)) if value_col else pd.Series(np.zeros(len(df)), index=df.index)
    tmpts = df[tmpts_col].apply(lambda v: _to_float(v, 0.0)) if tmpts_col else pd.Series(np.zeros(len(df)), index=df.index)
    ou    = df[ou_col].apply(lambda v: _to_float(v, 0.0)) if ou_col else pd.Series(np.zeros(len(df)), index=df.index)

    salary_k = sal / 1000.0
    value_k = np.where(salary_k > 0, proj / salary_k, 0.0)

    z_proj = zscore(proj)
    z_val  = zscore(pd.Series(value_k, index=df.index))
    z_use  = zscore(usage) if float(usage.abs().sum()) > 0 else pd.Series(np.zeros(len(df)), index=df.index)
    z_valcol = zscore(val) if float(val.abs().sum()) > 0 else pd.Series(np.zeros(len(df)), index=df.index)
    z_tmpts  = zscore(tmpts) if float(tmpts.abs().sum()) > 0 else pd.Series(np.zeros(len(df)), index=df.index)
    z_ou     = zscore(ou) if float(ou.abs().sum()) > 0 else pd.Series(np.zeros(len(df)), index=df.index)
    z_cheap  = zscore(-salary_k)

    lin = (
        1.05 * z_val +
        0.85 * z_proj +
        0.55 * z_use +
        0.35 * z_cheap +
        0.25 * z_valcol +
        0.20 * z_tmpts +
        0.10 * z_ou
    )

    base = sigmoid(lin)
    base = np.clip(base, 0.001, 0.999)

    mass = _target_total_own_mass(games)
    scaled = base * (mass / float(base.sum()))
    pred = np.clip(scaled, 0.0, 0.60)
    return pd.Series(pred, index=df.index)


# -----------------------------
# PARSE PLAYERS
# -----------------------------
def parse_players(df: pd.DataFrame, games: int, chalk_pctile: int, sneaky_pctile: int) -> List[Player]:
    name_col = _pick_col(df, ["name", "player", "player_name"])
    pos_col  = _pick_col(df, ["pos", "position", "positions", "dk_position"])
    team_col = _pick_col(df, ["team", "tm", "teamabbr", "team_abbrev", "abbrev"])
    sal_col  = _pick_col(df, ["salary", "sal", "dk_salary", "cost"])
    proj_col = _pick_col(df, ["proj", "projection", "projected_points", "points", "fpts", "dk_fp_projected"])

    usage_col = _pick_col(df, ["final_usage", "usage", "usg", "usage_rate"])
    value_col = _pick_col(df, ["value"])
    tmpts_col = _pick_col(df, ["tm_points", "team_points", "tm points", "team total"])
    ou_col    = _pick_col(df, ["o_u", "ou", "o/u", "total", "game_total", "over_under"])

    missing = [k for k, v in {"name": name_col, "pos": pos_col, "team": team_col, "salary": sal_col, "proj": proj_col}.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    tmp = df.copy()
    tmp["name"] = tmp[name_col].apply(_clean)
    tmp["team"] = tmp[team_col].apply(_clean)
    tmp["pos_raw"] = tmp[pos_col].apply(_clean).str.upper()

    tmp["salary"] = tmp[sal_col].apply(lambda x: _to_int(x, 0))
    tmp["proj"] = tmp[proj_col].apply(lambda x: _to_float(x, 0.0))

    tmp["usage"] = tmp[usage_col].apply(lambda x: _to_float(x, 0.0)) if usage_col else 0.0
    tmp["value"] = tmp[value_col].apply(lambda x: _to_float(x, 0.0)) if value_col else 0.0
    tmp["tm_points"] = tmp[tmpts_col].apply(lambda x: _to_float(x, 0.0)) if tmpts_col else 0.0
    tmp["ou_total"] = tmp[ou_col].apply(lambda x: _to_float(x, 0.0)) if ou_col else 0.0

    tmp = tmp[(tmp["name"] != "") & (tmp["team"] != "") & (tmp["pos_raw"] != "")]
    tmp = tmp[(tmp["salary"] > 0) & (tmp["proj"] > 0)]
    if tmp.empty:
        raise ValueError("No valid NBA players after cleanup.")

    tmp["pred_own"] = project_ownership(
        tmp,
        salary_col="salary",
        proj_col="proj",
        usage_col="usage" if usage_col else None,
        value_col="value" if value_col else None,
        tmpts_col="tm_points" if tmpts_col else None,
        ou_col="ou_total" if ou_col else None,
        games=games
    )
    tmp["pred_own_pct"] = (100.0 * tmp["pred_own"]).round(1)

    own_pcts = tmp["pred_own_pct"].astype(float)
    chalk_cut = float(np.percentile(own_pcts, chalk_pctile))
    sneaky_cut = float(np.percentile(own_pcts, sneaky_pctile))
    tmp["is_chalk"] = (tmp["pred_own_pct"] >= chalk_cut).astype(int)
    tmp["is_sneaky"] = (tmp["pred_own_pct"] <= sneaky_cut).astype(int)

    players: List[Player] = []
    for _, r in tmp.iterrows():
        parts = re.split(r"[/,;\s]+", str(r["pos_raw"]).upper())
        parts = [p for p in parts if p in {"PG", "SG", "SF", "PF", "C"}]
        if not parts:
            continue
        players.append(Player(
            name=str(r["name"]),
            team=str(r["team"]),
            positions=tuple(sorted(set(parts))),
            salary=int(r["salary"]),
            proj=float(r["proj"]),
            usage=float(r["usage"]),
            value=float(r["value"]),
            tm_points=float(r["tm_points"]),
            ou_total=float(r["ou_total"]),
            pred_own=float(r["pred_own"]),
            is_chalk=int(r["is_chalk"]),
            is_sneaky=int(r["is_sneaky"]),
        ))

    if not players:
        raise ValueError("No valid NBA players after position parsing.")
    return players


# -----------------------------
# CONTEST PROFILES
# -----------------------------
def contest_profile(contest_type: str) -> Dict[str, float]:
    ct = (contest_type or "gpp_large").lower().strip()
    if ct in ("cash", "doubleup", "h2h"):
        return dict(min_chalk=3, max_chalk=6, min_sneaky=0, own_penalty=12.0, lev_weight=0.12, base_random=0.35)
    if ct in ("gpp_small", "small", "single_entry", "se"):
        return dict(min_chalk=2, max_chalk=4, min_sneaky=1, own_penalty=22.0, lev_weight=0.22, base_random=0.65)
    if ct in ("gpp_medium", "medium"):
        return dict(min_chalk=2, max_chalk=4, min_sneaky=2, own_penalty=26.0, lev_weight=0.28, base_random=0.80)
    return dict(min_chalk=1, max_chalk=3, min_sneaky=2, own_penalty=30.0, lev_weight=0.35, base_random=0.95)


# -----------------------------
# OPTIMIZE ONE LINEUP
# -----------------------------
def optimize_one_lineup(
    players: List[Player],
    salary_cap: int,
    min_salary_spend: int,
    util_salary_cap: Optional[int],
    min_unique: int,
    previous_sets: List[Set[int]],
    banned: Set[int],
    own_penalty: float,
    lev_weight: float,
    min_chalk: int,
    max_chalk: int,
    min_sneaky: int,
    randomness: float,
    seed: Optional[int],
    soft_exposure_penalty: float = 0.0,
    exposure_counts: Optional[Dict[int, int]] = None,
    num_lineups_total: int = 0,
) -> Optional[List[int]]:

    if seed is not None:
        random.seed(seed)

    n = len(players)
    elig = [eligible_slots(p) for p in players]

    # decision vars x[(i,slot)] only if eligible and not banned
    x: Dict[Tuple[int, str], pulp.LpVariable] = {}
    for i in range(n):
        if i in banned:
            continue
        for s in DK_SLOTS:
            if s in elig[i]:
                x[(i, s)] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")

    prob = pulp.LpProblem("NBA_DK", pulp.LpMaximize)

    # Fill each slot
    for s in DK_SLOTS:
        prob += pulp.lpSum(x[(i, s)] for i in range(n) if (i, s) in x) == 1, f"fill_{s}"

    # each player <= 1
    selected = []
    for i in range(n):
        sel = pulp.lpSum(x[(i, s)] for s in DK_SLOTS if (i, s) in x)
        selected.append(sel)
        prob += sel <= 1, f"one_{i}"

    # salary
    total_sal = pulp.lpSum(players[i].salary * selected[i] for i in range(n))
    prob += total_sal <= salary_cap, "cap"
    prob += total_sal >= min_salary_spend, "min_spend"

    # sanity
    prob += pulp.lpSum(selected[i] for i in range(n) if "PG" in players[i].positions) >= 1, "need_pg"
    prob += pulp.lpSum(selected[i] for i in range(n) if "PF" in players[i].positions) >= 1, "need_pf"

    # util salary cap
    if util_salary_cap is not None:
        prob += pulp.lpSum(players[i].salary * x[(i, "UTIL")] for i in range(n) if (i, "UTIL") in x) <= util_salary_cap, "util_cap"

    # chalk/sneaky constraints
    chalk_sum = pulp.lpSum(players[i].is_chalk * selected[i] for i in range(n))
    sneaky_sum = pulp.lpSum(players[i].is_sneaky * selected[i] for i in range(n))
    prob += chalk_sum >= min_chalk, "min_chalk"
    prob += chalk_sum <= max_chalk, "max_chalk"
    prob += sneaky_sum >= min_sneaky, "min_sneaky"

    # uniqueness vs previous (true player overlap)
    if previous_sets and min_unique > 0:
        max_overlap = LINEUP_SIZE - min_unique
        for j, prev in enumerate(previous_sets, start=1):
            prob += pulp.lpSum(selected[i] for i in prev) <= max_overlap, f"uniq_{j}"

    # objective noise
    noise = [random.uniform(-randomness, randomness) for _ in range(n)] if randomness > 0 else [0.0] * n

    # soft exposure penalty (optional)
    exp_term = 0
    if soft_exposure_penalty > 0 and exposure_counts is not None and num_lineups_total > 0:
        for i in range(n):
            rate = exposure_counts.get(i, 0) / float(max(1, num_lineups_total))
            exp_term += (-soft_exposure_penalty * rate) * selected[i]

    prob += (
        pulp.lpSum(
            (
                players[i].proj
                - own_penalty * players[i].pred_own
                + lev_weight * (players[i].proj * (1.0 - players[i].pred_own))
                + noise[i]
            ) * selected[i]
            for i in range(n)
        )
        + exp_term
    ), "objective"

    solver = pulp.PULP_CBC_CMD(
        msg=CBC_MSG,
        timeLimit=CBC_TIME_LIMIT_SEC,
        gapRel=CBC_GAP_REL,
        threads=CBC_THREADS,
    )
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        return None

    chosen: List[int] = []
    for s in DK_SLOTS:
        pick = None
        for i in range(n):
            v = x.get((i, s))
            if v is not None and v.value() == 1:
                pick = i
                break
        if pick is None:
            return None
        chosen.append(pick)

    return chosen


# -----------------------------
# OUTPUT
# -----------------------------
def lineups_to_df(players: List[Player], lineups: List[List[int]], contest_type: str) -> pd.DataFrame:
    rows = []
    for lu in lineups:
        row: Dict[str, object] = {}
        total_sal = 0
        total_proj = 0.0
        team_counts: Dict[str, int] = {}
        chalk_ct = 0
        sneaky_ct = 0
        own_sum = 0.0

        for slot, idx in zip(DK_SLOTS, lu):
            p = players[idx]
            row[f"{slot}_name"] = p.name
            row[f"{slot}_team"] = p.team
            row[f"{slot}_salary"] = int(p.salary)
            row[f"{slot}_proj"] = float(round(p.proj, 2))

            row[f"{slot}_own_pct"] = float(round(100.0 * p.pred_own, 1))
            row[f"{slot}_is_chalk"] = int(p.is_chalk)
            row[f"{slot}_is_sneaky"] = int(p.is_sneaky)

            total_sal += int(p.salary)
            total_proj += float(p.proj)
            own_sum += float(100.0 * p.pred_own)

            if p.team:
                team_counts[p.team] = team_counts.get(p.team, 0) + 1
            chalk_ct += int(p.is_chalk)
            sneaky_ct += int(p.is_sneaky)

        row["team_counts"] = "; ".join([f"{t}:{c}" for t, c in sorted(team_counts.items(), key=lambda x: (-x[1], x[0]))])
        row["total_salary"] = total_sal
        row["total_proj"] = round(total_proj, 2)
        row["total_own_sum"] = round(own_sum, 1)
        row["chalk_ct"] = chalk_ct
        row["sneaky_ct"] = sneaky_ct
        row["contest_type"] = contest_type
        rows.append(row)

    return pd.DataFrame(rows).sort_values("total_proj", ascending=False).reset_index(drop=True)


# -----------------------------
# PUBLIC API FOR FLASK
# -----------------------------
def generate_nba_df(
    num_lineups: int = 20,
    min_unique: int = MIN_UNIQUE_DEFAULT,
    min_salary_spend: int = MIN_SALARY_SPEND_DEFAULT,
    randomness: float = 0.8,
    salary_cap: int = DK_SALARY_CAP,
    csv_url: Optional[str] = None,
    games: int = 7,
    util_salary_cap: Optional[int] = UTIL_SALARY_CAP_DEFAULT,
    chalk_pctile: int = CHALK_PCTILE_DEFAULT,
    sneaky_pctile: int = SNEAKY_PCTILE_DEFAULT,
    max_player_exposure: float = MAX_PLAYER_EXPOSURE_DEFAULT,
    soft_exposure_penalty: float = SOFT_EXPOSURE_PENALTY_DEFAULT,
    contest_type: str = "gpp_large",
    seed: Optional[int] = 7,
) -> pd.DataFrame:

    url = csv_url or CSV_URL_DEFAULT
    df = fetch_csv_to_df(url)  # always fresh (cache-busted)
    players = parse_players(df, games=games, chalk_pctile=chalk_pctile, sneaky_pctile=sneaky_pctile)

    prof = contest_profile(contest_type)
    base_random = float(prof["base_random"])
    randomness_final = 0.6 * base_random + 0.4 * max(0.0, float(randomness))

    min_chalk = int(prof["min_chalk"])
    max_chalk = int(prof["max_chalk"])
    min_sneaky = int(prof["min_sneaky"])
    own_penalty = float(prof["own_penalty"])
    lev_weight = float(prof["lev_weight"])

    # clamp if pool small
    chalk_total = sum(p.is_chalk for p in players)
    sneaky_total = sum(p.is_sneaky for p in players)
    min_chalk = min(min_chalk, chalk_total)
    min_sneaky = min(min_sneaky, sneaky_total)
    max_chalk = max(max_chalk, min_chalk)

    n = len(players)
    cap_ct = max(1, int(np.floor(max(0.05, min(1.0, float(max_player_exposure))) * int(num_lineups))))
    exposure = {i: 0 for i in range(n)}

    final_lineups: List[List[int]] = []
    previous_sets: List[Set[int]] = []

    for relax_util, uniq_relax, min_sal_relax in RELAX_LADDER:
        final_lineups = []
        previous_sets = []
        exposure = {i: 0 for i in range(n)}

        util_cap_try = None if relax_util else util_salary_cap
        uniq_try = max(0, int(min_unique) - int(uniq_relax))
        min_sal_try = max(0, int(min_salary_spend) - int(min_sal_relax))

        for li in range(int(num_lineups)):
            banned = {i for i, ct in exposure.items() if ct >= cap_ct}

            built = None
            for attempt in range(int(RETRIES_PER_LINEUP) + 1):
                # ramp randomness + tiny per-lineup jitter
                rand_i = randomness_final + 0.12 * attempt + 0.05 * (li / max(1, int(num_lineups) - 1))

                built = optimize_one_lineup(
                    players=players,
                    salary_cap=int(salary_cap),
                    min_salary_spend=min_sal_try,
                    util_salary_cap=util_cap_try,
                    min_unique=uniq_try,
                    previous_sets=previous_sets,
                    banned=banned,
                    own_penalty=own_penalty,
                    lev_weight=lev_weight,
                    min_chalk=min_chalk,
                    max_chalk=max_chalk,
                    min_sneaky=min_sneaky,
                    randomness=rand_i,
                    seed=None if seed is None else int(seed) + li * 10 + attempt,
                    soft_exposure_penalty=float(soft_exposure_penalty),
                    exposure_counts=exposure,
                    num_lineups_total=int(num_lineups),
                )
                if built is not None:
                    break

            if built is None:
                break

            final_lineups.append(built)
            chosen_set = set(built)
            previous_sets.append(chosen_set)
            for idx in chosen_set:
                exposure[idx] += 1

        if final_lineups:
            break

    if not final_lineups:
        raise RuntimeError(
            "No NBA lineups generated.\n"
            "Try: lower min_salary_spend, set min_unique=0/1, increase util_salary_cap, or contest_type=cash."
        )

    return lineups_to_df(players, final_lineups, contest_type=contest_type)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_url", type=str, default=CSV_URL_DEFAULT)
    ap.add_argument("--num_lineups", type=int, default=20)
    ap.add_argument("--games", type=int, default=7)

    ap.add_argument("--salary_cap", type=int, default=DK_SALARY_CAP)
    ap.add_argument("--min_salary_spend", type=int, default=MIN_SALARY_SPEND_DEFAULT)
    ap.add_argument("--min_unique", type=int, default=MIN_UNIQUE_DEFAULT)
    ap.add_argument("--randomness", type=float, default=0.8)
    ap.add_argument("--util_salary_cap", type=int, default=UTIL_SALARY_CAP_DEFAULT if UTIL_SALARY_CAP_DEFAULT else -1)

    ap.add_argument("--contest_type", type=str, default="gpp_large",
                    help="cash | gpp_small | gpp_medium | gpp_large (default)")

    ap.add_argument("--chalk_pctile", type=int, default=CHALK_PCTILE_DEFAULT)
    ap.add_argument("--sneaky_pctile", type=int, default=SNEAKY_PCTILE_DEFAULT)
    ap.add_argument("--max_player_exposure", type=float, default=MAX_PLAYER_EXPOSURE_DEFAULT)
    ap.add_argument("--soft_exposure_penalty", type=float, default=SOFT_EXPOSURE_PENALTY_DEFAULT)
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()
    util_cap = None if args.util_salary_cap is None or args.util_salary_cap < 0 else args.util_salary_cap

    out_df = generate_nba_df(
        num_lineups=args.num_lineups,
        min_unique=args.min_unique,
        min_salary_spend=args.min_salary_spend,
        randomness=args.randomness,
        salary_cap=args.salary_cap,
        csv_url=args.csv_url,
        games=args.games,
        util_salary_cap=util_cap,
        chalk_pctile=args.chalk_pctile,
        sneaky_pctile=args.sneaky_pctile,
        max_player_exposure=args.max_player_exposure,
        soft_exposure_penalty=args.soft_exposure_penalty,
        contest_type=args.contest_type,
        seed=args.seed
    )

    cols = ["total_proj", "total_salary", "total_own_sum", "chalk_ct", "sneaky_ct", "contest_type", "team_counts"]
    print(out_df[cols].head(20).to_string(index=False))
    print("\nSample slot columns: PG_own_pct, PG_is_chalk, PG_is_sneaky ...")

if __name__ == "__main__":
    main()
