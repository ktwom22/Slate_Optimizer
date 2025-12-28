"""
NBA DraftKings Classic — DIVERSIFIED FAST GPP BUILDER (ONE FILE)

What this fixes (your issue):
✅ Stops identical chalk/value repeats by adding:
   - Global exposure caps (max % a player can appear across N lineups)
   - Per-lineup variability in chalk/sneaky requirements
   - Optional team-stack pattern rotation (2-2 vs 3-2 vs 2-2 default)
✅ Fixes uniqueness bug: previous code tracked SLOT indices, not PLAYER indices.
✅ Adds optional "ban over-cap" + "soft exposure penalty" modes
✅ Keeps generate_nba_df(...) for Flask + CLI

Install:
  pip install pandas requests pulp numpy

Run:
  python -u NBA.py --num_lineups 20
"""

import argparse
import math
import os
import random
import re
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple

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

MIN_SALARY_SPEND_DEFAULT = 49500
UTIL_SALARY_CAP_DEFAULT: Optional[int] = 5500

# Chalk/sneaky defaults (will be varied per lineup if enabled)
MIN_CHALK_DEFAULT = 2
MAX_CHALK_DEFAULT = 4
MIN_SNEAKY_DEFAULT = 2
CHALK_PCTILE_DEFAULT = 85
SNEAKY_PCTILE_DEFAULT = 25

OWN_PENALTY_DEFAULT = 28.0
LEVERAGE_WEIGHT_DEFAULT = 0.35

# Diversity controls
MAX_PLAYER_EXPOSURE_DEFAULT = 0.45   # max % of lineups any player can appear (hard ban once exceeded)
SOFT_EXPOSURE_PENALTY_DEFAULT = 0.0  # set to >0 to softly discourage high-exposure players (optional)
VARIABILITY_DEFAULT = True           # vary chalk/sneaky + objective weights per lineup
STACK_MODE_DEFAULT = "rotate"        # "2-2", "3-2", "rotate"

# Speed knobs
CBC_TIME_LIMIT_SEC = float(os.environ.get("NBA_CBC_TIME_LIMIT", "2.0"))
RETRIES_PER_LINEUP = int(os.environ.get("NBA_RETRIES", "3"))
RELAX_LADDER = [
    # (util_cap_relax, uniqueness_relax_steps, min_salary_relax)
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
    positions: Tuple[str, ...]   # base positions e.g. ("PG","SG")
    salary: int
    proj: float
    usage: float
    pred_own: float              # 0..1
    is_chalk: int
    is_sneaky: int


# -----------------------------
# HELPERS
# -----------------------------
def _normalize_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    return c.strip("_")


def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {orig: _normalize_col(orig) for orig in df.columns}
    inv: Dict[str, List[str]] = {}
    for orig, norm in norm_map.items():
        inv.setdefault(norm, []).append(orig)
    for cand in candidates:
        cn = _normalize_col(cand)
        if cn in inv:
            return inv[cn][0]
    return None


def clean_name(s: str) -> str:
    s = re.sub(r"&nbsp;?", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def _to_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        s = str(x).replace("$", "").replace(",", "").strip()
        if s.lower() in ("", "nan", "none"):
            return default
        v = float(s)
        # DK shorthand support (6 => 6000)
        if 0 < v <= 100:
            return int(round(v * 1000))
        return int(round(v))
    except Exception:
        return default


def _to_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        s = str(x).replace(",", "").replace("%", "").strip()
        if s.lower() in ("", "nan", "none"):
            return default
        return float(s)
    except Exception:
        return default


def fetch_csv_to_df(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def target_total_ownership_mass(games: int) -> float:
    g = max(2, min(12, int(games)))
    return 8.0 * (1.0 + 0.15 * (g - 1))


def predict_ownership(df: pd.DataFrame, salary_col: str, proj_col: str, usage_col: Optional[str], games: int) -> pd.Series:
    sal = df[salary_col].apply(lambda v: _to_float(v, 0.0))
    proj = df[proj_col].apply(lambda v: _to_float(v, 0.0))
    usage = df[usage_col].apply(lambda v: _to_float(v, 0.0)) if usage_col else pd.Series(np.zeros(len(df)))

    salary_k = sal / 1000.0
    value_k = np.where(salary_k > 0, proj / salary_k, 0.0)
    cheapness = -zscore(salary_k)

    z_proj = zscore(proj)
    z_val = zscore(pd.Series(value_k, index=df.index))
    z_use = zscore(usage) if usage.abs().sum() > 0 else pd.Series(np.zeros(len(df)), index=df.index)
    z_chp = zscore(cheapness)

    lin = 1.25 * z_val + 0.95 * z_proj + 0.55 * z_use + 0.30 * z_chp
    base = sigmoid(lin)
    base = np.clip(base, 0.001, 0.999)

    mass = target_total_ownership_mass(games)
    scaled = base * (mass / base.sum())
    pred = np.clip(scaled, 0.0, 0.60)
    return pd.Series(pred, index=df.index)


def eligible_slots(player: Player) -> Set[str]:
    base = set(player.positions)
    slots = set(base)
    if "PG" in base or "SG" in base:
        slots.add("G")
    if "SF" in base or "PF" in base:
        slots.add("F")
    slots.add("UTIL")
    return slots


# -----------------------------
# PARSE + CLEAN + TIERS
# -----------------------------
def parse_players(
    df: pd.DataFrame,
    games: int,
    chalk_pctile: int,
    sneaky_pctile: int,
) -> Tuple[List[Player], pd.DataFrame]:
    name_col = _pick_first(df, ["name", "player", "player_name", "nickname"])
    pos_col = _pick_first(df, ["pos", "position", "positions", "dk_position", "roster_position"])
    team_col = _pick_first(df, ["team", "teamabbrev", "team_abbrev", "teamabbr", "tm"])
    sal_col = _pick_first(df, ["salary", "sal", "dk_salary", "cost"])
    proj_col = _pick_first(df, ["proj", "projection", "fpts", "fp", "points", "projected_points"])
    usage_col = _pick_first(df, ["final_usage", "usage", "usg", "usg_pct", "usage_rate"])

    missing = [k for k, v in {"name": name_col, "pos": pos_col, "team": team_col, "salary": sal_col, "proj": proj_col}.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    df = df.copy()
    df["name"] = df[name_col].apply(clean_name)
    df["team"] = df[team_col].astype(str).str.strip().replace({"nan": "", "None": "", "": ""})
    df["pos_raw"] = df[pos_col].astype(str).str.strip().replace({"nan": "", "None": "", "": ""})
    df["salary"] = df[sal_col].apply(lambda v: _to_int(v, 0))
    df["proj"] = df[proj_col].apply(lambda v: _to_float(v, 0.0))
    df["usage"] = df[usage_col].apply(lambda v: _to_float(v, 0.0)) if usage_col else 0.0

    df = df[(df["name"] != "") & (df["team"] != "") & (df["pos_raw"] != "")]
    df = df[(df["salary"] > 0) & (df["proj"] > 0)]
    if df.empty:
        raise ValueError("After cleanup, no valid NBA players remain.")

    df["pred_own"] = predict_ownership(df, sal_col, proj_col, usage_col, games)
    df["pred_own_pct"] = (100.0 * df["pred_own"]).round(1)

    own_pcts = df["pred_own_pct"].astype(float)
    chalk_cut = float(np.percentile(own_pcts, chalk_pctile))
    sneaky_cut = float(np.percentile(own_pcts, sneaky_pctile))

    df["is_chalk"] = (df["pred_own_pct"] >= chalk_cut).astype(int)
    df["is_sneaky"] = (df["pred_own_pct"] <= sneaky_cut).astype(int)

    players: List[Player] = []
    for _, row in df.iterrows():
        pos_parts = re.split(r"[/,;\s]+", str(row["pos_raw"]).upper())
        pos_parts = [p for p in pos_parts if p in {"PG", "SG", "SF", "PF", "C"}]
        if not pos_parts:
            continue
        players.append(Player(
            name=str(row["name"]),
            team=str(row["team"]),
            positions=tuple(sorted(set(pos_parts))),
            salary=int(row["salary"]),
            proj=float(row["proj"]),
            usage=float(row["usage"]),
            pred_own=float(row["pred_own"]),
            is_chalk=int(row["is_chalk"]),
            is_sneaky=int(row["is_sneaky"]),
        ))

    if not players:
        raise ValueError("No valid players after position parsing.")

    analysis = df[["name", "team", "pos_raw", "salary", "proj", "usage", "pred_own_pct", "is_chalk", "is_sneaky"]].copy()
    analysis.rename(columns={"pos_raw": "pos"}, inplace=True)
    return players, analysis


# -----------------------------
# STACK MODES
# -----------------------------
def choose_stack_mode(stack_mode: str, li: int) -> str:
    sm = (stack_mode or "rotate").lower().strip()
    if sm in ("2-2", "3-2"):
        return sm
    # rotate pattern: 2-2, 2-2, 3-2, 2-2, 3-2 ...
    return "3-2" if (li % 3 == 2) else "2-2"


def add_team_stack_constraints(
    prob: pulp.LpProblem,
    selected: List[pulp.LpAffineExpression],
    players: List[Player],
    mode: str,
) -> None:
    teams = sorted(set(p.team for p in players))
    team_players: Dict[str, List[int]] = {t: [] for t in teams}
    for i, p in enumerate(players):
        team_players[p.team].append(i)

    if mode == "2-2":
        # exactly two teams have exactly 2 players; others <=1
        team_used = {t: pulp.LpVariable(f"team_used_{t}", cat="Binary") for t in teams}
        for t, idxs in team_players.items():
            team_count = pulp.lpSum(selected[i] for i in idxs)
            prob += team_count >= 2 * team_used[t], f"team_min_if_used_{t}"
            prob += team_count <= 2 * team_used[t] + 1 * (1 - team_used[t]), f"team_max_if_not_used_{t}"
        prob += pulp.lpSum(team_used[t] for t in teams) == 2, "exactly_two_stack_teams"
        return

    if mode == "3-2":
        # one team exactly 3, one team exactly 2, others <=1
        team3 = {t: pulp.LpVariable(f"team3_{t}", cat="Binary") for t in teams}
        team2 = {t: pulp.LpVariable(f"team2_{t}", cat="Binary") for t in teams}
        for t, idxs in team_players.items():
            team_count = pulp.lpSum(selected[i] for i in idxs)
            # enforce exact counts when flagged
            prob += team_count >= 3 * team3[t], f"team3_lb_{t}"
            prob += team_count <= 3 * team3[t] + 10 * (1 - team3[t]), f"team3_ub_{t}"
            prob += team_count >= 2 * team2[t], f"team2_lb_{t}"
            prob += team_count <= 2 * team2[t] + 10 * (1 - team2[t]), f"team2_ub_{t}"

            # if not chosen as 3-team or 2-team, cap at 1
            prob += team_count <= 1 + 10 * (team3[t] + team2[t]), f"others_cap_{t}"

            # can't be both
            prob += team3[t] + team2[t] <= 1, f"no_both_{t}"

        prob += pulp.lpSum(team3[t] for t in teams) == 1, "exactly_one_team3"
        prob += pulp.lpSum(team2[t] for t in teams) == 1, "exactly_one_team2"
        return


# -----------------------------
# VARIABILITY (chalk/sneaky + weights)
# -----------------------------
def lineup_profile(
    li: int,
    min_chalk: int,
    max_chalk: int,
    min_sneaky: int,
    own_penalty: float,
    leverage_weight: float,
    variability: bool,
) -> Tuple[int, int, int, float, float]:
    if not variability:
        return min_chalk, max_chalk, min_sneaky, own_penalty, leverage_weight

    # rotate profiles every 5 lineups
    r = li % 5
    if r == 0:
        # contrarian
        mc, xc, ms = max(0, min_chalk - 2), max(0, max_chalk - 2), min_sneaky + 1
        op = own_penalty * random.uniform(1.05, 1.35)
        lw = leverage_weight * random.uniform(1.05, 1.25)
    elif r == 1:
        # balanced
        mc, xc, ms = min_chalk, max_chalk, min_sneaky
        op = own_penalty * random.uniform(0.95, 1.15)
        lw = leverage_weight * random.uniform(0.95, 1.15)
    elif r == 2:
        # slightly chalky
        mc, xc, ms = min_chalk + 1, max_chalk + 1, max(0, min_sneaky - 1)
        op = own_penalty * random.uniform(0.85, 1.05)
        lw = leverage_weight * random.uniform(0.85, 1.05)
    elif r == 3:
        # very balanced but higher sneaky
        mc, xc, ms = max(0, min_chalk - 1), max_chalk, min_sneaky + 1
        op = own_penalty * random.uniform(0.95, 1.20)
        lw = leverage_weight * random.uniform(1.00, 1.20)
    else:
        # projection-ish
        mc, xc, ms = min_chalk, max_chalk + 1, max(0, min_sneaky - 1)
        op = own_penalty * random.uniform(0.80, 1.00)
        lw = leverage_weight * random.uniform(0.80, 1.00)

    if xc < mc:
        xc = mc
    return mc, xc, ms, op, lw


# -----------------------------
# OPTIMIZER (one lineup)
# -----------------------------
def optimize_one_lineup(
    players: List[Player],
    salary_cap: int,
    min_salary_spend: int,
    min_unique_vs_previous: int,
    previous_player_sets: List[Set[int]],
    randomness: float,
    util_salary_cap: Optional[int],
    own_penalty: float,
    leverage_weight: float,
    min_chalk: int,
    max_chalk: int,
    min_sneaky: int,
    seed: Optional[int],
    banned_players: Optional[Set[int]] = None,
    soft_exposure_penalty: float = 0.0,
    exposure_counts: Optional[Dict[int, int]] = None,
    num_lineups_total: int = 0,
    stack_mode: str = "2-2",
) -> Optional[List[int]]:

    if seed is not None:
        random.seed(seed)

    n = len(players)
    elig = [eligible_slots(p) for p in players]
    teams = sorted(set(p.team for p in players))
    if len(teams) < 2:
        return None

    # Build decision vars
    x: Dict[Tuple[int, str], pulp.LpVariable] = {}
    for i in range(n):
        for s in DK_SLOTS:
            if s in elig[i]:
                x[(i, s)] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")

    prob = pulp.LpProblem("NBA_DK_GPP_DIVERSE", pulp.LpMaximize)

    # Fill each slot exactly once
    for s in DK_SLOTS:
        prob += pulp.lpSum(x[(i, s)] for i in range(n) if (i, s) in x) == 1, f"fill_{s}"

    # Each player at most once
    selected = [pulp.lpSum(x[(i, s)] for s in DK_SLOTS if (i, s) in x) for i in range(n)]
    for i in range(n):
        prob += selected[i] <= 1, f"one_slot_{i}"

    # Salary
    total_salary = pulp.lpSum(players[i].salary * selected[i] for i in range(n))
    prob += total_salary <= salary_cap, "salary_cap"
    prob += total_salary >= min_salary_spend, "min_salary_spend"

    # Must have >=1 PG and >=1 PF
    prob += pulp.lpSum(selected[i] for i in range(n) if "PG" in players[i].positions) >= 1, "at_least_one_pg"
    prob += pulp.lpSum(selected[i] for i in range(n) if "PF" in players[i].positions) >= 1, "at_least_one_pf"

    # Stack constraints (2-2 or 3-2)
    add_team_stack_constraints(prob, selected, players, mode=stack_mode)

    # PG + PF same-team exists
    team_players: Dict[str, List[int]] = {t: [] for t in teams}
    for i, p in enumerate(players):
        team_players[p.team].append(i)

    y = {t: pulp.LpVariable(f"y_pg_pf_{t}", cat="Binary") for t in teams}
    for t, idxs in team_players.items():
        pg_count_t = pulp.lpSum(selected[i] for i in idxs if "PG" in players[i].positions)
        pf_count_t = pulp.lpSum(selected[i] for i in idxs if "PF" in players[i].positions)
        prob += pg_count_t >= y[t], f"pg_needed_for_pair_{t}"
        prob += pf_count_t >= y[t], f"pf_needed_for_pair_{t}"
    prob += pulp.lpSum(y[t] for t in teams) >= 1, "require_pg_pf_pair"

    # UTIL salary cap (optional)
    if util_salary_cap is not None:
        prob += pulp.lpSum(players[i].salary * x[(i, "UTIL")] for i in range(n) if (i, "UTIL") in x) <= util_salary_cap, "util_salary_cap"

    # Chalk/sneaky constraints
    chalk_sum = pulp.lpSum(players[i].is_chalk * selected[i] for i in range(n))
    sneaky_sum = pulp.lpSum(players[i].is_sneaky * selected[i] for i in range(n))
    prob += chalk_sum >= min_chalk, "min_chalk"
    prob += chalk_sum <= max_chalk, "max_chalk"
    prob += sneaky_sum >= min_sneaky, "min_sneaky"

    # Ban players (hard exposure cap)
    if banned_players:
        for i in banned_players:
            if 0 <= i < n:
                prob += selected[i] == 0, f"ban_{i}"

    # Uniqueness across lineups (FIXED: compare actual player indices)
    lineup_size = len(DK_SLOTS)
    if previous_player_sets and min_unique_vs_previous > 0:
        max_overlap = lineup_size - min_unique_vs_previous
        for li, prev in enumerate(previous_player_sets, start=1):
            prob += pulp.lpSum(selected[i] for i in prev) <= max_overlap, f"uniq_prev_{li}"

    # Objective
    noise = [random.uniform(-randomness, randomness) for _ in range(n)] if randomness > 0 else [0.0] * n

    # Soft exposure penalty (optional)
    exp_term = 0
    if soft_exposure_penalty > 0 and exposure_counts is not None and num_lineups_total > 0:
        # penalty proportional to (current_exposure_rate)
        for i in range(n):
            ct = exposure_counts.get(i, 0)
            rate = ct / float(num_lineups_total)
            exp_term += (-soft_exposure_penalty * rate) * selected[i]

    prob += pulp.lpSum(
        (
            players[i].proj
            - own_penalty * players[i].pred_own
            + leverage_weight * (players[i].proj * (1.0 - players[i].pred_own))
            + noise[i]
        ) * selected[i]
        for i in range(n)
    ) + exp_term, "objective"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=CBC_TIME_LIMIT_SEC)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        return None

    # Extract chosen players by slot
    chosen_by_slot: List[int] = []
    for s in DK_SLOTS:
        chosen = None
        for i in range(n):
            var = x.get((i, s))
            if var is not None and var.value() == 1:
                chosen = i
                break
        if chosen is None:
            return None
        chosen_by_slot.append(chosen)

    return chosen_by_slot


def summarize_lineup(players: List[Player], lineup_idxs: List[int]) -> Dict[str, object]:
    total_salary = sum(players[i].salary for i in lineup_idxs)
    total_proj = sum(players[i].proj for i in lineup_idxs)

    team_counts: Dict[str, int] = {}
    for i in lineup_idxs:
        t = players[i].team
        team_counts[t] = team_counts.get(t, 0) + 1
    team_counts = dict(sorted(team_counts.items(), key=lambda x: (-x[1], x[0])))

    chalk_ct = sum(players[i].is_chalk for i in lineup_idxs)
    sneaky_ct = sum(players[i].is_sneaky for i in lineup_idxs)

    return {
        "total_salary": total_salary,
        "total_proj": round(total_proj, 2),
        "team_counts": "; ".join([f"{t}:{c}" for t, c in team_counts.items()]),
        "chalk": chalk_ct,
        "sneaky": sneaky_ct,
    }


def lineups_to_df(players: List[Player], lineups: List[List[int]]) -> pd.DataFrame:
    rows = []
    for lu in lineups:
        row: Dict[str, object] = {}
        for slot, idx in zip(DK_SLOTS, lu):
            p = players[idx]
            row[f"{slot}_name"] = p.name
            row[f"{slot}_team"] = p.team
            row[f"{slot}_salary"] = p.salary
            row[f"{slot}_proj"] = round(p.proj, 2)

        meta = summarize_lineup(players, lu)
        row["team_counts"] = meta["team_counts"]
        row["total_salary"] = meta["total_salary"]
        row["total_proj"] = meta["total_proj"]
        row["chalk_ct"] = meta["chalk"]
        row["sneaky_ct"] = meta["sneaky"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("total_proj", ascending=False).reset_index(drop=True)


# -----------------------------
# PUBLIC API FOR FLASK
# -----------------------------
def generate_nba_df(
    num_lineups: int = 20,
    min_unique: int = 2,
    min_salary_spend: int = MIN_SALARY_SPEND_DEFAULT,
    randomness: float = 0.8,
    salary_cap: int = DK_SALARY_CAP,
    csv_url: Optional[str] = None,
    games: int = 7,
    util_salary_cap: Optional[int] = UTIL_SALARY_CAP_DEFAULT,
    min_chalk: int = MIN_CHALK_DEFAULT,
    max_chalk: int = MAX_CHALK_DEFAULT,
    min_sneaky: int = MIN_SNEAKY_DEFAULT,
    chalk_pctile: int = CHALK_PCTILE_DEFAULT,
    sneaky_pctile: int = SNEAKY_PCTILE_DEFAULT,
    own_penalty: float = OWN_PENALTY_DEFAULT,
    leverage_weight: float = LEVERAGE_WEIGHT_DEFAULT,
    seed: Optional[int] = 7,
    max_player_exposure: float = MAX_PLAYER_EXPOSURE_DEFAULT,
    soft_exposure_penalty: float = SOFT_EXPOSURE_PENALTY_DEFAULT,
    variability: bool = VARIABILITY_DEFAULT,
    stack_mode: str = STACK_MODE_DEFAULT,
) -> pd.DataFrame:

    url = csv_url or CSV_URL_DEFAULT
    df = fetch_csv_to_df(url)
    players, _analysis = parse_players(df, games=games, chalk_pctile=chalk_pctile, sneaky_pctile=sneaky_pctile)

    # Auto-fix counts if pool small
    chalk_total = sum(p.is_chalk for p in players)
    sneaky_total = sum(p.is_sneaky for p in players)
    min_chalk = min(min_chalk, chalk_total)
    min_sneaky = min(min_sneaky, sneaky_total)
    max_chalk = max(max_chalk, min_chalk)

    n = len(players)
    player_counts: Dict[int, int] = {i: 0 for i in range(n)}
    cap = max(1, int(math.floor(max(0.05, min(1.0, max_player_exposure)) * num_lineups)))

    lineups: List[List[int]] = []
    previous_player_sets: List[Set[int]] = []

    # Relax ladder to avoid hangs
    for relax_util, relax_uniq_steps, relax_min_sal in RELAX_LADDER:
        util_cap_try = None if relax_util else util_salary_cap
        uniq_try = max(0, min_unique - relax_uniq_steps)
        min_sal_try = max(0, min_salary_spend - relax_min_sal)

        lineups = []
        previous_player_sets = []
        player_counts = {i: 0 for i in range(n)}

        for li in range(num_lineups):
            built = None

            # hard ban players already at exposure cap
            banned = {i for i, ct in player_counts.items() if ct >= cap}

            # lineup-specific profile
            mc_i, xc_i, ms_i, op_i, lw_i = lineup_profile(
                li=li,
                min_chalk=min_chalk,
                max_chalk=max_chalk,
                min_sneaky=min_sneaky,
                own_penalty=own_penalty,
                leverage_weight=leverage_weight,
                variability=variability,
            )

            # stack mode rotation
            mode_i = choose_stack_mode(stack_mode, li)

            for attempt in range(RETRIES_PER_LINEUP):
                rand_i = randomness + (0.15 * attempt) + (0.15 * (li / max(1, num_lineups - 1)))

                built = optimize_one_lineup(
                    players=players,
                    salary_cap=salary_cap,
                    min_salary_spend=min_sal_try,
                    min_unique_vs_previous=uniq_try,
                    previous_player_sets=previous_player_sets,
                    randomness=rand_i,
                    util_salary_cap=util_cap_try,
                    own_penalty=op_i,
                    leverage_weight=lw_i,
                    min_chalk=mc_i,
                    max_chalk=xc_i,
                    min_sneaky=ms_i,
                    seed=None if seed is None else seed + li * 10 + attempt,
                    banned_players=banned,
                    soft_exposure_penalty=soft_exposure_penalty,
                    exposure_counts=player_counts,
                    num_lineups_total=num_lineups,
                    stack_mode=mode_i,
                )
                if built is not None:
                    break

            if built is None:
                break

            # store + update exposure
            lineups.append(built)
            chosen_set = set(built)
            previous_player_sets.append(chosen_set)
            for idx in chosen_set:
                player_counts[idx] += 1

        if len(lineups) > 0:
            break

    if not lineups:
        raise RuntimeError(
            "No NBA lineups generated.\n"
            "Try loosening: lower min_salary_spend, set min_unique=1, reduce min_sneaky, increase util_salary_cap, or set stack_mode=2-2."
        )

    return lineups_to_df(players, lineups)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_url", type=str, default=CSV_URL_DEFAULT)
    ap.add_argument("--games", type=int, default=7)
    ap.add_argument("--num_lineups", type=int, default=20)

    ap.add_argument("--salary_cap", type=int, default=DK_SALARY_CAP)
    ap.add_argument("--min_salary_spend", type=int, default=MIN_SALARY_SPEND_DEFAULT)
    ap.add_argument("--min_unique", type=int, default=2)
    ap.add_argument("--randomness", type=float, default=0.8)

    ap.add_argument("--util_salary_cap", type=int, default=UTIL_SALARY_CAP_DEFAULT if UTIL_SALARY_CAP_DEFAULT else -1)

    ap.add_argument("--min_chalk", type=int, default=MIN_CHALK_DEFAULT)
    ap.add_argument("--max_chalk", type=int, default=MAX_CHALK_DEFAULT)
    ap.add_argument("--min_sneaky", type=int, default=MIN_SNEAKY_DEFAULT)

    ap.add_argument("--chalk_pctile", type=int, default=CHALK_PCTILE_DEFAULT)
    ap.add_argument("--sneaky_pctile", type=int, default=SNEAKY_PCTILE_DEFAULT)

    ap.add_argument("--own_penalty", type=float, default=OWN_PENALTY_DEFAULT)
    ap.add_argument("--leverage_weight", type=float, default=LEVERAGE_WEIGHT_DEFAULT)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--max_player_exposure", type=float, default=MAX_PLAYER_EXPOSURE_DEFAULT)
    ap.add_argument("--soft_exposure_penalty", type=float, default=SOFT_EXPOSURE_PENALTY_DEFAULT)
    ap.add_argument("--variability", action="store_true", help="Enable per-lineup chalk/sneaky + weight variability")
    ap.add_argument("--no_variability", action="store_true", help="Disable variability")
    ap.add_argument("--stack_mode", type=str, default=STACK_MODE_DEFAULT, help="2-2, 3-2, rotate")

    args = ap.parse_args()

    util_cap = None if args.util_salary_cap is None or args.util_salary_cap < 0 else args.util_salary_cap

    variability = True
    if args.no_variability:
        variability = False
    elif args.variability:
        variability = True

    out_df = generate_nba_df(
        num_lineups=args.num_lineups,
        min_unique=args.min_unique,
        min_salary_spend=args.min_salary_spend,
        randomness=args.randomness,
        salary_cap=args.salary_cap,
        csv_url=args.csv_url,
        games=args.games,
        util_salary_cap=util_cap,
        min_chalk=args.min_chalk,
        max_chalk=args.max_chalk,
        min_sneaky=args.min_sneaky,
        chalk_pctile=args.chalk_pctile,
        sneaky_pctile=args.sneaky_pctile,
        own_penalty=args.own_penalty,
        leverage_weight=args.leverage_weight,
        seed=args.seed,
        max_player_exposure=args.max_player_exposure,
        soft_exposure_penalty=args.soft_exposure_penalty,
        variability=variability,
        stack_mode=args.stack_mode,
    )

    print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
