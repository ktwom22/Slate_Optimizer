"""
NHL DraftKings Classic Optimizer (ONE FILE) — nhl_optimizer.py

Uses YOUR CSV:
https://docs.google.com/spreadsheets/d/e/2PACX-1vTJOqq7cId7F4U7QmfyXR8yMb6wc88PV7c_QHbCHb8a0f-tezW0gvLadFx-gSkNCw/pub?gid=562844939&single=true&output=csv

DK NHL Classic roster:
C, C, W, W, W, D, D, G, UTIL  (Salary cap 50k)

Key features:
✅ EXACT stack types selected by user (SKATERS ONLY)
   Examples: 3-2-2, 4-3, 3-3-2, 2-2-2, 3-2, 2-2, etc.
   - Stack logic applies to SKATERS ONLY (C/W/D/UTIL). Goalie excluded from stack counts.
   - "Exact stack" means: those stack team skater counts are exact, and all other teams max 1 skater.

✅ NBA-like diversification:
   - Projected ownership (estimated from salary + proj + value-ish + usage-ish fields)
   - Chalk/sneaky flags
   - Contest-type modes (single/3max/20max/150max) that change exposure + uniqueness defaults
   - Hard player exposure caps
   - Hard CHEAP SKATER exposure caps
   - True uniqueness vs previous lineups using PLAYER indices (not slots)

Install:
  pip install pandas requests pulp numpy

Run:
  python -u nhl_optimizer.py --num_lineups 20 --stack_type 3-2-2 --mode 20max

Exposes:
  generate_nhl_df(...) -> pandas DataFrame (DK-style output columns)
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
    "NHL_CSV_URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vTJOqq7cId7F4U7QmfyXR8yMb6wc88PV7c_QHbCHb8a0f-tezW0gvLadFx-gSkNCw/pub?gid=562844939&single=true&output=csv"
)

DK_SALARY_CAP = 50000
DK_SLOTS = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]
LINEUP_SIZE = len(DK_SLOTS)

SKATER_SLOTS = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "UTIL"]
GOALIE_SLOT = "G"

# speed knobs
CBC_TIME_LIMIT_SEC = float(os.environ.get("NHL_CBC_TIME_LIMIT", "1.6"))
CBC_GAP_REL = float(os.environ.get("NHL_CBC_GAP_REL", "0.09"))
CBC_THREADS = int(os.environ.get("NHL_CBC_THREADS", "1"))
CBC_MSG = bool(int(os.environ.get("NHL_CBC_MSG", "0")))

MAX_SOLVES_PER_LINEUP = int(os.environ.get("NHL_MAX_SOLVES_PER_LU", "4"))

# Ownership modeling defaults
CHALK_PCTILE_DEFAULT = 85
SNEAKY_PCTILE_DEFAULT = 25

OWN_SALARY_WEIGHT = 0.55
OWN_PROJ_WEIGHT = 1.05
OWN_VALUE_WEIGHT = 0.65
OWN_USAGE_WEIGHT = 0.35   # if we can find "usage" proxy
OWN_FORM_WEIGHT = 0.20    # if we can find L5/L10-ish proxy


# Contest mode presets (mirrors your NBA logic)
MODE_PRESETS = {
    "single": {
        "max_player_exposure": 0.85,
        "cheap_cap_exposure": 0.80,
        "min_unique": 1,
        "randomness": 0.35,
        "own_penalty": 16.0,
        "leverage_weight": 0.15,
    },
    "3max": {
        "max_player_exposure": 0.65,
        "cheap_cap_exposure": 0.55,
        "min_unique": 2,
        "randomness": 0.55,
        "own_penalty": 22.0,
        "leverage_weight": 0.25,
    },
    "20max": {
        "max_player_exposure": 0.45,
        "cheap_cap_exposure": 0.35,
        "min_unique": 3,
        "randomness": 0.85,
        "own_penalty": 28.0,
        "leverage_weight": 0.33,
    },
    "150max": {
        "max_player_exposure": 0.30,
        "cheap_cap_exposure": 0.22,
        "min_unique": 4,
        "randomness": 1.05,
        "own_penalty": 32.0,
        "leverage_weight": 0.40,
    },
}


# -----------------------------
# DATA MODEL
# -----------------------------
@dataclass(frozen=True)
class Player:
    name: str
    team: str
    pos: str        # C / W / D / G
    salary: int
    proj: float
    opp: str

    pred_own: float   # 0..1
    is_chalk: int
    is_sneaky: int

    # Optional signals used for ownership estimation (if available)
    value: float
    usage: float
    form: float


# -----------------------------
# CSV
# -----------------------------
def fetch_csv_to_df(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


# -----------------------------
# PARSING HELPERS
# -----------------------------
def clean_text(x) -> str:
    s = str(x)
    s = s.replace("\t", " ")
    s = re.sub(r"&nbsp;?", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    inv = {norm_key(c): c for c in df.columns}
    for cand in candidates:
        k = norm_key(cand)
        if k in inv:
            return inv[k]
    return None


_num_re = re.compile(r"[-+]?\d*\.?\d+")


def parse_float(x, default=0.0) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    s = str(x).replace(",", "").replace("$", "").replace("%", "").strip()
    m = _num_re.search(s)
    if not m:
        return default
    try:
        return float(m.group(0))
    except Exception:
        return default


def parse_salary(x, default=0) -> int:
    v = parse_float(x, float(default))
    if v <= 0:
        return default
    if v <= 100:
        return int(round(v * 1000))
    return int(round(v))


def normalize_pos(p: str) -> str:
    p = clean_text(p).upper()
    if p in ("LW", "RW", "W"):
        return "W"
    if p.startswith("C"):
        return "C"
    if p.startswith("D"):
        return "D"
    if p.startswith("G"):
        return "G"
    return p


def find_proj_col(df: pd.DataFrame) -> Optional[str]:
    direct = ["proj", "projection", "fpts", "fp", "points", "projected_points", "dk_fp_projected"]
    col = pick_col(df, direct)
    if col:
        return col
    for c in df.columns:
        k = norm_key(c)
        if ("proj" in k or "project" in k) and ("fp" in k or "fpt" in k or "points" in k):
            return c
    return None


def zscore(arr: np.ndarray) -> np.ndarray:
    sd = arr.std()
    if sd <= 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / sd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_ownership(tmp: pd.DataFrame,
                      salary_col: str,
                      proj_col: str,
                      value_col: Optional[str],
                      usage_col: Optional[str],
                      form_col: Optional[str]) -> np.ndarray:
    sal = tmp[salary_col].to_numpy(dtype=float)
    proj = tmp[proj_col].to_numpy(dtype=float)

    value = tmp[value_col].to_numpy(dtype=float) if value_col else np.zeros(len(tmp))
    usage = tmp[usage_col].to_numpy(dtype=float) if usage_col else np.zeros(len(tmp))
    form = tmp[form_col].to_numpy(dtype=float) if form_col else np.zeros(len(tmp))

    sal_k = np.where(sal > 0, sal / 1000.0, 0.0)
    cheap = zscore(-sal_k)

    # fallback value if none
    if np.any(value):
        val_signal = zscore(value)
    else:
        val_signal = zscore(np.where(sal_k > 0, proj / np.maximum(1e-6, sal_k), 0.0))

    lin = (
        OWN_SALARY_WEIGHT * cheap +
        OWN_PROJ_WEIGHT * zscore(proj) +
        OWN_VALUE_WEIGHT * val_signal +
        OWN_USAGE_WEIGHT * (zscore(usage) if np.any(usage) else 0.0) +
        OWN_FORM_WEIGHT * (zscore(form) if np.any(form) else 0.0)
    )

    base = sigmoid(lin)
    base = np.clip(base, 0.001, 0.999)

    # scale to a “reasonable” ownership mass for the slate pool
    target_mass = min(20.0, max(10.0, 11.5 + 0.05 * len(tmp)))
    scaled = base * (target_mass / base.sum())
    pred = np.clip(scaled, 0.0, 0.60)
    return pred


def parse_players(df: pd.DataFrame, chalk_pctile: int, sneaky_pctile: int, verbose: bool = False) -> List[Player]:
    pos_col = pick_col(df, ["pos", "position", "dk_position", "roster_position"])
    name_col = pick_col(df, ["name", "player", "player_name"])
    team_col = pick_col(df, ["team", "teamabbr", "team_abbrev", "tm"])
    sal_col = pick_col(df, ["salary", "sal", "dk_salary", "cost"])
    proj_col = find_proj_col(df)
    opp_col = pick_col(df, ["opp", "opponent", "vs", "opponent_team"])

    # optional columns (we’ll use what exists)
    value_col = pick_col(df, ["value", "pts_per_dollar", "proj_per_k", "dk_value"])
    usage_col = pick_col(df, ["final_usage", "usage", "usuage", "toi", "minutes", "pp_line", "power_play_line"])
    l5_col = pick_col(df, ["l5", "l5_avg", "last5", "last_5"])
    l10_col = pick_col(df, ["l10", "l10_avg", "last10", "last_10"])

    missing = []
    if not pos_col: missing.append("pos")
    if not name_col: missing.append("name")
    if not team_col: missing.append("team")
    if not sal_col: missing.append("salary")
    if not proj_col: missing.append("proj")
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    tmp = df.copy()
    tmp["name"] = tmp[name_col].apply(clean_text)
    tmp["team"] = tmp[team_col].apply(clean_text)
    tmp["pos"] = tmp[pos_col].apply(normalize_pos)
    tmp["salary"] = tmp[sal_col].apply(lambda v: parse_salary(v, 0))
    tmp["proj"] = tmp[proj_col].apply(lambda v: parse_float(v, 0.0))
    tmp["opp"] = tmp[opp_col].apply(clean_text) if opp_col else ""

    # optional numeric signals
    tmp["value"] = tmp[value_col].apply(lambda v: parse_float(v, 0.0)) if value_col else 0.0
    tmp["usage"] = tmp[usage_col].apply(lambda v: parse_float(v, 0.0)) if usage_col else 0.0
    l5 = tmp[l5_col].apply(lambda v: parse_float(v, 0.0)) if l5_col else pd.Series([0.0] * len(tmp))
    l10 = tmp[l10_col].apply(lambda v: parse_float(v, 0.0)) if l10_col else pd.Series([0.0] * len(tmp))
    tmp["form"] = (0.65 * l5 + 0.35 * l10).astype(float)

    # filter
    tmp = tmp[(tmp["name"] != "") & (tmp["team"] != "") & (tmp["pos"] != "")]
    tmp = tmp[tmp["pos"].isin(["C", "W", "D", "G"])]
    tmp = tmp[(tmp["salary"] > 0) & (tmp["proj"] > 0)]
    if tmp.empty:
        raise ValueError("No valid NHL players after cleanup.")

    # ownership + chalk/sneaky
    tmp["pred_own"] = predict_ownership(tmp, "salary", "proj",
                                        "value" if value_col else None,
                                        "usage" if usage_col else None,
                                        "form" if (l5_col or l10_col) else None)
    tmp["pred_own_pct"] = (100.0 * tmp["pred_own"]).round(1)

    own_pcts = tmp["pred_own_pct"].to_numpy(dtype=float)
    chalk_cut = float(np.percentile(own_pcts, chalk_pctile))
    sneaky_cut = float(np.percentile(own_pcts, sneaky_pctile))
    tmp["is_chalk"] = (tmp["pred_own_pct"] >= chalk_cut).astype(int)
    tmp["is_sneaky"] = (tmp["pred_own_pct"] <= sneaky_cut).astype(int)

    players: List[Player] = []
    for _, r in tmp.iterrows():
        players.append(Player(
            name=str(r["name"]),
            team=str(r["team"]),
            pos=str(r["pos"]),
            salary=int(r["salary"]),
            proj=float(r["proj"]),
            opp=str(r["opp"]),

            pred_own=float(r["pred_own"]),
            is_chalk=int(r["is_chalk"]),
            is_sneaky=int(r["is_sneaky"]),

            value=float(r["value"]) if value_col else 0.0,
            usage=float(r["usage"]) if usage_col else 0.0,
            form=float(r["form"]) if (l5_col or l10_col) else 0.0,
        ))

    if verbose:
        print(
            f"Detected columns -> POS:{pos_col} | NAME:{name_col} | TEAM:{team_col} | SALARY:{sal_col} | PROJ:{proj_col} | OPP:{opp_col or 'None'}",
            flush=True
        )
        if value_col: print(f"Using Value col: {value_col}", flush=True)
        if usage_col: print(f"Using Usage proxy col: {usage_col}", flush=True)
        if l5_col or l10_col: print(f"Using Form cols: {l5_col or 'None'} / {l10_col or 'None'}", flush=True)

    return players


# -----------------------------
# ELIGIBILITY
# -----------------------------
def eligible_slots(pos: str) -> Set[str]:
    pos = pos.upper()
    if pos == "C":
        return {"C1", "C2", "UTIL"}
    if pos == "W":
        return {"W1", "W2", "W3", "UTIL"}
    if pos == "D":
        return {"D1", "D2", "UTIL"}
    if pos == "G":
        return {"G"}
    return set()


# -----------------------------
# STACK TYPE PARSER
# -----------------------------
def parse_stack_type(stack_type: str) -> List[int]:
    if stack_type is None:
        return []
    s = str(stack_type).strip()
    if not s:
        return []
    s = s.replace(",", "-").replace(" ", "-")
    parts = [p for p in s.split("-") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p.strip()))
        except Exception:
            raise ValueError(f"Invalid stack_type: {stack_type}. Use like '3-2-2'.")
    out = [x for x in out if x > 0]
    return out


# -----------------------------
# OPTIMIZER (single lineup)
# -----------------------------
def optimize_one_nhl(
    players: List[Player],
    salary_cap: int,
    min_salary_spend: int,
    min_unique_vs_previous: int,
    previous_lineups: List[Set[int]],
    randomness: float,
    seed: Optional[int],

    # exact stack on SKATERS ONLY
    stack_counts: List[int],

    avoid_goalie_vs_opp_skaters: bool,

    # diversification controls
    own_penalty: float,
    leverage_weight: float,
    min_chalk: int,
    max_chalk: int,
    min_sneaky: int,

    banned: Set[int],
    cheap_banned: Set[int],
) -> Optional[List[int]]:
    if seed is not None:
        random.seed(seed)

    n = len(players)
    teams = sorted(set(p.team for p in players))
    team_idxs: Dict[str, List[int]] = {t: [] for t in teams}
    for i, p in enumerate(players):
        team_idxs[p.team].append(i)

    # vars x[(i,slot)] only if eligible
    x: Dict[Tuple[int, str], pulp.LpVariable] = {}
    elig_by_slot: Dict[str, List[int]] = {s: [] for s in DK_SLOTS}
    for i, p in enumerate(players):
        for s in eligible_slots(p.pos):
            x[(i, s)] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")
            elig_by_slot[s].append(i)

    for s in DK_SLOTS:
        if not elig_by_slot[s]:
            return None

    prob = pulp.LpProblem("NHL_DK_CLASSIC_DIVERSE", pulp.LpMaximize)

    # fill slots
    for s in DK_SLOTS:
        prob += pulp.lpSum(x[(i, s)] for i in elig_by_slot[s]) == 1, f"fill_{s}"

    # each player at most once
    selected = [pulp.lpSum(x.get((i, s), 0) for s in DK_SLOTS) for i in range(n)]
    for i in range(n):
        prob += selected[i] <= 1, f"one_{i}"

    # salary
    total_salary = pulp.lpSum(players[i].salary * selected[i] for i in range(n))
    prob += total_salary <= salary_cap, "salary_cap"
    prob += total_salary >= min_salary_spend, "min_salary_spend"

    # bans (exposure caps)
    for i in banned:
        if 0 <= i < n:
            prob += selected[i] == 0, f"ban_{i}"
    for i in cheap_banned:
        if 0 <= i < n:
            prob += selected[i] == 0, f"cheap_ban_{i}"

    # -----------------------------
    # EXACT STACK (SKATERS ONLY) — GOALIE EXCLUDED
    # -----------------------------
    # skater selection = all slots except goalie slot
    skater_selected = {
        i: pulp.lpSum(x.get((i, s), 0) for s in SKATER_SLOTS)
        for i in range(n)
    }
    prob += pulp.lpSum(skater_selected[i] for i in range(n)) == 8, "skaters_eq_8"

    team_skaters_count = {t: pulp.lpSum(skater_selected[i] for i in team_idxs[t]) for t in teams}

    if stack_counts:
        reqs = [r for r in stack_counts if r >= 2]  # stacks are 2+
        if not reqs:
            raise ValueError("stack_type must include at least one 2+ stack (e.g., 3-2-2).")
        if sum(reqs) > 8:
            raise ValueError(f"stack_type {stack_counts} impossible: stack sum > 8 skaters.")

        M = 8
        a = {(t, j): pulp.LpVariable(f"a_{t}_{j}", cat="Binary") for t in teams for j in range(len(reqs))}

        for j, req in enumerate(reqs):
            prob += pulp.lpSum(a[(t, j)] for t in teams) == 1, f"assign_{j}_{req}"

        for t in teams:
            prob += pulp.lpSum(a[(t, j)] for j in range(len(reqs))) <= 1, f"team_one_stackslot_{t}"

        for t in teams:
            for j, req in enumerate(reqs):
                prob += team_skaters_count[t] - req <= M * (1 - a[(t, j)]), f"stack_eq_ub_{t}_{j}"
                prob += req - team_skaters_count[t] <= M * (1 - a[(t, j)]), f"stack_eq_lb_{t}_{j}"

        # all other teams are one-offs (<=1 skater)
        for t in teams:
            prob += team_skaters_count[t] <= 1 + M * pulp.lpSum(a[(t, j)] for j in range(len(reqs))), f"oneoff_cap_{t}"

    # -----------------------------
    # Chalk / Sneaky constraints (on ALL players, including goalie)
    # -----------------------------
    chalk_sum = pulp.lpSum(players[i].is_chalk * selected[i] for i in range(n))
    sneaky_sum = pulp.lpSum(players[i].is_sneaky * selected[i] for i in range(n))
    prob += chalk_sum >= min_chalk, "min_chalk"
    prob += chalk_sum <= max_chalk, "max_chalk"
    prob += sneaky_sum >= min_sneaky, "min_sneaky"

    # -----------------------------
    # Uniqueness vs previous lineups
    # -----------------------------
    if previous_lineups and min_unique_vs_previous > 0:
        max_overlap = LINEUP_SIZE - min_unique_vs_previous
        for li, prev in enumerate(previous_lineups, start=1):
            prob += pulp.lpSum(selected[i] for i in prev) <= max_overlap, f"uniq_prev_{li}"

    # -----------------------------
    # Goalie vs opposing skaters (optional, if opp exists)
    # -----------------------------
    if avoid_goalie_vs_opp_skaters:
        goalie_team = {t: pulp.LpVariable(f"goalie_{t}", cat="Binary") for t in teams}
        for t in teams:
            g_idxs = [i for i in team_idxs[t] if players[i].pos == "G"]
            if g_idxs:
                prob += goalie_team[t] == pulp.lpSum(selected[i] for i in g_idxs), f"goalie_team_eq_{t}"
            else:
                prob += goalie_team[t] == 0, f"goalie_team_eq_{t}"

        for t in teams:
            opp = ""
            for i in team_idxs[t]:
                if players[i].pos == "G" and players[i].opp:
                    opp = players[i].opp
                    break
            if opp and opp in team_skaters_count:
                # if you roster goalie from team t, limit skaters from opponent opp
                prob += team_skaters_count[opp] <= 2 + 8 * (1 - goalie_team[t]), f"goalie_avoid_{t}_vs_{opp}"

    # -----------------------------
    # Objective: proj - own_penalty*own + leverage + noise
    # -----------------------------
    noise = [random.uniform(-randomness, randomness) for _ in range(n)] if randomness > 0 else [0.0] * n
    prob += pulp.lpSum(
        (
            players[i].proj
            - own_penalty * players[i].pred_own
            + leverage_weight * (players[i].proj * (1.0 - players[i].pred_own))
            + noise[i]
        ) * selected[i]
        for i in range(n)
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

    lineup: List[int] = []
    for s in DK_SLOTS:
        pick = None
        for i in elig_by_slot[s]:
            v = x.get((i, s))
            if v is not None and v.value() == 1:
                pick = i
                break
        if pick is None:
            return None
        lineup.append(pick)

    return lineup


# -----------------------------
# OUTPUT
# -----------------------------
def summarize_lineup(players: List[Player], lu: List[int]) -> Dict[str, object]:
    total_salary = sum(players[i].salary for i in lu)
    total_proj = sum(players[i].proj for i in lu)
    total_own = sum(players[i].pred_own for i in lu)
    chalk_ct = sum(players[i].is_chalk for i in lu)
    sneaky_ct = sum(players[i].is_sneaky for i in lu)

    # team counts for full lineup
    team_counts: Dict[str, int] = {}
    for i in lu:
        team_counts[players[i].team] = team_counts.get(players[i].team, 0) + 1

    # stack template SKATERS ONLY
    skater_team_counts: Dict[str, int] = {}
    for slot, idx in zip(DK_SLOTS, lu):
        if slot == "G":
            continue
        t = players[idx].team
        skater_team_counts[t] = skater_team_counts.get(t, 0) + 1
    stack_template = "-".join(str(v) for v in sorted(skater_team_counts.values(), reverse=True))

    return {
        "total_salary": total_salary,
        "total_proj": round(total_proj, 2),
        "total_own": round(total_own, 3),
        "chalk_ct": int(chalk_ct),
        "sneaky_ct": int(sneaky_ct),
        "team_counts": "; ".join([f"{t}:{c}" for t, c in sorted(team_counts.items(), key=lambda x: (-x[1], x[0]))]),
        "stack_template": stack_template,
    }


def lineups_to_df(players: List[Player], built: List[List[int]], mode: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for lu in built:
        row: Dict[str, object] = {"mode": mode}
        for slot, idx in zip(DK_SLOTS, lu):
            p = players[idx]
            row[f"{slot}_name"] = p.name
            row[f"{slot}_team"] = p.team
            row[f"{slot}_pos"] = p.pos
            row[f"{slot}_salary"] = int(p.salary)
            row[f"{slot}_proj"] = float(round(p.proj, 2))

            # NEW: ownership + chalk/sneaky per player
            row[f"{slot}_own"] = float(round(p.pred_own, 4))
            row[f"{slot}_chalk"] = int(p.is_chalk)
            row[f"{slot}_sneaky"] = int(p.is_sneaky)

        meta = summarize_lineup(players, lu)
        row.update(meta)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("total_proj", ascending=False).reset_index(drop=True)


# -----------------------------
# BUILD MANY LINEUPS
# -----------------------------
def generate_nhl_df(
    num_lineups: int = 20,
    min_unique: int = 2,
    min_salary_spend: int = 47000,
    randomness: float = 0.9,
    salary_cap: int = DK_SALARY_CAP,
    csv_url: Optional[str] = None,
    seed: Optional[int] = 7,
    avoid_goalie_vs_opp_skaters: bool = True,
    stack_type: str = "3-2-2",
    chalk_pctile: int = CHALK_PCTILE_DEFAULT,
    sneaky_pctile: int = SNEAKY_PCTILE_DEFAULT,

    # accept BOTH like NBA (so app can pass contest_type)
    mode: str = "20max",
    contest_type: Optional[str] = None,

    verbose: bool = False,
) -> pd.DataFrame:
    if contest_type is not None:
        mode = contest_type
    mode = (mode or "20max").strip().lower()
    if mode not in MODE_PRESETS:
        mode = "20max"

    preset = MODE_PRESETS[mode]

    # apply mode defaults (but allow caller overrides by passing bigger values)
    min_unique = max(min_unique, preset["min_unique"])
    randomness = max(randomness, preset["randomness"])

    own_penalty = preset["own_penalty"]
    leverage_weight = preset["leverage_weight"]

    # mode-based chalk/sneaky requirements
    if mode == "single":
        min_chalk, max_chalk, min_sneaky = 2, 6, 0
    elif mode == "3max":
        min_chalk, max_chalk, min_sneaky = 2, 5, 1
    elif mode == "20max":
        min_chalk, max_chalk, min_sneaky = 2, 4, 2
    else:  # 150max
        min_chalk, max_chalk, min_sneaky = 1, 4, 3

    url = csv_url or CSV_URL_DEFAULT
    df = fetch_csv_to_df(url)
    players = parse_players(df, chalk_pctile=chalk_pctile, sneaky_pctile=sneaky_pctile, verbose=verbose)

    stack_counts = parse_stack_type(stack_type)

    # exposure caps (HARD)
    n = len(players)
    cap = max(1, int(math.floor(float(preset["max_player_exposure"]) * num_lineups)))
    cheap_cap = max(1, int(math.floor(float(preset["cheap_cap_exposure"]) * num_lineups)))

    # cheap SKATERS only (don’t apply cheap cap to goalies)
    skater_salaries = np.array([p.salary for p in players if p.pos != "G"], dtype=float)
    cheap_thr = float(np.percentile(skater_salaries, 35)) if len(skater_salaries) else 0.0
    cheap_idxs = {i for i, p in enumerate(players) if (p.pos != "G" and p.salary <= cheap_thr)}

    counts: Dict[int, int] = {i: 0 for i in range(n)}
    cheap_counts: Dict[int, int] = {i: 0 for i in range(n)}

    built: List[List[int]] = []
    prev_sets: List[Set[int]] = []

    print("Step 1/4: Fetching CSV...", flush=True)
    print(f"CSV loaded: rows={len(df)} cols={len(df.columns)}", flush=True)
    print("Step 2/4: Parsing players...", flush=True)
    print(f"Player pool: {len(players)} players", flush=True)
    print("Step 3/4: Solving lineups (CBC)...", flush=True)

    # relax ladder to prevent “hangs”
    relax = [
        (min_unique, min_salary_spend),
        (max(0, min_unique - 1), min_salary_spend),
        (max(0, min_unique - 1), max(0, min_salary_spend - 800)),
        (max(0, min_unique - 2), max(0, min_salary_spend - 1500)),
    ]

    for uniq_try, minsal_try in relax:
        built = []
        prev_sets = []
        counts = {i: 0 for i in range(n)}
        cheap_counts = {i: 0 for i in range(n)}

        for li in range(num_lineups):
            # hard ban once exposure is hit
            banned = {i for i, ct in counts.items() if ct >= cap}
            cheap_banned = {i for i in cheap_idxs if cheap_counts.get(i, 0) >= cheap_cap}

            lineup = None
            for attempt in range(MAX_SOLVES_PER_LINEUP):
                rand_i = (randomness
                          + 0.10 * (li / max(1, num_lineups - 1))
                          + 0.12 * attempt)

                lineup = optimize_one_nhl(
                    players=players,
                    salary_cap=salary_cap,
                    min_salary_spend=minsal_try,
                    min_unique_vs_previous=uniq_try,
                    previous_lineups=prev_sets,
                    randomness=rand_i,
                    seed=None if seed is None else seed + li * 10 + attempt,

                    stack_counts=stack_counts,
                    avoid_goalie_vs_opp_skaters=avoid_goalie_vs_opp_skaters,

                    own_penalty=own_penalty,
                    leverage_weight=leverage_weight,
                    min_chalk=min_chalk,
                    max_chalk=max_chalk,
                    min_sneaky=min_sneaky,

                    banned=banned,
                    cheap_banned=cheap_banned,
                )
                if lineup is not None:
                    break

            if lineup is None:
                break

            built.append(lineup)
            s = set(lineup)
            prev_sets.append(s)

            for idx in s:
                counts[idx] += 1
                if idx in cheap_idxs:
                    cheap_counts[idx] += 1

        if len(built) == num_lineups:
            break

    if not built:
        raise RuntimeError(
            f"No feasible NHL lineups generated for stack_type={stack_type}.\n"
            "Try loosening:\n"
            "  - lower min_salary_spend (e.g. 45000)\n"
            "  - set min_unique to 0 or 1\n"
            "  - choose a different stack_type\n"
            "  - set avoid_goalie_vs_opp_skaters=False\n"
        )

    print("Step 4/4: Done.", flush=True)
    return lineups_to_df(players, built, mode=mode)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_url", type=str, default=CSV_URL_DEFAULT)
    ap.add_argument("--num_lineups", type=int, default=20)
    ap.add_argument("--salary_cap", type=int, default=DK_SALARY_CAP)
    ap.add_argument("--min_salary_spend", type=int, default=47000)
    ap.add_argument("--min_unique", type=int, default=2)
    ap.add_argument("--randomness", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stack_type", type=str, default="3-2-2")
    ap.add_argument("--mode", type=str, default="20max", help="single,3max,20max,150max")
    ap.add_argument("--no_goalie_opp_rule", action="store_true")
    ap.add_argument("--cbc_msg", action="store_true")
    args = ap.parse_args()

    global CBC_MSG
    if args.cbc_msg:
        CBC_MSG = True

    out_df = generate_nhl_df(
        num_lineups=args.num_lineups,
        min_unique=args.min_unique,
        min_salary_spend=args.min_salary_spend,
        randomness=args.randomness,
        salary_cap=args.salary_cap,
        csv_url=args.csv_url,
        seed=args.seed,
        stack_type=args.stack_type,
        avoid_goalie_vs_opp_skaters=(not args.no_goalie_opp_rule),
        mode=args.mode,
        verbose=True,
    )

    print("\nTop 10 lineups:\n", flush=True)
    print(out_df.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
