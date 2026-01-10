"""
NHL DraftKings Classic Optimizer — nhl_optimizer.py
VERSION: 2026-01-10-STABLE
"""

import argparse
import hashlib
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pulp
import requests

# -----------------------------
# VERSION / DEBUG MARKERS
# -----------------------------
VERSION_TAG = "2026-01-10-STABLE"
print(f"=== LOADED nhl_optimizer.py VERSION: {VERSION_TAG} ===", flush=True)

# -----------------------------
# DEFAULTS / ENV
# -----------------------------
CSV_URL_DEFAULT = os.environ.get(
    "NHL_CSV_URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vTJOqq7cId7F4U7QmfyXR8yMb6wc88PV7c_QHbCHb8a0f-tezW0gvLadFx-gSkNCw/pub?gid=562844939&single=true&output=csv",
)

DK_SALARY_CAP = 50000
DK_SLOTS = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]
LINEUP_SIZE = len(DK_SLOTS)
SKATER_SLOTS = [s for s in DK_SLOTS if s != "G"]

# SPEED KNOBS - These prevent the "Hanging" issue
CBC_TIME_LIMIT_SEC = 15.0  # Cap solve time per lineup
CBC_GAP_REL = 0.05        # 5% gap (Allows solver to finish once it's 'close enough')
CBC_THREADS = 4           # Parallel processing
CBC_MSG = False

# -----------------------------
# HARD BANS
# -----------------------------
BANNED_NAMES = [""]

def normalize_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

BANNED_NORM = {normalize_name(x) for x in BANNED_NAMES}

@dataclass(frozen=True)
class Player:
    name: str
    name_norm: str
    team: str
    pos: str
    salary: int
    proj: float
    opp: str

# -----------------------------
# PARSING HELPERS
# -----------------------------
def clean_text(x) -> str:
    s = str(x).replace("\t", " ")
    s = re.sub(r"&nbsp;?", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return "" if s.lower() in ("nan", "none", "") else s

def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    key_map = {c: norm_key(c) for c in df.columns}
    for cand in candidates:
        ck = norm_key(cand)
        for orig, k in key_map.items():
            if k == ck: return orig
    return None

def parse_float(x, default=0.0) -> float:
    try:
        s = str(x).replace(",", "").replace("$", "").strip()
        m = re.search(r"[-+]?\d*\.?\d+", s)
        return float(m.group(0)) if m else default
    except: return default

def parse_salary(x, default=0) -> int:
    v = parse_float(x)
    return int(v * 1000) if 0 < v <= 100 else int(v)

def normalize_pos(p: str) -> str:
    p = clean_text(p).upper()
    if p in ("LW", "RW", "W"): return "W"
    if p.startswith("C"): return "C"
    if p.startswith("D"): return "D"
    if p.startswith("G"): return "G"
    return p

# -----------------------------
# CORE LOGIC
# -----------------------------
def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # Normalize all column names in the actual CSV
    actual_cols = {norm_key(c): c for c in df.columns}

    # Try direct matches first
    for cand in candidates:
        ck = norm_key(cand)
        if ck in actual_cols:
            return actual_cols[ck]

    # Try partial matches (e.g., 'dk_fp_proj' contains 'proj')
    for cand in candidates:
        ck = norm_key(cand)
        for k_norm, original_name in actual_cols.items():
            if ck in k_norm:
                return original_name
    return None


def parse_players(df: pd.DataFrame, verbose: bool = True) -> List[Player]:
    # 1. Identify Columns
    pos_col = pick_col(df, ["pos", "position", "dk_position", "roster_position"])
    name_col = pick_col(df, ["name", "player", "player_name", "nickname"])
    team_col = pick_col(df, ["team", "teamabbr", "team_abbrev", "tm"])
    sal_col = pick_col(df, ["salary", "sal", "dk_salary", "cost"])
    proj_col = pick_col(df, ["proj", "projection", "fpts", "fp", "points", "projected"])
    opp_col = pick_col(df, ["opp", "opponent", "vs"])

    # 2. CRITICAL ERROR CHECK: If any required col is None, stop and print columns
    missing = []
    if not pos_col: missing.append("Position")
    if not name_col: missing.append("Name")
    if not team_col: missing.append("Team")
    if not sal_col: missing.append("Salary")
    if not proj_col: missing.append("Projection")

    if missing:
        available = ", ".join(df.columns.tolist())
        raise ValueError(f"COULD NOT FIND COLUMNS: {missing}. \nAvailable columns in your CSV are: [{available}]")

    # 3. Process Data
    tmp = df.copy()
    tmp["name"] = tmp[name_col].apply(clean_text)
    tmp["team"] = tmp[team_col].apply(clean_text)
    tmp["pos"] = tmp[pos_col].apply(normalize_pos)
    tmp["salary"] = tmp[sal_col].apply(parse_salary)
    tmp["proj"] = tmp[proj_col].apply(parse_float)
    tmp["opp"] = tmp[opp_col].apply(clean_text) if opp_col else ""

    # Rest of the filtering...
    tmp = tmp[(tmp["salary"] > 0) & (tmp["proj"] > 0)]
    tmp = tmp[tmp["pos"].isin(["C", "W", "D", "G"])]

    tmp["name_norm"] = tmp["name"].apply(normalize_name)
    tmp = tmp[~tmp["name_norm"].isin(BANNED_NORM)]

    players = []
    for _, r in tmp.iterrows():
        players.append(Player(
            name=r["name"], name_norm=r["name_norm"], team=r["team"],
            pos=r["pos"], salary=r["salary"], proj=r["proj"], opp=r["opp"]
        ))

    if not players:
        raise ValueError("No players left after filtering. Check if salary/proj are > 0.")

    return players

def eligible_slots(pos: str) -> Set[str]:
    if pos == "C": return {"C1", "C2", "UTIL"}
    if pos == "W": return {"W1", "W2", "W3", "UTIL"}
    if pos == "D": return {"D1", "D2", "UTIL"}
    if pos == "G": return {"G"}
    return set()

def optimize_one_nhl(
    players: List[Player],
    salary_cap: int,
    min_salary_spend: int,
    min_unique_vs_previous: int,
    previous_lineups: List[Set[int]],
    randomness: float,
    seed: Optional[int],
    stack_counts: List[int],
    avoid_goalie_vs_opp_skaters: bool = True,
) -> Optional[List[int]]:

    if seed is not None: random.seed(seed)
    n = len(players)
    teams = sorted(set(p.team for p in players))

    prob = pulp.LpProblem("NHL", pulp.LpMaximize)

    # Decision Vars: x[player_index, slot_name]
    x = {}
    for i, p in enumerate(players):
        for s in eligible_slots(p.pos):
            x[(i, s)] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")

    # 1. Fill every slot
    for s in DK_SLOTS:
        prob += pulp.lpSum(x[(i, s)] for i in range(n) if (i, s) in x) == 1

    # 2. Player at most once
    selected = {}
    for i in range(n):
        selected[i] = pulp.lpSum(x[(i, s)] for s in DK_SLOTS if (i, s) in x)
        prob += selected[i] <= 1

    # 3. Salary
    prob += pulp.lpSum(players[i].salary * selected[i] for i in range(n)) <= salary_cap
    prob += pulp.lpSum(players[i].salary * selected[i] for i in range(n)) >= min_salary_spend

    # 4. Stacking Logic (Skaters Only)
    team_skaters = {t: pulp.lpSum(selected[i] for i in range(n) if players[i].team == t and players[i].pos != "G") for t in teams}

    if stack_counts:
        reqs = [r for r in stack_counts if r >= 2]
        M = 8
        a = {(t, j): pulp.LpVariable(f"a_{t}_{j}", cat="Binary") for t in teams for j in range(len(reqs))}

        for j in range(len(reqs)):
            prob += pulp.lpSum(a[(t, j)] for t in teams) == 1
        for t in teams:
            prob += pulp.lpSum(a[(t, j)] for j in range(len(reqs))) <= 1
            for j, req in enumerate(reqs):
                prob += team_skaters[t] >= req * a[(t, j)]

    # 5. Uniqueness
    for prev in previous_lineups:
        prob += pulp.lpSum(selected[i] for i in prev) <= (LINEUP_SIZE - min_unique_vs_previous)

    # 6. Objective
    noise = [random.uniform(-randomness, randomness) for _ in range(n)]
    prob += pulp.lpSum((players[i].proj + noise[i]) * selected[i] for i in range(n))

    # SOLVER - The secret sauce to stop the hanging
    solver = pulp.PULP_CBC_CMD(
        msg=CBC_MSG,
        timeLimit=CBC_TIME_LIMIT_SEC,
        gapRel=CBC_GAP_REL,
        threads=CBC_THREADS,
        options=['presolve on', 'heuristics on']
    )

    if prob.solve(solver) != 1: return None

    return [i for i in range(n) if pulp.value(selected[i]) > 0.5]

# -----------------------------
# PUBLIC API
# -----------------------------
def generate_nhl_df(
        num_lineups: int = 20,
        min_unique: int = 2,
        min_salary_spend: int = 47000,
        randomness: float = 0.9,
        salary_cap: int = DK_SALARY_CAP,
        csv_url: Optional[str] = None,
        seed: Optional[int] = 7,
        stack_type: str = "3-2",
        **kwargs
) -> pd.DataFrame:
    """
    Generates optimized NHL lineups and formats them for the Flask results.html template.
    """
    url = csv_url or CSV_URL_DEFAULT
    print(f"Step 1/4: Fetching CSV from {url[:50]}...", flush=True)

    # Use the established parsing logic to get the player pool
    df = pd.read_csv(url)
    players = parse_players(df)

    # Convert "3-2-2" or "3-2" string into list of integers [3, 2, 2]
    stack_counts = [int(x) for x in stack_type.replace("-", ",").split(",") if x.strip()]

    built = []
    prev_sets = []

    # Define the exact slot order expected by results.html and DraftKings
    slot_names = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]

    print(f"Step 3/4: Solving {num_lineups} lineups...", flush=True)
    for li in range(num_lineups):
        # Pass parameters to the solver
        lu_indices = optimize_one_nhl(
            players=players,
            salary_cap=salary_cap,
            min_salary_spend=min_salary_spend,
            min_unique_vs_previous=min_unique,
            previous_lineups=prev_sets,
            randomness=randomness,
            seed=(seed + li) if seed is not None else None,
            stack_counts=stack_counts
        )

        if lu_indices:
            built.append(lu_indices)
            prev_sets.append(set(lu_indices))
            print(f"  Lineup {li + 1} found.", flush=True)
        else:
            print(f"  Lineup {li + 1} could not be solved. Stopping.", flush=True)
            break

    # Format for results.html - Mapping solver output to template columns
    rows = []
    for lu in built:
        lineup_dict = {
            "total_salary": sum(players[i].salary for i in lu),
            "total_proj": round(sum(players[i].proj for i in lu), 2),
            "stack_template": stack_type  # Helps frontend show stack info
        }

        # lu is the list of indices in the order of slot_names
        for idx, p_idx in enumerate(lu):
            if idx >= len(slot_names):
                break

            p = players[p_idx]
            slot = slot_names[idx]  # e.g., "C1", "W2"

            # These keys MUST match what your results.html loops through
            lineup_dict[f"{slot}_name"] = p.name
            lineup_dict[f"{slot}_team"] = p.team
            lineup_dict[f"{slot}_pos"] = p.pos
            lineup_dict[f"{slot}_salary"] = p.salary
            lineup_dict[f"{slot}_proj"] = round(p.proj, 2)

        rows.append(lineup_dict)

    print("Step 4/4: Formatting complete.", flush=True)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # CLI fallback
    print(generate_nhl_df(num_lineups=2).head())