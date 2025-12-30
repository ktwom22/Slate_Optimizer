"""
NHL DraftKings Classic Optimizer (ONE FILE) — nhl_optimizer.py

Uses YOUR CSV (default env NHL_CSV_URL):
https://docs.google.com/spreadsheets/d/e/2PACX-1vTJOqq7cId7F4U7QmfyXR8yMb6wc88PV7c_QHbCHb8a0f-tezW0gvLadFx-gSkNCw/pub?gid=562844939&single=true&output=csv

Roster:
C, C, W, W, W, D, D, G, UTIL  (Salary cap 50k)

Key features:
✅ Exact stack type on SKATERS ONLY (goalie excluded)
✅ Uniqueness across lineups (min_unique)
✅ Randomness for diversity
✅ HARD BAN by normalized name (fixes "Auston still showing up" even with hidden chars)
✅ Optional OUT/status filtering if your CSV has those columns
✅ Debug prints prove whether the CSV still contains him and which file Flask is using

Install:
  pip install pandas requests pulp

Run:
  python -u nhl_optimizer.py --num_lineups 20 --stack_type 3-2-2
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
VERSION_TAG = "2025-12-30-NHL-A"
print(f"=== LOADED nhl_optimizer.py VERSION: {VERSION_TAG} ===", flush=True)
print("FILE:", __file__, flush=True)

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

# speed knobs
CBC_TIME_LIMIT_SEC = float(os.environ.get("NHL_CBC_TIME_LIMIT", "2.0"))
CBC_GAP_REL = float(os.environ.get("NHL_CBC_GAP_REL", "0.08"))
CBC_THREADS = int(os.environ.get("NHL_CBC_THREADS", "1"))
CBC_MSG = bool(int(os.environ.get("NHL_CBC_MSG", "0")))
MAX_SOLVES_PER_LINEUP = int(os.environ.get("NHL_MAX_SOLVES_PER_LU", "5"))

# -----------------------------
# HARD BANS (normalize names)
# -----------------------------
BANNED_NAMES = [
    "Auston Matthews",
    # add more here if you want:
    # "Connor McDavid",
]

def normalize_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("\u00A0", " ")  # NBSP
    s = re.sub(r"[^\w\s]", " ", s.lower())  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

BANNED_NORM = {normalize_name(x) for x in BANNED_NAMES}

# -----------------------------
# DATA MODEL
# -----------------------------
@dataclass(frozen=True)
class Player:
    name: str
    name_norm: str
    team: str
    pos: str        # C / W / D / G
    salary: int
    proj: float
    opp: str        # optional


# -----------------------------
# CSV
# -----------------------------
def fetch_csv_to_df(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers={"Cache-Control": "no-cache"})
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
    key_map = {c: norm_key(c) for c in df.columns}
    inv: Dict[str, List[str]] = {}
    for orig, k in key_map.items():
        inv.setdefault(k, []).append(orig)
    for cand in candidates:
        ck = norm_key(cand)
        if ck in inv:
            return inv[ck][0]
    return None

_num_re = re.compile(r"[-+]?\d*\.?\d+")

def parse_float(x, default=0.0) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    s = str(x).replace(",", "").replace("$", "").strip()
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

def detect_out_col(df: pd.DataFrame) -> Optional[str]:
    """
    Tries to find any column that looks like injury/status/out.
    We'll treat values containing: out, o, inactive, scratch, dnp as OUT.
    """
    for c in df.columns:
        k = norm_key(c)
        if any(x in k for x in ["out", "status", "inj", "injury", "inactive", "scratch", "dnp"]):
            return c
    return None

def is_out_value(v) -> bool:
    s = clean_text(v).lower()
    if not s:
        return False
    # common markers
    return any(tok in s for tok in ["out", "inactive", "scratch", "dnp", "ir", "o"])

def parse_players(df: pd.DataFrame, verbose: bool = True) -> List[Player]:
    pos_col  = pick_col(df, ["pos", "position", "dk_position", "roster_position"])
    name_col = pick_col(df, ["name", "player", "player_name"])
    team_col = pick_col(df, ["team", "teamabbr", "team_abbrev", "tm"])
    sal_col  = pick_col(df, ["salary", "sal", "dk_salary", "cost"])
    proj_col = find_proj_col(df)
    opp_col  = pick_col(df, ["opp", "opponent", "vs", "opponent_team"])
    out_col  = detect_out_col(df)

    missing = []
    if not pos_col:  missing.append("pos")
    if not name_col: missing.append("name")
    if not team_col: missing.append("team")
    if not sal_col:  missing.append("salary")
    if not proj_col: missing.append("proj")
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # DEBUG signature so you can see if the CSV changed
    sig = hashlib.md5((str(list(df.columns)) + f"|{len(df)}").encode("utf-8")).hexdigest()[:10]
    print(f"[DEBUG] CSV rows={len(df)} cols={len(df.columns)} sig={sig}", flush=True)

    tmp = df.copy()
    tmp["name"] = tmp[name_col].apply(clean_text)
    tmp["name_norm"] = tmp["name"].apply(normalize_name)
    tmp["team"] = tmp[team_col].apply(clean_text)
    tmp["pos"] = tmp[pos_col].apply(normalize_pos)
    tmp["salary"] = tmp[sal_col].apply(lambda v: parse_salary(v, 0))
    tmp["proj"] = tmp[proj_col].apply(lambda v: parse_float(v, 0.0))
    tmp["opp"] = tmp[opp_col].apply(clean_text) if opp_col else ""

    # DEBUG: find Auston BEFORE any filtering
    auston_before = tmp[tmp["name_norm"] == normalize_name("Auston Matthews")]
    print(f"[DEBUG] Auston rows BEFORE filters: {len(auston_before)}", flush=True)
    if len(auston_before) > 0:
        cols_show = ["name", "team", "pos", "salary", "proj"]
        if out_col:
            cols_show.append(out_col)
        print(auston_before[cols_show].head(5).to_string(index=False), flush=True)

    # basic cleanup
    tmp = tmp[(tmp["name"] != "") & (tmp["team"] != "") & (tmp["pos"] != "")]
    tmp = tmp[tmp["pos"].isin(["C", "W", "D", "G"])]
    tmp = tmp[(tmp["salary"] > 0) & (tmp["proj"] > 0)]

    # OUT/status filtering if column exists
    if out_col:
        mask_out = tmp[out_col].apply(is_out_value)
        out_ct = int(mask_out.sum())
        if out_ct > 0:
            print(f"[DEBUG] Filtering OUT via column '{out_col}': removing {out_ct} rows", flush=True)
        tmp = tmp[~mask_out]

    # HARD BAN by name normalization
    ban_ct = int(tmp["name_norm"].isin(BANNED_NORM).sum())
    if ban_ct > 0:
        print(f"[DEBUG] HARD BAN removing {ban_ct} rows by name match", flush=True)
    tmp = tmp[~tmp["name_norm"].isin(BANNED_NORM)]

    # DEBUG: find Auston AFTER filters
    auston_after = tmp[tmp["name_norm"] == normalize_name("Auston Matthews")]
    print(f"[DEBUG] Auston rows AFTER filters: {len(auston_after)}", flush=True)

    if tmp.empty:
        raise ValueError("No valid NHL players after cleanup/bans/out filtering.")

    players: List[Player] = []
    for _, r in tmp.iterrows():
        players.append(Player(
            name=str(r["name"]),
            name_norm=str(r["name_norm"]),
            team=str(r["team"]),
            pos=str(r["pos"]),
            salary=int(r["salary"]),
            proj=float(r["proj"]),
            opp=str(r["opp"]) if isinstance(r["opp"], str) else "",
        ))

    if verbose:
        print(
            f"Detected columns -> POS:{pos_col} | NAME:{name_col} | TEAM:{team_col} | SALARY:{sal_col} | PROJ:{proj_col} | OPP:{opp_col or 'None'} | OUTCOL:{out_col or 'None'}",
            flush=True
        )
        for p in players[:8]:
            print(f"  {p.pos} {p.name:<24} {p.team:<4} sal={p.salary:<5} proj={p.proj:<5.2f} opp={p.opp}", flush=True)

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
    parts = [p for p in s.split("-") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(int(p.strip()))
        except Exception:
            raise ValueError(f"Invalid stack_type: {stack_type}. Use like '3-2-2'.")
    return [x for x in out if x > 0]


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
    stack_counts: List[int],                    # exact stacks on SKATERS ONLY
    avoid_goalie_vs_opp_skaters: bool = True,   # if opp exists
) -> Optional[List[int]]:

    if seed is not None:
        random.seed(seed)

    n = len(players)
    teams = sorted(set(p.team for p in players))
    if len(teams) < 2:
        return None

    team_idxs: Dict[str, List[int]] = {t: [] for t in teams}
    for i, p in enumerate(players):
        team_idxs[p.team].append(i)

    # decision vars x[(i,slot)]
    x: Dict[Tuple[int, str], pulp.LpVariable] = {}
    elig_by_slot: Dict[str, List[int]] = {s: [] for s in DK_SLOTS}
    for i, p in enumerate(players):
        for s in eligible_slots(p.pos):
            x[(i, s)] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")
            elig_by_slot[s].append(i)

    # must be able to fill each slot
    for s in DK_SLOTS:
        if not elig_by_slot[s]:
            return None

    prob = pulp.LpProblem("NHL_DK_CLASSIC", pulp.LpMaximize)

    # fill each slot exactly once
    for s in DK_SLOTS:
        prob += pulp.lpSum(x[(i, s)] for i in elig_by_slot[s]) == 1, f"fill_{s}"

    # each player at most once
    selected = [pulp.lpSum(x.get((i, s), 0) for s in DK_SLOTS) for i in range(n)]
    for i in range(n):
        prob += selected[i] <= 1, f"one_slot_{i}"

    # salary
    total_salary = pulp.lpSum(players[i].salary * selected[i] for i in range(n))
    prob += total_salary <= salary_cap, "salary_cap"
    prob += total_salary >= min_salary_spend, "min_salary_spend"

    # -----------------------------
    # EXACT STACK (SKATERS ONLY) — goalie excluded
    # -----------------------------
    skater_selected = {
        i: pulp.lpSum(x.get((i, s), 0) for s in SKATER_SLOTS)
        for i in range(n)
    }
    prob += pulp.lpSum(skater_selected[i] for i in range(n)) == 8, "skaters_eq_8"

    team_skaters_count = {
        t: pulp.lpSum(skater_selected[i] for i in team_idxs[t])
        for t in teams
    }

    if stack_counts:
        reqs = [r for r in stack_counts if r >= 2]
        if not reqs:
            raise ValueError("stack_type must include at least one 2+ stack (e.g., 3-2-2).")
        if sum(reqs) > 8:
            raise ValueError(f"stack_type {stack_counts} impossible: sum(reqs) > 8 skaters.")

        M = 8
        a = {(t, j): pulp.LpVariable(f"a_{t}_{j}", cat="Binary")
             for t in teams for j in range(len(reqs))}

        for j, req in enumerate(reqs):
            prob += pulp.lpSum(a[(t, j)] for t in teams) == 1, f"assign_stack_{j}_{req}"

        for t in teams:
            prob += pulp.lpSum(a[(t, j)] for j in range(len(reqs))) <= 1, f"team_one_stackslot_{t}"

        for t in teams:
            for j, req in enumerate(reqs):
                prob += team_skaters_count[t] - req <= M * (1 - a[(t, j)]), f"stack_eq_ub_{t}_{j}"
                prob += req - team_skaters_count[t] <= M * (1 - a[(t, j)]), f"stack_eq_lb_{t}_{j}"

        for t in teams:
            prob += team_skaters_count[t] <= 1 + M * pulp.lpSum(a[(t, j)] for j in range(len(reqs))), f"oneoff_cap_{t}"

    # -----------------------------
    # Uniqueness vs previous lineups
    # -----------------------------
    if previous_lineups and min_unique_vs_previous > 0:
        max_overlap = LINEUP_SIZE - min_unique_vs_previous
        for li, prev in enumerate(previous_lineups, start=1):
            prob += pulp.lpSum(selected[i] for i in prev) <= max_overlap, f"uniq_prev_{li}"

    # -----------------------------
    # Goalie vs opposing skaters avoidance (if opp exists)
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
            # try to find opponent tag for team t
            opp = ""
            for i in team_idxs[t]:
                if players[i].opp:
                    opp = players[i].opp
                    break
            if opp and opp in team_skaters_count:
                prob += team_skaters_count[opp] <= 2 + 8 * (1 - goalie_team[t]), f"goalie_avoids_opp_{t}_vs_{opp}"

    # -----------------------------
    # Absolute last line of defense: never allow banned names
    # -----------------------------
    for i, p in enumerate(players):
        if p.name_norm in BANNED_NORM:
            prob += selected[i] == 0, f"hard_ban_{i}"

    # -----------------------------
    # objective with noise
    # -----------------------------
    noise = [random.uniform(-randomness, randomness) for _ in range(n)] if randomness > 0 else [0.0] * n
    prob += pulp.lpSum((players[i].proj + noise[i]) * selected[i] for i in range(n)), "objective"

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

    team_counts: Dict[str, int] = {}
    for i in lu:
        t = players[i].team
        team_counts[t] = team_counts.get(t, 0) + 1

    team_counts_sorted = dict(sorted(team_counts.items(), key=lambda x: (-x[1], x[0])))
    stack_template = "-".join(str(v) for v in sorted(team_counts_sorted.values(), reverse=True))

    return {
        "total_salary": total_salary,
        "total_proj": round(total_proj, 2),
        "team_counts": team_counts_sorted,
        "stack_template": stack_template,
    }

def lineups_to_df(players: List[Player], built: List[List[int]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for lu in built:
        row: Dict[str, object] = {}
        for slot, idx in zip(DK_SLOTS, lu):
            p = players[idx]
            row[f"{slot}_name"] = p.name
            row[f"{slot}_team"] = p.team
            row[f"{slot}_pos"] = p.pos
            row[f"{slot}_salary"] = int(p.salary)
            row[f"{slot}_proj"] = float(round(p.proj, 2))

        meta = summarize_lineup(players, lu)
        row["total_salary"] = meta["total_salary"]
        row["total_proj"] = meta["total_proj"]
        row["team_counts"] = "; ".join([f"{t}:{c}" for t, c in meta["team_counts"].items()])
        row["stack_template"] = meta["stack_template"]
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
    verbose: bool = False,
) -> pd.DataFrame:

    url = csv_url or CSV_URL_DEFAULT
    print("Step 1/4: Fetching CSV...", flush=True)
    df = fetch_csv_to_df(url)
    print(f"CSV loaded: rows={len(df)} cols={len(df.columns)}", flush=True)

    print("Step 2/4: Parsing players...", flush=True)
    players = parse_players(df, verbose=verbose)
    print(f"Player pool: {len(players)} players", flush=True)

    stack_counts = parse_stack_type(stack_type)

    print("Step 3/4: Solving lineups (CBC)...", flush=True)
    built: List[List[int]] = []
    prev_sets: List[Set[int]] = []

    for li in range(num_lineups):
        lineup = None
        solves = 0

        while solves < MAX_SOLVES_PER_LINEUP and lineup is None:
            solves += 1
            lineup = optimize_one_nhl(
                players=players,
                salary_cap=salary_cap,
                min_salary_spend=min_salary_spend,
                min_unique_vs_previous=min_unique,
                previous_lineups=prev_sets,
                randomness=randomness + (0.12 * (solves - 1)) + (0.05 * (li / max(1, num_lineups - 1))),
                seed=None if seed is None else seed + li * 10 + solves,
                stack_counts=stack_counts,
                avoid_goalie_vs_opp_skaters=avoid_goalie_vs_opp_skaters,
            )

        if lineup is None:
            raise RuntimeError(
                f"No feasible NHL lineups for stack_type={stack_type}.\n"
                "Try loosening:\n"
                "  - lower min_salary_spend (e.g. 45000)\n"
                "  - set min_unique to 0 or 1\n"
                "  - increase randomness (1.2)\n"
                "  - choose a different stack_type\n"
                "  - set avoid_goalie_vs_opp_skaters=False\n"
            )

        # Extra safety check (should never trigger now)
        for idx in lineup:
            if players[idx].name_norm in BANNED_NORM:
                raise RuntimeError(f"BANNED PLAYER STILL SELECTED: {players[idx].name}")

        built.append(lineup)
        prev_sets.append(set(lineup))

    print("Step 4/4: Done.", flush=True)
    return lineups_to_df(players, built)


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
        verbose=True,
    )

    print("\nTop 10 lineups:\n", flush=True)
    print(out_df.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
