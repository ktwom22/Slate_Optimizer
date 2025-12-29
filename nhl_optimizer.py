"""
NHL DraftKings Classic Optimizer (ONE FILE) — nhl_optimizer.py

Uses YOUR CSV:
https://docs.google.com/spreadsheets/d/e/2PACX-1vTJOqq7cId7F4U7QmfyXR8yMb6wc88PV7c_QHbCHb8a0f-tezW0gvLadFx-gSkNCw/pub?gid=562844939&single=true&output=csv

DK NHL Classic roster:
C, C, W, W, W, D, D, G, UTIL  (Salary cap 50k)

Built around your “Perfect Lineups” insights:
- Strong preference for TEAM stacks:
  - Default: Force at least one 3-player team stack
  - Plus: Force a second 2-player stack (different team) when feasible (3-2)
  - Falls back to 2-2 / 2-2-2 / single 3-stack with feasibility ladder
- Position-stack preference on stacked team:
  - Bias toward C+D and G+RW style correlations by bonus terms
- Uniqueness across lineups (min_unique)
- Randomness to diversify
- Optional: cap per-team exposure across lineups
- Optional: limit goalies vs opposing skaters (if OPP exists)

Install:
  pip install pandas requests pulp

Run:
  python -u nhl_optimizer.py --num_lineups 20

Exposes:
  generate_nhl_df(...) -> pandas DataFrame (DK-style output columns)
"""

import argparse
import os
import random
import re
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple

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

# speed knobs
CBC_TIME_LIMIT_SEC = float(os.environ.get("NHL_CBC_TIME_LIMIT", "2.0"))
CBC_GAP_REL = float(os.environ.get("NHL_CBC_GAP_REL", "0.08"))
CBC_THREADS = int(os.environ.get("NHL_CBC_THREADS", "1"))
CBC_MSG = bool(int(os.environ.get("NHL_CBC_MSG", "0")))

MAX_SOLVES_PER_LINEUP = int(os.environ.get("NHL_MAX_SOLVES_PER_LU", "5"))

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
    opp: str        # optional


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
    # common NHL labels: C, W, LW, RW, D, G
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


def parse_players(df: pd.DataFrame, verbose: bool = True) -> List[Player]:
    pos_col = pick_col(df, ["pos", "position", "dk_position", "roster_position"])
    name_col = pick_col(df, ["name", "player", "player_name"])
    team_col = pick_col(df, ["team", "teamabbr", "team_abbrev", "tm"])
    sal_col = pick_col(df, ["salary", "sal", "dk_salary", "cost"])
    proj_col = find_proj_col(df)
    opp_col = pick_col(df, ["opp", "opponent", "vs", "opponent_team"])

    missing = []
    if not pos_col: missing.append("pos")
    if not name_col: missing.append("name")
    if not team_col: missing.append("team")
    if not sal_col: missing.append("salary")
    if not proj_col: missing.append("proj")
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    df = df.copy()
    df["name"] = df[name_col].apply(clean_text)
    df["team"] = df[team_col].apply(clean_text)
    df["pos"] = df[pos_col].apply(normalize_pos)
    df["salary"] = df[sal_col].apply(lambda v: parse_salary(v, 0))
    df["proj"] = df[proj_col].apply(lambda v: parse_float(v, 0.0))
    df["opp"] = df[opp_col].apply(clean_text) if opp_col else ""

    df = df[(df["name"] != "") & (df["team"] != "") & (df["pos"] != "")]
    df = df[df["pos"].isin(["C", "W", "D", "G"])]
    df = df[(df["salary"] > 0) & (df["proj"] > 0)]
    if df.empty:
        raise ValueError("No valid NHL players after cleanup.")

    players: List[Player] = []
    for _, r in df.iterrows():
        players.append(Player(
            name=str(r["name"]),
            team=str(r["team"]),
            pos=str(r["pos"]),
            salary=int(r["salary"]),
            proj=float(r["proj"]),
            opp=str(r["opp"]) if isinstance(r["opp"], str) else "",
        ))

    if verbose:
        print(
            f"Detected columns -> POS:{pos_col} | NAME:{name_col} | TEAM:{team_col} | SALARY:{sal_col} | PROJ:{proj_col} | OPP:{opp_col or 'None'}",
            flush=True
        )
        print("Sample players:", flush=True)
        for p in players[:10]:
            print(f"  {p.pos:<1} {p.name:<25} {p.team:<4} sal={p.salary:<5} proj={p.proj:<5.2f} opp={p.opp}", flush=True)

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

    # stacking targets (feasibility ladder will relax)
    require_primary_stack: int,     # 3 => require a 3-stack from one team, 2 => require a 2-stack, 0 => none
    require_secondary_stack: int,   # 2 => require second team 2-stack (if primary is 3), 0 => none

    # optional opp rules
    avoid_goalie_vs_opp_skaters: bool,  # if opp exists

    # correlation bonus weights (not hard rules)
    bonus_c_d: float,
    bonus_g_rw: float,
) -> Optional[List[int]]:
    if seed is not None:
        random.seed(seed)

    n = len(players)
    teams = sorted(set(p.team for p in players))
    team_idxs: Dict[str, List[int]] = {t: [] for t in teams}
    for i, p in enumerate(players):
        team_idxs[p.team].append(i)

    # decision vars x[(i,slot)] only if eligible
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

    # team counts (skaters only for stacks, goalies excluded from stacking counts)
    team_skaters_count: Dict[str, pulp.LpAffineExpression] = {}
    for t in teams:
        idxs = [i for i in team_idxs[t] if players[i].pos != "G"]
        team_skaters_count[t] = pulp.lpSum(selected[i] for i in idxs) if idxs else 0

    # Primary + secondary stack enforcement
    if require_primary_stack > 0:
        prim = {t: pulp.LpVariable(f"prim_{t}", cat="Binary") for t in teams}
        prob += pulp.lpSum(prim[t] for t in teams) == 1, "one_primary_team"
        M = 9
        for t in teams:
            prob += team_skaters_count[t] >= require_primary_stack * prim[t], f"prim_lb_{t}"
            prob += team_skaters_count[t] <= (require_primary_stack + M * (1 - prim[t])), f"prim_ub_{t}"

        if require_secondary_stack > 0:
            sec = {t: pulp.LpVariable(f"sec_{t}", cat="Binary") for t in teams}
            prob += pulp.lpSum(sec[t] for t in teams) == 1, "one_secondary_team"
            for t in teams:
                prob += sec[t] <= 1 - prim[t], f"sec_not_primary_{t}"
                prob += team_skaters_count[t] >= require_secondary_stack * sec[t], f"sec_lb_{t}"
                prob += team_skaters_count[t] <= (require_secondary_stack + M * (1 - sec[t])), f"sec_ub_{t}"

    # goalie vs opposing skaters avoidance (if OPP exists)
    if avoid_goalie_vs_opp_skaters:
        # pick goalie team binary
        goalie_team = {t: pulp.LpVariable(f"goalie_{t}", cat="Binary") for t in teams}
        for t in teams:
            g_idxs = [i for i in team_idxs[t] if players[i].pos == "G"]
            if g_idxs:
                prob += goalie_team[t] == pulp.lpSum(selected[i] for i in g_idxs), f"goalie_team_eq_{t}"
            else:
                prob += goalie_team[t] == 0, f"goalie_team_eq_{t}"

        # if goalie from team A has opp B, then limit skaters from B (soft cap 2)
        for t in teams:
            # find an opp for team t if any player has it
            opp = ""
            for i in team_idxs[t]:
                if players[i].opp:
                    opp = players[i].opp
                    break
            if opp and opp in team_skaters_count:
                prob += team_skaters_count[opp] <= 2 + 9 * (1 - goalie_team[t]), f"goalie_avoids_opp_{t}_vs_{opp}"

    # objective with noise + bonuses
    noise = [random.uniform(-randomness, randomness) for _ in range(n)] if randomness > 0 else [0.0] * n
    base = pulp.lpSum((players[i].proj + noise[i]) * selected[i] for i in range(n))

    # C + D correlation bonus (prefer at least one C and one D from same team)
    bonus = 0
    for t in teams:
        c_t = pulp.lpSum(selected[i] for i in team_idxs[t] if players[i].pos == "C")
        d_t = pulp.lpSum(selected[i] for i in team_idxs[t] if players[i].pos == "D")
        # binary indicator via "min" linearization
        y_cd = pulp.LpVariable(f"y_cd_{t}", cat="Binary")
        prob += c_t >= y_cd, f"cd_c_{t}"
        prob += d_t >= y_cd, f"cd_d_{t}"
        bonus += bonus_c_d * y_cd

    # G + RW correlation proxy: we don't have RW/LW separately (often just W),
    # so we approximate: goalie + at least one W from same team.
    for t in teams:
        g_t = pulp.lpSum(selected[i] for i in team_idxs[t] if players[i].pos == "G")
        w_t = pulp.lpSum(selected[i] for i in team_idxs[t] if players[i].pos == "W")
        y_gw = pulp.LpVariable(f"y_gw_{t}", cat="Binary")
        prob += g_t >= y_gw, f"gw_g_{t}"
        prob += w_t >= y_gw, f"gw_w_{t}"
        bonus += bonus_g_rw * y_gw

    prob += base + bonus, "objective"

    solver = pulp.PULP_CBC_CMD(
        msg=CBC_MSG,
        timeLimit=CBC_TIME_LIMIT_SEC,
        gapRel=CBC_GAP_REL,
        threads=CBC_THREADS,
    )
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        return None

    # read lineup by slot
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
    team_counts = dict(sorted(team_counts.items(), key=lambda x: (-x[1], x[0])))
    stack_template = "-".join(str(v) for v in sorted(team_counts.values(), reverse=True))
    return {
        "total_salary": total_salary,
        "total_proj": round(total_proj, 2),
        "team_counts": team_counts,
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
# BUILD MANY LINEUPS (with ladder)
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
    verbose: bool = False,
) -> pd.DataFrame:
    url = csv_url or CSV_URL_DEFAULT
    df = fetch_csv_to_df(url)
    players = parse_players(df, verbose=verbose)

    # Feasibility ladder based on your stack tendencies:
    # Start with 3-2 (primary 3, secondary 2), then relax to 2-2, then 2-2-2-ish,
    # then just "at least 2 stack", then no stack.
    ladder = [
        # (primary_stack, secondary_stack, min_unique, min_sal, bonus_cd, bonus_gw)
        (3, 2, min_unique, min_salary_spend, 0.80, 0.25),             # 3-2 stack + C/D preference
        (2, 2, min_unique, min_salary_spend, 0.70, 0.20),             # 2-2 stack
        (3, 0, max(0, min_unique - 1), min_salary_spend, 0.70, 0.20), # single 3-stack
        (2, 0, max(0, min_unique - 1), max(45000, min_salary_spend - 1500), 0.60, 0.15),
        (0, 0, max(0, min_unique - 2), max(44000, min_salary_spend - 2500), 0.40, 0.10),
    ]

    built: List[List[int]] = []
    prev_sets: List[Set[int]] = []

    for prim, sec, uniq, min_sal, b_cd, b_gw in ladder:
        built = []
        prev_sets = []
        ok = True

        for li in range(num_lineups):
            lineup = None
            solves = 0
            while solves < MAX_SOLVES_PER_LINEUP and lineup is None:
                solves += 1
                lineup = optimize_one_nhl(
                    players=players,
                    salary_cap=salary_cap,
                    min_salary_spend=min_sal,
                    min_unique_vs_previous=uniq,
                    previous_lineups=prev_sets,
                    randomness=randomness + (0.12 * (solves - 1)),
                    seed=None if seed is None else seed + li * 10 + solves,
                    require_primary_stack=prim,
                    require_secondary_stack=sec,
                    avoid_goalie_vs_opp_skaters=avoid_goalie_vs_opp_skaters,
                    bonus_c_d=b_cd,
                    bonus_g_rw=b_gw,
                )

            if lineup is None:
                ok = False
                break

            built.append(lineup)
            prev_sets.append(set(lineup))

        if ok and built:
            break

    if not built:
        raise RuntimeError(
            "No NHL lineups generated.\n"
            "Try loosening:\n"
            "  - lower min_salary_spend (e.g. 45000)\n"
            "  - set min_unique to 0 or 1\n"
            "  - increase randomness (1.2)\n"
            "  - set avoid_goalie_vs_opp_skaters=False (if OPP column is unreliable)\n"
        )

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
        avoid_goalie_vs_opp_skaters=(not args.no_goalie_opp_rule),
        verbose=True,
    )

    print("\nTop 10 lineups:\n", flush=True)
    print(out_df.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
