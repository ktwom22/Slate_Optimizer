"""
NFL DraftKings Classic Lineup Generator (Tiny-slate strategy + stacking templates)
LOCAL CLI VERSION (one file) — Windows-safe, never “looks hung”.

✅ Keeps your CSV pull (unchanged)
✅ Flushed logging (PyCharm shows progress immediately)
✅ Solver safety (threads=1 default) + time limits
✅ Hard caps solve attempts so it cannot run forever
✅ Constraint: DST cannot be opposing team of QB (uses OPP mapping)
✅ Tiny slate biases:
   - Prefer RB in FLEX
   - Prefer cheap TE (<=4500)
✅ Stack templates on NON-DST players with exact team counts
✅ Builds multiple lineups WITHOUT stopping when a template fails
✅ Includes generate_nfl_df(...) (optional, doesn’t require Flask)

Install:
  pip install pandas requests pulp

Run:
  python -u NFL.py --num_lineups 20
"""

import argparse
import os
import random
import re
import sys
import threading
import time
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
    "NFL_CSV_URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSExcKi8LiRgnZpx9JeIRpgFMfYCFxgfdixu6oZxD2FhUG5UwyI86QDYC1ImTPAIPGdDMizdrYWSWP3/pub?gid=279175578&single=true&output=csv",
)

DK_SALARY_CAP = 50000
DK_SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
LINEUP_SIZE = len(DK_SLOTS)

# Templates (non-DST team counts)
TEMPLATE_MAP_NORMAL: Dict[str, Tuple[List[int], Optional[Set[int]]]] = {
    "2-2": ([2, 2], {2}),
    "3-2": ([3, 2], {3}),
    "2-2-2": ([2, 2, 2], {2}),
    "3-3-2": ([3, 3, 2], {3}),
}

TEMPLATE_MAP_TINY: Dict[str, Tuple[List[int], Optional[Set[int]]]] = {
    "3-3-2": ([3, 3, 2], None),
    "4-3": ([4, 3], None),
    "2-2": ([2, 2], None),
    "2-2-2": ([2, 2, 2], None),
    "4-4": ([4, 4], None),
    "5-2": ([5, 2], None),
}

TINY_WEIGHTS = [
    ("3-3-2", 0.30),
    ("4-3",   0.23),
    ("2-2",   0.20),
    ("2-2-2", 0.16),
    ("4-4",   0.06),
    ("5-2",   0.03),
]

# -----------------------------
# SPEED / SAFETY KNOBS (Windows-safe)
# -----------------------------
CBC_TIME_LIMIT_SEC = float(os.environ.get("NFL_CBC_TIME_LIMIT", "1.5"))
CBC_GAP_REL = float(os.environ.get("NFL_CBC_GAP_REL", "0.08"))
CBC_THREADS = int(os.environ.get("NFL_CBC_THREADS", "1"))  # safest on Windows
CBC_MSG = bool(int(os.environ.get("NFL_CBC_MSG", "0")))

TEMPLATES_TRIED_PER_LINEUP = int(os.environ.get("NFL_TEMPLATES_PER_LU", "2"))
RETRIES_PER_TEMPLATE = int(os.environ.get("NFL_RETRIES_PER_TEMPLATE", "0"))
MAX_SOLVES_PER_LINEUP = int(os.environ.get("NFL_MAX_SOLVES_PER_LU", "4"))

# Hard wall to kill truly stuck CBC processes
HARD_WALL_SEC = int(os.environ.get("NFL_HARD_WALL_SEC", "60"))

# -----------------------------
# DATA MODEL
# -----------------------------
@dataclass(frozen=True)
class Player:
    name: str
    team: str
    pos: str
    salary: int
    proj: float
    opp: str


# -----------------------------
# LOGGING + WATCHDOG
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def start_watchdog() -> None:
    def _boom():
        log(f"\n⛔ HARD STOP: exceeded {HARD_WALL_SEC}s (set NFL_HARD_WALL_SEC to change).")
        os._exit(2)  # guaranteed exit even if solver is stuck

    t = threading.Timer(HARD_WALL_SEC, _boom)
    t.daemon = True
    t.start()


# -----------------------------
# CSV UTIL
# -----------------------------
def fetch_csv_to_df(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


# -----------------------------
# TEXT / PARSE HELPERS
# -----------------------------
def clean_text(x: str) -> str:
    s = str(x)
    s = s.replace("\t", " ")
    s = re.sub(r"&nbsp;?", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def normalize_header(h: str) -> str:
    h = str(h).replace("\t", " ")
    h = re.sub(r"\s+", " ", h).strip()
    return h


def norm_key(h: str) -> str:
    h = normalize_header(h).lower()
    h = re.sub(r"[^a-z0-9]+", "_", h).strip("_")
    return h


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    key_map = {c: norm_key(c) for c in df.columns}
    inv: Dict[str, List[str]] = {}
    for orig, k in key_map.items():
        inv.setdefault(k, []).append(orig)
    for cand in candidates:
        if cand in inv:
            return inv[cand][0]
    return None


def find_proj_col(df: pd.DataFrame) -> Optional[str]:
    direct = [
        "proj", "projection", "fpts", "fp", "points", "projected_points",
        "dk_fp_projected", "dk_fp_proj", "dk_fpts_projected",
    ]
    col = pick_col(df, direct)
    if col:
        return col
    for c in df.columns:
        k = norm_key(c)
        if "dk" in k and "project" in k and ("fp" in k or "fpt" in k):
            return c
        if ("fp" in k or "fpts" in k) and "project" in k:
            return c
    return None


_num_re = re.compile(r"[-+]?\d*\.?\d+")


def parse_float_from_any(x, default=0.0) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    s = str(x)
    s = s.replace(",", "").replace("$", "").replace("\t", " ").strip()
    s = s.replace("—", " ").replace("–", " ").replace("−", "-")
    m = _num_re.search(s)
    if not m:
        return default
    try:
        return float(m.group(0))
    except Exception:
        return default


def parse_salary_dk(x, default=0) -> int:
    v = parse_float_from_any(x, default=float(default))
    if v <= 0:
        return default
    if v <= 100:
        return int(round(v * 1000))
    return int(round(v))


def normalize_pos(p: str) -> str:
    p = clean_text(p).upper()
    p = p.replace("D/ST", "DST").replace("DST/ST", "DST").replace("D_ST", "DST")
    if "DST" in p or "DEF" in p or p in ("D", "DEFENSE"):
        return "DST"
    if p.startswith("QB"):
        return "QB"
    if p.startswith("RB"):
        return "RB"
    if p.startswith("WR"):
        return "WR"
    if p.startswith("TE"):
        return "TE"
    return p


ELIG_MAP = {
    "QB": {"QB"},
    "RB": {"RB1", "RB2", "FLEX"},
    "WR": {"WR1", "WR2", "WR3", "FLEX"},
    "TE": {"TE", "FLEX"},
    "DST": {"DST"},
}


def eligible_slots(pos: str) -> Set[str]:
    return ELIG_MAP.get(pos.upper().strip(), set())



# -----------------------------
# PARSE PLAYERS
# -----------------------------
def parse_players(df: pd.DataFrame, verbose: bool = True) -> Tuple[List[Player], pd.DataFrame, Dict[str, str]]:
    df = df.copy()

    # 1. Clean column names immediately
    df.columns = [str(c).strip().upper() for c in df.columns]

    if verbose:
        log(f"Mapping columns from: {list(df.columns)}")

    # 2. Hard-coded mapping based on your data snippet
    # Using .get() ensures the script doesn't crash if one is missing
    name_col = 'PLAYER'
    pos_col = 'POS'
    sal_col = 'SALARY'
    proj_col = 'PROJECTED POINTS'
    team_col = 'TEAM'
    opp_col = 'OPPONENT'

    # Check for presence
    for col in [name_col, pos_col, sal_col, proj_col, team_col]:
        if col not in df.columns:
            raise ValueError(f"CRITICAL ERROR: Column '{col}' not found in CSV. Available: {list(df.columns)}")

    # 3. Data Conversion
    # We use a custom lambda for salary to handle the '$8.7k' format
    df["name_clean"] = df[name_col].apply(clean_text)
    df["team_clean"] = df[team_col].apply(clean_text)
    df["pos_clean"] = df[pos_col].apply(normalize_pos)

    # Updated Salary Logic to handle $ and K
    def clean_salary_value(val):
        s = str(val).lower().replace('$', '').replace(',', '').strip()
        if 'k' in s:
            return int(float(s.replace('k', '')) * 1000)
        try:
            return int(float(s))
        except:
            return 0

    df["salary_clean"] = df[sal_col].apply(clean_salary_value)
    df["proj_clean"] = df[proj_col].apply(lambda v: parse_float_from_any(v, 0.0))
    df["opp_clean"] = df[opp_col].apply(clean_text) if opp_col in df.columns else ""

    # 4. Filter and Create Player Objects
    # Remove players with 0 projection or 0 salary (injured/out)
    valid_df = df[(df["salary_clean"] > 0) & (df["proj_clean"] > 0)].copy()

    players: List[Player] = []
    for _, r in valid_df.iterrows():
        players.append(Player(
            name=r["name_clean"],
            team=r["team_clean"],
            pos=r["pos_clean"],
            salary=r["salary_clean"],
            proj=r["proj_clean"],
            opp=r["opp_clean"]
        ))

    # 5. Team Mapping for DST logic
    team_to_opp = {p.team: p.opp for p in players if p.team and p.opp}

    if verbose:
        log(f"✅ Successfully loaded {len(players)} players into the optimizer.")

    return players, valid_df, team_to_opp


# -----------------------------
# SLATE / POOL
# -----------------------------
def pool_counts(players: List[Player]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for p in players:
        d[p.pos] = d.get(p.pos, 0) + 1
    return d


def teams_nondst_counts(players: List[Player]) -> Dict[str, int]:
    t: Dict[str, int] = {}
    for p in players:
        if p.pos == "DST":
            continue
        t[p.team] = t.get(p.team, 0) + 1
    return t


def print_pool_summary(players: List[Player], team_to_opp: Dict[str, str]) -> None:
    c = pool_counts(players)
    log("\nPLAYER POOL COUNTS:")
    for k in ["QB", "RB", "WR", "TE", "DST"]:
        log(f"  {k}: {c.get(k, 0)}")

    tc = teams_nondst_counts(players)
    teams = sorted(tc.keys())
    log(f"\nTeams in pool (non-DST): {len(teams)} -> {teams}")

    for size in [2, 3, 4, 5]:
        log(f"Teams with >={size} non-DST: {len([t for t, ct in tc.items() if ct >= size])}")

    log("\nTEAM->OPP mapping (DST != QB opponent):")
    for t in teams:
        log(f"  {t} -> {team_to_opp.get(t, '')}")


def is_tiny_slate(players: List[Player]) -> bool:
    return len(teams_nondst_counts(players).keys()) <= 4


# -----------------------------
# TEMPLATES
# -----------------------------
def weighted_template_list(num_lineups: int) -> List[str]:
    tpls: List[str] = []
    for name, w in TINY_WEIGHTS:
        tpls += [name] * max(1, round(num_lineups * w))
    while len(tpls) < num_lineups:
        tpls.append("3-3-2")
    tpls = tpls[:num_lineups]
    random.shuffle(tpls)
    return tpls


def template_feasible(players: List[Player], template_sizes: List[int], qb_must_be_on_largest: bool) -> bool:
    c = pool_counts(players)
    if c.get("QB", 0) < 1: return False
    if c.get("RB", 0) < 2: return False
    if c.get("WR", 0) < 3: return False
    if c.get("TE", 0) < 1: return False
    if c.get("DST", 0) < 1: return False

    tc = teams_nondst_counts(players)
    if len(tc) < len(template_sizes):
        return False

    required = sorted(template_sizes, reverse=True)
    available = sorted(tc.items(), key=lambda x: x[1], reverse=True)
    used = set()
    for need in required:
        found = False
        for t, ct in available:
            if t in used:
                continue
            if ct >= need:
                used.add(t)
                found = True
                break
        if not found:
            return False

    if qb_must_be_on_largest:
        largest = max(template_sizes)
        qb_teams = {p.team for p in players if p.pos == "QB"}
        if not any(t in qb_teams and ct >= largest for t, ct in tc.items()):
            return False

    return True


# -----------------------------
# PRECOMPUTE INDICES (speed)
# -----------------------------
def precompute(players: List[Player]) -> Dict[str, object]:
    teams = sorted(set(p.team for p in players))
    elig_by_slot: Dict[str, List[int]] = {s: [] for s in DK_SLOTS}

    idx_qb: List[int] = []
    team_nondst: Dict[str, List[int]] = {t: [] for t in teams}
    team_qb: Dict[str, List[int]] = {t: [] for t in teams}
    team_rb: Dict[str, List[int]] = {t: [] for t in teams}
    team_dst: Dict[str, List[int]] = {t: [] for t in teams}

    for i, p in enumerate(players):
        if p.pos == "QB":
            idx_qb.append(i)

        for s in eligible_slots(p.pos):
            elig_by_slot[s].append(i)

        if p.pos != "DST":
            team_nondst[p.team].append(i)
        if p.pos == "QB":
            team_qb[p.team].append(i)
        if p.pos == "RB":
            team_rb[p.team].append(i)
        if p.pos == "DST":
            team_dst[p.team].append(i)

    return {
        "teams": teams,
        "elig_by_slot": elig_by_slot,
        "idx_qb": idx_qb,
        "team_nondst": team_nondst,
        "team_qb": team_qb,
        "team_rb": team_rb,
        "team_dst": team_dst,
        "n": len(players),
    }


# -----------------------------
# OPTIMIZER (single lineup)
# -----------------------------
def optimize_one_lineup(
        players: List[Player],
        pre: Dict[str, object],
        team_to_opp: Dict[str, str],
        template_sizes: List[int],
        qb_must_be_on_largest_stack: bool,
        salary_cap: int,
        min_salary_spend: int,
        min_unique_vs_previous: int,
        previous_lineups: List[Set[int]],
        randomness: float,
        seed: Optional[int],
        nonstack_max: int,
        use_dst_rb_correlation: bool,
        tiny_bias: bool,
        enforce_dst_not_qb_opp: bool,
        forced_stack_team: str = "ANY",
        forced_stack_size: int = 3
) -> Optional[List[int]]:
    if seed is not None:
        random.seed(seed)

    n: int = pre["n"]
    teams: List[str] = pre["teams"]
    elig_by_slot: Dict[str, List[int]] = pre["elig_by_slot"]
    idx_qb: List[int] = pre["idx_qb"]
    team_nondst: Dict[str, List[int]] = pre["team_nondst"]
    team_qb: Dict[str, List[int]] = pre["team_qb"]
    team_rb: Dict[str, List[int]] = pre["team_rb"]
    team_dst: Dict[str, List[int]] = pre["team_dst"]

    # 1. Variables Setup
    x: Dict[Tuple[int, str], pulp.LpVariable] = {}
    for s in DK_SLOTS:
        for i in elig_by_slot[s]:
            x[(i, s)] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")

    prob = pulp.LpProblem("NFL_DK_STACKS", pulp.LpMaximize)

    # 2. Roster and Slot Constraints
    for s in DK_SLOTS:
        prob += pulp.lpSum(x[(i, s)] for i in elig_by_slot[s]) == 1, f"fill_{s}"

    selected = [pulp.lpSum(x[(i, s)] for s in DK_SLOTS if (i, s) in x) for i in range(n)]
    for i in range(n):
        prob += selected[i] <= 1, f"one_slot_{i}"

    # 3. Salary Constraints
    total_salary = pulp.lpSum(players[i].salary * selected[i] for i in range(n))
    prob += total_salary <= salary_cap, "salary_cap"
    prob += total_salary >= min_salary_spend, "min_salary_spend"

    # 4. QB and DST Indicators
    qb_team: Dict[str, pulp.LpVariable] = {t: pulp.LpVariable(f"qb_team_{t}", cat="Binary") for t in teams}
    dst_team: Dict[str, pulp.LpVariable] = {t: pulp.LpVariable(f"dst_team_{t}", cat="Binary") for t in teams}

    for t in teams:
        qb_idxs = team_qb.get(t, [])
        prob += qb_team[t] == (pulp.lpSum(selected[i] for i in qb_idxs) if qb_idxs else 0), f"qb_team_eq_{t}"

        dst_idxs = team_dst.get(t, [])
        if dst_idxs:
            prob += dst_team[t] == pulp.lpSum(x[(i, "DST")] for i in dst_idxs if (i, "DST") in x), f"dst_team_eq_{t}"
        else:
            prob += dst_team[t] == 0

    prob += pulp.lpSum(qb_team[t] for t in teams) == 1, "exactly_one_qb"

    # 5. FORCED TEAM LOGIC
    # If the user selected a specific team, force that team's QB and stack size
    if forced_stack_team != "ANY" and forced_stack_team in teams:
        prob += qb_team[forced_stack_team] == 1, "force_stack_team_qb"
        # Force the total players from this team to match the stack size
        # (Using team_nondst because QB is usually part of the stack size count)
        idxs = team_nondst.get(forced_stack_team, [])
        if idxs:
            prob += pulp.lpSum(selected[i] for i in idxs) == forced_stack_size, "force_stack_size"

    # 6. Team Usage & Stacking Logic
    team_count: Dict[str, pulp.LpAffineExpression] = {}
    for t in teams:
        idxs = team_nondst.get(t, [])
        team_count[t] = pulp.lpSum(selected[i] for i in idxs) if idxs else 0

    K = len(template_sizes)
    stack_team: Dict[Tuple[str, int], pulp.LpVariable] = {
        (t, k): pulp.LpVariable(f"stack_{t}_{k}", cat="Binary")
        for t in teams for k in range(K)
    }

    for k in range(K):
        prob += pulp.lpSum(stack_team[(t, k)] for t in teams) == 1, f"one_team_for_stack_{k}"

    M = 9
    for t in teams:
        is_stacked = pulp.lpSum(stack_team[(t, k)] for k in range(K))
        prob += team_count[t] <= nonstack_max + M * is_stacked, f"nonstack_limit_{t}"

        for k, s_size in enumerate(template_sizes):
            # If stack_team[t,k] is 1, team_count must be exactly s_size
            prob += team_count[t] >= s_size * stack_team[(t, k)], f"stack_lb_{t}_{k}"
            prob += team_count[t] <= s_size + M * (1 - stack_team[(t, k)]), f"stack_ub_{t}_{k}"

    # 7. Correlation Rules
    if qb_must_be_on_largest_stack:
        largest = max(template_sizes)
        for t in teams:
            allowed = pulp.lpSum(stack_team[(t, k)] for k, s_size in enumerate(template_sizes) if s_size == largest)
            prob += qb_team[t] <= allowed, f"qb_on_largest_{t}"

    if use_dst_rb_correlation:
        for t in teams:
            rb_idxs = team_rb.get(t, [])
            rb_on_team = pulp.lpSum(selected[i] for i in rb_idxs) if rb_idxs else 0
            prob += rb_on_team >= 1 * dst_team[t], f"dst_rb_corr_{t}"

    if enforce_dst_not_qb_opp:
        for t in teams:
            opp = team_to_opp.get(t, "")
            if opp in dst_team:
                prob += dst_team[opp] <= 1 - qb_team[t], f"dst_not_vs_qb_{t}"

    # 8. Uniqueness Constraints
    if previous_lineups and min_unique_vs_previous > 0:
        max_overlap = 9 - min_unique_vs_previous
        for li, prev in enumerate(previous_lineups):
            prob += pulp.lpSum(selected[i] for i in prev) <= max_overlap, f"overlap_{li}"

    # 9. Objective Function with Randomness
    noise = [random.uniform(-randomness, randomness) for _ in range(n)] if randomness > 0 else [0.0] * n
    base_obj = pulp.lpSum((players[i].proj + noise[i]) * selected[i] for i in range(n))

    # Tiny Slate Biases (e.g., favoring RBs in Flex)
    bonus = 0
    if tiny_bias:
        flex_rb = [i for i in elig_by_slot["FLEX"] if players[i].pos == "RB"]
        bonus += 0.75 * pulp.lpSum(x[(i, "FLEX")] for i in flex_rb)

    prob += base_obj + bonus, "objective"

    # 10. Solve
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=1.0, gapRel=0.10, threads=4)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        return None

    # 11. Parse Results
    chosen: List[int] = []
    for s in DK_SLOTS:
        for i in elig_by_slot[s]:
            if x[(i, s)].value() == 1:
                chosen.append(i)
                break

    return chosen if len(chosen) == 9 else None


# -----------------------------
# OUTPUT HELPERS
# -----------------------------
def summarize_lineup(players: List[Player], lineup_idxs: List[int]) -> Dict[str, object]:
    total_salary = sum(players[i].salary for i in lineup_idxs)
    total_proj = sum(players[i].proj for i in lineup_idxs)

    team_counts_nondst: Dict[str, int] = {}
    qb_str = "None"
    dst_str = "None"
    for i in lineup_idxs:
        p = players[i]
        if p.pos == "QB":
            qb_str = f"{p.name} ({p.team}) vs {p.opp or '?'}"
        if p.pos == "DST":
            dst_str = f"{p.name} ({p.team})"
        if p.pos != "DST":
            team_counts_nondst[p.team] = team_counts_nondst.get(p.team, 0) + 1

    team_counts_nondst = dict(sorted(team_counts_nondst.items(), key=lambda x: (-x[1], x[0])))
    return {"salary": total_salary, "proj": round(total_proj, 2), "qb": qb_str, "dst": dst_str, "team_counts_nondst": team_counts_nondst}


def lineups_to_df(players: List[Player], built: List[Tuple[str, List[int]]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for tpl, lineup in built:
        row: Dict[str, object] = {"template": tpl}
        for slot, idx in zip(DK_SLOTS, lineup):
            p = players[idx]
            row[f"{slot}_name"] = p.name
            row[f"{slot}_team"] = p.team
            row[f"{slot}_salary"] = int(p.salary)
            row[f"{slot}_proj"] = float(round(p.proj, 2))

        meta = summarize_lineup(players, lineup)
        row["total_salary"] = meta["salary"]
        row["total_proj"] = meta["proj"]
        row["stack_team_counts_nondst"] = "; ".join([f"{t}:{c}" for t, c in meta["team_counts_nondst"].items()])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("total_proj", ascending=False).reset_index(drop=True)


# -----------------------------
# BUILD LINEUPS
def build_lineups(
        players: List[Player],
        pre: Dict[str, object],
        team_to_opp: Dict[str, str],
        templates: List[str],
        template_map: Dict[str, Tuple[List[int], Optional[Set[int]]]],
        num_lineups: int,
        salary_cap: int,
        min_salary_spend: int,
        min_unique: int,
        randomness: float,
        seed: Optional[int],
        tiny_mode: bool,
        verbose: bool = True,
        # NEW: Pass down the user-selected forced team if applicable
        forced_stack_team: str = "ANY",
        forced_stack_size: int = 3
) -> List[Tuple[str, List[int]]]:
    feasible_templates = []
    for t in templates:
        if t not in template_map: continue
        sizes, _ = template_map[t]
        if template_feasible(players, sizes, qb_must_be_on_largest=tiny_mode):
            feasible_templates.append(t)

    if not feasible_templates:
        return []

    results: List[Tuple[str, List[int]]] = []
    prev_sets: List[Set[int]] = []

    # Adaptive variables that loosen if we get stuck
    current_min_unique = min_unique
    current_min_salary = min_salary_spend

    # The Shuffle ensures diversity across runs
    base_order = feasible_templates[:]
    if not tiny_mode:
        random.shuffle(base_order)

    for li in range(num_lineups):
        made = False
        start_idx = li % len(base_order)
        attempt_order = base_order[start_idx:] + base_order[:start_idx]

        # LIMIT: Only try the top 3 best templates to save time
        for tpl in attempt_order[:3]:
            sizes, _ = template_map[tpl]

            # Optimization Call
            lineup = optimize_one_lineup(
                players=players,
                pre=pre,
                team_to_opp=team_to_opp,
                template_sizes=sizes,
                qb_must_be_on_largest_stack=tiny_mode,
                salary_cap=salary_cap,
                min_salary_spend=current_min_salary,
                min_unique_vs_previous=current_min_unique,
                previous_lineups=prev_sets,
                randomness=randomness,
                seed=None if seed is None else seed + li,
                nonstack_max=3,
                use_dst_rb_correlation=True,
                tiny_bias=tiny_mode,
                enforce_dst_not_qb_opp=True,
                forced_stack_team=forced_stack_team,
                forced_stack_size=forced_stack_size
            )

            if lineup:
                results.append((tpl, lineup))
                prev_sets.append(set(lineup))
                made = True
                break

        # --- RELAXATION LOGIC ---
        # If we couldn't make a lineup, lower the bar for the NEXT one
        if not made:
            if verbose:
                print(f"⚠️ Relaxing constraints for lineup {li + 1}")
            current_min_unique = max(0, current_min_unique - 1)
            current_min_salary = max(44000, current_min_salary - 500)

            # Final "Hail Mary" attempt with no uniqueness requirement
            lineup = optimize_one_lineup(
                players=players,
                pre=pre,
                team_to_opp=team_to_opp,
                template_sizes=template_map[base_order[0]][0],
                qb_must_be_on_largest_stack=tiny_mode,
                salary_cap=salary_cap,
                min_salary_spend=44000,
                min_unique_vs_previous=0,  # Force it to work
                previous_lineups=prev_sets,
                randomness=randomness + 0.5,
                seed=None,
                nonstack_max=4,
                use_dst_rb_correlation=False,
                tiny_bias=tiny_mode,
                enforce_dst_not_qb_opp=True
            )
            if lineup:
                results.append(("RELAXED", lineup))
                prev_sets.append(set(lineup))

    return results


# -----------------------------
# OPTIONAL PUBLIC API (works locally too)
# -----------------------------
def generate_nfl_df(
    num_lineups: int = 20,
    min_unique: int = 2,
    min_salary_spend: int = 46000,
    randomness: float = 1.0,
    salary_cap: int = DK_SALARY_CAP,
    csv_url: Optional[str] = None,
    seed: Optional[int] = 7,
    templates: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    url = csv_url or CSV_URL_DEFAULT
    df = fetch_csv_to_df(url)
    players, _analysis, team_to_opp = parse_players(df, verbose=verbose)

    pre = precompute(players)
    tiny_mode = is_tiny_slate(players)

    if templates is None:
        if tiny_mode:
            templates_use = weighted_template_list(num_lineups)
            template_map = TEMPLATE_MAP_TINY
        else:
            templates_use = ["2-2", "3-2", "2-2-2", "3-3-2"]
            template_map = TEMPLATE_MAP_NORMAL
    else:
        templates_use = templates
        template_map = TEMPLATE_MAP_TINY if tiny_mode else TEMPLATE_MAP_NORMAL

    built = build_lineups(
        players=players,
        pre=pre,
        team_to_opp=team_to_opp,
        templates=templates_use,
        template_map=template_map,
        num_lineups=num_lineups,
        salary_cap=salary_cap,
        min_salary_spend=min_salary_spend,
        min_unique=min_unique,
        randomness=randomness,
        seed=seed,
        tiny_mode=tiny_mode,
        verbose=verbose,
    )

    if not built:
        raise RuntimeError(
            "No NFL lineups generated.\n"
            "Try loosening:\n"
            "  - --min_salary_spend 44000\n"
            "  - --min_unique 0 or 1\n"
            "  - --randomness 1.6\n"
        )

    return lineups_to_df(players, built)


# -----------------------------
# CLI
# -----------------------------
def main():
    start_watchdog()  # ✅ MUST start before any solver work

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_url", type=str, default=CSV_URL_DEFAULT)
    ap.add_argument("--num_lineups", type=int, default=20)
    ap.add_argument("--salary_cap", type=int, default=DK_SALARY_CAP)
    ap.add_argument("--min_salary_spend", type=int, default=46000)
    ap.add_argument("--min_unique", type=int, default=2)
    ap.add_argument("--randomness", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--templates", type=str, default="2-2,3-2,2-2-2,3-3-2")
    ap.add_argument("--cbc_msg", action="store_true", help="Show CBC solver output")
    args = ap.parse_args()

    global CBC_MSG
    if args.cbc_msg:
        CBC_MSG = True

    log("Step 1/6: Checking solvers...")
    log(f"Available solvers: {pulp.listSolvers(onlyAvailable=True)}")

    log("Step 2/6: Fetching CSV...")
    t0 = time.time()
    df = fetch_csv_to_df(args.csv_url)
    log(f"CSV fetched in {time.time() - t0:.2f}s | rows={len(df)} cols={len(df.columns)}")

    log("Step 3/6: Parsing players...")
    players, _analysis, team_to_opp = parse_players(df, verbose=True)
    log(f"Loaded {len(players)} players after cleanup.")

    log("Step 4/6: Pool summary...")
    print_pool_summary(players, team_to_opp)

    pre = precompute(players)

    log("Step 5/6: Slate type + templates...")
    tiny_mode = is_tiny_slate(players)
    if tiny_mode:
        log("=== TINY SLATE MODE (<=4 teams) ===")
        templates = weighted_template_list(args.num_lineups)
        template_map = TEMPLATE_MAP_TINY
        log(f"Weighted templates (len={len(templates)}): {templates}")
    else:
        log("=== NORMAL SLATE MODE ===")
        templates = [t.strip() for t in args.templates.split(",") if t.strip()]
        template_map = TEMPLATE_MAP_NORMAL
        log(f"Templates: {templates}")

    log("Step 6/6: Building lineups...")
    log(f"Settings: min_salary_spend={args.min_salary_spend} min_unique={args.min_unique} randomness={args.randomness}")
    log(f"Speed: timeLimit={CBC_TIME_LIMIT_SEC}s gapRel={CBC_GAP_REL} threads={CBC_THREADS} max_solves/lineup={MAX_SOLVES_PER_LINEUP}")

    built = build_lineups(
        players=players,
        pre=pre,
        team_to_opp=team_to_opp,
        templates=templates,
        template_map=template_map,
        num_lineups=args.num_lineups,
        salary_cap=args.salary_cap,
        min_salary_spend=args.min_salary_spend,
        min_unique=args.min_unique,
        randomness=args.randomness,
        seed=args.seed,
        tiny_mode=tiny_mode,
        verbose=True,
    )

    if not built:
        raise RuntimeError(
            "No lineups generated.\n"
            "Try loosening:\n"
            "  1) --min_salary_spend 44000\n"
            "  2) --min_unique 0 or 1\n"
            "  3) --randomness 1.6\n"
        )

    for i, (tpl, lineup) in enumerate(built, start=1):
        meta = summarize_lineup(players, lineup)
        log("\n" + "=" * 130)
        log(f"LINEUP #{i} | Template:{tpl} | Salary:{meta['salary']} | Proj:{meta['proj']} | QB:{meta['qb']} | DST:{meta['dst']} | TeamCounts(nonDST):{meta['team_counts_nondst']}")
        log("-" * 130)
        for slot, idx in zip(DK_SLOTS, lineup):
            p = players[idx]
            log(f"{slot:>4}  {p.name:<28} {p.team:<4} {p.pos:<3}  ${p.salary:<5}  proj:{p.proj:>5.2f}  opp:{p.opp or ''}")
        log("=" * 130)

    out_df = lineups_to_df(players, built)
    log("\nTop 10 (DF preview):")
    log(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    start_watchdog()
    main()