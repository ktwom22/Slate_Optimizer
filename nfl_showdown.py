"""
NFL DraftKings SHOWDOWN Optimizer (ONE FILE) — nfl_showdown.py

✅ Uses YOUR Google Sheet CSV:
   https://docs.google.com/spreadsheets/d/e/2PACX-1vSExcKi8LiRgnZpx9JeIRpgFMfYCFxgfdixu6oZxD2FhUG5UwyI86QDYC1ImTPAIPGdDMizdrYWSWP3/pub?gid=1791525610&single=true&output=csv

✅ DK Showdown rules:
   - 6 slots: CPT + 5 FLEX
   - CPT salary = 1.5x, CPT points = 1.5x
   - Same player cannot appear twice (no CPT+FLEX duplicates)

✅ Correlation + uniqueness rules inspired by Showdown strategy:
   - Default: NO K at CPT
   - Default: NO DST at CPT (can allow)
   - CPT WR/TE => strongly prefers QB from same team (toggle)
   - CPT QB => discourages same-team K (toggle)
   - Max K+DST combined (default 2)
   - If roster DST, cap opposing team players (default 3)

✅ Diversity:
   - Uniqueness across lineups (min_unique)
   - Randomness noise
   - Optional max CPT exposure per player
   - Optional max FLEX exposure per player
   - Salary cap window (min & optional max spend)

Install:
  pip install pandas requests pulp

Run:
  python -u nfl_showdown.py --num_lineups 20

This module exposes:
  - generate_showdown_df(...) -> pandas DataFrame (DK-style output)
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
# CONFIG
# -----------------------------
CSV_URL_DEFAULT = os.environ.get(
    "NFL_SHOWDOWN_CSV_URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSExcKi8LiRgnZpx9JeIRpgFMfYCFxgfdixu6oZxD2FhUG5UwyI86QDYC1ImTPAIPGdDMizdrYWSWP3/pub?gid=1791525610&single=true&output=csv",
)

SALARY_CAP_DEFAULT = 50000
SLOTS = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]
LINEUP_SIZE = len(SLOTS)

# speed knobs
CBC_TIME_LIMIT_SEC = float(os.environ.get("SHOWDOWN_CBC_TIME_LIMIT", "2.0"))
CBC_GAP_REL = float(os.environ.get("SHOWDOWN_CBC_GAP_REL", "0.08"))
CBC_THREADS = int(os.environ.get("SHOWDOWN_CBC_THREADS", "1"))
CBC_MSG = bool(int(os.environ.get("SHOWDOWN_CBC_MSG", "0")))

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
# IO
# -----------------------------
def fetch_csv_to_df(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


# -----------------------------
# PARSING HELPERS
# -----------------------------
def _norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _clean_text(x) -> str:
    s = str(x)
    s = s.replace("\t", " ")
    s = re.sub(r"&nbsp;?", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


_num_re = re.compile(r"[-+]?\d*\.?\d+")


def _to_float(x, default=0.0) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    s = str(x).replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.replace("—", " ").replace("–", " ").replace("−", "-")
    m = _num_re.search(s)
    if not m:
        return default
    try:
        return float(m.group(0))
    except Exception:
        return default


def _to_salary(x, default=0) -> int:
    v = _to_float(x, float(default))
    if v <= 0:
        return default
    # DK shorthand support (6.2 => 6200)
    if v <= 100:
        return int(round(v * 1000))
    return int(round(v))


def _normalize_pos(p: str) -> str:
    p = _clean_text(p).upper()
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
    if p.startswith("K"):
        return "K"
    return p


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    key_map = {c: _norm_key(c) for c in df.columns}
    inv: Dict[str, List[str]] = {}
    for orig, k in key_map.items():
        inv.setdefault(k, []).append(orig)
    for cand in candidates:
        ck = _norm_key(cand)
        if ck in inv:
            return inv[ck][0]
    return None


def _find_proj_col(df: pd.DataFrame) -> Optional[str]:
    direct = [
        "proj", "projection", "projected_points", "fpts", "fp", "points",
        "dk_fp_projected", "dk_fp_proj", "dk_fpts_projected",
    ]
    col = _pick_col(df, direct)
    if col:
        return col
    for c in df.columns:
        k = _norm_key(c)
        if ("proj" in k or "project" in k) and ("fp" in k or "fpt" in k or "points" in k):
            return c
    return None


def parse_players(df: pd.DataFrame, verbose: bool = True) -> Tuple[List[Player], Dict[str, str]]:
    """
    Returns:
      players: list[Player]
      team_to_opp: mapping team -> opp (best-effort)
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    pos_col = _pick_col(df, ["pos", "position", "dk_position", "roster_position"])
    name_col = _pick_col(df, ["name", "player", "player_name", "nickname"])
    team_col = _pick_col(df, ["team", "teamabbr", "team_abbrev", "tm"])
    sal_col = _pick_col(df, ["salary", "sal", "dk_salary", "cost"])
    proj_col = _find_proj_col(df)
    opp_col = _pick_col(df, ["opp", "opponent", "vs", "opponent_team"])

    missing = []
    if not pos_col: missing.append("pos")
    if not name_col: missing.append("name")
    if not team_col: missing.append("team")
    if not sal_col: missing.append("salary")
    if not proj_col: missing.append("proj")
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    df["name"] = df[name_col].apply(_clean_text)
    df["team"] = df[team_col].apply(_clean_text)
    df["pos"] = df[pos_col].apply(_normalize_pos)
    df["salary"] = df[sal_col].apply(lambda v: _to_salary(v, 0))
    df["proj"] = df[proj_col].apply(lambda v: _to_float(v, 0.0))
    df["opp"] = df[opp_col].apply(_clean_text) if opp_col else ""

    # keep only relevant showdown positions (QB/RB/WR/TE/K/DST)
    df = df[(df["name"] != "") & (df["team"] != "") & (df["pos"] != "")]
    df = df[df["pos"].isin(["QB", "RB", "WR", "TE", "K", "DST"])]
    df = df[(df["salary"] > 0) & (df["proj"] > 0)]
    if df.empty:
        raise ValueError("No valid players after cleanup (salary/proj/pos filters).")

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

    # team->opp mapping best effort (prefer QB rows)
    team_to_opp: Dict[str, str] = {}
    for p in players:
        if p.pos == "QB" and p.opp:
            team_to_opp[p.team] = p.opp
    for p in players:
        if p.team not in team_to_opp and p.opp:
            team_to_opp[p.team] = p.opp
    # make it symmetric when possible
    for a, b in list(team_to_opp.items()):
        if b and b not in team_to_opp:
            team_to_opp[b] = a

    if verbose:
        print(f"Detected columns -> POS:{pos_col} | NAME:{name_col} | TEAM:{team_col} | SALARY:{sal_col} | PROJ:{proj_col} | OPP:{opp_col or 'None'}", flush=True)
        print("Sample players:", flush=True)
        for p in players[:10]:
            print(f"  {p.pos:<3} {p.name:<25} {p.team:<4} sal={p.salary:<5} proj={p.proj:<5.2f} opp={p.opp}", flush=True)

    return players, team_to_opp


# -----------------------------
# SHOWDOWN OPTIMIZER (single lineup)
# -----------------------------
def optimize_one_showdown(
    players: List[Player],
    team_to_opp: Dict[str, str],
    salary_cap: int,
    min_salary_spend: int,
    max_salary_spend: Optional[int],
    min_unique_vs_previous: int,
    previous_lineups: List[Set[int]],
    randomness: float,
    seed: Optional[int],

    # CPT pool controls
    allow_k_cpt: bool,
    allow_dst_cpt: bool,

    # correlation toggles
    enforce_cpt_wrte_with_qb: bool,
    discourage_cpt_qb_with_same_team_k: bool,

    # roster construction rules
    max_k_dst_total: int,
    dst_max_opp_players: int,

    # exposure caps (optional)
    cpt_exposure_count: Optional[Dict[int, int]],
    flex_exposure_count: Optional[Dict[int, int]],
    max_cpt_exposure: Optional[int],
    max_flex_exposure: Optional[int],
) -> Optional[List[int]]:
    """
    Returns lineup indices in slot order: [CPT, FLEX1..FLEX5] as player indices.
    """
    if seed is not None:
        random.seed(seed)

    n = len(players)
    all_idx = list(range(n))

    # CPT eligibility
    def cpt_ok(p: Player) -> bool:
        if p.pos == "K" and not allow_k_cpt:
            return False
        if p.pos == "DST" and not allow_dst_cpt:
            return False
        return True

    cpt_idx = [i for i in all_idx if cpt_ok(players[i])]
    if not cpt_idx:
        return None

    # build by team indices
    teams = sorted(set(p.team for p in players))
    team_players: Dict[str, List[int]] = {t: [] for t in teams}
    for i, p in enumerate(players):
        team_players[p.team].append(i)

    # variables: x[(i, slot)] only created if eligible
    x: Dict[Tuple[int, str], pulp.LpVariable] = {}
    # CPT vars only for cpt_idx
    for i in cpt_idx:
        x[(i, "CPT")] = pulp.LpVariable(f"x_{i}_CPT", cat="Binary")
    # FLEX vars for all players
    for s in SLOTS[1:]:
        for i in all_idx:
            x[(i, s)] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")

    prob = pulp.LpProblem("NFL_DK_SHOWDOWN", pulp.LpMaximize)

    # fill each slot exactly once
    prob += pulp.lpSum(x[(i, "CPT")] for i in cpt_idx) == 1, "fill_CPT"
    for s in SLOTS[1:]:
        prob += pulp.lpSum(x[(i, s)] for i in all_idx) == 1, f"fill_{s}"

    # each player at most once across all slots
    selected = {}
    for i in all_idx:
        selected[i] = (
            x.get((i, "CPT"), 0) +
            pulp.lpSum(x[(i, s)] for s in SLOTS[1:])
        )
        prob += selected[i] <= 1, f"one_slot_{i}"

    # salary (CPT is 1.5x)
    total_salary = pulp.lpSum(
        (1.5 * players[i].salary) * x.get((i, "CPT"), 0) +
        players[i].salary * pulp.lpSum(x[(i, s)] for s in SLOTS[1:])
        for i in all_idx
    )
    prob += total_salary <= salary_cap, "salary_cap"
    prob += total_salary >= min_salary_spend, "min_salary_spend"
    if max_salary_spend is not None:
        prob += total_salary <= max_salary_spend, "max_salary_spend"

    # team counts for constraints
    team_count = {t: pulp.lpSum(selected[i] for i in idxs) for t, idxs in team_players.items()}

    # position counts (K/DST limits)
    k_count = pulp.lpSum(selected[i] for i in all_idx if players[i].pos == "K")
    dst_count = pulp.lpSum(selected[i] for i in all_idx if players[i].pos == "DST")
    prob += (k_count + dst_count) <= max_k_dst_total, "max_k_dst_total"

    # DST opp cap rule: if DST chosen from team T, limit opposing team players
    if dst_max_opp_players is not None and dst_max_opp_players >= 0:
        dst_team = {t: pulp.LpVariable(f"dst_{t}", cat="Binary") for t in teams}
        for t in teams:
            idxs = team_players[t]
            dst_idxs = [i for i in idxs if players[i].pos == "DST"]
            if dst_idxs:
                prob += dst_team[t] == pulp.lpSum(selected[i] for i in dst_idxs), f"dst_team_eq_{t}"
            else:
                prob += dst_team[t] == 0, f"dst_team_eq_{t}"

        for t in teams:
            opp = team_to_opp.get(t, "")
            if opp and opp in team_players:
                prob += team_count[opp] <= dst_max_opp_players + 6 * (1 - dst_team[t]), f"dst_caps_opp_{t}_vs_{opp}"

    # correlation: CPT WR/TE implies QB same team (optional hard rule)
    if enforce_cpt_wrte_with_qb:
        for t in teams:
            qb_on_t = pulp.lpSum(selected[i] for i in team_players[t] if players[i].pos == "QB")
            cpt_wrte_on_t = pulp.lpSum(
                x.get((i, "CPT"), 0)
                for i in team_players[t]
                if players[i].pos in ("WR", "TE")
            )
            prob += qb_on_t >= cpt_wrte_on_t, f"cpt_wrte_implies_qb_{t}"

    # discourage CPT QB with same-team K: implement as a penalty (soft) OR hard rule.
    # We'll do it as a soft penalty in the objective.
    # But if you want it hard, uncomment and use:
    # if discourage_cpt_qb_with_same_team_k:
    #   for t in teams:
    #       cpt_qb_t = pulp.lpSum(x.get((i,"CPT"),0) for i in team_players[t] if players[i].pos=="QB")
    #       k_t = pulp.lpSum(selected[i] for i in team_players[t] if players[i].pos=="K")
    #       prob += k_t <= 6*(1-cpt_qb_t), f"no_k_with_cpt_qb_{t}"

    # uniqueness vs previous lineups
    if previous_lineups and min_unique_vs_previous > 0:
        max_overlap = LINEUP_SIZE - min_unique_vs_previous
        for li, prev in enumerate(previous_lineups, start=1):
            prob += pulp.lpSum(selected[i] for i in prev) <= max_overlap, f"uniq_prev_{li}"

    # exposure caps (if counts passed in)
    if cpt_exposure_count is not None and max_cpt_exposure is not None:
        for i in cpt_idx:
            if cpt_exposure_count.get(i, 0) >= max_cpt_exposure:
                prob += x.get((i, "CPT"), 0) == 0, f"cap_cpt_{i}"
    if flex_exposure_count is not None and max_flex_exposure is not None:
        for i in all_idx:
            if flex_exposure_count.get(i, 0) >= max_flex_exposure:
                prob += pulp.lpSum(x[(i, s)] for s in SLOTS[1:]) == 0, f"cap_flex_{i}"

    # objective: CPT 1.5x proj, FLEX 1x proj
    noise = [random.uniform(-randomness, randomness) for _ in range(n)] if randomness > 0 else [0.0] * n
    base_points = pulp.lpSum(
        (1.5 * (players[i].proj + noise[i])) * x.get((i, "CPT"), 0) +
        (players[i].proj + noise[i]) * pulp.lpSum(x[(i, s)] for s in SLOTS[1:])
        for i in all_idx
    )

    # soft penalty to reduce CPT QB + same-team K
    penalty = 0
    if discourage_cpt_qb_with_same_team_k:
        for t in teams:
            cpt_qb_t = pulp.lpSum(x.get((i, "CPT"), 0) for i in team_players[t] if players[i].pos == "QB")
            k_t = pulp.lpSum(selected[i] for i in team_players[t] if players[i].pos == "K")
            penalty += 0.75 * cpt_qb_t * k_t  # small penalty

    prob += base_points - penalty, "objective"

    solver = pulp.PULP_CBC_CMD(
        msg=CBC_MSG,
        timeLimit=CBC_TIME_LIMIT_SEC,
        gapRel=CBC_GAP_REL,
        threads=CBC_THREADS,
    )
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        return None

    # read solution
    chosen: List[int] = []
    # CPT
    cpt_pick = None
    for i in cpt_idx:
        v = x.get((i, "CPT"))
        if v is not None and v.value() == 1:
            cpt_pick = i
            break
    if cpt_pick is None:
        return None
    chosen.append(cpt_pick)

    # FLEX slots
    for s in SLOTS[1:]:
        pick = None
        for i in all_idx:
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
def _lineup_totals(players: List[Player], lineup: List[int]) -> Tuple[int, float]:
    cpt = lineup[0]
    flex = lineup[1:]
    sal = int(round(1.5 * players[cpt].salary + sum(players[i].salary for i in flex)))
    pts = float(round(1.5 * players[cpt].proj + sum(players[i].proj for i in flex), 2))
    return sal, pts


def _team_counts(players: List[Player], lineup: List[int]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for i in lineup:
        t = players[i].team
        d[t] = d.get(t, 0) + 1
    return dict(sorted(d.items(), key=lambda x: (-x[1], x[0])))


def lineups_to_df(players: List[Player], built: List[List[int]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for lu in built:
        cpt = lu[0]
        flex = lu[1:]
        total_salary, total_proj = _lineup_totals(players, lu)
        team_counts = _team_counts(players, lu)
        stack = "-".join(str(v) for v in sorted(team_counts.values(), reverse=True))

        row: Dict[str, object] = {}
        # CPT columns (note: CPT salary/proj shown as MULTIPLIED values)
        row["CPT_name"] = players[cpt].name
        row["CPT_team"] = players[cpt].team
        row["CPT_pos"] = players[cpt].pos
        row["CPT_salary"] = int(round(1.5 * players[cpt].salary))
        row["CPT_proj"] = float(round(1.5 * players[cpt].proj, 2))

        # FLEX columns
        for k, idx in enumerate(flex, start=1):
            p = players[idx]
            row[f"FLEX{k}_name"] = p.name
            row[f"FLEX{k}_team"] = p.team
            row[f"FLEX{k}_pos"] = p.pos
            row[f"FLEX{k}_salary"] = p.salary
            row[f"FLEX{k}_proj"] = float(round(p.proj, 2))

        row["total_salary"] = total_salary
        row["total_proj"] = total_proj
        row["team_counts"] = "; ".join([f"{t}:{c}" for t, c in team_counts.items()])
        row["stack_template"] = stack
        rows.append(row)

    return pd.DataFrame(rows).sort_values("total_proj", ascending=False).reset_index(drop=True)


# -----------------------------
# PUBLIC API (for Flask)
# -----------------------------
def generate_showdown_df(
    num_lineups: int = 20,
    min_unique: int = 2,
    min_salary_spend: int = 44000,
    max_salary_spend: Optional[int] = None,
    randomness: float = 0.8,
    salary_cap: int = SALARY_CAP_DEFAULT,
    csv_url: Optional[str] = None,
    seed: Optional[int] = 7,

    allow_k_cpt: bool = False,
    allow_dst_cpt: bool = False,

    enforce_cpt_wrte_with_qb: bool = True,
    discourage_cpt_qb_with_same_team_k: bool = True,

    max_k_dst_total: int = 2,
    dst_max_opp_players: int = 3,

    max_cpt_exposure: Optional[int] = None,
    max_flex_exposure: Optional[int] = None,

    verbose: bool = False,
) -> pd.DataFrame:
    """
    Returns a DK-style DataFrame for showdown with CPT/FLEX columns.
    """
    url = csv_url or CSV_URL_DEFAULT
    df = fetch_csv_to_df(url)
    players, team_to_opp = parse_players(df, verbose=verbose)

    built: List[List[int]] = []
    prev_sets: List[Set[int]] = []

    cpt_exposure_count: Dict[int, int] = {}
    flex_exposure_count: Dict[int, int] = {}

    # small relax ladder for feasibility
    relaxes = [
        # (uniq_relax, min_sal_relax, randomness_bump)
        (0, 0, 0.0),
        (1, 0, 0.1),
        (2, 800, 0.2),
        (3, 1500, 0.25),
    ]

    for uniq_relax, min_sal_relax, rand_bump in relaxes:
        built = []
        prev_sets = []
        cpt_exposure_count = {}
        flex_exposure_count = {}

        uniq_try = max(0, min_unique - uniq_relax)
        min_sal_try = max(0, min_salary_spend - min_sal_relax)
        rand_try = max(0.0, randomness + rand_bump)

        ok = True
        for li in range(num_lineups):
            lu = optimize_one_showdown(
                players=players,
                team_to_opp=team_to_opp,
                salary_cap=salary_cap,
                min_salary_spend=min_sal_try,
                max_salary_spend=max_salary_spend,
                min_unique_vs_previous=uniq_try,
                previous_lineups=prev_sets,
                randomness=rand_try,
                seed=None if seed is None else seed + li,

                allow_k_cpt=allow_k_cpt,
                allow_dst_cpt=allow_dst_cpt,

                enforce_cpt_wrte_with_qb=enforce_cpt_wrte_with_qb,
                discourage_cpt_qb_with_same_team_k=discourage_cpt_qb_with_same_team_k,

                max_k_dst_total=max_k_dst_total,
                dst_max_opp_players=dst_max_opp_players,

                cpt_exposure_count=cpt_exposure_count if max_cpt_exposure is not None else None,
                flex_exposure_count=flex_exposure_count if max_flex_exposure is not None else None,
                max_cpt_exposure=max_cpt_exposure,
                max_flex_exposure=max_flex_exposure,
            )
            if lu is None:
                ok = False
                break

            built.append(lu)
            prev_sets.append(set(lu))

            # update exposure counts
            cpt_exposure_count[lu[0]] = cpt_exposure_count.get(lu[0], 0) + 1
            for idx in lu[1:]:
                flex_exposure_count[idx] = flex_exposure_count.get(idx, 0) + 1

        if ok and built:
            break

    if not built:
        raise RuntimeError(
            "No showdown lineups generated.\n"
            "Try loosening:\n"
            "  - lower min_salary_spend (e.g. 42000)\n"
            "  - set min_unique lower (0 or 1)\n"
            "  - increase randomness (1.0)\n"
            "  - allow DST CPT if player pool is small\n"
        )

    return lineups_to_df(players, built)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_url", type=str, default=CSV_URL_DEFAULT)
    ap.add_argument("--num_lineups", type=int, default=20)
    ap.add_argument("--salary_cap", type=int, default=SALARY_CAP_DEFAULT)
    ap.add_argument("--min_salary_spend", type=int, default=44000)
    ap.add_argument("--max_salary_spend", type=int, default=-1)
    ap.add_argument("--min_unique", type=int, default=2)
    ap.add_argument("--randomness", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--allow_k_cpt", action="store_true")
    ap.add_argument("--allow_dst_cpt", action="store_true")
    ap.add_argument("--no_cpt_wrte_qb", action="store_true", help="Disable CPT WR/TE => QB same team rule")
    ap.add_argument("--no_qb_k_penalty", action="store_true", help="Disable CPT QB + same-team K penalty")
    ap.add_argument("--max_k_dst_total", type=int, default=2)
    ap.add_argument("--dst_max_opp_players", type=int, default=3)

    ap.add_argument("--max_cpt_exposure", type=int, default=-1)
    ap.add_argument("--max_flex_exposure", type=int, default=-1)

    ap.add_argument("--cbc_msg", action="store_true", help="Show CBC solver output")
    args = ap.parse_args()

    global CBC_MSG
    if args.cbc_msg:
        CBC_MSG = True

    max_sal = None if args.max_salary_spend is None or args.max_salary_spend < 0 else args.max_salary_spend
    max_cpt_exp = None if args.max_cpt_exposure is None or args.max_cpt_exposure < 0 else args.max_cpt_exposure
    max_flex_exp = None if args.max_flex_exposure is None or args.max_flex_exposure < 0 else args.max_flex_exposure

    out_df = generate_showdown_df(
        num_lineups=args.num_lineups,
        min_unique=args.min_unique,
        min_salary_spend=args.min_salary_spend,
        max_salary_spend=max_sal,
        randomness=args.randomness,
        salary_cap=args.salary_cap,
        csv_url=args.csv_url,
        seed=args.seed,

        allow_k_cpt=args.allow_k_cpt,
        allow_dst_cpt=args.allow_dst_cpt,

        enforce_cpt_wrte_with_qb=(not args.no_cpt_wrte_qb),
        discourage_cpt_qb_with_same_team_k=(not args.no_qb_k_penalty),

        max_k_dst_total=args.max_k_dst_total,
        dst_max_opp_players=args.dst_max_opp_players,

        max_cpt_exposure=max_cpt_exp,
        max_flex_exposure=max_flex_exp,

        verbose=True,
    )

    print("\nTop 10 lineups:\n", flush=True)
    print(out_df.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
