"""
NFL DraftKings SHOWDOWN — DIVERSIFIED BUILDER (ONE FILE)

✅ Pulls your Google Sheet CSV (gid=1791525610)
✅ CPT slot: salary * 1.5 AND projected points * 1.5
✅ Roster: 1 CPT + 5 FLEX (total 6)
✅ Salary cap: 50,000
✅ No duplicates (CPT can't also be FLEX)
✅ Exposure caps across lineups + real uniqueness
✅ Optional team rules:
   - max_players_per_team (default 5 = basically off)
   - optional "no CPT vs DST" style toggles (off by default)

Install:
  pip install pandas requests pulp numpy

Run:
  python -u NFL_Showdown.py --num_lineups 20
"""

import argparse
import math
import os
import random
import re
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np
import pulp
import requests


# -----------------------------
# DEFAULTS
# -----------------------------
CSV_URL_DEFAULT = os.environ.get(
    "NFL_SD_CSV_URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vSExcKi8LiRgnZpx9JeIRpgFMfYCFxgfdixu6oZxD2FhUG5UwyI86QDYC1ImTPAIPGdDMizdrYWSWP3/pub?gid=1791525610&single=true&output=csv",
)

SALARY_CAP = 50000
ROSTER_SLOTS = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]

# CPT multipliers (DK showdown)
CPT_SAL_MULT = 1.5
CPT_PROJ_MULT = 1.5

# Speed knobs
CBC_TIME_LIMIT_SEC = float(os.environ.get("NFL_SD_CBC_TIME_LIMIT", "2.0"))
RETRIES_PER_LINEUP = int(os.environ.get("NFL_SD_RETRIES", "4"))
RELAX_LADDER = [
    # (uniq_relax_steps, min_salary_relax)
    (0, 0),
    (1, 0),
    (1, 500),
    (2, 1200),
]

# Diversity defaults
MAX_PLAYER_EXPOSURE_DEFAULT = 0.55     # hard cap exposure (ban over cap)
SOFT_EXPOSURE_PENALTY_DEFAULT = 0.0    # optional soft penalty, set to e.g. 6.0
DEFAULT_MIN_UNIQUE = 2

DEFAULT_MIN_SALARY_SPEND = 48500  # showdown usually spends high; relax ladder will lower if needed


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
# UTIL
# -----------------------------
def _normalize_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    return c.strip("_")


def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    inv: Dict[str, List[str]] = {}
    for orig in df.columns:
        inv.setdefault(_normalize_col(orig), []).append(orig)
    for cand in candidates:
        key = _normalize_col(cand)
        if key in inv:
            return inv[key][0]
    return None


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
    v = _to_float(x, default=float(default))
    if v <= 0:
        return default
    # allow 6.2 -> 6200 shorthand
    if v <= 100:
        return int(round(v * 1000))
    return int(round(v))


def clean_text(x: str) -> str:
    s = re.sub(r"&nbsp;?", " ", str(x))
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def normalize_pos(p: str) -> str:
    p = clean_text(p).upper()
    p = p.replace("D/ST", "DST").replace("D_ST", "DST").replace("DST/ST", "DST")
    if p in ("DEF", "DEFENSE") or "DST" in p:
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


def fetch_csv_to_df(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


# -----------------------------
# PARSE PLAYERS
# -----------------------------
def parse_players(df: pd.DataFrame) -> Tuple[List[Player], pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # try common columns
    name_col = _pick_first(df, ["name", "player", "player_name", "nickname"])
    pos_col = _pick_first(df, ["pos", "position"])
    team_col = _pick_first(df, ["team", "teamabbr", "team_abbrev", "tm"])
    sal_col = _pick_first(df, ["salary", "sal", "dk_salary", "cost"])
    proj_col = _pick_first(df, ["proj", "projection", "fpts", "fp", "points", "projected_points", "dk_fp_projected"])
    opp_col = _pick_first(df, ["opp", "opponent", "opponent_team", "vs"])

    missing = [k for k, v in {"name": name_col, "pos": pos_col, "team": team_col, "salary": sal_col, "proj": proj_col}.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    df["name"] = df[name_col].apply(clean_text)
    df["pos"] = df[pos_col].apply(normalize_pos)
    df["team"] = df[team_col].apply(clean_text)
    df["salary"] = df[sal_col].apply(lambda v: _to_salary(v, 0))
    df["proj"] = df[proj_col].apply(lambda v: _to_float(v, 0.0))
    df["opp"] = df[opp_col].apply(clean_text) if opp_col else ""

    df = df[(df["name"] != "") & (df["team"] != "") & (df["pos"] != "")]
    df = df[(df["salary"] > 0) & (df["proj"] > 0)]
    if df.empty:
        raise ValueError("No valid players after cleanup.")

    players: List[Player] = [
        Player(
            name=str(r["name"]),
            team=str(r["team"]),
            pos=str(r["pos"]),
            salary=int(r["salary"]),
            proj=float(r["proj"]),
            opp=str(r["opp"]) if isinstance(r["opp"], str) else "",
        )
        for _, r in df.iterrows()
    ]

    # TEAM->OPP mapping for optional constraints
    team_to_opp: Dict[str, str] = {}
    for p in players:
        if p.opp and p.team not in team_to_opp:
            team_to_opp[p.team] = p.opp
    for a, b in list(team_to_opp.items()):
        if b and b not in team_to_opp:
            team_to_opp[b] = a

    analysis = df[["name", "team", "pos", "opp", "salary", "proj"]].copy()
    return players, analysis, team_to_opp


# -----------------------------
# OPTIMIZE ONE SHOWDOWN LINEUP
# -----------------------------
def optimize_one_showdown(
    players: List[Player],
    salary_cap: int,
    min_salary_spend: int,
    min_unique_vs_previous: int,
    prev_player_sets: List[Set[int]],
    randomness: float,
    seed: Optional[int],
    banned_players: Optional[Set[int]] = None,
    soft_exposure_penalty: float = 0.0,
    exposure_counts: Optional[Dict[int, int]] = None,
    num_lineups_total: int = 0,
    max_players_per_team: int = 5,
) -> Optional[Tuple[int, List[int]]]:
    """
    Returns (cpt_idx, flex_idxs[5]) or None
    """
    if seed is not None:
        random.seed(seed)

    n = len(players)
    if n < 6:
        return None

    teams = sorted(set(p.team for p in players))
    team_players: Dict[str, List[int]] = {t: [] for t in teams}
    for i, p in enumerate(players):
        team_players[p.team].append(i)

    # decision variables
    cpt = [pulp.LpVariable(f"cpt_{i}", cat="Binary") for i in range(n)]
    flex = [pulp.LpVariable(f"flex_{i}", cat="Binary") for i in range(n)]

    prob = pulp.LpProblem("NFL_SHOWDOWN", pulp.LpMaximize)

    # exactly 1 CPT
    prob += pulp.lpSum(cpt) == 1, "one_cpt"
    # exactly 5 flex
    prob += pulp.lpSum(flex) == 5, "five_flex"
    # cannot use same player twice
    for i in range(n):
        prob += cpt[i] + flex[i] <= 1, f"no_dup_{i}"

    # salary cap with CPT 1.5x
    total_salary = pulp.lpSum(players[i].salary * (CPT_SAL_MULT * cpt[i] + 1.0 * flex[i]) for i in range(n))
    prob += total_salary <= salary_cap, "salary_cap"
    prob += total_salary >= min_salary_spend, "min_salary_spend"

    # team max (optional; default 5 is basically "no restriction" because roster size is 6)
    if max_players_per_team is not None:
        for t, idxs in team_players.items():
            prob += pulp.lpSum(cpt[i] + flex[i] for i in idxs) <= max_players_per_team, f"team_max_{t}"

    # hard bans (exposure cap)
    if banned_players:
        for i in banned_players:
            if 0 <= i < n:
                prob += cpt[i] == 0, f"ban_cpt_{i}"
                prob += flex[i] == 0, f"ban_flex_{i}"

    # uniqueness vs previous lineups (compare actual selected player indices)
    if prev_player_sets and min_unique_vs_previous > 0:
        max_overlap = 6 - min_unique_vs_previous
        for li, prev in enumerate(prev_player_sets, start=1):
            prob += pulp.lpSum((cpt[i] + flex[i]) for i in prev) <= max_overlap, f"uniq_{li}"

    # objective: proj with CPT 1.5x + noise
    noise = [random.uniform(-randomness, randomness) for _ in range(n)] if randomness > 0 else [0.0] * n
    base_obj = pulp.lpSum(
        (players[i].proj * (CPT_PROJ_MULT * cpt[i] + 1.0 * flex[i]) + noise[i] * (cpt[i] + flex[i]))
        for i in range(n)
    )

    # optional soft exposure penalty
    exp_term = 0
    if soft_exposure_penalty > 0 and exposure_counts is not None and num_lineups_total > 0:
        for i in range(n):
            rate = exposure_counts.get(i, 0) / float(num_lineups_total)
            exp_term += (-soft_exposure_penalty * rate) * (cpt[i] + flex[i])

    prob += base_obj + exp_term, "objective"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=CBC_TIME_LIMIT_SEC)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        return None

    cpt_idx = None
    flex_idxs: List[int] = []
    for i in range(n):
        if cpt[i].value() == 1:
            cpt_idx = i
        if flex[i].value() == 1:
            flex_idxs.append(i)

    if cpt_idx is None or len(flex_idxs) != 5:
        return None

    return cpt_idx, flex_idxs


# -----------------------------
# OUTPUT
# -----------------------------
def summarize_showdown(players: List[Player], cpt_idx: int, flex_idxs: List[int]) -> Dict[str, object]:
    chosen = [cpt_idx] + flex_idxs
    salary = int(round(players[cpt_idx].salary * CPT_SAL_MULT + sum(players[i].salary for i in flex_idxs)))
    proj = float(round(players[cpt_idx].proj * CPT_PROJ_MULT + sum(players[i].proj for i in flex_idxs), 2))

    team_counts: Dict[str, int] = {}
    for i in chosen:
        t = players[i].team
        team_counts[t] = team_counts.get(t, 0) + 1
    team_counts = dict(sorted(team_counts.items(), key=lambda x: (-x[1], x[0])))

    return {"total_salary": salary, "total_proj": proj, "team_counts": "; ".join([f"{t}:{c}" for t, c in team_counts.items()])}


def lineups_to_df(players: List[Player], built: List[Tuple[int, List[int]]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cpt_idx, flex_idxs in built:
        row: Dict[str, object] = {}

        cpt = players[cpt_idx]
        row["CPT_name"] = cpt.name
        row["CPT_team"] = cpt.team
        row["CPT_pos"] = cpt.pos
        row["CPT_salary"] = int(round(cpt.salary * CPT_SAL_MULT))
        row["CPT_proj"] = float(round(cpt.proj * CPT_PROJ_MULT, 2))

        for j, idx in enumerate(flex_idxs, start=1):
            p = players[idx]
            row[f"FLEX{j}_name"] = p.name
            row[f"FLEX{j}_team"] = p.team
            row[f"FLEX{j}_pos"] = p.pos
            row[f"FLEX{j}_salary"] = int(p.salary)
            row[f"FLEX{j}_proj"] = float(round(p.proj, 2))

        meta = summarize_showdown(players, cpt_idx, flex_idxs)
        row["total_salary"] = meta["total_salary"]
        row["total_proj"] = meta["total_proj"]
        row["team_counts"] = meta["team_counts"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("total_proj", ascending=False).reset_index(drop=True)


# -----------------------------
# PUBLIC API (for Flask)
# -----------------------------
def generate_nfl_showdown_df(
    num_lineups: int = 20,
    min_unique: int = DEFAULT_MIN_UNIQUE,
    min_salary_spend: int = DEFAULT_MIN_SALARY_SPEND,
    randomness: float = 1.0,
    salary_cap: int = SALARY_CAP,
    csv_url: Optional[str] = None,
    seed: Optional[int] = 7,
    max_player_exposure: float = MAX_PLAYER_EXPOSURE_DEFAULT,
    soft_exposure_penalty: float = SOFT_EXPOSURE_PENALTY_DEFAULT,
    max_players_per_team: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    url = csv_url or CSV_URL_DEFAULT
    df = fetch_csv_to_df(url)
    players, _analysis, _team_to_opp = parse_players(df)

    n = len(players)
    counts: Dict[int, int] = {i: 0 for i in range(n)}
    cap = max(1, int(math.floor(max(0.05, min(1.0, max_player_exposure)) * num_lineups)))

    built: List[Tuple[int, List[int]]] = []
    prev_sets: List[Set[int]] = []

    for uniq_relax, min_sal_relax in RELAX_LADDER:
        built = []
        prev_sets = []
        counts = {i: 0 for i in range(n)}

        uniq_try = max(0, min_unique - uniq_relax)
        min_sal_try = max(0, min_salary_spend - min_sal_relax)

        for li in range(num_lineups):
            banned = {i for i, ct in counts.items() if ct >= cap}

            lu = None
            for attempt in range(RETRIES_PER_LINEUP):
                rand_i = randomness + 0.15 * attempt + 0.20 * (li / max(1, num_lineups - 1))

                lu = optimize_one_showdown(
                    players=players,
                    salary_cap=salary_cap,
                    min_salary_spend=min_sal_try,
                    min_unique_vs_previous=uniq_try,
                    prev_player_sets=prev_sets,
                    randomness=rand_i,
                    seed=None if seed is None else seed + li * 10 + attempt,
                    banned_players=banned,
                    soft_exposure_penalty=soft_exposure_penalty,
                    exposure_counts=counts,
                    num_lineups_total=num_lineups,
                    max_players_per_team=max_players_per_team,
                )
                if lu is not None:
                    break

            if lu is None:
                break

            cpt_idx, flex_idxs = lu
            built.append((cpt_idx, flex_idxs))

            chosen_set = set([cpt_idx] + flex_idxs)
            prev_sets.append(chosen_set)
            for idx in chosen_set:
                counts[idx] += 1

            if verbose:
                meta = summarize_showdown(players, cpt_idx, flex_idxs)
                print(f"Lineup {li+1}/{num_lineups} | salary={meta['total_salary']} proj={meta['total_proj']} teams={meta['team_counts']}", flush=True)

        if built:
            break

    if not built:
        raise RuntimeError(
            "No showdown lineups generated.\n"
            "Try: lower min_salary_spend (e.g. 47000), set min_unique=1, increase randomness, or raise max_player_exposure."
        )

    return lineups_to_df(players, built)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_url", type=str, default=CSV_URL_DEFAULT)
    ap.add_argument("--num_lineups", type=int, default=20)
    ap.add_argument("--min_unique", type=int, default=DEFAULT_MIN_UNIQUE)
    ap.add_argument("--salary_cap", type=int, default=SALARY_CAP)
    ap.add_argument("--min_salary_spend", type=int, default=DEFAULT_MIN_SALARY_SPEND)
    ap.add_argument("--randomness", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--max_player_exposure", type=float, default=MAX_PLAYER_EXPOSURE_DEFAULT)
    ap.add_argument("--soft_exposure_penalty", type=float, default=SOFT_EXPOSURE_PENALTY_DEFAULT)
    ap.add_argument("--max_players_per_team", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out = generate_nfl_showdown_df(
        num_lineups=args.num_lineups,
        min_unique=args.min_unique,
        min_salary_spend=args.min_salary_spend,
        randomness=args.randomness,
        salary_cap=args.salary_cap,
        csv_url=args.csv_url,
        seed=args.seed,
        max_player_exposure=args.max_player_exposure,
        soft_exposure_penalty=args.soft_exposure_penalty,
        max_players_per_team=args.max_players_per_team,
        verbose=args.verbose,
    )

    print(out.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
