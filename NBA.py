# nba_optimizer.py
# NBA DraftKings Classic — fast diversified builder with projected ownership + chalk/sneaky
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
    # (keep your current url here)
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vTF0d2pT0myrD7vjzsB2IrEzMa3o1lylX5_GYyas_5UISsgOud7WffGDxSVq6tJhS45UaxFOX_FolyT/pub?gid=2055904356&single=true&output=csv"
)

DK_SALARY_CAP = 50000
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

MIN_SALARY_SPEND_DEFAULT = 49500
UTIL_SALARY_CAP_DEFAULT: Optional[int] = 5500

# Solver knobs
CBC_TIME_LIMIT_SEC = float(os.environ.get("NBA_CBC_TIME_LIMIT", "1.8"))
CBC_GAP_REL = float(os.environ.get("NBA_CBC_GAP_REL", "0.10"))
CBC_THREADS = int(os.environ.get("NBA_CBC_THREADS", "1"))
CBC_MSG = bool(int(os.environ.get("NBA_CBC_MSG", "0")))

# Per-lineup retries (avoid hangs)
RETRIES_PER_LINEUP = int(os.environ.get("NBA_RETRIES", "2"))

# Relax ladder to prevent long stalls
RELAX_LADDER = [
    # (relax_util_cap, uniq_relax, min_salary_relax)
    (False, 0, 0),
    (True,  0, 0),
    (True,  1, 0),
    (True,  1, 800),
    (True,  2, 1500),
]

# Exposure cap
MAX_PLAYER_EXPOSURE_DEFAULT = 0.45  # max share of lineups any player can appear in (hard ban once exceeded)

# Chalk/sneaky percentiles
CHALK_PCTILE_DEFAULT = 85
SNEAKY_PCTILE_DEFAULT = 25

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
# UTIL
# -----------------------------
_num_re = re.compile(r"[-+]?\d*\.?\d+")

def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    return c.strip("_")

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        k = _norm_col(cand)
        if k in norm:
            return norm[k]
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
    if 0 < v <= 100:
        return int(round(v * 1000))
    return int(round(v))

def fetch_csv_to_df(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def eligible_slots(p: Player) -> Set[str]:
    base = set(p.positions)
    slots = set(base)
    if "PG" in base or "SG" in base:
        slots.add("G")
    if "SF" in base or "PF" in base:
        slots.add("F")
    slots.add("UTIL")
    return slots

def zscore(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=0))
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / sd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _target_total_own_mass(num_games: int) -> float:
    # roughly 8-11 total "ownership points" spread across pool depending on slate size
    g = max(2, min(12, int(num_games)))
    return 8.0 + 0.35 * (g - 1)

# -----------------------------
# OWNERSHIP MODEL (from your stats)
# -----------------------------
def project_ownership(df: pd.DataFrame,
                      salary_col: str,
                      proj_col: str,
                      usage_col: Optional[str],
                      value_col: Optional[str],
                      tmpts_col: Optional[str],
                      ou_col: Optional[str],
                      games: int) -> pd.Series:
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

    # If you have Value column, fold it in (it often mirrors proj/salary, but helps)
    z_valcol = zscore(val) if float(val.abs().sum()) > 0 else pd.Series(np.zeros(len(df)), index=df.index)
    z_tmpts  = zscore(tmpts) if float(tmpts.abs().sum()) > 0 else pd.Series(np.zeros(len(df)), index=df.index)
    z_ou     = zscore(ou) if float(ou.abs().sum()) > 0 else pd.Series(np.zeros(len(df)), index=df.index)

    # cheapness boosts ownership
    z_cheap = zscore(-salary_k)

    # linear score -> sigmoid -> scale to plausible total ownership mass
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
    pred = np.clip(scaled, 0.0, 0.60)  # cap at 60% (reasonable upper bound)
    return pd.Series(pred, index=df.index)

# -----------------------------
# PARSE PLAYERS (auto-detect cols)
# -----------------------------
def parse_players(df: pd.DataFrame,
                  games: int,
                  chalk_pctile: int,
                  sneaky_pctile: int) -> List[Player]:

    name_col = _pick_col(df, ["name", "player", "player_name"])
    pos_col  = _pick_col(df, ["pos", "position", "positions", "dk_position"])
    team_col = _pick_col(df, ["team", "tm", "teamabbr", "team_abbrev", "abbrev"])
    sal_col  = _pick_col(df, ["salary", "sal", "dk_salary", "abbrev_salary", "cost"])
    proj_col = _pick_col(df, ["proj", "projection", "projected_points", "points", "fpts", "dk_fp_projected"])

    # Your sheet often has these (optional):
    usage_col = _pick_col(df, ["final_usage", "usage", "usg", "usuage", "final usage"])
    value_col = _pick_col(df, ["value"])
    tmpts_col = _pick_col(df, ["tm_points", "team_points", "tm pts", "tm points"])
    ou_col    = _pick_col(df, ["o_u", "ou", "o/u", "total", "game_total"])

    required = {"name": name_col, "pos": pos_col, "team": team_col, "salary": sal_col, "proj": proj_col}
    missing = [k for k, v in required.items() if v is None]
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
        tmp, salary_col="salary", proj_col="proj",
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
def contest_profile(contest_type: str):
    ct = (contest_type or "gpp_large").lower().strip()

    # objective weights:
    #   maximize: proj + lev_weight*(proj*(1-own)) - own_penalty*own + noise
    # plus constraints on chalk/sneaky
    if ct in ("cash", "doubleup", "h2h"):
        return dict(min_chalk=3, max_chalk=6, min_sneaky=0, own_penalty=12.0, lev_weight=0.12, randomness=0.35)
    if ct in ("gpp_small", "small", "single_entry", "se"):
        return dict(min_chalk=2, max_chalk=4, min_sneaky=1, own_penalty=22.0, lev_weight=0.22, randomness=0.65)
    if ct in ("gpp_medium", "medium"):
        return dict(min_chalk=2, max_chalk=4, min_sneaky=2, own_penalty=26.0, lev_weight=0.28, randomness=0.80)
    # default: large-field / MME
    return dict(min_chalk=1, max_chalk=3, min_sneaky=2, own_penalty=30.0, lev_weight=0.35, randomness=0.95)

# -----------------------------
# OPTIMIZE ONE LINEUP
# -----------------------------
def optimize_one_lineup(players: List[Player],
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
                        seed: Optional[int]) -> Optional[List[int]]:

    if seed is not None:
        random.seed(seed)

    n = len(players)
    elig = [eligible_slots(p) for p in players]

    # decision vars by (player, slot) only if eligible
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

    # simple NBA sanity: at least one PG and one PF
    prob += pulp.lpSum(selected[i] for i in range(n) if "PG" in players[i].positions) >= 1, "need_pg"
    prob += pulp.lpSum(selected[i] for i in range(n) if "PF" in players[i].positions) >= 1, "need_pf"

    # util salary cap optional
    if util_salary_cap is not None:
        prob += pulp.lpSum(players[i].salary * x[(i, "UTIL")] for i in range(n) if (i, "UTIL") in x) <= util_salary_cap, "util_cap"

    # chalk/sneaky constraints
    chalk_sum = pulp.lpSum(players[i].is_chalk * selected[i] for i in range(n))
    sneaky_sum = pulp.lpSum(players[i].is_sneaky * selected[i] for i in range(n))
    prob += chalk_sum >= min_chalk, "min_chalk"
    prob += chalk_sum <= max_chalk, "max_chalk"
    prob += sneaky_sum >= min_sneaky, "min_sneaky"

    # uniqueness vs previous
    if previous_sets and min_unique > 0:
        max_overlap = len(DK_SLOTS) - min_unique
        for j, prev in enumerate(previous_sets, start=1):
            prob += pulp.lpSum(selected[i] for i in prev) <= max_overlap, f"uniq_{j}"

    # objective
    noise = [random.uniform(-randomness, randomness) for _ in range(n)] if randomness > 0 else [0.0] * n
    prob += pulp.lpSum(
        (
            players[i].proj
            - own_penalty * players[i].pred_own
            + lev_weight * (players[i].proj * (1.0 - players[i].pred_own))
            + noise[i]
        ) * selected[i]
        for i in range(n)
    )

    solver = pulp.PULP_CBC_CMD(
        msg=CBC_MSG,
        timeLimit=CBC_TIME_LIMIT_SEC,
        gapRel=CBC_GAP_REL,
        threads=CBC_THREADS
    )
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        return None

    chosen = []
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

        for slot, idx in zip(DK_SLOTS, lu):
            p = players[idx]
            row[f"{slot}_name"] = p.name
            row[f"{slot}_team"] = p.team
            row[f"{slot}_salary"] = int(p.salary)
            row[f"{slot}_proj"] = float(round(p.proj, 2))

            # NEW: per-slot ownership + chalk/sneaky flags
            row[f"{slot}_own_pct"] = float(round(100.0 * p.pred_own, 1))
            row[f"{slot}_is_chalk"] = int(p.is_chalk)
            row[f"{slot}_is_sneaky"] = int(p.is_sneaky)

            total_sal += int(p.salary)
            total_proj += float(p.proj)
            if p.team:
                team_counts[p.team] = team_counts.get(p.team, 0) + 1
            chalk_ct += int(p.is_chalk)
            sneaky_ct += int(p.is_sneaky)

        row["team_counts"] = "; ".join([f"{t}:{c}" for t, c in sorted(team_counts.items(), key=lambda x: (-x[1], x[0]))])
        row["total_salary"] = total_sal
        row["total_proj"] = round(total_proj, 2)
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
    min_unique: int = 2,
    min_salary_spend: int = MIN_SALARY_SPEND_DEFAULT,
    randomness: float = 0.8,
    salary_cap: int = DK_SALARY_CAP,
    csv_url: Optional[str] = None,
    games: int = 7,
    util_salary_cap: Optional[int] = UTIL_SALARY_CAP_DEFAULT,
    chalk_pctile: int = CHALK_PCTILE_DEFAULT,
    sneaky_pctile: int = SNEAKY_PCTILE_DEFAULT,
    max_player_exposure: float = MAX_PLAYER_EXPOSURE_DEFAULT,
    contest_type: str = "gpp_large",
    seed: Optional[int] = 7,
) -> pd.DataFrame:

    url = csv_url or CSV_URL_DEFAULT
    df = fetch_csv_to_df(url)
    players = parse_players(df, games=games, chalk_pctile=chalk_pctile, sneaky_pctile=sneaky_pctile)

    prof = contest_profile(contest_type)
    # allow form to override base randomness by adding requested randomness
    base_random = float(prof["randomness"])
    rand_use = max(0.0, float(randomness))  # user input
    # final randomness is blended
    randomness_final = 0.6 * base_random + 0.4 * rand_use

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
    cap_ct = max(1, int(np.floor(max(0.05, min(1.0, max_player_exposure)) * num_lineups)))
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

        for li in range(num_lineups):
            banned = {i for i, ct in exposure.items() if ct >= cap_ct}

            built = None
            for attempt in range(RETRIES_PER_LINEUP + 1):
                # small ramp in randomness per attempt to shake solver loose
                rand_i = randomness_final + 0.12 * attempt

                built = optimize_one_lineup(
                    players=players,
                    salary_cap=salary_cap,
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
                    seed=None if seed is None else seed + li * 10 + attempt
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
            "Try: lower min_salary_spend, set min_unique=0/1, increase util_salary_cap, or use contest_type=cash."
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
    ap.add_argument("--min_unique", type=int, default=2)
    ap.add_argument("--randomness", type=float, default=0.8)
    ap.add_argument("--util_salary_cap", type=int, default=UTIL_SALARY_CAP_DEFAULT if UTIL_SALARY_CAP_DEFAULT else -1)

    ap.add_argument("--contest_type", type=str, default="gpp_large",
                    help="cash | gpp_small | gpp_medium | gpp_large (default)")

    ap.add_argument("--chalk_pctile", type=int, default=CHALK_PCTILE_DEFAULT)
    ap.add_argument("--sneaky_pctile", type=int, default=SNEAKY_PCTILE_DEFAULT)
    ap.add_argument("--max_player_exposure", type=float, default=MAX_PLAYER_EXPOSURE_DEFAULT)
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
        contest_type=args.contest_type,
        seed=args.seed
    )

    # show key new columns
    cols = ["total_proj", "total_salary", "chalk_ct", "sneaky_ct", "contest_type", "team_counts"]
    print(out_df[cols].head(20).to_string(index=False))
    print("\nSample slot columns include: PG_own_pct, PG_is_chalk, PG_is_sneaky ...")

if __name__ == "__main__":
    main()
