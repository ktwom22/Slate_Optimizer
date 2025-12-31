# nhl_showdown.py
# NHL DraftKings Showdown (ONE FILE) — CPT + 5 FLEX
#
# ✅ Uses YOUR CSV (fresh fetch every click: cache-bust + no-cache + fingerprint)
# ✅ DK Showdown roster: CPT, FLEX1..FLEX5 (6 total)
# ✅ Salary cap: 50,000 (CPT counts 1.5x salary + 1.5x projection)
# ✅ Projects ownership from your sheet columns (ppg_projection, salary, lines, totals, implied score, fppg/value)
# ✅ Adds chalk/sneaky tiers + per-slot own% + lineup chalk/sneaky counts
# ✅ Contest profiles: cash | gpp_small | gpp_medium | gpp_large
#
# Install:
#   pip install pandas requests pulp numpy
#
# CLI:
#   python -u nhl_showdown.py --num_lineups 20 --contest_type gpp_large
#
# Flask:
#   import nhl_showdown as NHLSD
#   df = NHLSD.generate_nhl_showdown_df(num_lineups=20, contest_type="gpp_large")

import argparse
import hashlib
import os
import random
import re
import time
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import numpy as np
import pandas as pd
import pulp
import requests

# -----------------------------
# DEFAULTS / ENV
# -----------------------------
CSV_URL_DEFAULT = os.environ.get(
    "NHL_SD_CSV_URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRcEmUjeqWwtnYJFyT1T8CFRWR-sd-NEPZv4rZZ-BK8Rx3CYtWhDHb9ZbNMhkiaExaPHeqt-eoPNH2-/pub?gid=1320973499&single=true&output=csv"
)

DK_SALARY_CAP = 50000
DK_SLOTS = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]
LINEUP_SIZE = len(DK_SLOTS)

CPT_SAL_MULT = 1.5
CPT_PROJ_MULT = 1.5

# Solver knobs
CBC_TIME_LIMIT_SEC = float(os.environ.get("NHL_SD_CBC_TIME_LIMIT", "2.2"))
CBC_GAP_REL = float(os.environ.get("NHL_SD_CBC_GAP_REL", "0.10"))
CBC_THREADS = int(os.environ.get("NHL_SD_CBC_THREADS", "1"))
CBC_MSG = bool(int(os.environ.get("NHL_SD_CBC_MSG", "0")))

RETRIES_PER_LINEUP = int(os.environ.get("NHL_SD_RETRIES", "3"))

# Diversity knobs
MAX_PLAYER_EXPOSURE_DEFAULT = 0.55

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
    opp: str
    pos: str          # C/W/D/G
    salary: int
    proj: float       # ppg_projection

    reg_line: float
    pp_line: float

    implied_team_score: float
    game_total: float   # over_under

    l5: float
    l10: float
    season: float
    value: float

    pred_own: float   # 0..1
    is_chalk: int
    is_sneaky: int
    is_goalie: int    # 1 if goalie

# -----------------------------
# HELPERS
# -----------------------------
_num_re = re.compile(r"[-+]?\d*\.?\d+")

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

def zscore(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=0))
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / sd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _cache_bust(url: str) -> str:
    u = urlparse(url)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q["_cb"] = str(int(time.time() * 1000))
    return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(q), u.fragment))

def fetch_csv_to_df(url: str) -> pd.DataFrame:
    bust_url = _cache_bust(url)
    headers = {"Cache-Control": "no-cache", "Pragma": "no-cache"}
    r = requests.get(bust_url, timeout=30, headers=headers)
    r.raise_for_status()
    txt = r.text
    sig = hashlib.md5(txt.encode("utf-8")).hexdigest()[:10]
    print(f"[NHL SD] fetched bytes={len(txt)} md5={sig}", flush=True)
    return pd.read_csv(StringIO(txt))

def _target_total_own_mass(num_players: int) -> float:
    # Showdown ownership concentrates: aim ~6–8 ownership points across player pool
    p = max(18, min(70, int(num_players)))
    return 6.0 + 0.03 * (p - 18)

# -----------------------------
# OWNERSHIP MODEL (from YOUR stats)
# -----------------------------
def project_ownership(tmp: pd.DataFrame) -> pd.Series:
    # Core:
    proj = tmp["proj"].astype(float)
    sal = tmp["salary"].astype(float) / 1000.0
    value_from_proj = np.where(sal > 0, proj / sal, 0.0)

    # Lines: PP1 / Line1 tends to be chalkier. Lower is better.
    reg_line = tmp["reg_line"].astype(float)
    pp_line = tmp["pp_line"].astype(float)

    # Game context:
    its = tmp["implied_team_score"].astype(float)
    total = tmp["game_total"].astype(float)

    # Recent form:
    l5 = tmp["l5"].astype(float)
    l10 = tmp["l10"].astype(float)
    season = tmp["season"].astype(float)
    value_col = tmp["value"].astype(float)

    z_proj = zscore(proj)
    z_val  = zscore(pd.Series(value_from_proj, index=tmp.index))
    z_cheap = zscore(-sal)

    # Better lines => higher own: use -line as a positive signal
    z_reg = zscore(-reg_line) if reg_line.abs().sum() > 0 else pd.Series(np.zeros(len(tmp)), index=tmp.index)
    z_pp  = zscore(-pp_line) if pp_line.abs().sum() > 0 else pd.Series(np.zeros(len(tmp)), index=tmp.index)

    z_its = zscore(its) if its.abs().sum() > 0 else pd.Series(np.zeros(len(tmp)), index=tmp.index)
    z_tot = zscore(total) if total.abs().sum() > 0 else pd.Series(np.zeros(len(tmp)), index=tmp.index)

    z_l5 = zscore(l5) if l5.abs().sum() > 0 else pd.Series(np.zeros(len(tmp)), index=tmp.index)
    z_l10 = zscore(l10) if l10.abs().sum() > 0 else pd.Series(np.zeros(len(tmp)), index=tmp.index)
    z_season = zscore(season) if season.abs().sum() > 0 else pd.Series(np.zeros(len(tmp)), index=tmp.index)
    z_valuecol = zscore(value_col) if value_col.abs().sum() > 0 else pd.Series(np.zeros(len(tmp)), index=tmp.index)

    # Goalies: usually lower than elite skaters, but starting goalies can be popular.
    is_goalie = tmp["is_goalie"].astype(int)
    goalie_boost = pd.Series(np.where(is_goalie == 1, 0.15, 0.0), index=tmp.index)

    lin = (
        1.05 * z_val +
        0.85 * z_proj +
        0.30 * z_cheap +
        0.30 * z_pp +
        0.18 * z_reg +
        0.20 * z_its +
        0.10 * z_tot +
        0.10 * z_l5 +
        0.06 * z_l10 +
        0.05 * z_season +
        0.10 * z_valuecol +
        goalie_boost
    )

    base = sigmoid(lin)
    base = np.clip(base, 0.001, 0.999)

    mass = _target_total_own_mass(len(tmp))
    scaled = base * (mass / float(base.sum()))
    pred = np.clip(scaled, 0.0, 0.70)
    return pd.Series(pred, index=tmp.index)

# -----------------------------
# PARSE PLAYERS (YOUR HEADERS)
# -----------------------------
def parse_players(df: pd.DataFrame, chalk_pctile: int, sneaky_pctile: int) -> List[Player]:
    required = [
        "first_name","last_name","position","reg_line","pp_line","team","opp",
        "salary","ppg_projection","L5_fppg_avg","L10_fppg_avg","szn_fppg_avg","value_projection",
        "over_under","implied_team_score","starting_goalie"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}\nFound: {list(df.columns)}")

    tmp = df.copy()

    tmp["name"] = (tmp["first_name"].astype(str).apply(_clean) + " " + tmp["last_name"].astype(str).apply(_clean)).str.strip()
    tmp["team"] = tmp["team"].astype(str).apply(_clean)
    tmp["opp"] = tmp["opp"].astype(str).apply(_clean)

    tmp["pos"] = tmp["position"].astype(str).apply(_clean).str.upper()
    # normalize
    tmp.loc[tmp["pos"].isin(["LW","RW"]), "pos"] = "W"

    tmp["salary"] = tmp["salary"].apply(lambda x: _to_int(x, 0))
    tmp["proj"] = tmp["ppg_projection"].apply(lambda x: _to_float(x, 0.0))

    tmp["reg_line"] = tmp["reg_line"].apply(lambda x: _to_float(x, 0.0))
    tmp["pp_line"] = tmp["pp_line"].apply(lambda x: _to_float(x, 0.0))

    tmp["game_total"] = tmp["over_under"].apply(lambda x: _to_float(x, 0.0))
    tmp["implied_team_score"] = tmp["implied_team_score"].apply(lambda x: _to_float(x, 0.0))

    tmp["l5"] = tmp["L5_fppg_avg"].apply(lambda x: _to_float(x, 0.0))
    tmp["l10"] = tmp["L10_fppg_avg"].apply(lambda x: _to_float(x, 0.0))
    tmp["season"] = tmp["szn_fppg_avg"].apply(lambda x: _to_float(x, 0.0))
    tmp["value"] = tmp["value_projection"].apply(lambda x: _to_float(x, 0.0))

    # goalie flag:
    # - position == G
    # - OR starting_goalie non-empty / truthy
    tmp["starting_goalie"] = tmp["starting_goalie"].astype(str).apply(_clean)
    tmp["is_goalie"] = ((tmp["pos"] == "G") | (tmp["starting_goalie"] != "")).astype(int)

    # filter valid
    tmp = tmp[(tmp["name"] != "") & (tmp["team"] != "") & (tmp["opp"] != "")]
    tmp = tmp[(tmp["salary"] > 0) & (tmp["proj"] > 0)]
    tmp = tmp[tmp["pos"].isin(["C","W","D","G"])]
    if tmp.empty:
        raise ValueError("No valid players after cleanup.")

    tmp["pred_own"] = project_ownership(tmp)
    tmp["pred_own_pct"] = (100.0 * tmp["pred_own"]).round(1)

    own_pcts = tmp["pred_own_pct"].astype(float)
    chalk_cut = float(np.percentile(own_pcts, chalk_pctile))
    sneaky_cut = float(np.percentile(own_pcts, sneaky_pctile))

    tmp["is_chalk"] = (tmp["pred_own_pct"] >= chalk_cut).astype(int)
    tmp["is_sneaky"] = (tmp["pred_own_pct"] <= sneaky_cut).astype(int)

    players: List[Player] = []
    for _, r in tmp.iterrows():
        players.append(Player(
            name=str(r["name"]),
            team=str(r["team"]),
            opp=str(r["opp"]),
            pos=str(r["pos"]),
            salary=int(r["salary"]),
            proj=float(r["proj"]),
            reg_line=float(r["reg_line"]),
            pp_line=float(r["pp_line"]),
            implied_team_score=float(r["implied_team_score"]),
            game_total=float(r["game_total"]),
            l5=float(r["l5"]),
            l10=float(r["l10"]),
            season=float(r["season"]),
            value=float(r["value"]),
            pred_own=float(r["pred_own"]),
            is_chalk=int(r["is_chalk"]),
            is_sneaky=int(r["is_sneaky"]),
            is_goalie=int(r["is_goalie"]),
        ))

    return players

# -----------------------------
# CONTEST PROFILES
# -----------------------------
def contest_profile(contest_type: str) -> Dict[str, float]:
    ct = (contest_type or "gpp_large").lower().strip()
    # chalk/sneaky totals across 6 players
    if ct in ("cash", "h2h", "doubleup"):
        return dict(min_chalk=2, max_chalk=5, min_sneaky=0, own_penalty=10.0, lev_weight=0.10, base_random=0.25)
    if ct in ("gpp_small", "small", "single_entry", "se"):
        return dict(min_chalk=1, max_chalk=3, min_sneaky=1, own_penalty=18.0, lev_weight=0.20, base_random=0.55)
    if ct in ("gpp_medium", "medium"):
        return dict(min_chalk=1, max_chalk=3, min_sneaky=2, own_penalty=22.0, lev_weight=0.26, base_random=0.75)
    return dict(min_chalk=0, max_chalk=2, min_sneaky=2, own_penalty=26.0, lev_weight=0.32, base_random=0.90)

# -----------------------------
# OPTIMIZE ONE LINEUP
# -----------------------------
def optimize_one_showdown(
    players: List[Player],
    salary_cap: int,
    min_salary_spend: int,
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
) -> Optional[List[int]]:
    if seed is not None:
        random.seed(seed)

    n = len(players)

    cpt = {i: pulp.LpVariable(f"cpt_{i}", cat="Binary") for i in range(n) if i not in banned}
    flex = {i: pulp.LpVariable(f"flex_{i}", cat="Binary") for i in range(n) if i not in banned}

    prob = pulp.LpProblem("NHL_SHOWDOWN", pulp.LpMaximize)

    prob += pulp.lpSum(cpt[i] for i in cpt) == 1, "one_cpt"
    prob += pulp.lpSum(flex[i] for i in flex) == 5, "five_flex"

    for i in range(n):
        if i in banned:
            continue
        prob += cpt.get(i, 0) + flex.get(i, 0) <= 1, f"no_double_{i}"

    # Salary (CPT 1.5x)
    total_salary = pulp.lpSum(players[i].salary * flex.get(i, 0) for i in range(n)) + \
                   pulp.lpSum((players[i].salary * CPT_SAL_MULT) * cpt.get(i, 0) for i in range(n))
    prob += total_salary <= salary_cap, "cap"
    prob += total_salary >= min_salary_spend, "min_spend"

    # 2 teams represented
    teams = sorted(set(p.team for p in players))
    team_vars = {t: pulp.LpVariable(f"team_{t}", cat="Binary") for t in teams}
    for t in teams:
        idxs = [i for i in range(n) if players[i].team == t and i not in banned]
        if idxs:
            prob += pulp.lpSum(cpt.get(i, 0) + flex.get(i, 0) for i in idxs) >= team_vars[t], f"team_on_{t}"
        else:
            prob += team_vars[t] == 0, f"team_zero_{t}"
    prob += pulp.lpSum(team_vars[t] for t in teams) >= 2, "two_teams"

    def sel(i):
        return cpt.get(i, 0) + flex.get(i, 0)

    chalk_sum = pulp.lpSum(players[i].is_chalk * sel(i) for i in range(n))
    sneaky_sum = pulp.lpSum(players[i].is_sneaky * sel(i) for i in range(n))
    prob += chalk_sum >= min_chalk, "min_chalk"
    prob += chalk_sum <= max_chalk, "max_chalk"
    prob += sneaky_sum >= min_sneaky, "min_sneaky"

    # Uniqueness (overlap cap)
    if previous_sets and min_unique > 0:
        max_overlap = LINEUP_SIZE - min_unique
        for j, prev in enumerate(previous_sets, start=1):
            prob += pulp.lpSum(sel(i) for i in prev) <= max_overlap, f"uniq_{j}"

    noise = [random.uniform(-randomness, randomness) for _ in range(n)] if randomness > 0 else [0.0] * n

    # Objective: maximize projection but penalize ownership, add leverage, add noise
    prob += (
        pulp.lpSum(
            (
                players[i].proj
                - own_penalty * players[i].pred_own
                + lev_weight * (players[i].proj * (1.0 - players[i].pred_own))
                + noise[i]
            ) * flex.get(i, 0)
            for i in range(n)
        )
        +
        pulp.lpSum(
            (
                CPT_PROJ_MULT * players[i].proj
                - own_penalty * players[i].pred_own
                + lev_weight * ((CPT_PROJ_MULT * players[i].proj) * (1.0 - players[i].pred_own))
                + noise[i]
            ) * cpt.get(i, 0)
            for i in range(n)
        )
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

    cpt_idx = None
    flex_idxs: List[int] = []

    for i in range(n):
        if i in banned:
            continue
        if cpt.get(i) is not None and cpt[i].value() == 1:
            cpt_idx = i
        if flex.get(i) is not None and flex[i].value() == 1:
            flex_idxs.append(i)

    if cpt_idx is None or len(flex_idxs) != 5:
        return None

    return [cpt_idx] + flex_idxs

# -----------------------------
# OUTPUT DF (DK-style columns)
# -----------------------------
def lineups_to_df(players: List[Player], lineups: List[List[int]], contest_type: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for lu in lineups:
        row: Dict[str, object] = {}
        total_sal = 0
        total_proj = 0.0
        chalk_ct = 0
        sneaky_ct = 0
        own_sum = 0.0
        team_counts: Dict[str, int] = {}

        for slot, idx in zip(DK_SLOTS, lu):
            p = players[idx]
            is_cpt = (slot == "CPT")

            sal = int(round(p.salary * (CPT_SAL_MULT if is_cpt else 1.0)))
            prj = float(round(p.proj * (CPT_PROJ_MULT if is_cpt else 1.0), 2))

            row[f"{slot}_name"] = p.name
            row[f"{slot}_team"] = p.team
            row[f"{slot}_salary"] = sal
            row[f"{slot}_proj"] = prj

            row[f"{slot}_own_pct"] = float(round(100.0 * p.pred_own, 1))
            row[f"{slot}_is_chalk"] = int(p.is_chalk)
            row[f"{slot}_is_sneaky"] = int(p.is_sneaky)

            total_sal += sal
            total_proj += prj
            own_sum += float(100.0 * p.pred_own)

            team_counts[p.team] = team_counts.get(p.team, 0) + 1
            chalk_ct += int(p.is_chalk)
            sneaky_ct += int(p.is_sneaky)

        row["total_salary"] = total_sal
        row["total_proj"] = round(total_proj, 2)
        row["total_own_sum"] = round(own_sum, 1)
        row["chalk_ct"] = chalk_ct
        row["sneaky_ct"] = sneaky_ct
        row["contest_type"] = contest_type
        row["team_counts"] = "; ".join([f"{t}:{c}" for t, c in sorted(team_counts.items(), key=lambda x: (-x[1], x[0]))])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("total_proj", ascending=False).reset_index(drop=True)

# -----------------------------
# PUBLIC API FOR FLASK
# -----------------------------
def generate_nhl_showdown_df(
    num_lineups: int = 20,
    min_unique: int = 2,
    min_salary_spend: int = 48000,
    randomness: float = 1.0,
    salary_cap: int = DK_SALARY_CAP,
    csv_url: Optional[str] = None,
    chalk_pctile: int = CHALK_PCTILE_DEFAULT,
    sneaky_pctile: int = SNEAKY_PCTILE_DEFAULT,
    max_player_exposure: float = MAX_PLAYER_EXPOSURE_DEFAULT,
    contest_type: str = "gpp_large",
    seed: Optional[int] = 7,
) -> pd.DataFrame:

    url = csv_url or CSV_URL_DEFAULT
    df = fetch_csv_to_df(url)
    players = parse_players(df, chalk_pctile=chalk_pctile, sneaky_pctile=sneaky_pctile)

    prof = contest_profile(contest_type)
    base_random = float(prof["base_random"])
    randomness_final = 0.6 * base_random + 0.4 * max(0.0, float(randomness))

    min_chalk = int(prof["min_chalk"])
    max_chalk = int(prof["max_chalk"])
    min_sneaky = int(prof["min_sneaky"])
    own_penalty = float(prof["own_penalty"])
    lev_weight = float(prof["lev_weight"])

    # clamp if pool is small
    chalk_total = sum(p.is_chalk for p in players)
    sneaky_total = sum(p.is_sneaky for p in players)
    min_chalk = min(min_chalk, chalk_total)
    min_sneaky = min(min_sneaky, sneaky_total)
    max_chalk = max(max_chalk, min_chalk)

    n = len(players)
    cap_ct = max(1, int(np.floor(max(0.05, min(1.0, float(max_player_exposure))) * int(num_lineups))))
    exposure: Dict[int, int] = {i: 0 for i in range(n)}

    final_lineups: List[List[int]] = []
    previous_sets: List[Set[int]] = []

    for li in range(int(num_lineups)):
        banned = {i for i, ct in exposure.items() if ct >= cap_ct}

        built = None
        for attempt in range(int(RETRIES_PER_LINEUP) + 1):
            rand_i = randomness_final + 0.20 * attempt + 0.06 * (li / max(1, int(num_lineups) - 1))
            built = optimize_one_showdown(
                players=players,
                salary_cap=int(salary_cap),
                min_salary_spend=int(min_salary_spend),
                min_unique=int(min_unique),
                previous_sets=previous_sets,
                banned=banned,
                own_penalty=own_penalty,
                lev_weight=lev_weight,
                min_chalk=min_chalk,
                max_chalk=max_chalk,
                min_sneaky=min_sneaky,
                randomness=rand_i,
                seed=None if seed is None else int(seed) + li * 10 + attempt,
            )
            if built is not None:
                break

        if built is None:
            raise RuntimeError(
                "No feasible NHL Showdown lineups.\n"
                "Try: lower min_salary_spend, set min_unique=0/1, increase randomness, or contest_type=cash."
            )

        final_lineups.append(built)
        chosen_set = set(built)
        previous_sets.append(chosen_set)
        for idx in chosen_set:
            exposure[idx] += 1

    return lineups_to_df(players, final_lineups, contest_type=contest_type)

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_url", type=str, default=CSV_URL_DEFAULT)
    ap.add_argument("--num_lineups", type=int, default=20)

    ap.add_argument("--salary_cap", type=int, default=DK_SALARY_CAP)
    ap.add_argument("--min_salary_spend", type=int, default=48000)
    ap.add_argument("--min_unique", type=int, default=2)
    ap.add_argument("--randomness", type=float, default=1.0)

    ap.add_argument("--contest_type", type=str, default="gpp_large",
                    help="cash | gpp_small | gpp_medium | gpp_large (default)")

    ap.add_argument("--chalk_pctile", type=int, default=CHALK_PCTILE_DEFAULT)
    ap.add_argument("--sneaky_pctile", type=int, default=SNEAKY_PCTILE_DEFAULT)
    ap.add_argument("--max_player_exposure", type=float, default=MAX_PLAYER_EXPOSURE_DEFAULT)
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()

    out_df = generate_nhl_showdown_df(
        num_lineups=args.num_lineups,
        min_unique=args.min_unique,
        min_salary_spend=args.min_salary_spend,
        randomness=args.randomness,
        salary_cap=args.salary_cap,
        csv_url=args.csv_url,
        chalk_pctile=args.chalk_pctile,
        sneaky_pctile=args.sneaky_pctile,
        max_player_exposure=args.max_player_exposure,
        contest_type=args.contest_type,
        seed=args.seed
    )

    cols = ["total_proj", "total_salary", "total_own_sum", "chalk_ct", "sneaky_ct", "contest_type", "team_counts"]
    print(out_df[cols].head(25).to_string(index=False))
    print("\nSlot columns include: CPT_own_pct, CPT_is_chalk, CPT_is_sneaky ...")

if __name__ == "__main__":
    main()
