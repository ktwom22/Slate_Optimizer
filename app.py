import os
import traceback
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

import NFL
import NBA
import nfl_showdown
import nhl_optimizer as NHL  # change if your NHL module name differs
import nhl_showdown as NHLSD  # make sure this import exists at the top

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "local-dev")


# -----------------------------
# helpers
# -----------------------------
def _safe_int(x, default=0) -> int:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _require_fn(module, fn_name: str):
    fn = getattr(module, fn_name, None)
    if not callable(fn):
        raise RuntimeError(
            f"{module.__name__}.{fn_name}(...) not found.\n"
            f"Fix: in {module.__name__}.py, define `{fn_name}` and return a pandas DataFrame."
        )
    return fn


def df_to_lineups(df: pd.DataFrame, slots: list[str], meta_fields: dict[str, str]):
    """
    Expects DK-style columns:
      {SLOT}_name, {SLOT}_team, {SLOT}_salary, {SLOT}_proj

    Optional (if present):
      {SLOT}_own_pct  -> float (0-100)
      {SLOT}_is_chalk -> int (0/1)
      {SLOT}_is_sneaky-> int (0/1)

    Also supports optional lineup-level columns in meta_fields (like team_counts, chalk_ct, etc.)
    """
    if df is None or df.empty:
        raise RuntimeError("Generator returned no lineups (empty DataFrame).")

    # validate the generator output has the base required columns
    s0 = slots[0]
    needed = [f"{s0}_name", f"{s0}_team", f"{s0}_salary", f"{s0}_proj"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Generator output missing required DK-style columns for {s0}: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # detect optional player-level fields by checking the first slot
    has_own = f"{s0}_own_pct" in df.columns
    has_chalk = f"{s0}_is_chalk" in df.columns
    has_sneaky = f"{s0}_is_sneaky" in df.columns

    lineups = []
    for _, row in df.iterrows():
        rows = []
        total_salary = 0
        total_proj = 0.0
        team_usage = {}

        for s in slots:
            pname = str(row.get(f"{s}_name", "") or "")
            team = str(row.get(f"{s}_team", "") or "")
            sal = _safe_int(row.get(f"{s}_salary", 0))
            proj = _safe_float(row.get(f"{s}_proj", 0.0))

            r = {"pos": s, "player": pname, "team": team, "salary": sal, "proj": proj}

            # optional per-player ownership/chalk/sneaky
            if has_own:
                r["own"] = _safe_float(row.get(f"{s}_own_pct", 0.0), 0.0)
            if has_chalk:
                r["chalk"] = _safe_int(row.get(f"{s}_is_chalk", 0), 0)
            if has_sneaky:
                r["sneaky"] = _safe_int(row.get(f"{s}_is_sneaky", 0), 0)

            rows.append(r)

            total_salary += sal
            total_proj += proj

            if team:
                team_usage[team] = team_usage.get(team, 0) + 1

        stack_template = "-".join(str(count) for count in sorted(team_usage.values(), reverse=True)) if team_usage else ""

        meta = {}
        for k, col in meta_fields.items():
            meta[k] = row.get(col, "") if col in df.columns else ""

        # always include basic stack summary in meta
        meta["stack_template"] = stack_template
        meta["team_usage"] = team_usage

        # include flags so template can render columns safely
        meta["has_own"] = has_own
        meta["has_chalk"] = has_chalk
        meta["has_sneaky"] = has_sneaky

        lineups.append({
            "rows": rows,
            "total_salary": total_salary,
            "total_proj": round(total_proj, 2),
            "meta": meta,
        })

    return lineups


def _error(where: str, e: Exception, back_endpoint: str):
    print(f"\n=== ERROR in {where} ===")
    traceback.print_exc()
    flash(str(e))
    return redirect(url_for(back_endpoint))


# -----------------------------
# routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html", title="Optimizer Home")


@app.route("/nfl", methods=["GET", "POST"])
def nfl():
    if request.method == "GET":
        return render_template("nfl.html", title="NFL DK Classic")

    try:
        num_lineups = max(1, min(_safe_int(request.form.get("num_lineups", "20"), 20), 150))
        min_unique = max(0, min(_safe_int(request.form.get("min_unique", "2"), 2), 8))
        min_salary = max(0, _safe_int(request.form.get("min_salary", "46000"), 46000))
        randomness = max(0.0, min(_safe_float(request.form.get("randomness", "1.0"), 1.0), 3.0))

        gen = _require_fn(NFL, "generate_nfl_df")
        df = gen(
            num_lineups=num_lineups,
            min_unique=min_unique,
            min_salary_spend=min_salary,
            randomness=randomness,
        ).head(num_lineups)

        slots = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
        meta_fields = {"note": "note", "total_proj": "total_proj"}
        lineups = df_to_lineups(df, slots, meta_fields)

        return render_template("results.html", title="NFL Lineups", lineups=lineups, back_url=url_for("nfl"))

    except Exception as e:
        return _error("NFL /nfl", e, "nfl")


@app.route("/nba", methods=["GET", "POST"])
def nba():
    if request.method == "GET":
        return render_template("nba.html", title="NBA DK Classic")

    try:
        num_lineups = max(1, min(_safe_int(request.form.get("num_lineups", "20"), 20), 150))
        min_unique = max(0, min(_safe_int(request.form.get("min_unique", "2"), 2), 8))
        min_salary = max(0, _safe_int(request.form.get("min_salary", "49500"), 49500))
        randomness = max(0.0, min(_safe_float(request.form.get("randomness", "0.8"), 0.8), 3.0))

        # optional contest type (if you add it to the form)
        contest_type = request.form.get("contest_type", "gpp_large")

        gen = _require_fn(NBA, "generate_nba_df")
        df = gen(
            num_lineups=num_lineups,
            min_unique=min_unique,
            min_salary_spend=min_salary,
            randomness=randomness,
            contest_type=contest_type,
        ).head(num_lineups)

        slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        meta_fields = {
            "stack": "team_counts",
            "chalk_ct": "chalk_ct",
            "sneaky_ct": "sneaky_ct",
            "contest_type": "contest_type",
        }
        lineups = df_to_lineups(df, slots, meta_fields)

        return render_template("results.html", title="NBA Lineups", lineups=lineups, back_url=url_for("nba"))

    except Exception as e:
        return _error("NBA /nba", e, "nba")


@app.route("/nfl_showdown", methods=["GET", "POST"])
def nfl_showdown_route():
    if request.method == "GET":
        return render_template("nfl_showdown.html", title="NFL Showdown (DK)")

    try:
        num_lineups = max(1, min(_safe_int(request.form.get("num_lineups", "20"), 20), 150))
        min_unique = max(0, min(_safe_int(request.form.get("min_unique", "2"), 2), 8))
        min_salary = max(0, _safe_int(request.form.get("min_salary", "48000"), 48000))
        randomness = max(0.0, min(_safe_float(request.form.get("randomness", "1.0"), 1.0), 3.0))

        gen = _require_fn(nfl_showdown, "generate_nfl_showdown_df")
        df = gen(
            num_lineups=num_lineups,
            min_unique=min_unique,
            min_salary_spend=min_salary,
            randomness=randomness,
        ).head(num_lineups)

        slots = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]
        meta_fields = {}
        lineups = df_to_lineups(df, slots, meta_fields)

        return render_template(
            "results.html",
            title="NFL Showdown Lineups",
            lineups=lineups,
            back_url=url_for("nfl_showdown_route"),
        )

    except Exception as e:
        return _error("NFL Showdown /nfl_showdown", e, "nfl_showdown_route")


@app.route("/nhl", methods=["GET", "POST"])
def nhl():
    if request.method == "GET":
        return render_template("nhl.html", title="NHL DK Classic")

    try:
        num_lineups = max(1, min(_safe_int(request.form.get("num_lineups", "20"), 20), 150))
        min_unique = max(0, min(_safe_int(request.form.get("min_unique", "2"), 2), 8))
        min_salary = max(0, _safe_int(request.form.get("min_salary", "47000"), 47000))
        randomness = max(0.0, min(_safe_float(request.form.get("randomness", "1.0"), 1.0), 3.0))

        gen = _require_fn(NHL, "generate_nhl_df")
        df = gen(
            num_lineups=num_lineups,
            min_unique=min_unique,
            min_salary_spend=min_salary,
            randomness=randomness,
        ).head(num_lineups)

        slots = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]
        meta_fields = {"stack": "team_counts", "stack_template": "stack_template"}
        lineups = df_to_lineups(df, slots, meta_fields)

        return render_template("results.html", title="NHL Lineups", lineups=lineups, back_url=url_for("nhl"))

    except Exception as e:
        return _error("NHL /nhl", e, "nhl")


@app.route("/nhl_showdown", methods=["GET", "POST"])
def nhl_showdown_route():
    if request.method == "GET":
        return render_template("nhl_showdown.html", title="NHL Showdown (DK)")

    try:
        num_lineups = max(1, min(_safe_int(request.form.get("num_lineups", "20"), 20), 150))
        min_unique = max(0, min(_safe_int(request.form.get("min_unique", "2"), 2), 8))
        min_salary = max(0, _safe_int(request.form.get("min_salary", "48000"), 48000))
        randomness = max(0.0, min(_safe_float(request.form.get("randomness", "1.0"), 1.0), 3.0))
        contest_type = (request.form.get("contest_type", "gpp_large") or "gpp_large").strip().lower()

        gen = _require_fn(NHLSD, "generate_nhl_showdown_df")

        df = gen(
            num_lineups=num_lineups,
            min_unique=min_unique,
            min_salary_spend=min_salary,
            randomness=randomness,
            contest_type=contest_type,
        ).head(num_lineups)

        slots = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]
        meta_fields = {
            "contest_type": "contest_type",
            "chalk_ct": "chalk_ct",
            "sneaky_ct": "sneaky_ct",
            "stack": "team_counts",
        }

        lineups = df_to_lineups(df, slots, meta_fields)

        return render_template(
            "results.html",
            title="NHL Showdown Lineups",
            lineups=lineups,
            back_url=url_for("nhl_showdown_route"),
            csv_path="nhl_showdown",
        )

    except Exception as e:
        return _error("NHL Showdown /nhl_showdown", e, "nhl_showdown_route")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
