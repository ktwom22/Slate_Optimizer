import os
import traceback
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
from functools import wraps
import stripe
from dotenv import load_dotenv

import NFL
import NBA
import nfl_showdown
import nhl_optimizer as NHL  # change if your NHL module name differs


app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-key-123' # Change this for security later
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 1. Load the variables from .env
load_dotenv()

# 2. Assign the Secret Key from the .env file to Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# (Optional) You can also use os.getenv for your Flask Secret Key
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-if-env-is-missing")

db = SQLAlchemy(app)
# This creates the database tables automatically on Railway
with app.app_context():
    db.create_all()
    print("Database tables created successfully!")

login_manager = LoginManager(app)
login_manager.login_view = 'login' # Tells Flask where to send users who aren't logged in

# The User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_pro = db.Column(db.Boolean, default=False)
    pro_expiry = db.Column(db.DateTime, nullable=True)

    def has_access(self):
        # Returns True only if is_pro is True AND the date hasn't passed
        if self.is_pro and self.pro_expiry and self.pro_expiry > datetime.now():
            return True
        return False


# 1. This function MUST come before the routes
def pro_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # First check: Is the user logged in?
        if not current_user.is_authenticated:
            return redirect(url_for('login'))

        # Second check: Does the user have an active subscription?
        # This calls the has_access() method in your User model
        if not current_user.has_access():
            flash("Upgrade to Pro to access this optimizer.")
            return redirect(url_for('upgrade'))  # Ensure you have an 'upgrade' route

        return f(*args, **kwargs)

    return decorated_function

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
    # Accessible to all users
    return render_template("index.html", title="Optimizer Home")

# --- NFL CLASSIC ---
@app.route("/nfl", methods=["GET", "POST"])
@login_required
@pro_required
def nfl():
    if request.method == "GET":
        return render_template("nfl.html", title="NFL DK Classic")
    try:
        num_lineups = max(1, min(_safe_int(request.form.get("num_lineups", "20"), 20), 150))
        min_unique = max(0, min(_safe_int(request.form.get("min_unique", "2"), 2), 8))
        min_salary = max(0, _safe_int(request.form.get("min_salary", "46000"), 46000))
        randomness = max(0.0, min(_safe_float(request.form.get("randomness", "1.0"), 1.0), 3.0))

        gen = _require_fn(NFL, "generate_nfl_df")
        df = gen(num_lineups=num_lineups, min_unique=min_unique, min_salary_spend=min_salary, randomness=randomness).head(num_lineups)

        slots = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
        lineups = df_to_lineups(df, slots, {"note": "note", "total_proj": "total_proj"})
        return render_template("results.html", title="NFL Lineups", lineups=lineups, back_url=url_for("nfl"))
    except Exception as e:
        return _error("NFL /nfl", e, "nfl")

# --- NBA CLASSIC ---
@app.route("/nba", methods=["GET", "POST"])
@login_required
@pro_required
def nba():
    if request.method == "GET":
        return render_template("nba.html", title="NBA DK Classic")
    try:
        num_lineups = max(1, min(_safe_int(request.form.get("num_lineups", "20"), 20), 150))
        min_unique = max(0, min(_safe_int(request.form.get("min_unique", "2"), 2), 8))
        min_salary = max(0, _safe_int(request.form.get("min_salary", "49500"), 49500))
        randomness = max(0.0, min(_safe_float(request.form.get("randomness", "0.8"), 0.8), 3.0))
        contest_type = request.form.get("contest_type", "gpp_large")

        gen = _require_fn(NBA, "generate_nba_df")
        df = gen(num_lineups=num_lineups, min_unique=min_unique, min_salary_spend=min_salary, randomness=randomness, contest_type=contest_type).head(num_lineups)

        slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        lineups = df_to_lineups(df, slots, {"stack": "team_counts", "chalk_ct": "chalk_ct", "sneaky_ct": "sneaky_ct", "contest_type": "contest_type"})
        return render_template("results.html", title="NBA Lineups", lineups=lineups, back_url=url_for("nba"))
    except Exception as e:
        return _error("NBA /nba", e, "nba")

# --- NFL SHOWDOWN ---
@app.route("/nfl_showdown", methods=["GET", "POST"])
@login_required
@pro_required
def nfl_showdown_route():
    if request.method == "GET":
        return render_template("nfl_showdown.html", title="NFL Showdown (DK)")
    try:
        num_lineups = max(1, min(_safe_int(request.form.get("num_lineups"), 20), 150))
        min_unique = max(0, min(_safe_int(request.form.get("min_unique"), 2), 5))
        min_salary = _safe_int(request.form.get("min_salary"), 48000)
        randomness = _safe_float(request.form.get("randomness"), 1.0)

        gen = _require_fn(nfl_showdown, "generate_nfl_showdown_df")
        df = gen(num_lineups=num_lineups, min_unique=min_unique, min_salary_spend=min_salary, randomness=randomness)

        formatted_lineups = []
        for _, row in df.iterrows():
            lineup_data = {"total_salary": row["total_salary"], "total_proj": row["total_proj"], "meta": {"stack": row["stack_template"]}, "rows": []}
            lineup_data["rows"].append({"pos": "CPT", "player": row["CPT_name"], "team": row["CPT_team"], "salary": row["CPT_salary"], "proj": row["CPT_proj"]})
            for i in range(1, 6):
                lineup_data["rows"].append({"pos": "FLEX", "player": row[f"FLEX{i}_name"], "team": row[f"FLEX{i}_team"], "salary": row[f"FLEX{i}_salary"], "proj": row[f"FLEX{i}_proj"]})
            formatted_lineups.append(lineup_data)

        return render_template("results.html", title="NFL Showdown Results", lineups=formatted_lineups, back_url=url_for("nfl_showdown_route"))
    except Exception as e:
        return _error("NFL Showdown Error", e, "nfl_showdown_route")

# --- NHL CLASSIC ---
@app.route("/nhl", methods=["GET", "POST"])
@login_required
@pro_required
def nhl():
    if request.method == "GET":
        return render_template("nhl.html", title="NHL DK Classic")
    try:
        num_lineups = max(1, min(_safe_int(request.form.get("num_lineups", "20"), 20), 150))
        min_unique = max(0, min(_safe_int(request.form.get("min_unique", "2"), 2), 8))
        min_salary = max(0, _safe_int(request.form.get("min_salary", "47000"), 47000))
        randomness = max(0.0, min(_safe_float(request.form.get("randomness", "1.0"), 1.0), 3.0))

        gen = _require_fn(NHL, "generate_nhl_df")
        df = gen(num_lineups=num_lineups, min_unique=min_unique, min_salary_spend=min_salary, randomness=randomness).head(num_lineups)

        slots = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]
        lineups = df_to_lineups(df, slots, {"stack": "team_counts", "stack_template": "stack_template"})
        return render_template("results.html", title="NHL Lineups", lineups=lineups, back_url=url_for("nhl"))
    except Exception as e:
        return _error("NHL /nhl", e, "nhl")

# --- NHL SHOWDOWN ---
@app.route("/nhl_showdown", methods=["GET", "POST"])
@login_required
@pro_required
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
        df = gen(num_lineups=num_lineups, min_unique=min_unique, min_salary_spend=min_salary, randomness=randomness, contest_type=contest_type).head(num_lineups)

        slots = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]
        lineups = df_to_lineups(df, slots, {"contest_type": "contest_type", "chalk_ct": "chalk_ct", "sneaky_ct": "sneaky_ct", "stack": "team_counts"})
        return render_template("results.html", title="NHL Showdown Lineups", lineups=lineups, back_url=url_for("nhl_showdown_route"))
    except Exception as e:
        return _error("NHL Showdown /nhl_showdown", e, "nhl_showdown_route")


# --- LOGIN ROUTE ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        # NOTE: In a real app, use: if user and check_password_hash(user.password, password):
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('index'))

        flash('Invalid username or password')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))

        # 1. Create the user but set is_pro=False for now
        new_user = User(username=username, password=password, is_pro=False)
        db.session.add(new_user)
        db.session.commit()

        # Log them in so we know WHO is paying
        login_user(new_user)

        # 2. Create a Stripe Checkout Session
        try:
            checkout_session = stripe.checkout.Session.create(
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {'name': 'All-Sports Pro Access (Until April)'},
                        'unit_amount': 500,  # $5.00 in cents
                    },
                    'quantity': 1,
                }],
                mode='payment',
                # Where to go after success/cancel
                success_url=url_for('activate_pro', _external=True),
                cancel_url=url_for('upgrade', _external=True),
            )
            return redirect(checkout_session.url, code=303)
        except Exception as e:
            return str(e)

    return render_template('register.html')


@app.route('/upgrade')
@login_required
def upgrade():
    # If they are already pro, just send them home
    if current_user.has_access():
        return redirect(url_for('index'))

    # Otherwise, show them the payment page
    return render_template('upgrade.html')


# This route actually creates the Stripe session for existing users
@app.route('/create-checkout-session')
@login_required
def create_checkout_session():
    try:
        checkout_session = stripe.checkout.Session.create(
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {'name': 'All-Sports Pro Access'},
                    'unit_amount': 500,  # $5.00
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('activate_pro', _external=True),
            cancel_url=url_for('upgrade', _external=True),
        )
        return redirect(checkout_session.url, code=303)
    except Exception as e:
        return str(e)


@app.route('/activate_pro')
@login_required
def activate_pro():
    # 1. Identify the user who just finished paying
    user = User.query.get(current_user.id)

    # 2. Update their database record
    user.is_pro = True
    # We set the expiry to April 1st, 2026
    user.pro_expiry = datetime(2026, 4, 1)

    # 3. Save the changes to users.db
    db.session.commit()

    # 4. Give them a "High Five" and send them to the home page
    flash("Success! Your All-Sports Pro access is now active until April.")
    return redirect(url_for('index'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # This creates the users.db file automatically
    app.run(debug=True, host='0.0.0.0', port=5001)