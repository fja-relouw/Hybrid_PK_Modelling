"""
Hybrid Pharmacokinetic Modelling for Vancomycin MIPD
Shared pipeline: data loading, PK simulation, XGBoost tuning, plotting utilities.
Imported by model-specific scripts via: from shared_pipeline import *
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial output

from scipy.integrate import solve_ivp
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# ── Constants ──────────────────────────────────────────────────────────────────
SEED        = 1
TEST_SIZE   = 0.2
N_CV_FOLDS  = 10
N_TRIALS    = 500
INF_DURATION = 1.0   # infusion duration [hours]

FEATURE_COLS = [
    "Gender", "Age", "Scr.Baseline", "Weight", "SOFA", "APSIII",
    "Potassium", "Sodium", "Glucose", "Chloride", "Calcium", "Magnesium",
    "Phosphate", "BUN", "Bicarbonate", "Anion.Gap", "Hematocrit",
    "Hemoglobin", "MCHC", "MCV", "Platelet", "RBC", "RDW", "WBC",
    "Height", "AST", "ALP", "ALT", "Bilirubin.Total", "pH", "pO2",
    "pCO2", "Lactate", "Mean.Heart.Rate", "StdDev.Heart.Rate",
    "Mean.SBP", "StdDev.SBP", "Mean.DBP", "StdDev.DBP",
    "Mean.MBP", "StdDev.MBP", "Mean.Resp.Rate", "StdDev.Resp.Rate",
    "Mean.Temperature", "StdDev.Temperature", "Mean.SpO2", "StdDev.SpO2",
    "Urine.pH", "Urine.Gravity", "Albumin",
]

# Colour palette (consistent across all plots)
COL_REF  = "#000000"
COL_XGB  = "#0b655d"
COL_ALT  = "#9809A6"
COL_GRAY = "#666666"


# ── Data loading and outlier removal ───────────────────────────────────────────
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load dataset and remove MAP-BE outliers (|Z| > 5)."""
    df = pd.read_csv(csv_path)

    outlier_cols = ["MAPBE_CL", "MAPBE_V1"]
    df[outlier_cols] = df[outlier_cols].apply(pd.to_numeric, errors="coerce")

    z = np.abs(
        (df[outlier_cols] - df[outlier_cols].mean()) / df[outlier_cols].std()
    )
    df = df[(z <= 5).all(axis=1) | z.isna().all(axis=1)]
    df.reset_index(drop=True, inplace=True)

    return df


# ── Two-compartment PK model (ODE) ────────────────────────────────────────────
def _two_comp_ode(t, y, K_el, Q, Vc, Vp, dose_amount, inf_duration, dose_time):
    CENT, PERI = y
    dCENT = -K_el * CENT + Q * (PERI / Vp - CENT / Vc)
    dPERI =  Q * (CENT / Vc - PERI / Vp)
    if dose_time <= t <= dose_time + inf_duration:
        dCENT += dose_amount / inf_duration
    return [dCENT, dPERI]


def simulate_two_comp(Vc: float, CL: float, Vp: float, Q: float,
                      dosing_str: str, inf_duration: float,
                      tdm_time: float) -> float:
    """
    Simulate a two-compartment vancomycin PK model and return the
    predicted concentration at `tdm_time`.

    Args:
        Vc          : central volume [L]
        CL          : clearance [L/h]
        Vp          : peripheral volume [L]
        Q           : intercompartmental clearance [L/h]
        dosing_str  : comma-separated string of alternating time/amount pairs
        inf_duration: infusion duration [h]
        tdm_time    : time of TDM observation from first dose [h]

    Returns:
        Predicted plasma concentration [mg/L] at tdm_time.
    """
    K_el = CL / Vc

    # Parse dosing string and group simultaneous doses
    raw = [float(v) if "." in str(v) else int(v)
           for v in dosing_str.split(",")]
    doses_by_time: dict = {}
    for i in range(0, len(raw), 2):
        t, amt = raw[i], raw[i + 1]
        doses_by_time.setdefault(t, []).append(amt)

    sorted_times = sorted(doses_by_time)
    t_all, C_all = np.array([]), np.array([])
    CENT_prev, PERI_prev = 0.0, 0.0

    for k, dose_time in enumerate(sorted_times):
        total_amt = sum(doses_by_time[dose_time])
        t_end   = tdm_time if k == len(sorted_times) - 1 else sorted_times[k + 1]
        t_eval  = np.linspace(dose_time, t_end, 500)

        sol = solve_ivp(
            _two_comp_ode,
            (dose_time, t_end),
            [CENT_prev, PERI_prev],
            args=(K_el, Q, Vc, Vp, total_amt, inf_duration, dose_time),
            t_eval=t_eval,
        )
        CENT_prev, PERI_prev = sol.y[0][-1], sol.y[1][-1]
        t_all = np.concatenate([t_all, sol.t])
        C_all = np.concatenate([C_all, sol.y[0]])

    return float(np.interp(tdm_time, t_all, C_all / Vc))


# ── Prediction band ────────────────────────────────────────────────────────────
def prediction_band(x: np.ndarray, x_fit, y_fit,
                    deg: int = 1, conf: float = 0.95):
    """Return lower/upper 95 % prediction band arrays evaluated at x."""
    x_fit, y_fit = np.asarray(x_fit, float), np.asarray(y_fit, float)
    n      = x_fit.size
    t_val  = stats.t.ppf((1 + conf) / 2.0, n - 2)
    fit    = np.polyfit(x_fit, y_fit, deg)
    resid  = y_fit - np.polyval(fit, x_fit)
    s_res  = np.sqrt(np.sum(resid ** 2) / (n - deg - 1))
    se     = s_res * np.sqrt(
        1 + 1 / n + (x - x_fit.mean()) ** 2 / np.sum((x_fit - x_fit.mean()) ** 2)
    )
    delta  = t_val * se
    return np.polyval(fit, x) - delta, np.polyval(fit, x) + delta


# ── Plotting helpers ───────────────────────────────────────────────────────────
def _scatter_with_band(ax, x_pred, y_obs, color, alpha=0.25, marker="o"):
    ax.scatter(x_pred, y_obs, color=color, alpha=alpha, marker=marker,
               edgecolors="none")


def plot_distribution(values: pd.Series, xlabel: str, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(values, bins=25, color=COL_XGB, edgecolor="black")
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("Frequency [-]", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.grid(True)
    if title:
        ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_pred_obs(x_pred, y_obs, xlabel: str, ylabel: str,
                  axis_limit: float,
                  x_ref=None, y_ref=None,
                  label_main: str = "Model",
                  label_ref: str = "Reference") -> None:
    """
    Predicted vs observed scatter with regression line and prediction band.
    Optionally overlays a reference series.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    lim = [0, axis_limit]
    x_vals = np.linspace(0, axis_limit, 200)

    # Identity line
    ax.plot(lim, lim, color=COL_XGB, linestyle="-", linewidth=1.5)

    # Main series
    _scatter_with_band(ax, x_pred, y_obs, COL_XGB)
    slope, intercept = np.polyfit(np.asarray(x_pred, float),
                                  np.asarray(y_obs, float), 1)
    ax.plot(x_vals, np.polyval([slope, intercept], x_vals),
            color=COL_XGB, linestyle=(0, (5, 10)),
            label=f"{label_main}: y={slope:.2f}x+{intercept:.2f}")
    lo, hi = prediction_band(x_vals, x_pred, y_obs)
    ax.fill_between(x_vals, lo, hi, color=COL_XGB, alpha=0.1,
                    label=f"{label_main} 95% band")

    # Optional reference series
    if x_ref is not None and y_ref is not None:
        _scatter_with_band(ax, x_ref, y_ref, COL_REF, marker="x")
        s_r, i_r = np.polyfit(np.asarray(x_ref, float),
                               np.asarray(y_ref, float), 1)
        ax.plot(x_vals, np.polyval([s_r, i_r], x_vals),
                color=COL_REF, linestyle=(0, (5, 10)),
                label=f"{label_ref}: y={s_r:.2f}x+{i_r:.2f}")
        lo_r, hi_r = prediction_band(x_vals, x_ref, y_ref)
        ax.fill_between(x_vals, lo_r, hi_r, color=COL_REF, alpha=0.1,
                        label=f"{label_ref} 95% band")

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.tick_params(labelsize=15)
    ax.grid(True)
    ax.legend(fontsize=13, loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                             top_n: int | None = None,
                             title: str = "") -> None:
    df = importance_df.sort_values("Importance", ascending=False)
    if top_n is not None:
        df = df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(df["Feature"], df["Importance"], color=COL_XGB)
    ax.invert_yaxis()
    ax.set_xlabel("Importance [-]", fontsize=20)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=8 if top_n is None else 15)
    ax.grid(True)
    if title:
        ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# ── XGBoost tuning and training ────────────────────────────────────────────────
def _xgb_objective(trial, X_train, y_train, seed):
    params = {
        "max_depth":        trial.suggest_int("max_depth", 2, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "gamma":            trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "alpha":            trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "lambda":           trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "objective":        "reg:squarederror",
        "random_state":     seed,
        "n_jobs":           -1,
    }
    model  = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=N_CV_FOLDS, scoring="r2")
    return scores.mean()


def tune_and_train_xgb(X_train, y_train, X_test, y_test,
                        target_name: str = "target"):
    """
    Run Optuna hyperparameter search, refit on full training set,
    predict on test set, and print performance metrics.

    Returns:
        model            : fitted XGBRegressor
        y_pred           : test-set predictions (numpy array)
        importance_df    : DataFrame with Feature / Importance columns
    """
    random.seed(SEED)
    np.random.seed(SEED)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        lambda trial: _xgb_objective(trial, X_train, y_train, SEED),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"\nBest params for {target_name}: {study.best_params}")

    model = xgb.XGBRegressor(**study.best_params,
                              objective="reg:squarederror",
                              random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    importance_df = (
        pd.DataFrame({"Feature": X_train.columns,
                      "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return model, y_pred, importance_df


# ── Trough concentration simulation loop ──────────────────────────────────────
def simulate_troughs(indices, filtered_data: pd.DataFrame,
                     CL_vals, Vc_vals,
                     Vp: float, Q: float) -> tuple[pd.Series, pd.Series]:
    """
    Run two-compartment simulation for each patient index.

    Args:
        indices      : iterable of DataFrame index values to simulate
        filtered_data: full patient DataFrame
        CL_vals      : array-like of CL values aligned to indices
        Vc_vals      : array-like of Vc values aligned to indices
        Vp           : peripheral volume [L]  (model-specific constant)
        Q            : intercompartmental clearance [L/h]

    Returns:
        pred, obs : pd.Series of predicted / observed concentrations
    """
    pred, obs = pd.Series(dtype=float), pd.Series(dtype=float)

    for i, idx in enumerate(indices):
        row = filtered_data.loc[idx]
        try:
            c_pred = simulate_two_comp(
                Vc=float(Vc_vals[i]),
                CL=float(CL_vals[i]),
                Vp=Vp,
                Q=Q,
                dosing_str=row["dosing_list"],
                inf_duration=INF_DURATION,
                tdm_time=row["tdm_time_from_first_dose"],
            )
        except Exception as e:
            print(f"  Warning: simulation failed for index {idx}: {e}")
            c_pred = float("nan")

        pred.at[idx] = c_pred
        obs.at[idx]  = row["tdm_value"]

    return pred, obs


# ── Performance metrics helper ─────────────────────────────────────────────────
def print_metrics(label: str, y_obs, y_pred) -> None:
    y_obs, y_pred = np.asarray(y_obs, float), np.asarray(y_pred, float)
    rmse = mean_squared_error(y_obs, y_pred) ** 0.5
    mape = (np.abs((y_obs - y_pred) / y_obs)).mean() * 100
    print(f"{label} — RMSE: {rmse:.4f}  |  MAPE: {mape:.2f}%")


# ── Train/test split (fixed seed, reusable) ────────────────────────────────────
def make_split(filtered_data: pd.DataFrame, target_col: str):
    """Return X_train, X_test, y_train, y_test using the global seed."""
    X = filtered_data[FEATURE_COLS]
    y = filtered_data[target_col]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


def get_test_indices(filtered_data: pd.DataFrame) -> pd.Index:
    """Return the test-set indices (consistent with make_split)."""
    _, X_test, _, _ = train_test_split(
        filtered_data, filtered_data["MAPBE_CL"],
        test_size=TEST_SIZE, random_state=SEED,
    )
    return X_test.index
