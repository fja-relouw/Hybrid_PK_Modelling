"""
Hybrid Pharmacokinetic Modelling for Vancomycin MIPD
Shared pipeline: data loading, PK simulation, SR fitting, plotting utilities.
Imported by model-specific scripts via: from shared_pipeline_sr import *
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from scipy.integrate import solve_ivp
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit
from pysr import PySRRegressor


# ── Constants ──────────────────────────────────────────────────────────────────
SEED       = 111
N_SPLITS   = 3
TEST_SIZE  = 0.25
INF_DURATION = 1.0  # infusion duration [hours]

# Colour palette
COL_REF  = "#000000"
COL_SR   = "#0b655d"
COL_MAIN = "#0b655d"
COL_GRAY = "#666666"

# PySR shared settings (override per-target in model scripts if needed)
PYSR_SHARED = dict(
    binary_operators=["+", "-", "*", "/", "pow"],
    constraints={"pow": (-1, 1)},
    loss="loss(x, y) = (x - y)^2",
    progress=False,
    deterministic=True,
    procs=0,
    multithreading=False,
)


# ── Seed helper ────────────────────────────────────────────────────────────────
def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    """
    K_el = CL / Vc

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
        t_end  = tdm_time if k == len(sorted_times) - 1 else sorted_times[k + 1]
        t_eval = np.linspace(dose_time, t_end, 500)

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
    """Return lower/upper 95% prediction band arrays evaluated at x."""
    x_fit, y_fit = np.asarray(x_fit, float), np.asarray(y_fit, float)
    n     = x_fit.size
    t_val = stats.t.ppf((1 + conf) / 2.0, n - 2)
    fit   = np.polyfit(x_fit, y_fit, deg)
    resid = y_fit - np.polyval(fit, x_fit)
    s_res = np.sqrt(np.sum(resid ** 2) / (n - deg - 1))
    se    = s_res * np.sqrt(
        1 + 1 / n + (x - x_fit.mean()) ** 2 / np.sum((x_fit - x_fit.mean()) ** 2)
    )
    delta = t_val * se
    return np.polyval(fit, x) - delta, np.polyval(fit, x) + delta


# ── Reference trough simulation (full dataset) ─────────────────────────────────
def compute_ref_troughs(filtered_data: pd.DataFrame,
                        cov_CL: pd.Series, cov_Vc: pd.Series,
                        Vp_fn, Q: float) -> pd.Series:
    """
    Simulate reference troughs for every patient in filtered_data.

    Args:
        Vp_fn : callable(row) -> float, returns peripheral volume for a row.
                For a fixed Vp, pass: lambda row: VP_CONSTANT
    """
    c_trough = pd.Series(dtype=float)
    for idx, row in filtered_data.iterrows():
        try:
            c_pred = simulate_two_comp(
                Vc=cov_Vc[idx], CL=cov_CL[idx],
                Vp=Vp_fn(row), Q=Q,
                dosing_str=row["dosing_list"],
                inf_duration=INF_DURATION,
                tdm_time=row["tdm_time_from_first_dose"],
            )
        except Exception as e:
            print(f"  Warning: simulation failed for index {idx}: {e}")
            c_pred = float("nan")
        c_trough.at[idx] = c_pred
    return c_trough


# ── PySR fitting loop ──────────────────────────────────────────────────────────
def run_sr(filtered_data: pd.DataFrame,
           features: list[str],
           target_col: str,
           pysr_kwargs: dict,
           output_filename: str) -> tuple[list, list, list, list]:
    """
    Run PySR with ShuffleSplit cross-validation, evaluate all hall-of-fame
    equations on test folds, and save results to Excel.

    Returns:
        idx_trains, idx_tests : lists of train/test index arrays per fold
        (results saved to output_filename)
    """
    set_seeds(SEED)

    sr_data = filtered_data[features + [target_col]].dropna()
    X_data  = sr_data[features].to_numpy(dtype=np.float64)
    y_data  = sr_data[target_col].to_numpy(dtype=np.float64)

    n_features   = len(features)
    splitter     = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE,
                                random_state=SEED)
    final_results = {}
    idx_trains, idx_tests = [], []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_data), 1):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        idx_trains.append(train_idx)
        idx_tests.append(test_idx)

        sr = PySRRegressor(**{**PYSR_SHARED, **pysr_kwargs, "random_state": SEED})
        sr.fit(X_train, y_train)

        eq_df = sr.equations_.copy()
        eq_df["RMSE_train"] = (eq_df["loss"] ** 0.5).round(3)
        eq_df = eq_df.drop(columns=["sympy_format", "lambda_format",
                                     "loss", "score"])

        y_true_mean = np.mean(y_test)

        for i in range(len(eq_df)):
            eq_str  = str(eq_df["equation"].iloc[i]).replace("^", "**")
            sq_errs = []

            for j in range(len(X_test)):
                # Unpack features as x0..xN dynamically
                local_vars = {f"x{k}": X_test[j][k] for k in range(n_features)}
                try:
                    y_hat = eval(eq_str, {"__builtins__": {}}, local_vars)
                except Exception:
                    y_hat = float("nan")
                sq_errs.append((y_hat - y_test[j]) ** 2)

            sq_errs = np.array(sq_errs)
            rmse_eq = np.sqrt(np.nanmean(sq_errs))
            ss_res  = np.nansum(sq_errs)
            ss_tot  = np.sum((y_test - y_true_mean) ** 2)
            r2_eq   = 1 - ss_res / ss_tot

            eq_df.at[i, "RMSE_test"] = round(rmse_eq, 3)
            eq_df.at[i, "R2_test"]   = round(r2_eq, 2)

        final_results[f"equations_fold_{fold}"] = eq_df
        print(f"  Fold {fold} complete — {len(eq_df)} equations evaluated")

    with pd.ExcelWriter(output_filename) as writer:
        for sheet, df in final_results.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Results saved to {output_filename}")

    return idx_trains, idx_tests


# ── Reference fold metrics ─────────────────────────────────────────────────────
def fold_metrics(series_obs: pd.Series, series_pred: pd.Series,
                 fold_indices: list, split: str = "test",
                 label: str = "") -> tuple[list, list]:
    """
    Compute RMSE and R² for each fold given lists of integer index arrays.
    Prints results and returns (rmse_list, r2_list).
    """
    rmse_list, r2_list = [], []
    for fold_idx in fold_indices:
        obs  = series_obs.iloc[fold_idx]
        pred = series_pred.iloc[fold_idx]
        rmse = mean_squared_error(obs, pred) ** 0.5
        r2   = r2_score(obs, pred)
        rmse_list.append(round(rmse, 4))
        r2_list.append(round(r2, 4))

    tag = f"{label} {split}".strip()
    print(f"{tag} RMSE per fold : {rmse_list}")
    print(f"{tag} R²   per fold : {r2_list}")
    return rmse_list, r2_list


# ── Trough scatter plot (SR vs reference) ──────────────────────────────────────
def plot_sr_trough(c_pred_ref, c_obs_ref,
                   c_pred_sr, c_obs_sr,
                   label_ref: str = "Reference",
                   label_sr:  str = "SR") -> None:
    """
    Predicted vs observed trough plot comparing SR model against reference,
    with regression lines and 95% prediction bands.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    x_vals = np.linspace(0, 50, 200)

    # Scatter
    ax.scatter(c_pred_ref, c_obs_ref,
               color=COL_GRAY, alpha=0.5, edgecolors="none",
               label=f"{label_ref} data")
    ax.scatter(c_pred_sr, c_obs_sr,
               color=COL_SR, alpha=0.5, edgecolors="none",
               label=f"{label_sr} data")

    # Identity line
    ax.plot([0, 50], [0, 50], color="#999999", linestyle="-", linewidth=1)

    # Reference regression + band
    s_r, i_r = np.polyfit(np.asarray(c_pred_ref, float),
                           np.asarray(c_obs_ref, float), 1)
    ax.plot(x_vals, np.polyval([s_r, i_r], x_vals),
            color="#444444", linestyle="--", linewidth=2,
            label=f"{label_ref} fit: y={s_r:.2f}x+{i_r:.2f}")
    lo_r, hi_r = prediction_band(x_vals, c_pred_ref, c_obs_ref)
    ax.fill_between(x_vals, lo_r, hi_r,
                    color="#888888", alpha=0.12,
                    label=f"{label_ref} 95% band")

    # SR regression + band
    s_sr, i_sr = np.polyfit(np.asarray(c_pred_sr, float),
                             np.asarray(c_obs_sr, float), 1)
    ax.plot(x_vals, np.polyval([s_sr, i_sr], x_vals),
            color=COL_SR, linestyle="--", linewidth=2,
            label=f"{label_sr} fit: y={s_sr:.2f}x+{i_sr:.2f}")
    lo_sr, hi_sr = prediction_band(x_vals, c_pred_sr, c_obs_sr)
    ax.fill_between(x_vals, lo_sr, hi_sr,
                    color=COL_SR, alpha=0.12,
                    label=f"{label_sr} 95% band")

    ax.set_xlabel(r"Predicted C$_{\mathrm{trough}}$ [mg/L]",
                  fontsize=16, fontweight="medium")
    ax.set_ylabel(r"Observed C$_{\mathrm{trough}}$ [mg/L]",
                  fontsize=16, fontweight="medium")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=12, loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()


# ── Performance metrics helper ─────────────────────────────────────────────────
def print_metrics(label: str, y_obs, y_pred) -> None:
    y_obs  = np.asarray(y_obs, float)
    y_pred = np.asarray(y_pred, float)
    rmse   = mean_squared_error(y_obs, y_pred) ** 0.5
    mape   = np.abs((y_obs - y_pred) / y_obs).mean() * 100
    print(f"{label} — RMSE: {rmse:.4f}  |  MAPE: {mape:.2f}%")
