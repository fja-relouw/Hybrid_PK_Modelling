"""
Hybrid Pharmacokinetic Modelling for Vancomycin MIPD
Garibay et al. (2016) — Symbolic Regression Training and Evaluation
"""

from shared_pipeline_sr import *

# ── Model-specific constants ───────────────────────────────────────────────────
Q_GARIBAY  = 0.81   # intercompartmental clearance [L/h]
# Vp is weight-dependent for Garibay: 5.90 * weight
Vp_fn      = lambda row: 5.90 * row["Weight"]

CSV_PATH   = "ML_Dataset_GaribayFullFinal.csv"

# Features selected for SR (from prior feature importance analysis)
CL_FEATURES = ["Age", "Scr.Baseline", "Weight", "Gender",
                "BUN", "RDW", "Mean.Temperature", "Mean.SBP"]
VC_FEATURES = ["Weight", "Scr.Baseline"]

# PySR hyperparameters — CL and Vc differ in parsimony/maxsize/select_k
PYSR_CL = dict(
    populations=1,       # N=1000 for full run
    niterations=1,       # N=200  for full run
    select_k_features=8,
    parsimony=0.15,
    maxsize=15,
)
PYSR_VC = dict(
    populations=1,
    niterations=1,
    select_k_features=2,
    parsimony=20.0,
    maxsize=9,
)

# Output filenames
CL_OUTPUT = "garibay_SR_CL_final.xlsx"
VC_OUTPUT = "garibay_SR_Vc_final.xlsx"

# Best SR equations (from hall-of-fame, paste chosen equations here)
def sr_CL(row) -> float:
    return (
        (row["Weight"] / (
            ((row["Scr.Baseline"] * row["Age"]) + row["BUN"])
            / (row["Mean.SBP"] / row["RDW"])
        )) ** 0.63075936
    ) - row["Gender"]

def sr_Vc(row) -> float:
    return (row["Weight"] + -16.86333) - (1.0211724 ** row["Weight"])


# ── Load and prepare data ──────────────────────────────────────────────────────
filtered_data = load_and_clean(CSV_PATH)
filtered_data = filtered_data.dropna(
    subset=list(set(CL_FEATURES + VC_FEATURES)), how="any"
)
filtered_data.reset_index(drop=True, inplace=True)

CL = filtered_data["MAPBE_CL"]
Vc = filtered_data["MAPBE_V1"]


# ── Covariate-equation reference predictions ───────────────────────────────────
cov_CL = filtered_data["CRCL"] / 1000 * 60 * 0.49
cov_Vc = filtered_data.apply(
    lambda r: r["Weight"] * (1.07 if r["Age"] > 65 else 0.74), axis=1
)

print_metrics("Reference CL (full dataset)", CL, cov_CL)
print_metrics("Reference Vc (full dataset)", Vc, cov_Vc)


# ── Reference trough simulation (full dataset) ─────────────────────────────────
c_trough_ref = compute_ref_troughs(
    filtered_data, cov_CL, cov_Vc, Vp_fn=Vp_fn, Q=Q_GARIBAY
)
print_metrics("Reference Ctrough (full dataset)",
              filtered_data["tdm_value"], c_trough_ref)


# ── Symbolic Regression: CL ───────────────────────────────────────────────────
print("\n── SR CL ──")
idx_trains_CL, idx_tests_CL = run_sr(
    filtered_data=filtered_data,
    features=CL_FEATURES,
    target_col="MAPBE_CL",
    pysr_kwargs=PYSR_CL,
    output_filename=CL_OUTPUT,
)

fold_metrics(CL, cov_CL, idx_tests_CL,  split="test",  label="CL")
fold_metrics(CL, cov_CL, idx_trains_CL, split="train", label="CL")


# ── Symbolic Regression: Vc ───────────────────────────────────────────────────
print("\n── SR Vc ──")
idx_trains_Vc, idx_tests_Vc = run_sr(
    filtered_data=filtered_data,
    features=VC_FEATURES,
    target_col="MAPBE_V1",
    pysr_kwargs=PYSR_VC,
    output_filename=VC_OUTPUT,
)

fold_metrics(Vc, cov_Vc, idx_tests_Vc,  split="test",  label="Vc")
fold_metrics(Vc, cov_Vc, idx_trains_Vc, split="train", label="Vc")


# ── Trough simulation using best SR equations (fold 1 test set) ────────────────
print("\n── SR Trough (fold 1) ──")
fold1_indices = idx_tests_CL[0]

ref_ctrough_fold1 = pd.DataFrame({
    "Ctrough_pred": c_trough_ref.loc[fold1_indices],
    "Ctrough_obs":  filtered_data["tdm_value"].loc[fold1_indices],
})

c_trough_pred_sr = pd.Series(dtype=float)
c_trough_obs_sr  = pd.Series(dtype=float)

for idx in fold1_indices:
    row = filtered_data.loc[idx]
    try:
        c_pred = simulate_two_comp(
            Vc=sr_Vc(row), CL=sr_CL(row),
            Vp=Vp_fn(row), Q=Q_GARIBAY,
            dosing_str=row["dosing_list"],
            inf_duration=INF_DURATION,
            tdm_time=row["tdm_time_from_first_dose"],
        )
    except Exception as e:
        print(f"  Warning: simulation failed for index {idx}: {e}")
        c_pred = float("nan")
    c_trough_pred_sr.at[idx] = c_pred
    c_trough_obs_sr.at[idx]  = row["tdm_value"]

plot_sr_trough(
    c_pred_ref=ref_ctrough_fold1["Ctrough_pred"],
    c_obs_ref=ref_ctrough_fold1["Ctrough_obs"],
    c_pred_sr=c_trough_pred_sr,
    c_obs_sr=c_trough_obs_sr,
    label_ref="Reference",
    label_sr="SR",
)

print_metrics("Reference Ctrough (fold 1)",
              ref_ctrough_fold1["Ctrough_obs"],
              ref_ctrough_fold1["Ctrough_pred"])
print_metrics("SR Ctrough (fold 1)", c_trough_obs_sr, c_trough_pred_sr)
