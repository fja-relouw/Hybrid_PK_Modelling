"""
Hybrid Pharmacokinetic Modelling for Vancomycin MIPD
Garibay et al. (2016) — XGBoost Training and Evaluation
"""

from shared_pipeline_xgboost import *

# ── Model-specific constants ───────────────────────────────────────────────────
# Garibay peripheral volume and intercompartmental clearance
VP_GARIBAY = None   # computed per-patient: 5.90 * weight  (passed inline below)
Q_GARIBAY  = 0.81   # [L/h]

CSV_PATH   = "ML_Dataset_GaribayFullFinal.csv"


# ── Load data ──────────────────────────────────────────────────────────────────
filtered_data = load_and_clean(CSV_PATH)

CL = filtered_data["MAPBE_CL"]
Vc = filtered_data["MAPBE_V1"]


# ── Distributions ──────────────────────────────────────────────────────────────
plot_distribution(CL, r"Observed CL$_{\mathrm{MAP-BE}}$ [L/h]")
plot_distribution(Vc, r"Observed Vc$_{\mathrm{MAP-BE}}$ [L]")


# ── MAP-BE predicted vs observed trough ────────────────────────────────────────
plot_pred_obs(
    x_pred=filtered_data["MAPBE_C_pred"],
    y_obs=filtered_data["tdm_value"],
    xlabel=r"Predicted C$_{\mathrm{trough-MAP-BE}}$ [mg/L]",
    ylabel=r"Observed C$_{\mathrm{trough-MIMIC-IV}}$ [mg/L]",
    axis_limit=50,
    label_main="MAP-BE",
)


# ── Covariate-equation reference predictions ───────────────────────────────────
# CL: TVCL * CRCL  (CRCL converted from ml/min to L/h)
cov_CL = 0.49 * (filtered_data["CRCL"] / 1000 * 60)

# Vc: age-stratified typical value * body weight
cov_Vc = filtered_data.apply(
    lambda r: r["Weight"] * (1.07 if r["Age"] > 65 else 0.74), axis=1
)

plot_pred_obs(
    x_pred=cov_CL, y_obs=CL,
    xlabel=r"Predicted CL$_{\mathrm{covariate-equation}}$ [L/h]",
    ylabel=r"Observed CL$_{\mathrm{MAP-BE}}$ [L/h]",
    axis_limit=13, label_main="Covariate Equation",
)
print_metrics("Covariate CL (full dataset)", CL, cov_CL)

plot_pred_obs(
    x_pred=cov_Vc, y_obs=Vc,
    xlabel=r"Predicted Vc$_{\mathrm{covariate-equation}}$ [L]",
    ylabel=r"Observed Vc$_{\mathrm{MAP-BE}}$ [L]",
    axis_limit=201, label_main="Covariate Equation",
)
print_metrics("Covariate Vc (full dataset)", Vc, cov_Vc)


# ── Reference trough simulation (full dataset) ─────────────────────────────────
# Garibay Vp is weight-dependent; simulate_troughs accepts a scalar Vp,
# so we loop using the shared helper with per-patient Vp passed via CL_vals trick.
# Instead, we call simulate_two_comp directly inside a small wrapper here.

c_trough_ref = pd.Series(dtype=float)
for idx, row in filtered_data.iterrows():
    vp = 5.90 * row["Weight"]
    try:
        c_pred = simulate_two_comp(
            Vc=cov_Vc[idx], CL=cov_CL[idx],
            Vp=vp, Q=Q_GARIBAY,
            dosing_str=row["dosing_list"],
            inf_duration=INF_DURATION,
            tdm_time=row["tdm_time_from_first_dose"],
        )
    except Exception as e:
        print(f"  Warning: simulation failed for index {idx}: {e}")
        c_pred = float("nan")
    c_trough_ref.at[idx] = c_pred

plot_pred_obs(
    x_pred=c_trough_ref, y_obs=filtered_data["tdm_value"],
    xlabel=r"Predicted C$_{\mathrm{trough-covariate-equation}}$ [mg/L]",
    ylabel=r"Observed C$_{\mathrm{trough-MIMIC-IV}}$ [mg/L]",
    axis_limit=50, label_main="Covariate Equation",
)
print_metrics("Covariate Ctrough (full dataset)",
              filtered_data["tdm_value"], c_trough_ref)


# ── Test-set reference baselines ───────────────────────────────────────────────
test_idx = get_test_indices(filtered_data)

ref_test_CL = pd.DataFrame({
    "CL_pred": cov_CL.loc[test_idx],
    "CL_obs":  filtered_data["MAPBE_CL"].loc[test_idx],
})
ref_test_Vc = pd.DataFrame({
    "Vc_pred": cov_Vc.loc[test_idx],
    "Vc_obs":  filtered_data["MAPBE_V1"].loc[test_idx],
})
ref_test_Ctrough = pd.DataFrame({
    "Ctrough_pred": c_trough_ref.loc[test_idx],
    "Ctrough_obs":  filtered_data["tdm_value"].loc[test_idx],
})

print("\n── Reference performance on test set ──")
print_metrics("CL",     ref_test_CL["CL_obs"],          ref_test_CL["CL_pred"])
print_metrics("Vc",     ref_test_Vc["Vc_obs"],          ref_test_Vc["Vc_pred"])
print_metrics("Ctrough",ref_test_Ctrough["Ctrough_obs"],ref_test_Ctrough["Ctrough_pred"])


# ── XGBoost: CL ───────────────────────────────────────────────────────────────
print("\n── XGBoost CL ──")
X_train_CL, X_test_CL, y_train_CL, y_test_CL = make_split(filtered_data, "MAPBE_CL")
model_CL, CL_pred, importance_CL = tune_and_train_xgb(
    X_train_CL, y_train_CL, X_test_CL, y_test_CL, target_name="CL"
)

plot_feature_importance(importance_CL, title="Garibay — CL Feature Importance")
plot_feature_importance(importance_CL, top_n=10, title="Garibay — CL Top 10 Features")

plot_pred_obs(
    x_pred=CL_pred, y_obs=y_test_CL,
    xlabel=r"Predicted CL$_{\mathrm{XGBoost}}$ [L/h]",
    ylabel=r"Observed CL$_{\mathrm{MAP-BE}}$ [L/h]",
    axis_limit=13,
    x_ref=ref_test_CL["CL_pred"], y_ref=ref_test_CL["CL_obs"],
    label_main="XGBoost", label_ref="Covariate Equation",
)


# ── XGBoost: Vc ───────────────────────────────────────────────────────────────
print("\n── XGBoost Vc ──")
X_train_Vc, X_test_Vc, y_train_Vc, y_test_Vc = make_split(filtered_data, "MAPBE_V1")
model_Vc, Vc_pred, importance_Vc = tune_and_train_xgb(
    X_train_Vc, y_train_Vc, X_test_Vc, y_test_Vc, target_name="Vc"
)

plot_feature_importance(importance_Vc, title="Garibay — Vc Feature Importance")
plot_feature_importance(importance_Vc, top_n=10, title="Garibay — Vc Top 10 Features")

plot_pred_obs(
    x_pred=Vc_pred, y_obs=y_test_Vc,
    xlabel=r"Predicted Vc$_{\mathrm{XGBoost}}$ [L]",
    ylabel=r"Observed Vc$_{\mathrm{MAP-BE}}$ [L]",
    axis_limit=201,
    x_ref=ref_test_Vc["Vc_pred"], y_ref=ref_test_Vc["Vc_obs"],
    label_main="XGBoost", label_ref="Covariate Equation",
)


# ── XGBoost trough simulation ──────────────────────────────────────────────────
print("\n── Trough simulation (XGBoost CL/Vc) ──")
c_trough_xgb_pred = pd.Series(dtype=float)
c_trough_xgb_obs  = pd.Series(dtype=float)

for i, idx in enumerate(y_test_CL.index):
    row = filtered_data.loc[idx]
    vp  = 5.90 * row["Weight"]
    try:
        c_pred = simulate_two_comp(
            Vc=float(Vc_pred[i]), CL=float(CL_pred[i]),
            Vp=vp, Q=Q_GARIBAY,
            dosing_str=row["dosing_list"],
            inf_duration=INF_DURATION,
            tdm_time=row["tdm_time_from_first_dose"],
        )
    except Exception as e:
        print(f"  Warning: simulation failed for index {idx}: {e}")
        c_pred = float("nan")
    c_trough_xgb_pred.at[idx] = c_pred
    c_trough_xgb_obs.at[idx]  = row["tdm_value"]

plot_pred_obs(
    x_pred=c_trough_xgb_pred, y_obs=c_trough_xgb_obs,
    xlabel=r"Predicted C$_{\mathrm{trough}}$ [mg/L]",
    ylabel=r"Observed C$_{\mathrm{trough}}$ [mg/L]",
    axis_limit=50,
    x_ref=ref_test_Ctrough["Ctrough_pred"],
    y_ref=ref_test_Ctrough["Ctrough_obs"],
    label_main="XGBoost", label_ref="Covariate Equation",
)

print_metrics("XGBoost Ctrough", c_trough_xgb_obs, c_trough_xgb_pred)
