### Hybrid Pharmacokinetic Modelling for Vancomycin MIPD   ###
### Garibay et al. (2016) — MAP-BE Dataset Generation      ###

rm(list = ls())
gc()
source("shared_pipeline.R")   # loads libraries, cleans data, defines utilities



# ── PopPK Model: Garibay 2016 ───────────────────────────────────────────────────
# Two-compartment model.
# Covariates: body weight (BW), creatinine clearance (CRCL), age (AGE).
# CL scales linearly with CRCL.
# V1 uses an age-stratified typical value (≤65 vs >65 years), scaled by BW.
# V2 scales with BW; Q is fixed.

garibay_code <- "
$PARAM @annotated
TVCL       : 0.49 : Typical CL without furosemide [L/h per unit CRCL]
TVV1_OLD   : 1.07 : Typical V1 for age > 65 [L/kg]
TVV1_YOUNG : 0.74 : Typical V1 for age <= 65 [L/kg]
TVV2       : 5.9  : Typical V2 [L/kg]
TVQ        : 0.81 : Intercompartmental clearance Q [L/h]

ETA1 : 0 : IIV on CL
ETA2 : 0 : IIV on V1

$PARAM @annotated @covariates
BW   : 0 : Body weight [kg]
CRCL : 0 : Creatinine clearance [L/h]
AGE  : 0 : Age [years]

$OMEGA 0.1264 0.1484

$SIGMA
0.0362  // proportional
0       // additive

$CMT @annotated
CENT   : Central compartment [mg/L]
PERIPH : Peripheral compartment

$TABLE
double DV = (CENT / V1) * (1 + EPS(1)) + EPS(2);

$MAIN
double CL       = TVCL * CRCL * exp(ETA1 + ETA(1));
double TVV1_SEL = (AGE > 65) ? TVV1_OLD * BW : TVV1_YOUNG * BW;
double V1       = TVV1_SEL * exp(ETA2 + ETA(2));
double V2       = TVV2 * BW;
double Q        = TVQ;

double K10 = CL / V1;
double K12 = Q  / V1;
double K21 = Q  / V2;

$ODE
dxdt_CENT   = K21 * PERIPH - (K10 + K12) * CENT;
dxdt_PERIPH = K12 * CENT   -  K21        * PERIPH;

$CAPTURE DV CL V1 V2
"

garibay_model <- mcode("Garibay2016_vancomycinPopPK", garibay_code)



# ── Run MAP-BE ──────────────────────────────────────────────────────────────────
# CRCL is converted from ml/min to L/h (× 60 / 1000) to match model units.

covariate_fn <- function(i) {
  list(
    BW   = final_combined$Weight[i],
    CRCL = final_combined$CRCL[i] * 60 / 1000,
    AGE  = final_combined$Age[i]
  )
}

mapbe_results <- run_mapbe_parallel(garibay_model, covariate_fn)



# ── Export ──────────────────────────────────────────────────────────────────────
export_results(final_combined, mapbe_results, "ML_Dataset_GaribayFullFinal.csv")
