### Hybrid Pharmacokinetic Modelling for Vancomycin MIPD   ###
### Goti et al. (2019) — MAP-BE Dataset Generation         ###

rm(list = ls())
gc()
source("00_shared_pipeline.R")   # loads libraries, cleans data, defines utilities



# ── PopPK Model: Goti 2019 ──────────────────────────────────────────────────────
# Two-compartment model.
# Covariates: body weight (BW), creatinine clearance (CRCL), dialysis status (DIAL).
# CL scales with CRCL via a power function (exponent 0.8).
# V1 scales with BW relative to a 70 kg reference.
# Dialysis reduces TVCL by 30 % and TVV1 by 50 %.

goti_code <- "
$PARAM @annotated
TVCL : 4.5  : Typical CL [L/h]
TVV1 : 58.4 : Typical central volume [L]
TVV2 : 38.4 : Typical peripheral volume [L]
Q    : 6.5  : Intercompartmental clearance [L/h]

ETA1 : 0 : IIV on CL
ETA2 : 0 : IIV on V1
ETA3 : 0 : IIV on V2

$PARAM @annotated @covariates
BW   : 0 : Body weight [kg]
CRCL : 0 : Creatinine clearance [ml/min]
DIAL : 0 : Dialysis status (0 = no, 1 = yes)

$OMEGA 0.147 0.5103 0.2822

$SIGMA
0.0502 // proportional
3.4    // additive

$CMT @annotated
CENT   : Central compartment [mg/L]
PERIPH : Peripheral compartment

$TABLE
double DV = (CENT / V1) * (1 + EPS(1)) + EPS(2);

$MAIN
// Apply dialysis-related adjustments to typical values
double TVCL_adj = (DIAL == 1) ? TVCL * 0.7 : TVCL;
double TVV1_adj = (DIAL == 1) ? TVV1 * 0.5 : TVV1;

double CL  = TVCL_adj * exp(ETA1 + ETA(1)) * pow(CRCL / 120.0, 0.8);
double V1  = TVV1_adj * exp(ETA2 + ETA(2)) * (BW / 70.0);
double V2  = TVV2     * exp(ETA3 + ETA(3));

double K12 = Q  / V1;
double K21 = Q  / V2;
double K10 = CL / V1;

$ODE
dxdt_CENT   = K21 * PERIPH - (K10 + K12) * CENT;
dxdt_PERIPH = K12 * CENT   -  K21        * PERIPH;

$CAPTURE DV CL V1 V2
"

goti_model <- mcode("Goti2019_vancomycinPopPK", goti_code)



# ── Run MAP-BE ──────────────────────────────────────────────────────────────────
# CRCL is passed in ml/min as required by this model (no unit conversion).
# DIAL is not in demographics; defaults to 0 (non-dialysis) for all patients,
# consistent with the non-RRT filter applied in the shared pipeline.

covariate_fn <- function(i) {
  list(
    BW   = final_combined$Weight[i],
    CRCL = final_combined$CRCL[i],
    DIAL = 0L
  )
}

mapbe_results <- run_mapbe_parallel(goti_model, covariate_fn)



# ── Export ──────────────────────────────────────────────────────────────────────
export_results(final_combined, mapbe_results, "ML_Dataset_GotiFullFinal.csv")
