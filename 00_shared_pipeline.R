### Hybrid Pharmacokinetic Modelling for Vancomycin MIPD ###
### Shared Data Preparation and Utility Functions         ###
### Sourced by model-specific scripts                     ###

# ── Libraries ──────────────────────────────────────────────────────────────────
library(mapbayr)
library(mrgsolve)
library(dplyr)
library(ggplot2)
library(lubridate)
library(future.apply)



# ── Load raw data ───────────────────────────────────────────────────────────────
setwd("D:/code_advancedcovariateanalysis")

rrt_and_race          <- read.csv("rrt_and_race.csv")
demographics_original <- read.csv("demographics_original.csv")
vanco_administrations <- read.csv("vanco_administrations.csv")
vanco_tdm             <- read.csv("vanco_tdm.csv")



# ── Filter to non-RRT patients ──────────────────────────────────────────────────
non_rrt_ids <- rrt_and_race %>%
  filter(rrt == 0) %>%
  select(subject_id, hadm_id)



# ── Merge and clean demographics ────────────────────────────────────────────────
final_demographics <- non_rrt_ids %>%
  left_join(demographics_original,
            by = c("subject_id" = "Subject.ID", "hadm_id" = "Admission.ID")) %>%
  mutate(
    Gender       = ifelse(Gender == "F", 1, 0),
    Age          = as.numeric(Age),
    Weight       = as.numeric(Weight),
    Scr.Baseline = as.numeric(Scr.Baseline)
  ) %>%
  filter(!is.na(Weight), !is.na(Scr.Baseline), !is.na(Age), !is.na(Gender)) %>%
  mutate(
    CRCL = ifelse(
      Gender == 1,
      ((140 - Age) * Weight) / (0.814 * Scr.Baseline * 88.42) * 0.85,  # Female
      ((140 - Age) * Weight) / (0.814 * Scr.Baseline * 88.42)           # Male
    )
  )



# ── Merge TDM observations ──────────────────────────────────────────────────────
final_dataset <- final_demographics %>%
  left_join(
    vanco_tdm %>% select(hadm_id, subject_id, tdm_charttime, tdm_value),
    by = c("hadm_id", "subject_id")
  ) %>%
  filter(tdm_value != "NULL") %>%
  mutate(tdm_charttime = as.POSIXct(tdm_charttime, format = "%Y-%m-%d %H:%M:%S"))



# ── Prepare dosing records ──────────────────────────────────────────────────────
final_doses <- vanco_administrations %>%
  filter(
    subject_id %in% final_dataset$subject_id,
    hadm_id    %in% final_dataset$hadm_id
  ) %>%
  select(-all_of(c(
    "dose_given_unit", "product_amount_given", "product_unit",
    "event_txt", "relative_administration"
  ))) %>%
  mutate(administration_charttime = as.POSIXct(
    administration_charttime, format = "%Y-%m-%d %H:%M:%S"
  ))



# ── Calculate time-from-first-dose for TDM observations ─────────────────────────
first_dose_times <- final_doses %>%
  group_by(subject_id, hadm_id) %>%
  summarise(first_dose_time = min(administration_charttime), .groups = "drop")

final_dataset <- final_dataset %>%
  left_join(first_dose_times, by = c("subject_id", "hadm_id")) %>%
  mutate(
    tdm_time_from_first_dose = as.numeric(
      difftime(tdm_charttime, first_dose_time, units = "hours")
    )
  ) %>%
  select(-first_dose_time)



# ── Build compact dosing lists [time amount] per patient ────────────────────────
final_dosing_info <- final_doses %>%
  group_by(subject_id, hadm_id) %>%
  mutate(
    time_from_first = as.numeric(
      difftime(administration_charttime, min(administration_charttime), units = "hours")
    )
  ) %>%
  summarise(
    dosing_list = paste0("[", time_from_first, " ", dose_given, "]", collapse = " "),
    .groups = "drop"
  )



# ── Merge dosing into main dataset ──────────────────────────────────────────────
final_combined <- final_dataset %>%
  left_join(final_dosing_info, by = c("subject_id", "hadm_id")) %>%
  mutate(
    dosing_list = lapply(dosing_list, function(s) {
      as.numeric(unlist(strsplit(gsub("\\[|\\]", "", s), " ")))
    })
  ) %>%
  select(-c(tdm_charttime, Stay.ID, Administration.Chart.Time))



# ── Quality filters ─────────────────────────────────────────────────────────────

# Require at least 3 administered doses (length >= 6: 3 time + 3 amount values)
final_combined <- final_combined %>%
  filter(sapply(dosing_list, length) >= 5)

# TDM must occur at least 12 h after first dose
final_combined <- final_combined %>%
  filter(tdm_time_from_first_dose >= 12)

# TDM must occur more than 5 h after the most recent dose
final_combined <- final_combined %>%
  rowwise() %>%
  filter({
    last_dose_time <- dosing_list[length(dosing_list) - 1]
    (tdm_time_from_first_dose - last_dose_time) > 5
  }) %>%
  ungroup()



# ── Remove covariate outliers (|Z| > 5) ────────────────────────────────────────
features <- final_combined %>%
  select(Age:CRCL, tdm_value) %>%
  mutate(across(everything(), as.numeric))

z_scores_df <- as.data.frame(scale(features)) %>%
  bind_cols(final_combined %>% select(subject_id, hadm_id), .)

final_combined <- final_combined %>%
  filter(
    apply(
      z_scores_df %>% select(-subject_id, -hadm_id),
      1,
      function(x) all(abs(x[!is.na(x)]) <= 5)
    )
  )



# ── MAP-BE runner (shared utility) ─────────────────────────────────────────────
# Runs MAP Bayesian Estimation for a single patient row using a compiled model.
#
# Args:
#   i          : row index in final_combined
#   model      : compiled mrgsolve model object
#   covariates : named list of covariate values to pass to add_covariates()
#
# Returns a named list: CL, V1, C_pred  (NA on failure)

run_mapbe <- function(i, model, covariates) {
  patient_obs_conc <- final_combined$tdm_value[i]
  patient_obs_time <- final_combined$tdm_time_from_first_dose[i]
  dosing           <- final_combined$dosing_list[[i]]
  nr_doses         <- length(dosing) / 2

  est <- model

  # Add dose events
  for (j in seq_len(nr_doses)) {
    dose_time   <- dosing[2 * j - 1]
    dose_amount <- dosing[2 * j]
    est <- est %>%
      adm_rows(time = dose_time, amt = dose_amount, rate = dose_amount / 1.5, cmt = 1)
  }

  # Run estimation with error handling
  result <- tryCatch({
    est <- est %>%
      obs_rows(time = patient_obs_time, DV = patient_obs_conc, cmt = 1)

    est <- do.call(add_covariates, c(list(est), covariates))

    est <- est %>%
      mapbayest()

    list(
      CL     = get_param(est, "CL"),
      V1     = get_param(est, "V1"),
      C_pred = tail(est[["mapbay_tab"]][["IPRED"]], n = 1)
    )
  }, error = function(e) {
    warning(sprintf("MAP-BE failed for patient %d (row %d): %s", i, i, conditionMessage(e)))
    list(CL = NA_real_, V1 = NA_real_, C_pred = NA_real_)
  })

  return(result)
}



# ── Parallel MAP-BE runner (shared utility) ────────────────────────────────────
# Runs run_mapbe() over all rows in final_combined using multisession workers.
#
# Args:
#   model      : compiled mrgsolve model object
#   covariate_fn : function(i) returning a named list of covariates for row i

run_mapbe_parallel <- function(model, covariate_fn) {
  N <- nrow(final_combined)

  plan(multisession, workers = max(1L, parallel::detectCores() - 1L))
  on.exit(plan(sequential), add = TRUE)

  message(sprintf("Starting MAP-BE for %d patients across %d workers ...",
                  N, nbrOfWorkers()))

  results <- future_lapply(seq_len(N), function(i) {
    if (i %% 50 == 0 || i == 1 || i == N) {
      message(sprintf("  Patient %d / %d", i, N))
    }
    run_mapbe(i, model, covariate_fn(i))
  })

  list(
    MAPBE_CL     = sapply(results, `[[`, "CL"),
    MAPBE_V1     = sapply(results, `[[`, "V1"),
    MAPBE_C_pred = sapply(results, `[[`, "C_pred")
  )
}



# ── Export helper ───────────────────────────────────────────────────────────────
export_results <- function(data, mapbe_results, filename) {
  data$MAPBE_CL     <- mapbe_results$MAPBE_CL
  data$MAPBE_V1     <- mapbe_results$MAPBE_V1
  data$MAPBE_C_pred <- mapbe_results$MAPBE_C_pred

  data$dosing_list <- sapply(data$dosing_list, paste, collapse = ", ")

  write.csv(data, filename, row.names = FALSE)
  message("Exported: ", filename)
}
