# This R-script fits a linear mixed model with subject as a random factor
# and age as a fixed factor to mean waveform features.

library(data.table)
library(lme4)
library(ciTools)

# function: fitting linear mixed model and extracting estimates
fit_model <- function(df, formula) {
  lmm <- lmer(formula, df, REML = FALSE)
  summary1 <- summary(lmm)[["coefficients"]]
  t <- summary1[, "t value"]["age"]
  estimate <- summary1[, "Estimate"]["age"]
  return(list(t = t, estimate = estimate))
}

filelist = c('C3', 'C4', 'Pz')
for (file in filelist) {
  dt <- fread(sprintf('../results/model_bursts/burst_features_%s-lap.csv', file))

  # extract prediction intervals specifically for feature == frequency
  formula = frequency ~ age + (1 | subject)
  lmm = lmer(formula, data=dt)
  pred_ci = add_ci(data.frame(age = 10:250), lmm, alpha=0.05, type='parametric',
                  includeRanef=FALSE)
  df_name = sprintf("../results/model_bursts_%s-lap_predictions.csv", file)
  write.csv(pred_ci,df_name,row.names = FALSE)

  # estimate models for all features
  formula = waveform ~ age + (1 | subject)
  dt <- melt(dt, id.vars = c("age", "subject"), variable.name = "feature", value.name = "waveform")
  df <- dt[, fit_model(.SD, formula), by = feature]

  # set starting ages
  first_ages <- dt[, .(age = min(age)), by = subject]$age
  dt[, age_orig := age]

  # perform hierarchical bootstrapping
  df_boot_all <- data.frame()
  nrep <- 5000

  for (n in 1:nrep) {
    dt_resampled <- dt[,
                       age := sample(age_orig, .N) - min(age_orig) +
                         sample(first_ages, size = 1, replace = TRUE),
                       by = .(subject, feature)]
    df_boot <- dt_resampled[, fit_model(.SD, formula), by = feature]
    df_boot_all <- rbind(df_boot_all, df_boot)
  }

  # p-value calculation: count number of times estimate exceeds bootstrapped estimate
  df_boot_all[df, ":=" (t_real = i.t, estimate_real = i.estimate), on = "feature"]
  boot_result <- df_boot_all[, .(p = (1 + sum(abs(estimate_real) <= abs(estimate))) / (nrep + 1)),
                             by = feature]
  df_estimates <- merge(df, boot_result, by = c("feature"))
  print(df_estimates)
  df_name = sprintf("../results/model_bursts/model_bursts_%s-lap.csv", file)
  write.csv(df_estimates, df_name, row.names = FALSE)

}
