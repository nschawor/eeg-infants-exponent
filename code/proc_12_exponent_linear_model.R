# This R-script fits a linear mixed model with subject as a random factor
# and age as a fixed factor to exponents.

library(data.table)
library(lme4)
library(ciTools)

dt <- fread("../results/model_exponents/exponents_for_linear_model.csv")
dt <- melt(dt, id.vars = c("age", "subject"), variable.name = "channel",
                                              value.name = "exponent")

# specify model
formula = exponent ~ age + (1 | subject)

# save prediction intervals for channel PO3 for plotting
df <- dt[channel == "PO3"]
lmm <- lmer(formula, df, REML = FALSE)
pred_PO3 <- add_ci(data.frame(age = 10:250), lmm, alpha = 0.05,
                   type = "parametric", includeRanef = FALSE)
write.csv(pred_PO3, "../results/model_exponents/exponent_model_predictions.csv",
          row.names = FALSE)

# function: fitting linear mixed model and extracting estimates
fit_model <- function(df, formula) {
  lmm <- lmer(formula, df, REML = FALSE)
  summary1 <- summary(lmm)[["coefficients"]]
  t <- summary1[, "t value"]["age"]
  estimate <- summary1[, "Estimate"]["age"]
  return(list(t = t, estimate = estimate))
}

# fit model for each channel separately
df <- dt[, fit_model(.SD, formula), by = channel]

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
       by = .(subject, channel)]
  df_boot <- dt_resampled[, fit_model(.SD,formula), by = channel]
  df_boot_all <- rbind(df_boot_all, df_boot)
}

# p-value calculation: count number of times estimate exceeds bootstrapped estimate
df_boot_all[df, ":=" (t_real = i.t, estimate_real = i.estimate), on = "channel"]
boot_result <- df_boot_all[, .(p_estimate = (1 + sum(abs(t_real) <= abs(t))) / (1 + nrep)),
                           by = channel]
df_estimates <- merge(df, boot_result, by = c("channel"))
print(df_estimates)
df_file <- "../results/model_exponents/exponent_model_results.csv"
write.csv(df_estimates, df_file, row.names = FALSE)
