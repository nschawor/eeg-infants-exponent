# This R-script compares intercepts between sensorimotor and parietal
# electrodes. 

library(data.table)
library(lme4)
require(stringr)
require(tibble)
require(dplyr)

# read all files in directory results and combine into one data.table
filelist <- list.files("../results/model_bursts/", "burst_features_.*-lap.csv")
dt <- rbindlist(lapply(filelist,
                       function(fn) {
                         dt <- fread(file.path("../results/model_bursts/", fn))
                         dt[, electrode := str_sub(fn, 16, 17)]
                         return(dt)
                       }))
dt[, electrode := factor(electrode, levels = c("Pz", "C3", "C4"))]

# get vector of age at first session for every participant
first_ages <- dt[, .(age = min(age)), by = subject]$age

# number of permutation test steps
nrep <- 5000
results_features <- data.frame()
results_resampled <- data.frame()

# loop over all features in data table
feature <- colnames(dt)[3]

# use formula with interaction for electrode resampling and without interaction
# for age resampling
f <- as.formula(paste(feature, "age * electrode + (1|subject)", sep = "~"))
f2 <- as.formula(paste(feature, "age + electrode + (1|subject)", sep =
                        "~"))
lmm_interaction <- lmer(f, data = dt)
summary_interaction <- enframe(summary(lmm_interaction)[["coefficients"]][, 1])
lmm_simple <- lmer(f2, data = dt)
summary_simple <- enframe(summary(lmm_simple)[["coefficients"]][1:2, 1])
results_features <- summary_interaction %>%
  mutate(name = if_else(
    name == "age",
    "age:electrodeC3",
    if_else(name == "(Intercept)", "electrodeC3", name)
  )) %>%
  bind_rows(summary_simple) %>%
  mutate(feature = feature) %>%
  bind_rows(results_features)

for (n in 1:nrep) {
  # shuffle electrode labels for every infant at every session
  dt_electrode <- copy(dt)
  dt_electrode[, electrode := sample(electrode, .N), by = .(subject, age)]
  lmm_electrode <- lmer(f, data = dt_electrode)

  # take coefficients for electrodes and interactions (3:6) from these results
  summary_electrode <- enframe(summary(lmm_electrode)[["coefficients"]][3:6, 1])

  # shuffle age labels for every infant & electrode; take a random first age
  # apply age increments between sessions to new start age, but in random order
  dt_age <- copy(dt)
  dt_age[,
         age := sample(age, .N) - min(age) +
           sample(first_ages, size = 1, replace = TRUE),
         by = .(subject, electrode)]

  # fit model without interaction term for electrodes
  lmm_age <- lmer(f2, data = dt_age)

  # extract coefficients for Intercept (1) and age (2) from the model
  summary_age <- enframe(summary(lmm_age)[["coefficients"]][1:2, 1])
  results_resampled <- summary_age %>%
    bind_rows(summary_electrode) %>%
    mutate(rep = n, feature = feature) %>%
    bind_rows(results_resampled)
}

# add true (unshuffled) coefficients to resampling result for comparison
setDT(results_features)
setDT(results_resampled)
results_resampled[results_features, value_true := i.value, on = .(name, feature)]

# compute p-values for true coefficients by counting how often more extreme
# values appear in models with shuffled labels
results <- results_resampled[,
                             .(coefficient = value_true[1],
                             p_val = mean(sign(value) == sign(value_true) &
                                          abs(value) >= abs(value_true))),
                             by = .(feature, name)]

