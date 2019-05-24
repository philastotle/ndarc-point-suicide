################################################################################
# Suicide Attempts 
################################################################################
# Load dependencies
library(dplyr)
library(ggplot2)
library(foreign)
library(lattice)
library(mice)
library(VIM)

# Load dataset 
df <- read.spss('../../data/processed/core/PointCore.sav',
                to.data.frame = T,
                use.value.labels = F)

# select variables of interest
df <- df %>% 
  select(
    participant_id,
    wave,
    age,
    sex,
    maritalstatus,
    employ,
    employ_chnge_pain,
    income_wk,
    indig_yn,
    bmi,
    age,
    sf12_mcs,
    sf12_pcs,
    phq9_severity,
    gad_severity,
    pharm_opioids_dep_icd10,
    suicide_thoughts_12m,
    suicide_attempt_12m,
    mos_ss_avg,
    who_qol_q1,
    who_qol_q2,
    totalopioiddose,
    pods_tot,
    antidepressant_week,
    antipsychotic_week,
    bpi_pscore,
    wg_totscore,
    num_chronic_cond_12m,
    bpi_interference,
    pseq_score,
    slp9,
    orbit_cont,
    can_12m,
    alc_12m,
    cig_12m,
    time_pain_weeks)

# Find number of suicidal users across entire dataset
# 1. Find suicidal participants across all years
suicide_user_df <- df[which(df$suicide_attempt_12m == 1),]
# 2. Extract their id's
suicide_users_id <- suicide_user_df$participant_id
# 3. Create df of suicide users
suicide_users_df <- subset(df, participant_id %in% suicide_users_id)
cat('Number of suicidal users', sum(suicide_users_df$suicide_attempt_12m, na.rm=T))
# 4. Calculate number of suicide and total dependent users in each wave
print("Suicidal Attempts at each wave")
for (i in 0:5){
  tmp <- filter(suicide_users_df, wave==i)
  tmp2 <- filter(df, wave==i)
  tmp$wave <- factor(tmp$wave)
  cat('\nWave', i,'=', sum(tmp$suicide_attempt_12m==1, na.rm=T))
}


################################################################################
#### BASELINE
baseline <- df[which(df$wave == 0),]
baseline[baseline < 0] <- NA
# Model Building ---------------------------------------------------------------

# Based on model selection process we impute these variables that in a LR model
# Produced an AIC of 249.2

baseline <- baseline %>% 
  select(
    participant_id,
    wave,
    age,
    sex,
    maritalstatus,
    pharm_opioids_dep_icd10,
    suicide_thoughts_12m,
    suicide_attempt_12m,
    antipsychotic_week,
    wg_totscore,
    pseq_score,
    slp9,
    orbit_cont)

# Imputation -------------------------------------------------------------------

# Before we begin lets count the na values
sum(is.na(baseline)) # 3144 NA/ missing values

# Proportion of complete cases
100*sum(complete.cases(baseline))/nrow(baseline) # 13.9% are complete

# % missing for key variables,
# apply the aggr function to the data and sort the data by number of missing
aggr(baseline, sortVars=TRUE, numbers=TRUE)

#check the number of missing cases per column
# (aggr will show percentage missing)
colSums(is.na(baseline))

# Multiple Imputation via Iterative Chained Equations (M-I-C-E)
# i: choice of 'm' which is the number of iterations
# Where m is the number of imputated datasets
# ii: choice of method 
imp <- mice(baseline, seed=1, maxit=25, print=FALSE) #default parameters with higher chain convergence
summary(imp)
#check non-convergence with trace plot
# (convergence is how well they mix across m imputations)
plot(imp, "suicide_attempt_12m", layout = c(2, 1))
xyplot(imp, suicide_attempt_12m~age, xlab="suicide_attempt_12m", ylab="age")

#compare statistics of original and imputed
summary(baseline)
summary(complete(imp)) # imp2 has the closest mean for tv to the original

mice::complete(imp,0)[, 8] # view original suicide_attempts data
mice::complete(imp,1)[, 8] # imputed suicide attempts data

# Save imputed DS
imputed_df <- complete(imp)

#modelling the imputed data
logmod_full <- glm(suicide_attempt_12m ~ 
              age+
              sex+
              maritalstatus+
              pharm_opioids_dep_icd10+
              suicide_thoughts_12m+
              suicide_attempt_12m+
              antipsychotic_week+
              wg_totscore+
              pseq_score+
              slp9+
              orbit_cont, data = imputed_df)
summary(logmod_full)


## CALIBRATION
# add predicted probabilities to the data frame 
imputed_df %>% mutate(predprob=predict(logmod_full, type="response"),
                    linpred=predict(logmod_full)) %>%
  #group the data into bins based on the linear predictor fitted values 
  group_by(cut(linpred, breaks=unique(quantile(linpred, (1:50)/51)))) %>%
  
  # summarise by bin
  
  summarise(suicide_attempt_12m_bin=sum(suicide_attempt_12m), predprob_bin=mean(predprob), n_bin=n()) %>% # add the standard error of the mean predicted probaility for each bin
  mutate(se_predprob_bin=sqrt(predprob_bin*(1 - predprob_bin)/n_bin)) %>%
  # plot it with 95% confidence interval bars 
  ggplot(aes(x=predprob_bin, 
             y=suicide_attempt_12m_bin/n_bin, 
             ymin=suicide_attempt_12m_bin/n_bin - 1.96*se_predprob_bin,
             ymax=suicide_attempt_12m_bin/n_bin + 1.96*se_predprob_bin)) +
  geom_point() + geom_linerange(colour="orange", alpha=0.4) +
  geom_abline(intercept=0, slope=1) +
  labs(x="Predicted probability (binned)",
       y="Observed proportion (in each bin)")

## GOODNESS OF FIT
imputed_df %>% mutate(predprob=predict(logmod_full, type="response"),
                    linpred=predict(logmod_full)) %>%
  group_by(cut(linpred, breaks=unique(quantile(linpred, (1:50)/51)))) %>%
  summarise(suicide_attempt_12m_bin=sum(suicide_attempt_12m), predprob_bin=mean(predprob), n_bin=n()) %>%
  mutate(se_predprob_bin=sqrt(predprob_bin*(1 - predprob_bin)/n_bin)) -> hl_df

hl_stat <- with(hl_df, sum( (suicide_attempt_12m_bin - n_bin*predprob_bin)^2 /
                              (n_bin* predprob_bin*(1 - predprob_bin))))

hl <- c(hosmer_lemeshow_stat=hl_stat, hl_degrees_freedom=nrow(hl_df) - 1)
hl

# calculate p-value
c(p_val=1 - pchisq(hl[1], hl[2]))

print("Confusion matrix")
imputed_df %>% mutate(predprob=predict(logmod_full, type="response")) %>%
  mutate(pred_outcome=ifelse(predprob < 0.5, 0, 1)) -> imputed_df

xtabs(~ suicide_thoughts_12m + pred_outcome, data=imputed_df)

accuracy <- (6219 + 8) / (6219 + 0 + 1351 + 8)

cat('\nClassification accuracy is ', accuracy)

imputed_df$pred_outcome <- as.factor(imputed_df$pred_outcome)
imputed_df$suicide_attempt_12m <- as.factor(imputed_df$suicide_attempt_12m)

# add the predicted probailities to the data frame
thresholds <- seq(0.01, 0.5, by=0.01)
sensitivities <- numeric(length(thresholds))
specificities <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  pp <- ifelse(baseline$predprob < thresholds[i], "no", "yes")
  xx <- xtabs( ~ suicide_attempt_12m + pp, data=imputed_df)
  specificities[i] <- xx[1,1] / (xx[1,1] + xx[1,2])
  sensitivities[i] <- xx[2,2] / (xx[2,1] + xx[2,2])
}

# plot the ROC
plot(1 - specificities, sensitivities, type="l",
     xlab="1 - Specificity", ylab="Sensitivity")
abline(0,1, lty=2)