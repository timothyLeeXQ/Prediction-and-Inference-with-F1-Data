# Databricks notebook source
library(SparkR)
library(sparklyr)
library(dplyr)

# COMMAND ----------

SparkR::sparkR.session()
sc <- spark_connect(method = "databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LogReg 1 - Predicting 1st & 2nd

# COMMAND ----------

df_top2_dummy_source <- spark_read_csv (sc, name ="df_1or2_dummy", path="/mnt/xql2001-gr5069/processed/final_project/df_top2_dummy.csv")
df_top2_dummy <- collect(df_top2_dummy_source)
display(df_top2_dummy)

# COMMAND ----------

df_top2_dummy$`1_or_2` <- as.factor(df_top2_dummy$`1_or_2`)
df_top2_dummy$pit_strategy_1 <- as.factor(df_top2_dummy$pit_strategy_1)
df_top2_dummy$pit_strategy_2 <- as.factor(df_top2_dummy$pit_strategy_2)
df_top2_dummy$pit_strategy_3 <- as.factor(df_top2_dummy$pit_strategy_3)
df_top2_dummy$pit_strategy_3stops <- as.factor(df_top2_dummy$pit_strategy_3stops)
df_top2_dummy$pit_strategy_missing <- as.factor(df_top2_dummy$pit_strategy_missing)
df_top2_dummy$constructor_quality_not_winner <- as.factor(df_top2_dummy$constructor_quality_not_winner)
df_top2_dummy$constructor_quality_winner <- as.factor(df_top2_dummy$constructor_quality_winner)
df_top2_dummy$circuit_type_race <- as.factor(df_top2_dummy$circuit_type_race)
df_top2_dummy$circuit_type_street <- as.factor(df_top2_dummy$circuit_type_street)


# COMMAND ----------

glimpse(df_top2_dummy)

# COMMAND ----------

logreg_top2 <- glm(`1_or_2` ~ grid +
                   years_since_debut +
                   pit_strategy_1 +
                   pit_strategy_2 +
                   pit_strategy_3 +
                   pit_strategy_3stops +
                   pit_strategy_missing +
                   constructor_quality_not_winner +
                   constructor_quality_winner +
                   circuit_type_race +
                   circuit_type_street +
                   pit_strategy_1*grid +
                   pit_strategy_2*grid +
                   pit_strategy_3*grid +
                   pit_strategy_3stops*grid +
                   pit_strategy_missing*grid +
                   years_since_debut*grid +
                   years_since_debut*circuit_type_race + 
                   years_since_debut*circuit_type_street + 
                   years_since_debut*constructor_quality_not_winner +
                   years_since_debut*constructor_quality_winner,
                   data = df_top2_dummy,
                   family="binomial")
summary(logreg_top2)

# COMMAND ----------

# McFadden R2
logreg_top2_ll_null <- logreg_top2$null.deviance/-2
logreg_top2_ll_proposed <- logreg_top2$deviance/-2

logreg_top2_rsq <- (logreg_top2_ll_null - logreg_top2_ll_proposed)/
  logreg_top2_ll_null

cat("R-squared:", logreg_top2_rsq)

# COMMAND ----------

# Calculate P-value for overall model significance
logreg_top2_p_val <- 1 - pchisq(2*(logreg_top2_ll_proposed - logreg_top2_ll_null),
                                df = (length(logreg_top2$coefficients) - 1)
                                )

cat("p-value: ", logreg_top2_p_val)

# COMMAND ----------

# Get summary table and CIs for plotting
logreg_top2_CIs <- data.frame(confint(logreg_top2))
logreg_top2_CIs$names <- rownames(logreg_top2_CIs)
logreg_top2_results <- data.frame(summary(logreg_top2)$coefficients)
logreg_top2_results$names <- rownames(logreg_top2_results)


logreg_top2_results <- full_join(logreg_top2_results, logreg_top2_CIs, by = "names") %>%
  select(names, everything()) %>%
  arrange(names)
colnames(logreg_top2_results) <- c("var", "coef", "StdError", "z_score", "p_val", "lower_CI", "upper_CI")

# COMMAND ----------

logreg_top2_results_spark <- SparkR::as.DataFrame(logreg_top2_results)
write.df(logreg_top2_results_spark, "/mnt/xql2001-gr5069/interim/final_project/logreg_top2_results.csv", "csv", header = TRUE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Marginal effects grid position

# COMMAND ----------

logreg_top2_results$coef

# COMMAND ----------

logreg_top2_results$lower_CI

# COMMAND ----------

logreg_top2_results$upper_CI

# COMMAND ----------

grid_alone <- logreg_top2_results$coef[5]
grid_alone

# COMMAND ----------

grid_alone_low <- logreg_top2_results$lower_CI[5]
grid_alone_low

# COMMAND ----------

grid_alone_high <- logreg_top2_results$upper_CI[5]
grid_alone_high

# COMMAND ----------

grid_w_1_stop <- logreg_top2_results$coef[5] + logreg_top2_results$coef[6]
grid_w_1_stop

# COMMAND ----------

grid_w_1_stop_low <- logreg_top2_results$lower_CI[5] + logreg_top2_results$lower_CI[6]
grid_w_1_stop_low

# COMMAND ----------

grid_w_1_stop_high <- logreg_top2_results$upper_CI[5] + logreg_top2_results$upper_CI[6]
grid_w_1_stop_high

# COMMAND ----------

grid_w_2_stop <- logreg_top2_results$coef[5] + logreg_top2_results$coef[7]
grid_w_2_stop

# COMMAND ----------

grid_w_2_stop_low <- logreg_top2_results$lower_CI[5] + logreg_top2_results$lower_CI[7]
grid_w_2_stop_low

# COMMAND ----------

grid_w_2_stop_high <- logreg_top2_results$upper_CI[5] + logreg_top2_results$upper_CI[7]
grid_w_2_stop_high

# COMMAND ----------

grid_w_3_stop <- logreg_top2_results$coef[5] + logreg_top2_results$coef[8]
grid_w_3_stop

# COMMAND ----------

grid_w_3_stop_low <- logreg_top2_results$lower_CI[5] + logreg_top2_results$lower_CI[8]
grid_w_3_stop_low

# COMMAND ----------

grid_w_3_stop_high <- logreg_top2_results$upper_CI[5] + logreg_top2_results$upper_CI[8]
grid_w_3_stop_high

# COMMAND ----------

# MAGIC %md
# MAGIC ## LogReg 2 - Predicting 1st/2nd among only 1st or 2nd place

# COMMAND ----------

df_1or2_dummy_source <- spark_read_csv (sc, name ="df_1or2_dummy", path="/mnt/xql2001-gr5069/processed/final_project/df_1or2_dummy.csv")
df_1or2_dummy <- collect(df_1or2_dummy_source)
display(df_1or2_dummy)

# COMMAND ----------

df_1or2_dummy$positionOrder <- as.factor(df_1or2_dummy$positionOrder)
df_1or2_dummy$pit_strategy_1 <- as.factor(df_1or2_dummy$pit_strategy_1)
df_1or2_dummy$pit_strategy_2 <- as.factor(df_1or2_dummy$pit_strategy_2)
df_1or2_dummy$pit_strategy_3 <- as.factor(df_1or2_dummy$pit_strategy_3)
df_1or2_dummy$pit_strategy_3stops <- as.factor(df_1or2_dummy$pit_strategy_3stops)
df_1or2_dummy$pit_strategy_missing <- as.factor(df_1or2_dummy$pit_strategy_missing)
df_1or2_dummy$constructor_quality_not_winner <- as.factor(df_1or2_dummy$constructor_quality_not_winner)
df_1or2_dummy$constructor_quality_winner <- as.factor(df_1or2_dummy$constructor_quality_winner)
df_1or2_dummy$circuit_type_race <- as.factor(df_1or2_dummy$circuit_type_race)
df_1or2_dummy$circuit_type_street <- as.factor(df_1or2_dummy$circuit_type_street)

# COMMAND ----------

glimpse(df_1or2_dummy)

# COMMAND ----------

df_1or2_dummy$positionOrder <- relevel(df_1or2_dummy$positionOrder, ref = "1")

# COMMAND ----------

logreg_1or2 <- glm(`positionOrder` ~ grid +
                   years_since_debut +
                   pit_strategy_1 +
                   pit_strategy_2 +
                   pit_strategy_3 +
                   pit_strategy_3stops +
                   pit_strategy_missing +
                   constructor_quality_not_winner +
                   constructor_quality_winner +
                   circuit_type_race +
                   circuit_type_street +
                   pit_strategy_1*grid +
                   pit_strategy_2*grid +
                   pit_strategy_3*grid +
                   pit_strategy_3stops*grid +
                   pit_strategy_missing*grid +
                   years_since_debut*grid +
                   years_since_debut*circuit_type_race + 
                   years_since_debut*circuit_type_street + 
                   years_since_debut*constructor_quality_not_winner +
                   years_since_debut*constructor_quality_winner,
                   data = df_1or2_dummy,
                   family="binomial")
summary(logreg_1or2)

# COMMAND ----------

# McFadden R2
logreg_1or2_ll_null <- logreg_1or2$null.deviance/-2
logreg_1or2_ll_proposed <- logreg_1or2$deviance/-2

logreg_1or2_rsq <- (logreg_1or2_ll_null - logreg_1or2_ll_proposed)/
  logreg_1or2_ll_null

cat("R-squared:", logreg_1or2_rsq)

# COMMAND ----------

# Calculate P-value for overall model significance
logreg_1or2_p_val <- 1 - pchisq(2*(logreg_1or2_ll_proposed - logreg_1or2_ll_null),
                                df = (length(logreg_1or2$coefficients) - 1)
                                )

cat("p-value: ", logreg_1or2_p_val)

# COMMAND ----------

# Get summary table and CIs for plotting
logreg_1or2_CIs <- data.frame(confint(logreg_1or2))
logreg_1or2_CIs$names <- rownames(logreg_1or2_CIs)
logreg_1or2_results <- data.frame(summary(logreg_1or2)$coefficients)
logreg_1or2_results$names <- rownames(logreg_1or2_results)


logreg_1or2_results <- full_join(logreg_1or2_results, logreg_1or2_CIs, by = "names") %>%
  select(names, everything()) %>%
  arrange(names)
colnames(logreg_1or2_results) <- c("var", "coef", "StdError", "z_score", "p_val", "lower_CI", "upper_CI")

# COMMAND ----------

logreg_1or2_results$coef

# COMMAND ----------

logreg_1or2_results$lower_CI

# COMMAND ----------

logreg_1or2_results$upper_CI

# COMMAND ----------

grid_alone <- logreg_1or2_results$coef[5]
grid_alone

# COMMAND ----------

grid_alone_low <- logreg_1or2_results$lower_CI[5]
grid_alone_low

# COMMAND ----------

grid_alone_high <- logreg_1or2_results$upper_CI[5]
grid_alone_high

# COMMAND ----------

grid_w_1_stop <- logreg_1or2_results$coef[5] + logreg_1or2_results$coef[6]
grid_w_1_stop

# COMMAND ----------

grid_w_1_stop_low <- logreg_1or2_results$lower_CI[5] + logreg_1or2_results$lower_CI[6]
grid_w_1_stop_low

# COMMAND ----------

grid_w_1_stop_high <- logreg_1or2_results$upper_CI[5] + logreg_1or2_results$upper_CI[6]
grid_w_1_stop_high

# COMMAND ----------

logreg_1or2_results_spark <- SparkR::as.DataFrame(logreg_1or2_results)
write.df(logreg_1or2_results_spark, "/mnt/xql2001-gr5069/interim/final_project/logreg_1or2_results.csv", "csv", header = TRUE)