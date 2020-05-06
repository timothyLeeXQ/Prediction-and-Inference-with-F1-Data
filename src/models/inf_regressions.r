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

logreg_1or2_results_spark <- SparkR::as.DataFrame(logreg_1or2_results)
write.df(logreg_1or2_results_spark, "/mnt/xql2001-gr5069/interim/final_project/logreg_1or2_results.csv", "csv", header = TRUE)
