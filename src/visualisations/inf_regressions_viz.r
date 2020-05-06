# Databricks notebook source
library(SparkR)
library(sparklyr)
library(ggplot2)

# COMMAND ----------

SparkR::sparkR.session()
sc <- spark_connect(method = "databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LogReg 1 - Predicting 1st & 2nd

# COMMAND ----------

logreg_top2_results_source <- spark_read_csv (sc, name ="logreg_top2_results", path="/mnt/xql2001-gr5069/interim/final_project/logreg_top2_results.csv")
logreg_top2_results <- collect(logreg_top2_results_source)
display(logreg_top2_results)

# COMMAND ----------

# Visualising significance
ggplot(logreg_top2_results) + 
  geom_point(aes(x = p_val, y = var)) + 
  geom_vline(xintercept = 0.05) + 
  annotate("text",
           x = 0.13,
           y = "years_since_debut:constructor_quality_winner1",
           label = " p = 0.05") +
  labs(x = "p-value",
       y = "Term",
       caption="Nominal variables with missing values are those
that were used as the baseline") +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"))

# COMMAND ----------

# Visualising coefficients
ggplot(logreg_top2_results) + 
  geom_point(aes(x = coef, y = var)) + 
  geom_point(aes(x = lower_CI, y = var), shape = 4) + 
  geom_point(aes(x = upper_CI, y = var), shape = 4) + 
  labs(x = "Coefficient",
       y = "Term",
       caption="Nominal variables with missing values are those
that were used as the baseline") +
    geom_vline(xintercept = 0) + 
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## LogReg 2 - Predicting 1st/2nd among only 1st or 2nd place

# COMMAND ----------

logreg_1or2_results_source <- spark_read_csv (sc, name ="logreg_1or2_results", path="/mnt/xql2001-gr5069/interim/final_project/logreg_1or2_results.csv")
logreg_1or2_results <- collect(logreg_1or2_results_source)
display(logreg_1or2_results)

# COMMAND ----------

# Visualising significance
ggplot(logreg_1or2_results) + 
  geom_point(aes(x = p_val, y = var)) + 
  geom_vline(xintercept = 0.05) + 
  annotate("text",
           x = 0.13,
           y = "years_since_debut:constructor_quality_winner1",
           label = " p = 0.05") +
  labs(x = "p-value",
       y = "Term",
       caption="Nominal variables with missing values are those
that were used as the baseline") +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"))

# COMMAND ----------

# Visualising coefficients
ggplot(logreg_1or2_results) + 
  geom_point(aes(x = coef, y = var)) + 
  geom_point(aes(x = lower_CI, y = var), shape = 4) + 
  geom_point(aes(x = upper_CI, y = var), shape = 4) + 
  labs(x = "Coefficient",
       y = "Term",
       caption="Nominal variables with missing values are those
that were used as the baseline") +
    geom_vline(xintercept = 0) + 
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"))
