# Databricks notebook source
library(SparkR)
library(sparklyr)
library(ggplot2)
library(dplyr)

# COMMAND ----------

SparkR::sparkR.session()
sc <- spark_connect(method = "databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Runs

# COMMAND ----------

model_runs_source <- spark_read_csv(sc, name ="model_runs", path="/mnt/xql2001-gr5069/processed/final_project/mlflow_output/model_runs_mlflow.csv")
model_runs <- collect(model_runs_source)
display(model_runs)

# COMMAND ----------

# Visualising results
model_runs %>%
  filter(Name != "test run rf") %>%
  filter(Name != "test run gb") %>%
  ggplot() + 
  geom_point(aes(x = Name, y = precision)) + 
  labs(x = "Algorithm and Sampling method",
       y = "Precision") +
  coord_flip() +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Best Model Feature Importances

# COMMAND ----------

feat_imp_source <- spark_read_csv(sc, name ="model_runs", path="/mnt/xql2001-gr5069/processed/final_project/mlflow_output/feature-importance_best_model.csv")
feat_imp <- collect(feat_imp_source)
display(feat_imp)

# COMMAND ----------

# Visualising significance
ggplot(feat_imp) + 
  geom_col(aes(x = Feature, y = Importance * 100)) + 
  labs(x = "Feature",
       y = "Relative Importance (Percentage)") +
  coord_flip() +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"))
