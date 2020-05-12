# Databricks notebook source
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

#w q1 train sets
pd_train_w_q1_wo_resamp = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_w_q1_wo_resamp.csv',header=True).toPandas()

#test sets
pd_test_w_q1 = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_test_w_q1.csv',header=True).toPandas()

# COMMAND ----------

# test set dfs to work with
pd_test_w_q1_working = pd_test_w_q1.copy(deep = True)
pd_test_w_q1_working['Second'] = pd_test_w_q1_working['Second'].astype(int)

# Test sets to give sklearn
pd_test_w_q1_4_predict = pd_test_w_q1.drop(['raceId', 'driverId', 'constructorId', 'Second'], axis = 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Testing Setup

# COMMAND ----------

# Model Test
rf = RandomForestClassifier(n_estimators = 1000, max_depth = 5, max_features = 'sqrt', random_state = 123)
rf_model = rf.fit(X = pd_train_w_q1_wo_resamp.drop(['Second'], axis = 1),
                  y = pd_train_w_q1_wo_resamp[['Second']].values.ravel())
pd_test_w_q1_working['predict_proba'] = rf_model.predict_proba(pd_test_w_q1_4_predict)[:, 1]

# COMMAND ----------

# Get max probability per race
max_prob_per_race = pd.DataFrame(pd_test_w_q1_working.groupby(['raceId']).predict_proba.max())
pd_test_w_q1_working = pd_test_w_q1_working\
  .merge(max_prob_per_race, on = 'raceId', how = 'inner', suffixes = ('_driver', '_max'))

# COMMAND ----------

# Classify driver with max probability for that race as 1
pd_test_w_q1_working['pred'] = pd_test_w_q1_working['predict_proba_driver']\
  .eq(pd_test_w_q1_working['predict_proba_max']).astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Check if classification procedure is correct

# COMMAND ----------

pd_test_w_q1_working['pred'].value_counts()

# COMMAND ----------

pd_test_w_q1_working['Second'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Results test

# COMMAND ----------

confusion_matrix(pd_test_w_q1_working['Second'], pd_test_w_q1_working['pred'])

# COMMAND ----------

precision_score(pd_test_w_q1_working['Second'], pd_test_w_q1_working['pred'])

# COMMAND ----------

# Feature Names
pd_train_w_q1_wo_resamp.drop(['Second'], axis = 1).columns

# COMMAND ----------

# Feature importances
rf_model.feature_importances_
