# Databricks notebook source
dbutils.library.installPyPI("imbalanced-learn", "0.6.2")

# COMMAND ----------

import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train

# COMMAND ----------

df_pred_train = spark.read.csv('/mnt/xql2001-gr5069/interim/final_project/df_pred_train.csv',header=True)
display(df_pred_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### There are 8 types of train sets
# MAGIC * W/ q1_ms - No resampling, random undersampling, random oversampling, SMOTE
# MAGIC * W/o q1_ms - No resampling, random undersampling, random oversampling, SMOTE

# COMMAND ----------

# MAGIC %md
# MAGIC ##### w/q1_ms - no resampling

# COMMAND ----------

df_pred_train_w_q1_wo_resamp = df_pred_train.drop('raceId', 'driverId', 'constructorId', 'year', 'q1_ms_OG')
display(df_pred_train_w_q1_wo_resamp)

# COMMAND ----------

# MAGIC %md
# MAGIC Get a pd df to dummy code constructor quality, and to make the rest

# COMMAND ----------

pd_pred_train_w_q1_wo_resamp = df_pred_train_w_q1_wo_resamp.toPandas()
pd_pred_train_w_q1_wo_resamp = pd.get_dummies(pd_pred_train_w_q1_wo_resamp, columns = ['constructor_quality'])

# COMMAND ----------

spark_pred_train_w_q1_wo_resamp = spark.createDataFrame(pd_pred_train_w_q1_wo_resamp)
#display(spark_pred_train_w_q1_wo_resamp)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### w/q1_ms - random undersampling

# COMMAND ----------

rus = RandomUnderSampler(random_state=0)

df_pred_train_w_q1_undersamp_X, df_pred_train_w_q1_undersamp_y = rus.\
  fit_sample(X = pd_pred_train_w_q1_wo_resamp.drop(['Second'], axis = 1),
             y = pd_pred_train_w_q1_wo_resamp[['Second']])

# COMMAND ----------

df_pred_train_w_q1_undersamp = df_pred_train_w_q1_undersamp_X.merge(df_pred_train_w_q1_undersamp_y,
                                                                    how='inner',
                                                                    left_index=True,
                                                                    right_index=True,
                                                                    sort=False)

# COMMAND ----------

# Check if join worked fine
df_pred_train_w_q1_undersamp.describe(include = 'all')

# COMMAND ----------

spark_pred_train_w_q1_undersamp = spark.createDataFrame(df_pred_train_w_q1_undersamp)
# display(spark_pred_train_w_q1_undersamp)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### w/q1_ms - random oversampling

# COMMAND ----------

ros = RandomOverSampler(random_state=0)
df_pred_train_w_q1_oversamp_X, df_pred_train_w_q1_oversamp_y = ros.\
  fit_sample(X = pd_pred_train_w_q1_wo_resamp.drop(['Second'], axis = 1),
             y = pd_pred_train_w_q1_wo_resamp[['Second']])

# COMMAND ----------

df_pred_train_w_q1_oversamp = df_pred_train_w_q1_oversamp_X.merge(df_pred_train_w_q1_oversamp_y,
                                                                  how='inner',
                                                                  left_index=True,
                                                                  right_index=True,
                                                                  sort=False)

# COMMAND ----------

# Check if join worked fine
df_pred_train_w_q1_oversamp.describe(include = 'all')

# COMMAND ----------

spark_pred_train_w_q1_oversamp = spark.createDataFrame(df_pred_train_w_q1_oversamp)
# display(spark_pred_train_w_q1_oversamp)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### w/q1_ms - SMOTE

# COMMAND ----------

smote = SMOTE(sampling_strategy='minority', random_state = 0)
df_pred_train_w_q1_smote_X, df_pred_train_w_q1_smote_y = smote.\
  fit_sample(X = pd_pred_train_w_q1_wo_resamp.drop(['Second'], axis = 1),
             y = pd_pred_train_w_q1_wo_resamp[['Second']])

# COMMAND ----------

df_pred_train_w_q1_smote = df_pred_train_w_q1_smote_X.merge(df_pred_train_w_q1_smote_y,
                                                            how='inner',
                                                            left_index=True,
                                                            right_index=True,
                                                            sort=False)

# COMMAND ----------

df_pred_train_w_q1_smote.describe(include = 'all')

# COMMAND ----------

spark_pred_train_w_q1_smote = spark.createDataFrame(df_pred_train_w_q1_smote)
#display(spark_pred_train_w_q1_smote)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### w/o q1_ms - No resampling

# COMMAND ----------

spark_pred_train_wo_q1_wo_resamp = spark_pred_train_w_q1_wo_resamp.drop('q1_ms_impute')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### w/o q1_ms - random undersampling

# COMMAND ----------

spark_pred_train_wo_q1_undersamp = spark_pred_train_w_q1_undersamp.drop('q1_ms_impute')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### w/o q1_ms - random oversampling

# COMMAND ----------

spark_pred_train_wo_q1_oversamp = spark_pred_train_w_q1_oversamp.drop('q1_ms_impute')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### w/o q1_ms - SMOTE

# COMMAND ----------

spark_pred_train_wo_q1_smote = spark_pred_train_w_q1_smote.drop('q1_ms_impute')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test

# COMMAND ----------

df_pred_test = spark.read.csv('/mnt/xql2001-gr5069/interim/final_project/df_pred_test.csv',header=True)
display(df_pred_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### There are 2 types of test sets
# MAGIC * W/ q1_ms
# MAGIC * W/o q1_ms
# MAGIC 
# MAGIC We need to keep IDs - probably need it to match test output original 
# MAGIC race data to get 1 winner per race.
# MAGIC But need to dummy code constructor_quality

# COMMAND ----------

# MAGIC %md
# MAGIC ##### w/ q1_ms

# COMMAND ----------

df_pred_test_w_q1 = df_pred_test.drop('year', 'q1_ms_OG')
display(df_pred_test_w_q1)

# COMMAND ----------

pd_pred_test_w_q1 = df_pred_test_w_q1.toPandas()
pd_pred_test_w_q1 = pd.get_dummies(pd_pred_test_w_q1, columns = ['constructor_quality'])
spark_pred_test_w_q1 = spark.createDataFrame(pd_pred_test_w_q1)
display(spark_pred_test_w_q1)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### w/o q1_ms

# COMMAND ----------

df_pred_test_wo_q1 = df_pred_test.drop('year', 'q1_ms_OG', 'q1_ms_impute')
display(df_pred_test_wo_q1)

# COMMAND ----------

pd_pred_test_wo_q1 = df_pred_test_wo_q1.toPandas()
pd_pred_test_wo_q1 = pd.get_dummies(pd_pred_test_wo_q1, columns = ['constructor_quality'])
spark_pred_test_wo_q1 = spark.createDataFrame(pd_pred_test_wo_q1)
display(spark_pred_test_wo_q1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to S3

# COMMAND ----------

#w q1 train sets
spark_pred_train_w_q1_wo_resamp.coalesce(1)\
  .write.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_w_q1_wo_resamp.csv', header=True)
spark_pred_train_w_q1_undersamp.coalesce(1)\
  .write.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_w_q1_undersamp.csv', header=True)
spark_pred_train_w_q1_oversamp.coalesce(1)\
  .write.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_w_q1_oversamp.csv', header=True)
spark_pred_train_w_q1_smote.coalesce(1)\
  .write.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_w_q1_smote.csv', header=True)

#w/o q1 train sets
spark_pred_train_wo_q1_wo_resamp.coalesce(1)\
  .write.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_wo_q1_wo_resamp.csv', header=True)
spark_pred_train_wo_q1_undersamp.coalesce(1)\
  .write.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_wo_q1_undersamp.csv', header=True)
spark_pred_train_wo_q1_oversamp.coalesce(1)\
  .write.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_wo_q1_oversamp.csv', header=True)
spark_pred_train_wo_q1_smote.coalesce(1)\
  .write.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_wo_q1_smote.csv', header=True)

#test sets
spark_pred_test_w_q1.coalesce(1)\
  .write.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_test_w_q1.csv', header=True)
spark_pred_test_wo_q1.coalesce(1)\
  .write.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_test_wo_q1.csv', header=True)
