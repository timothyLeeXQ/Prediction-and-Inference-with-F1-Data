# Databricks notebook source
dbutils.library.installPyPI("mlflow", "1.0.0")

# COMMAND ----------

import mlflow.sklearn
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

#w q1 train sets
pd_train_w_q1_wo_resamp = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_w_q1_wo_resamp.csv',header=True).toPandas()
pd_train_w_q1_undersamp = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_w_q1_undersamp.csv',header=True).toPandas()
pd_train_w_q1_oversamp = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_w_q1_oversamp.csv',header=True).toPandas()
pd_train_w_q1_smote = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_w_q1_smote.csv',header=True).toPandas()

#w/o q1 train sets
pd_train_wo_q1_wo_resamp = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_wo_q1_wo_resamp.csv',header=True).toPandas()
pd_train_wo_q1_undersamp = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_wo_q1_undersamp.csv',header=True).toPandas()
pd_train_wo_q1_oversamp = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_wo_q1_oversamp.csv',header=True).toPandas()
pd_train_wo_q1_smote = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_train_wo_q1_smote.csv',header=True).toPandas()

#test sets
pd_test_w_q1 = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_test_w_q1.csv',header=True).toPandas()
pd_test_wo_q1 = spark.read.csv('/mnt/xql2001-gr5069/processed/final_project/modelsets/spark_pred_test_wo_q1.csv',header=True).toPandas()

# COMMAND ----------

# test set dfs to work with
pd_test_w_q1_working = pd_test_w_q1.copy(deep = True)
pd_test_w_q1_working['Second'] = pd_test_w_q1_working['Second'].astype(int)
pd_test_wo_q1_working = pd_test_wo_q1.copy(deep = True)
pd_test_wo_q1_working['Second'] = pd_test_wo_q1_working['Second'].astype(int)

# Test sets to give sklearn
pd_test_w_q1_4_predict = pd_test_w_q1.drop(['raceId', 'driverId', 'constructorId', 'Second'], axis = 1)
pd_test_wo_q1_4_predict = pd_test_wo_q1.drop(['raceId', 'driverId', 'constructorId', 'Second'], axis = 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML Flow Setup

# COMMAND ----------

def log_rf(run_name, params, train, test_4_model, test_w_Ids):
  # :::::::::::: DESCRIPTION ::::::::::::
  # This function is used to fit a Random Forest model to the F1 data
  # and generate  predicted second place positions based on the most 
  # likely second place finish for each race.
  #
  # Model parameters and feature importances, are also logged in MLFlow
  #
  # ::::::::: INPUTS ::::::::::::
  # 1. run_name - A string, the name of the run registered in MLFlow
  # 2. Params - A dictionary of the parameters to pass to the model in 
  # the form {'parameter': value}
  # 3. train - The training data, to pass to model.fit() for an Sklearn 
  # model object
  # 4. test_4_model - The test data, to pass to model.predict_proba()
  # for an Sklearn model object. Should only contain features, without 
  # reference Ids.
  # 5. test_w_Ids - The test data with Ids, to group predicted probabilities
  # by raceId and compute the most likely second place finish in each race.
  #
  # ::::::::: OUTPUT ::::::::::::
  # Prints Precision Score
  # Model, parameters, and feature importance logged in MLFlow
  
  with mlflow.start_run(run_name = run_name) as run:
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
    
    # Create model, train it
    model = RandomForestClassifier(**params)
    model.fit(train.drop(['Second'], axis = 1),
              train[['Second']].values.ravel())
    
    # Log model
    # mlflow.sklearn.log_model(model, "random-forest-model")
    
    # Log params
    [mlflow.log_param(param, value) for param, value in params.items()]
    
    # Prediction
    test_w_Ids['predict_proba'] = model.predict_proba(test_4_model)[:, 1]
    max_prob_per_race = pd.DataFrame(test_w_Ids.groupby(['raceId']).predict_proba.max())
    test_w_Ids = test_w_Ids\
      .merge(max_prob_per_race, on = 'raceId', how = 'inner', suffixes = ('_driver', '_max'))
    
    # Classify driver with max probability for that race as 1
    test_w_Ids['pred'] = test_w_Ids['predict_proba_driver']\
      .eq(test_w_Ids['predict_proba_max']).astype(int)

    # Create metrics
    precision = precision_score(test_w_Ids['Second'], test_w_Ids['pred'])
    
    # Print metrics
    print("precision: {}".format(precision))
    
    # Log metrics
    mlflow.log_metric("precision", precision)
    
    # Create feature importance
    importance = pd.DataFrame(list(zip(train.drop(['Second'], axis = 1).columns,
                                       model.feature_importances_)), 
                                columns=["Feature", "Importance"]
                              ).sort_values("Importance", ascending=False)
    
    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    temp_name = temp.name
    try:
      importance.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-importance.csv")
    finally:
      temp.close() # Delete the temp file

    return run.info.run_uuid

# COMMAND ----------

def log_gb(run_name, params, train, test_4_model, test_w_Ids):
  # :::::::::::: DESCRIPTION ::::::::::::
  # This function is used to fit a Random Forest model to the F1 data
  # and generate  predicted second place positions based on the most 
  # likely second place finish for each race.
  #
  # Model parameters and feature importances, are also logged in MLFlow
  #
  # ::::::::: INPUTS ::::::::::::
  # 1. run_name - A string, the name of the run registered in MLFlow
  # 2. Params - A dictionary of the parameters to pass to the model in 
  # the form {'parameter': value}
  # 3. train - The training data, to pass to model.fit() for an Sklearn 
  # model object
  # 4. test_4_model - The test data, to pass to model.predict_proba()
  # for an Sklearn model object. Should only contain features, without 
  # reference Ids.
  # 5. test_w_Ids - The test data with Ids, to group predicted probabilities
  # by raceId and compute the most likely second place finish in each race.
  #
  # ::::::::: OUTPUT ::::::::::::
  # Prints Precision Score
  # Model, parameters, and feature importance logged in MLFlow
  
  with mlflow.start_run(run_name = run_name) as run:
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
    
    # Create model, train it
    model = GradientBoostingClassifier(**params)
    model.fit(train.drop(['Second'], axis = 1),
              train[['Second']].values.ravel())
    
    # Log model
    # mlflow.sklearn.log_model(model, "random-forest-model")
    
    # Log params
    [mlflow.log_param(param, value) for param, value in params.items()]
    
    # Prediction
    test_w_Ids['predict_proba'] = model.predict_proba(test_4_model)[:, 1]
    max_prob_per_race = pd.DataFrame(test_w_Ids.groupby(['raceId']).predict_proba.max())
    test_w_Ids = test_w_Ids\
      .merge(max_prob_per_race, on = 'raceId', how = 'inner', suffixes = ('_driver', '_max'))
    
    # Classify driver with max probability for that race as 1
    test_w_Ids['pred'] = test_w_Ids['predict_proba_driver']\
      .eq(test_w_Ids['predict_proba_max']).astype(int)

    # Create metrics
    precision = precision_score(test_w_Ids['Second'], test_w_Ids['pred'])
    
    # Print metrics
    print("precision: {}".format(precision))
    
    # Log metrics
    mlflow.log_metric("precision", precision)
    
    # Create feature importance
    importance = pd.DataFrame(list(zip(train.drop(['Second'], axis = 1).columns,
                                       model.feature_importances_)), 
                                columns=["Feature", "Importance"]
                              ).sort_values("Importance", ascending=False)
    
    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    temp_name = temp.name
    try:
      importance.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-importance.csv")
    finally:
      temp.close() # Delete the temp file

    return run.info.run_uuid

# COMMAND ----------

# MAGIC %md
# MAGIC ### Testing MLFlow implementation

# COMMAND ----------

params_test_rf = {'n_estimators': 1000,
                  'max_depth': 5,
                  'max_features': 'sqrt',
                  'random_state': 123
                 }

log_rf('test run rf', params_test_rf, pd_train_w_q1_wo_resamp, pd_test_w_q1_4_predict, pd_test_w_q1_working)

# COMMAND ----------

params_test_gb = {'loss': 'deviance',
                  'learning_rate': 0.1,
                  'n_estimators': 1000,
                  'max_depth': 5,
                  'max_features': 'sqrt',
                  'random_state': 123
                 }

log_gb('test run gb', params_test_gb, pd_train_w_q1_wo_resamp, pd_test_w_q1_4_predict, pd_test_w_q1_working)

# COMMAND ----------

# MAGIC %md
# MAGIC #### It works!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Runs

# COMMAND ----------

# MAGIC %md
# MAGIC ### No Resampling, w/ q1 data 
# MAGIC 
# MAGIC **RF**
# MAGIC 
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123
# MAGIC 
# MAGIC **GB**
# MAGIC * loss - 'deviance'
# MAGIC * learning_rate - 0.3, 0.5, 0.7, 0.9
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123

# COMMAND ----------

for depth in np.arange(1, 11):
  for feature_no in ['sqrt', None]:
    params_rf_w_q1_wo_resamp = {'n_estimators': 1000,
                                'max_depth': depth,
                                'max_features': feature_no,
                                'random_state': 123
                               }
    log_rf('rf_w_q1_wo_resamp',
           params_rf_w_q1_wo_resamp,
           pd_train_w_q1_wo_resamp,
           pd_test_w_q1_4_predict,
           pd_test_w_q1_working)

# COMMAND ----------

for depth in np.arange(1, 11):
  for learning_rate in [0.3, 0.5, 0.7, 0.9]:
    for feature_no in ['sqrt', None]:
      params_gb_w_q1_wo_resamp = {'loss': 'deviance',
                                  'learning_rate': learning_rate,
                                  'n_estimators': 1000,
                                  'max_depth': depth,
                                  'max_features': feature_no,
                                  'random_state': 123
                                 }
      log_gb('gb_w_q1_wo_resamp',
             params_gb_w_q1_wo_resamp,
             pd_train_w_q1_wo_resamp,
             pd_test_w_q1_4_predict,
             pd_test_w_q1_working)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Undersampling, w/ q1 data 
# MAGIC 
# MAGIC **RF**
# MAGIC 
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123
# MAGIC 
# MAGIC **GB**
# MAGIC * loss - 'deviance'
# MAGIC * learning_rate - 0.3, 0.5, 0.7, 0.9
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123

# COMMAND ----------

for depth in np.arange(1, 11):
  for feature_no in ['sqrt', None]:
    params_rf_w_q1_undersamp = {'n_estimators': 1000,
                                'max_depth': depth,
                                'max_features': feature_no,
                                'random_state': 123
                               }
    log_rf('rf_w_q1_undersamp',
           params_rf_w_q1_undersamp,
           pd_train_w_q1_undersamp,
           pd_test_w_q1_4_predict,
           pd_test_w_q1_working)

# COMMAND ----------

for depth in np.arange(1, 11):
  for learning_rate in [0.3, 0.5, 0.7, 0.9]:
    for feature_no in ['sqrt', None]:
      params_gb_w_q1_undersamp = {'loss': 'deviance',
                                  'learning_rate': learning_rate,
                                  'n_estimators': 1000,
                                  'max_depth': depth,
                                  'max_features': feature_no,
                                  'random_state': 123
                                 }
      log_gb('gb_w_q1_undersamp',
             params_gb_w_q1_undersamp,
             pd_train_w_q1_undersamp,
             pd_test_w_q1_4_predict,
             pd_test_w_q1_working)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Oversampling, w/ q1 data 
# MAGIC 
# MAGIC **RF**
# MAGIC 
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123
# MAGIC 
# MAGIC **GB**
# MAGIC * loss - 'deviance'
# MAGIC * learning_rate - 0.3, 0.5, 0.7, 0.9
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123

# COMMAND ----------

for depth in np.arange(1, 11):
  for feature_no in ['sqrt', None]:
    params_rf_w_q1_oversamp = {'n_estimators': 1000,
                               'max_depth': depth,
                               'max_features': feature_no,
                               'random_state': 123
                              }
    log_rf('rf_w_q1_oversamp',
           params_rf_w_q1_oversamp,
           pd_train_w_q1_oversamp,
           pd_test_w_q1_4_predict,
           pd_test_w_q1_working)

# COMMAND ----------

for depth in np.arange(1, 11):
  for learning_rate in [0.3, 0.5, 0.7, 0.9]:
    for feature_no in ['sqrt', None]:
      params_gb_w_q1_oversamp = {'loss': 'deviance',
                                 'learning_rate': learning_rate,
                                 'n_estimators': 1000,
                                 'max_depth': depth,
                                 'max_features': feature_no,
                                 'random_state': 123
                                }
      log_gb('gb_w_q1_oversamp',
             params_gb_w_q1_oversamp,
             pd_train_w_q1_oversamp,
             pd_test_w_q1_4_predict,
             pd_test_w_q1_working)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SMOTE, w/ q1 data 
# MAGIC 
# MAGIC **RF**
# MAGIC 
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123
# MAGIC 
# MAGIC **GB**
# MAGIC * loss - 'deviance'
# MAGIC * learning_rate - 0.3, 0.5, 0.7, 0.9
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123

# COMMAND ----------

for depth in np.arange(1, 11):
  for feature_no in ['sqrt', None]:
    params_rf_w_q1_smote = {'n_estimators': 1000,
                            'max_depth': depth,
                            'max_features': feature_no,
                            'random_state': 123
                           }
    log_rf('rf_w_q1_smote',
           params_rf_w_q1_smote,
           pd_train_w_q1_smote,
           pd_test_w_q1_4_predict,
           pd_test_w_q1_working)

# COMMAND ----------

for depth in np.arange(1, 11):
  for learning_rate in [0.3, 0.5, 0.7, 0.9]:
    for feature_no in ['sqrt', None]:
      params_gb_w_q1_smote = {'loss': 'deviance',
                              'learning_rate': learning_rate,
                              'n_estimators': 1000,
                              'max_depth': depth,
                              'max_features': feature_no,
                              'random_state': 123
                             }
      log_gb('gb_w_q1_smote',
             params_gb_w_q1_smote,
             pd_train_w_q1_smote,
             pd_test_w_q1_4_predict,
             pd_test_w_q1_working)

# COMMAND ----------

# MAGIC %md
# MAGIC Have tried the code above twice, cluster times out after 2 hours and shuts down. 
# MAGIC However, ML Flow managed to log the previous runs, so we can continue 
# MAGIC from where the cluster shut down.

# COMMAND ----------

for depth in np.arange(9, 11):
  for learning_rate in [0.3, 0.5, 0.7, 0.9]:
    for feature_no in ['sqrt', None]:
      params_gb_w_q1_smote = {'loss': 'deviance',
                              'learning_rate': learning_rate,
                              'n_estimators': 1000,
                              'max_depth': depth,
                              'max_features': feature_no,
                              'random_state': 123
                             }
      log_gb('gb_w_q1_smote',
             params_gb_w_q1_smote,
             pd_train_w_q1_smote,
             pd_test_w_q1_4_predict,
             pd_test_w_q1_working)

# COMMAND ----------

# MAGIC %md
# MAGIC ### No Resampling, w/o q1 data 
# MAGIC 
# MAGIC **RF**
# MAGIC 
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123
# MAGIC 
# MAGIC **GB**
# MAGIC * loss - 'deviance'
# MAGIC * learning_rate - 0.3, 0.5, 0.7, 0.9
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123

# COMMAND ----------

for depth in np.arange(1, 11):
  for feature_no in ['sqrt', None]:
    params_rf_wo_q1_wo_resamp = {'n_estimators': 1000,
                                 'max_depth': depth,
                                 'max_features': feature_no,
                                 'random_state': 123
                                }
    log_rf('rf_wo_q1_wo_resamp',
           params_rf_wo_q1_wo_resamp,
           pd_train_wo_q1_wo_resamp,
           pd_test_wo_q1_4_predict,
           pd_test_wo_q1_working)

# COMMAND ----------

for depth in np.arange(1, 11):
  for learning_rate in [0.3, 0.5, 0.7, 0.9]:
    for feature_no in ['sqrt', None]:
      params_gb_wo_q1_wo_resamp = {'loss': 'deviance',
                                   'learning_rate': learning_rate,
                                   'n_estimators': 1000,
                                   'max_depth': depth,
                                   'max_features': feature_no,
                                   'random_state': 123
                                  }
      log_gb('gb_wo_q1_wo_resamp',
             params_gb_wo_q1_wo_resamp,
             pd_train_wo_q1_wo_resamp,
             pd_test_wo_q1_4_predict,
             pd_test_wo_q1_working)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Undersampling, w/o q1 data 
# MAGIC 
# MAGIC **RF**
# MAGIC 
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123
# MAGIC 
# MAGIC **GB**
# MAGIC * loss - 'deviance'
# MAGIC * learning_rate - 0.3, 0.5, 0.7, 0.9
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123

# COMMAND ----------

for depth in np.arange(1, 11):
  for feature_no in ['sqrt', None]:
    params_rf_wo_q1_undersamp = {'n_estimators': 1000,
                                 'max_depth': depth,
                                 'max_features': feature_no,
                                 'random_state': 123
                                }
    log_rf('rf_wo_q1_undersamp',
           params_rf_wo_q1_undersamp,
           pd_train_wo_q1_undersamp,
           pd_test_wo_q1_4_predict,
           pd_test_wo_q1_working)

# COMMAND ----------

for depth in np.arange(1, 11):
  for learning_rate in [0.3, 0.5, 0.7, 0.9]:
    for feature_no in ['sqrt', None]:
      params_gb_wo_q1_undersamp = {'loss': 'deviance',
                                   'learning_rate': learning_rate,
                                   'n_estimators': 1000,
                                   'max_depth': depth,
                                   'max_features': feature_no,
                                   'random_state': 123
                                  }
      log_gb('gb_wo_q1_undersamp',
             params_gb_wo_q1_undersamp,
             pd_train_wo_q1_undersamp,
             pd_test_wo_q1_4_predict,
             pd_test_wo_q1_working)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Oversampling, w/o q1 data 
# MAGIC 
# MAGIC **RF**
# MAGIC 
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123
# MAGIC 
# MAGIC **GB**
# MAGIC * loss - 'deviance'
# MAGIC * learning_rate - 0.3, 0.5, 0.7, 0.9
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123

# COMMAND ----------

for depth in np.arange(1, 11):
  for feature_no in ['sqrt', None]:
    params_rf_wo_q1_oversamp = {'n_estimators': 1000,
                               'max_depth': depth,
                               'max_features': feature_no,
                               'random_state': 123
                              }
    log_rf('rf_wo_q1_oversamp',
           params_rf_wo_q1_oversamp,
           pd_train_wo_q1_oversamp,
           pd_test_wo_q1_4_predict,
           pd_test_wo_q1_working)

# COMMAND ----------

for depth in np.arange(1, 11):
  for learning_rate in [0.3, 0.5, 0.7, 0.9]:
    for feature_no in ['sqrt', None]:
      params_gb_wo_q1_oversamp = {'loss': 'deviance',
                                 'learning_rate': learning_rate,
                                 'n_estimators': 1000,
                                 'max_depth': depth,
                                 'max_features': feature_no,
                                 'random_state': 123
                                }
      log_gb('gb_wo_q1_oversamp',
             params_gb_wo_q1_oversamp,
             pd_train_wo_q1_oversamp,
             pd_test_wo_q1_4_predict,
             pd_test_wo_q1_working)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SMOTE, w/o q1 data 
# MAGIC 
# MAGIC **RF**
# MAGIC 
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123
# MAGIC 
# MAGIC **GB**
# MAGIC * loss - 'deviance'
# MAGIC * learning_rate - 0.3, 0.5, 0.7, 0.9
# MAGIC * n_estimators - 1000
# MAGIC * max_depth - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# MAGIC * max_features - 'sqrt', None
# MAGIC * random_state - 123

# COMMAND ----------

for depth in np.arange(1, 11):
  for feature_no in ['sqrt', None]:
    params_rf_wo_q1_smote = {'n_estimators': 1000,
                            'max_depth': depth,
                            'max_features': feature_no,
                            'random_state': 123
                           }
    log_rf('rf_wo_q1_smote',
           params_rf_wo_q1_smote,
           pd_train_wo_q1_smote,
           pd_test_wo_q1_4_predict,
           pd_test_wo_q1_working)

# COMMAND ----------

for depth in np.arange(1, 11):
  for learning_rate in [0.3, 0.5, 0.7, 0.9]:
    for feature_no in ['sqrt', None]:
      params_gb_wo_q1_smote = {'loss': 'deviance',
                              'learning_rate': learning_rate,
                              'n_estimators': 1000,
                              'max_depth': depth,
                              'max_features': feature_no,
                              'random_state': 123
                             }
      log_gb('gb_wo_q1_smote',
             params_gb_wo_q1_smote,
             pd_train_wo_q1_smote,
             pd_test_wo_q1_4_predict,
             pd_test_wo_q1_working)
