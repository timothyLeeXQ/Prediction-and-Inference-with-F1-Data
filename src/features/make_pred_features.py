# Databricks notebook source
from pyspark.sql import functions as F
import pandas as pd

# Requires install scikit-learn==0.22.2.post1
from sklearn.impute import KNNImputer

# COMMAND ----------

df_races = spark.read.csv('/mnt/ne-gr5069/raw/races.csv',header=True)
display(df_races)

# COMMAND ----------

df_drivers = spark.read.csv('/mnt/ne-gr5069/raw/drivers.csv',header=True)
display(df_drivers)

# COMMAND ----------

df_constructors = spark.read.csv('/mnt/ne-gr5069/raw/constructors.csv',header=True)
display(df_constructors)

# COMMAND ----------

df_qualifying = spark.read.csv('/mnt/ne-gr5069/raw/qualifying.csv',header=True)
display(df_qualifying)

# COMMAND ----------

df_results = spark.read.csv('/mnt/ne-gr5069/raw/results.csv',header=True)
display(df_results)

# COMMAND ----------

df_results.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table generation
# MAGIC 
# MAGIC Unfortunately we cannot use the initial table inherited from
# MAGIC master table from inf features (df_inf_master). 
# MAGIC 
# MAGIC It removed data where where grid position is 0, referring to crashes,
# MAGIC DNQ, or car faults.
# MAGIC 
# MAGIC * Makes sense for inferential task - it is self-explanatory that 
# MAGIC DNQ/car fault means you can't win #2, so its inclusion is not theoretically
# MAGIC valuable.
# MAGIC * For prediction we want the model to pick this up on its own - removing
# MAGIC DNQ/car fault data from the test set based on statusId and just saying they
# MAGIC didn't get #2 feels like doing the model's work for it.
# MAGIC 
# MAGIC Also, it only includes races up to 2010
# MAGIC 
# MAGIC Hence, build data from scratch from df_results
# MAGIC * Filter to include only races from 1950 - 2017 (inclusive) - use df_races
# MAGIC 
# MAGIC Needed references to cross reference other tables and build features:
# MAGIC * Race ID
# MAGIC * Driver ID
# MAGIC * Constructor ID
# MAGIC 
# MAGIC Features already present:
# MAGIC * Starting Grid Position (need to deal with position 0 entries)
# MAGIC 
# MAGIC 
# MAGIC Features to build:
# MAGIC * **Outcome** - Number 2 - Build from positionOrder
# MAGIC * Constructor quality (Need to dummy code) - build from df_constructors 
# MAGIC and constructor champion list
# MAGIC * Driver Experience - build from df_races
# MAGIC * Driver Average Finishing Position - Build from df_results
# MAGIC * Constructor Average Finishing Position - Build from df_results
# MAGIC * Average Qualifying lap time - Build from df_qualifying
# MAGIC * Q3 Qualifying lap time - Build from df_qualifying
# MAGIC 
# MAGIC Save to S3

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter to include only races from 1950-2017 (inclusive)
# MAGIC ### Drop irrelevant columns

# COMMAND ----------

df_results_filtered = df_results.select('raceId', 'driverId', 'constructorId', 'grid', 'positionOrder')\
  .join(df_races.select('raceId', 'year'), on = ['raceId'], how = 'left')

df_results_filtered = df_results_filtered\
  .filter(df_results_filtered['year'] >= 1950)\
  .filter(df_results_filtered['year'] <= 2017)
              
display(df_results_filtered)

# COMMAND ----------

df_results_filtered.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outcome - Number 2

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check Missing Values before if/else

# COMMAND ----------

display(df_results_filtered.groupby('positionOrder').count())
#display(df_results_filtered.groupby('positionOrder').count().agg(F.sum("count")))

# Looks good

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create outcome and drop inf outcomes

# COMMAND ----------

df_pred_w_outcome = df_results_filtered.withColumn('Second',
                                                   F.when(F.col('positionOrder') == 2, 1).otherwise(0))

display(df_pred_w_outcome)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Missing Values and Join check

# COMMAND ----------

display(df_pred_w_outcome.groupby('Second').count())
df_pred_w_outcome.count()

# Looks good

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grid Position
# MAGIC - Deal with 0 position entries by replacing them with the
# MAGIC positionOrder positions
# MAGIC   - This seems appropriate as the positionOrder values are the
# MAGIC inferred finishing position of these entries, tacked to the back of the
# MAGIC grid. This befits DNQs, crashouts, and technical faults.
# MAGIC   - None of the 0 positions have a podium finish, and only 11 finished in 
# MAGIC the top 15

# COMMAND ----------

df_pred_grid_fixed = df_pred_w_outcome.withColumn('grid_fixed',
                                                   F.when(F.col('grid') != '0', 
                                                          F.col('grid'))\
                                                  .otherwise(F.col('positionOrder')))

display(df_pred_grid_fixed)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Missing values and manipulation check

# COMMAND ----------

display(df_pred_grid_fixed.groupby('grid', 'grid_fixed').count())


# Here we are checking that the 0 grid values have been properly assigned to their
# position value and are not 0

# Looks good

# COMMAND ----------

df_pred_grid_fixed.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Remove grid and Position Order to prevent duplication and the outcome variable being in the features

# COMMAND ----------


df_pred_grid_fixed = df_pred_grid_fixed.drop('grid', 'positionOrder')
display(df_pred_grid_fixed)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Constructor Quality

# COMMAND ----------

# Because of the chassis and engine partnerships, etc. that create duplicates or counfounds
# for winning constructors, the wiki page links a Are the best way of pinpointing constructors
# Who have actually won in the past
constructors_won_urls = ['http://en.wikipedia.org/wiki/Scuderia_Ferrari',
                         'http://en.wikipedia.org/wiki/Williams_Grand_Prix_Engineering',
                         'http://en.wikipedia.org/wiki/McLaren',
                         'http://en.wikipedia.org/wiki/Team_Lotus',
                         'http://en.wikipedia.org/wiki/Mercedes-Benz_in_Formula_One',
                         'http://en.wikipedia.org/wiki/Red_Bull_Racing',
                         'http://en.wikipedia.org/wiki/Cooper_Car_Company',
                         'http://en.wikipedia.org/wiki/Brabham',
                         'http://en.wikipedia.org/wiki/Renault_in_Formula_One',
                         'http://en.wikipedia.org/wiki/Vanwall',
                         'http://en.wikipedia.org/wiki/BRM',
                         'http://en.wikipedia.org/wiki/Matra',
                         'http://en.wikipedia.org/wiki/Tyrrell_Racing',
                         'http://en.wikipedia.org/wiki/Benetton_Formula',
                         'http://en.wikipedia.org/wiki/Brawn_GP']

constructors_won = df_constructors.where(F.col("url").isin(constructors_won_urls)).toPandas()
constructors_won = [int(id) for id in constructors_won['constructorId'].tolist()]

# COMMAND ----------

df_pred_grid_fixed_const = df_pred_grid_fixed\
  .withColumn('constructor_quality',
              F.when(F.col('constructorId').isin(constructors_won), 'winner')\
              .otherwise('not_winner'))

display(df_pred_grid_fixed_const)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Missing Values and Joins

# COMMAND ----------

# Check for missing constructor IDs
# display(df_constructors.groupby('constructorId').count())
#display(df_pred_grid_fixed_const.groupby('constructorId').count())
#display(df_pred_grid_fixed_const.groupby('constructor_quality').count())
display(df_pred_grid_fixed_const.groupby('constructorId').count().agg(F.sum("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Driver Experience

# COMMAND ----------

driver_debut_year = df_pred_grid_fixed_const\
  .groupby('driverId')\
  .agg(F.min(F.col('year')).alias('debut_year'))
display(driver_debut_year)

# COMMAND ----------

df_pred_grid_fixed_const_driver_debut = df_pred_grid_fixed_const.join(driver_debut_year, on = ['driverId'], how = 'left')\
  .withColumn("years_since_debut", F.col('year') - F.col("debut_year"))\
  .drop('debut_year')
display(df_pred_grid_fixed_const_driver_debut)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Missing Values and Joins

# COMMAND ----------

# Check for missing driver IDs
#display(df_pred_grid_fixed_const_driver_debut.groupby('driverId').count())
display(df_pred_grid_fixed_const_driver_debut.groupby('driverId').count().agg(F.sum("count")))

# Looks good

# COMMAND ----------

# MAGIC %md
# MAGIC ### Driver Average Finishing Position

# COMMAND ----------

driver_avg_finish = df_results\
  .groupby('driverId')\
  .agg(F.mean(F.col('positionOrder')).alias('driver_avg_finish'))
display(driver_avg_finish)

# COMMAND ----------

df_pred_grid_fixed_const_driver_debut_finish = df_pred_grid_fixed_const_driver_debut\
  .join(driver_avg_finish, on = ['driverId'], how = 'left')
display(df_pred_grid_fixed_const_driver_debut_finish)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Missing Data and Joins

# COMMAND ----------

# Check for missing driver IDs
# display(df_pred_grid_fixed_const_driver_debut_finish.groupby('driverId').count())
# display(df_pred_grid_fixed_const_driver_debut_finish.groupby('driverId').count().agg(F.sum("count")))
display(df_pred_grid_fixed_const_driver_debut_finish.describe('driver_avg_finish'))

# Looks Good

# COMMAND ----------

# MAGIC %md
# MAGIC ### Constructor Average Finishing Position

# COMMAND ----------

const_avg_finish = df_results\
  .groupby('constructorId')\
  .agg(F.mean(F.col('positionOrder')).alias('const_avg_finish'))
display(const_avg_finish)

# COMMAND ----------

df_pred_wo_quals = df_pred_grid_fixed_const_driver_debut_finish\
  .join(const_avg_finish, on = ['constructorId'], how = 'left')
display(df_pred_wo_quals)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Missing Data

# COMMAND ----------

# Check for missing driver IDs
# display(df_pred_wo_quals.groupby('constructorId').count())
# display(df_pred_wo_quals.groupby('constructorId').count().agg(F.sum("count")))
display(df_pred_wo_quals.describe('const_avg_finish'))

# Looks good

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q3 Qualifying Lap Time

# COMMAND ----------

# MAGIC %md
# MAGIC Replace missing values with none

# COMMAND ----------

df_qualifying_missing_fix = df_qualifying.withColumn('q1_fix',
                                                     F.when(F.col('q1') == '\\N', None)\
                                                     .otherwise(F.col('q1')))
display(df_qualifying_missing_fix)

# COMMAND ----------

# MAGIC %md
# MAGIC Turn times to milliseconds

# COMMAND ----------

split_q1 = F.split(df_qualifying_missing_fix['q1_fix'], '\D')

qual_times = df_qualifying_missing_fix\
  .withColumn('q1_ms', (split_q1.getItem(0) * 60 * 1000) + (split_q1.getItem(1) * 1000) + (split_q1.getItem(2)))\
  .select('raceId', 'driverId', 'q1_ms')

display(qual_times)

# COMMAND ----------

# MAGIC %md
# MAGIC Check to see if values make sense

# COMMAND ----------

display(qual_times.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Only 8000+ entries suggests something is wrong. I suspect like pitstops, we only have 
# MAGIC qualifying data from more recent years

# COMMAND ----------

display(qual_times.join(df_races.select('raceId', 'year'), on = ['raceId'], how = 'left').describe())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Ok I should have checked this earlier, and thankfully I did check, we only have 
# MAGIC qualifying lap data from 1994 onwards. This is a problem as the majority of our
# MAGIC training set will be missing this feature.
# MAGIC 
# MAGIC I think imputation is going to be a problem, because chances are races from earlier in
# MAGIC F1 history have slower laptimes. Missingness is far from random here. 
# MAGIC 
# MAGIC There is also a small amount of missing data in q1_ms, likely to be people who didn't 
# MAGIC even manage to complete q1 qualifying for some reason. This is a small number, and
# MAGIC imputation might work here given the relationship with the fixed grid position.
# MAGIC 
# MAGIC As for the earlier races, it's safest to do without the feature, but we can try to
# MAGIC impute the values with the other features and see which works better. In effect,
# MAGIC we use that as an additional iteration parameter.

# COMMAND ----------

df_pred_full = df_pred_wo_quals.join(qual_times, on = ['raceId', 'driverId'], how = 'left')

display(df_pred_full)

# COMMAND ----------

pd_df_pred_full = df_pred_full.toPandas()

# COMMAND ----------

pd_df_pred_full.head()

# COMMAND ----------

df_4_impute = pd.get_dummies(pd_df_pred_full.drop(['raceId', 'driverId', 'constructorId', 'year', 'Second'],
                                                  axis = 1),
                             columns = ['constructor_quality'])

knn_imputer = KNNImputer(n_neighbors=1)

array_missing_imputed = knn_imputer.fit_transform(df_4_impute)
df_missing_imputed = pd.DataFrame(array_missing_imputed, columns = df_4_impute.columns)

# COMMAND ----------

pd_df_pred_full_w_impute = pd_df_pred_full.merge(df_missing_imputed[['q1_ms']],
                                                 how='left',
                                                 left_index=True,
                                                 right_index=True,
                                                 sort=False,
                                                 suffixes=('_OG', '_impute'))

# COMMAND ----------

pd_df_pred_full_w_impute.describe(include = 'all')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Train/Test Split

# COMMAND ----------

spark_df_pred_full_w_impute = spark_df_missing_imputed = spark.createDataFrame(pd_df_pred_full_w_impute)
display(spark_df_pred_full_w_impute)

# COMMAND ----------

df_pred_train = spark_df_pred_full_w_impute.filter(spark_df_pred_full_w_impute['year'] <= 2010)
display(df_pred_train.describe('year'))

# COMMAND ----------

df_pred_test = spark_df_pred_full_w_impute.filter(spark_df_pred_full_w_impute['year'] > 2010)
display(df_pred_test.describe('year'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to S3

# COMMAND ----------

df_pred_train.coalesce(1).write.csv('/mnt/xql2001-gr5069/interim/final_project/df_pred_train.csv', header=True)
df_pred_test.coalesce(1).write.csv('/mnt/xql2001-gr5069/interim/final_project/df_pred_test.csv', header=True)
