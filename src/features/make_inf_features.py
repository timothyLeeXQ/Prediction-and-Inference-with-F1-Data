# Databricks notebook source
# MAGIC %md
# MAGIC ## Packages

# COMMAND ----------

from pyspark.sql.types import DateType, IntegerType, DoubleType
from pyspark.sql import functions as F
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw Data

# COMMAND ----------

df_drivers = spark.read.csv('/mnt/ne-gr5069/raw/drivers.csv',header=True)
display(df_drivers)

# COMMAND ----------

df_races = spark.read.csv('/mnt/ne-gr5069/raw/races.csv',header=True)
display(df_races)

# COMMAND ----------

df_constructors = spark.read.csv('/mnt/ne-gr5069/raw/constructors.csv',header=True)
display(df_constructors)

# COMMAND ----------

df_circuits = spark.read.csv('/mnt/ne-gr5069/raw/circuits.csv',header=True)
display(df_circuits)

# COMMAND ----------

df_races = spark.read.csv('/mnt/ne-gr5069/raw/races.csv',header=True)
display(df_races)

# COMMAND ----------

df_results = spark.read.csv('/mnt/ne-gr5069/raw/results.csv',header=True)
display(df_results)

# COMMAND ----------

df_pitstops = spark.read.csv('/mnt/ne-gr5069/raw/pit_stops.csv',header=True)
display(df_pitstops)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table generation
# MAGIC 
# MAGIC Filter to include only races from 1950 - 2017 (inclusive)
# MAGIC 
# MAGIC Needed references to cross reference other tables and build features:
# MAGIC * Race ID
# MAGIC * Driver ID
# MAGIC * Constructor ID
# MAGIC 
# MAGIC Outcomes
# MAGIC * Outcome 1 - First or Second
# MAGIC * Outcome 2 - positionOrder (remember to filter for 1 or 2 when running analysis)
# MAGIC 
# MAGIC Base Features
# MAGIC * Grid Position
# MAGIC 
# MAGIC Features to generate/join
# MAGIC * Driver skill - Years since debut
# MAGIC * Number of pit stops
# MAGIC * Constructor quality - whether the constructor has won a championship
# MAGIC * Circuit
# MAGIC 
# MAGIC Interactions
# MAGIC * Driver skill x grid position
# MAGIC * Driver x track
# MAGIC * Driver x constructor
# MAGIC * Pit stops x driver 

# COMMAND ----------

df_refs_with_outcome = df_results.select('raceId', 'driverId', 'constructorId', 'grid', 'positionOrder')\
  .withColumn('1_or_2',
              F.when(F.col('positionOrder') > 2, 'lose')\
              .otherwise('win'))\
  .join(df_races.select('raceId', 'year'), on = ['raceId'], how = 'left')

df_refs_with_outcome = df_refs_with_outcome\
  .filter(df_refs_with_outcome['year'] >= 1950)\
  .filter(df_refs_with_outcome['year'] <= 2017)
              
display(df_refs_with_outcome)

# COMMAND ----------

df_refs_with_outcome.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Missing Values - positionOrder and grid position

# COMMAND ----------

#Use to check for missing values in positionOrder and grid
# display(df_refs_with_outcome.groupby('grid').count())
# display(df_refs_with_outcome.groupby('grid').count().agg(F.sum("count")))
# display(df_refs_with_outcome.groupby('positionOrder').count())
display(df_refs_with_outcome.groupby('positionOrder').count().agg(F.sum("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC There are missing values in grid position, labelled 0
# MAGIC 
# MAGIC These are probably DNQs or crashes during qualifying. 
# MAGIC * All but a handful have unrealistic finishing positions of high 20s or 30s,
# MAGIC suggesting that these positions were inferred.
# MAGIC * None of these position 0s had a podium finish, while only 11 finished in 
# MAGIC the top 15.
# MAGIC 
# MAGIC I think it's safe to remove these values from the analysis.

# COMMAND ----------

df_refs_with_outcome = df_refs_with_outcome.filter(df_refs_with_outcome['grid'] != 0)
display(df_refs_with_outcome)

# COMMAND ----------

df_refs_with_outcome.count()
#1566 missing values removed

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Driver Skill

# COMMAND ----------

driver_debut_year = df_refs_with_outcome\
  .groupby('driverId')\
  .agg(F.min(F.col('year')).alias('debut_year'))
display(driver_debut_year)

# COMMAND ----------

df_refs_outcome_driver_exp = df_refs_with_outcome.join(driver_debut_year, on = ['driverId'], how = 'left')\
  .withColumn("years_since_debut", F.col('year') - F.col("debut_year"))\
  .drop('debut_year', 'year')
display(df_refs_outcome_driver_exp)

# COMMAND ----------

df_refs_outcome_driver_exp.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing values

# COMMAND ----------

# Check for missing driver IDs
#display(df_drivers.groupby('driverId').count())
#display(df_refs_outcome_driver_exp.groupby('driverId').count())
display(df_refs_outcome_driver_exp.groupby('driverId').count().agg(F.sum("count")))

# Don't detect any missing values

# COMMAND ----------

# Check for missing driver debut values
# display(df_refs_outcome_driver_exp.groupby('years_since_debut').count())
display(df_refs_outcome_driver_exp.groupby('years_since_debut').count().agg(F.sum("count")))

# Don't detect any missing values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Number of Pit Stops

# COMMAND ----------

pit_strat = df_pitstops.groupby('raceId', 'driverId').agg(F.max(F.col('stop')).alias('num_pits'))\
  .withColumn('pit_strategy',
              F.when(F.col('num_pits') > 3, '>3-stops')\
              .otherwise(F.col('num_pits')))\
  .select('raceId', 'driverId', 'pit_strategy')
display(pit_strat)

# COMMAND ----------

df_refs_outcome_driver_exp_pit = df_refs_outcome_driver_exp.join(pit_strat, on= ['raceId', 'driverId'], how = 'left')
display(df_refs_outcome_driver_exp_pit)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Missing Values

# COMMAND ----------

display(df_refs_outcome_driver_exp_pit.groupby('pit_strategy').count())
#df_refs_outcome_driver_exp_pit.count()

# COMMAND ----------

df_pitstops.select('raceId').distinct().count()

# COMMAND ----------

df_refs_outcome_driver_exp.select('raceId').distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC We have a lot of null values here. This seems to be because 
# MAGIC we only have pitstop data for a minority of races
# MAGIC (see code just above)
# MAGIC 
# MAGIC Removing these null values leads to questions as to 
# MAGIC whether the pattern of missingless in races with 
# MAGIC available pitstop data is a random sample. 
# MAGIC 
# MAGIC Spoiler, it is not, given that the raceIds of data with pitstops 
# MAGIC ranges from 841 to 1030 (get this by sorting df pitstops), omitting 
# MAGIC 1-840 and 1031-1040. These IDs mostly correspond to races
# MAGIC in the 2010s.
# MAGIC 
# MAGIC Imputation is a bad strategy for an inferential model, since 
# MAGIC we should understand what's happening by inference first before
# MAGIC trying to impute strategies used. Also, as there is a missingness
# MAGIC pattern that isn't random, imputation might result in a bias in the
# MAGIC pit strategy data.
# MAGIC 
# MAGIC I think it is best to leave null values as their own nominal class.
# MAGIC This could correspond to 'old strategies' of races before the 2010s.

# COMMAND ----------

df_refs_outcome_driver_exp_pit = df_refs_outcome_driver_exp_pit.fillna({'pit_strategy':'missing'})
display(df_refs_outcome_driver_exp_pit)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Constructor

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

len(constructors_won)

# COMMAND ----------

df_refs_outcome_driver_exp_pit_const = df_refs_outcome_driver_exp_pit\
  .withColumn('constructor_quality',F.when(F.col('constructorId').isin(constructors_won), 'winner')\
              .otherwise('not_winner'))

display(df_refs_outcome_driver_exp_pit_const)

# COMMAND ----------

df_refs_outcome_driver_exp_pit_const.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing Values

# COMMAND ----------

# Check for missing constructor IDs
# display(df_constructors.groupby('constructorId').count())
# display(df_refs_outcome_driver_exp_pit_const.groupby('constructorId').count())
display(df_refs_outcome_driver_exp_pit_const.groupby('constructorId').count().agg(F.sum("count")))

# Don't seem to be any missing values

# COMMAND ----------

# Check for missing constructor_quality
#display(df_refs_outcome_driver_exp_pit_const.groupby('constructor_quality').count())
display(df_refs_outcome_driver_exp_pit_const.groupby('constructor_quality').count().agg(F.sum("count")))

# Don't seem to be any missing values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Circuit

# COMMAND ----------

# Like with the constructors, differences in how circuits are named makes wiki pages
# useful for identifying the actual circuits. Long Beach, Albert Park (Melbourne),
# and Las Vegas (Caesars Palace) had to be manually edited to fix the links
street_circuit_urls = ['http://en.wikipedia.org/wiki/Adelaide_Street_Circuit',
                       'http://en.wikipedia.org/wiki/Melbourne_Grand_Prix_Circuit',
                       'http://en.wikipedia.org/wiki/Baku_City_Circuit',
                       'http://en.wikipedia.org/wiki/Las_Vegas,_Nevada',
                       'http://en.wikipedia.org/wiki/Circuit_de_Monaco',
                       'http://en.wikipedia.org/wiki/Monsanto_Park_Circuit',
                       'http://en.wikipedia.org/wiki/Circuit_Gilles_Villeneuve',
                       'http://en.wikipedia.org/wiki/Circuito_da_Boavista',
                       'http://en.wikipedia.org/wiki/Fair_Park',
                       'http://en.wikipedia.org/wiki/Detroit_street_circuit',
                       'http://en.wikipedia.org/wiki/Long_Beach,_California',
                       'http://en.wikipedia.org/wiki/Hanoi_Street_Circuit',
                       'http://en.wikipedia.org/wiki/Marina_Bay_Street_Circuit',
                       'http://en.wikipedia.org/wiki/Montju%C3%AFc_circuit',
                       'http://en.wikipedia.org/wiki/Pedralbes_Circuit',
                       'http://en.wikipedia.org/wiki/Phoenix_street_circuit',
                       'http://en.wikipedia.org/wiki/Valencia_Street_Circuit']

street_circuits = df_circuits.where(F.col("url").isin(street_circuit_urls)).toPandas()
street_circuits = [int(id) for id in street_circuits['circuitId'].tolist()]

# COMMAND ----------

df_refs_outcome_driver_exp_pit_const_circuit = df_refs_outcome_driver_exp_pit_const\
  .join(df_races.select('raceId', 'circuitId'), on = ['raceId'], how = 'left')\
  .withColumn('circuit_type',F.when(F.col('circuitId').isin(street_circuits), 'street')\
              .otherwise('race'))

display(df_refs_outcome_driver_exp_pit_const_circuit)

# COMMAND ----------

df_refs_outcome_driver_exp_pit_const_circuit.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing Values

# COMMAND ----------

# Check for missing circuit IDs
# display(df_circuits.groupby('circuitId').count())
# display(df_refs_outcome_driver_exp_pit_const_circuit.groupby('circuitId').count())
display(df_refs_outcome_driver_exp_pit_const_circuit.groupby('circuitId').count().agg(F.sum("count")))

# Don't seem to be any missing values

# COMMAND ----------

# Check for missing circuit types
# display(df_refs_outcome_driver_exp_pit_const_circuit.groupby('circuit_type').count())
display(df_refs_outcome_driver_exp_pit_const_circuit.groupby('circuit_type').count().agg(F.sum("count")))

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Processed DFs and store in S3

# COMMAND ----------

display(df_refs_outcome_driver_exp_pit_const_circuit)

# COMMAND ----------

# MAGIC %md 
# MAGIC Borrowing this for predictive task

# COMMAND ----------

df_refs_outcome_driver_exp_pit_const_circuit.coalesce(1).write.csv('/mnt/xql2001-gr5069/interim/final_project/df_inf_master.csv', header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LogReg 1 - Predicting 1st & 2nd

# COMMAND ----------

df_top2 = df_refs_outcome_driver_exp_pit_const_circuit.select('grid',
                                                              'years_since_debut', 
                                                              'pit_strategy',
                                                              'constructor_quality',
                                                              'circuit_type',
                                                              '1_or_2')
display(df_top2)

# COMMAND ----------

pd_top2 = df_top2.toPandas()
pd_top2_dummy = pd.get_dummies(pd_top2, columns = ['pit_strategy', 'constructor_quality', 'circuit_type'])
display(pd_top2_dummy)

# COMMAND ----------

df_top2_dummy = spark.createDataFrame(pd_top2_dummy)
df_top2_dummy.coalesce(1).write.csv('/mnt/xql2001-gr5069/processed/final_project/df_top2_dummy.csv', header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LogReg 2 - Predicting 1st/2nd among only 1st or 2nd place

# COMMAND ----------

df_1or2 = df_refs_outcome_driver_exp_pit_const_circuit\
  .filter(df_refs_outcome_driver_exp_pit_const_circuit['positionOrder'] <= 2)\
  .select('grid',
          'years_since_debut', 
          'pit_strategy',
          'constructor_quality',
          'circuit_type',
          'positionOrder')
  
display(df_1or2)

# COMMAND ----------

pd_1or2 = df_1or2.toPandas()
pd_1or2_dummy = pd.get_dummies(pd_1or2, columns = ['pit_strategy', 'constructor_quality', 'circuit_type'])
display(pd_1or2_dummy)

# COMMAND ----------

df_1or2_dummy = spark.createDataFrame(pd_1or2_dummy)
df_1or2_dummy.coalesce(1).write.csv('/mnt/xql2001-gr5069/processed/final_project/df_1or2_dummy.csv', header=True)

# COMMAND ----------


