# Databricks notebook source
# MAGIC %md
# MAGIC # Code for AWS Mount

# COMMAND ----------

ACCESS_KEY = ''
# Encode the Secret Key as that can contain '/'
SECRET_KEY = ''.replace("/", "%2F")
AWS_BUCKET_NAME_RAW = 'ne-gr5069'
MOUNT_NAME_RAW = 'ne-gr5069'
AWS_BUCKET_NAME_PROC = 'xql2001-gr5069'
MOUNT_NAME_PROC = 'xql2001-gr5069'

# COMMAND ----------

dbutils.fs.mount('s3a://%s:%s@%s' % (ACCESS_KEY, SECRET_KEY, AWS_BUCKET_NAME_RAW),
                 '/mnt/%s' % MOUNT_NAME_RAW)
display(dbutils.fs.ls('/mnt/%s' % MOUNT_NAME_RAW))

# COMMAND ----------

dbutils.fs.mount('s3a://%s:%s@%s' % (ACCESS_KEY, SECRET_KEY, AWS_BUCKET_NAME_PROC),
                 '/mnt/%s' % MOUNT_NAME_PROC)
display(dbutils.fs.ls('/mnt/%s' % MOUNT_NAME_PROC))

# COMMAND ----------


