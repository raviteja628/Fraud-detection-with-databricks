# Databricks notebook source
# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

# DBTITLE 1,Leverage built-in visualizations to explore your dataset
# MAGIC %sql
# MAGIC select 
# MAGIC   is_fraud,
# MAGIC   count(1) as `Transactions`, 
# MAGIC   sum(amount) as `Total Amount` 
# MAGIC from gold_transactions
# MAGIC group by is_fraud
# MAGIC
# MAGIC --Visualization Pie chart: Keys: is_fraud, Values: [Transactions, Total Amount]

# COMMAND ----------

# MAGIC %md
# MAGIC As expected, financial fraud is by nature very imbalanced between fraudulant and normal transactions

# COMMAND ----------

# DBTITLE 1,Run analysis using your usual python plot libraries
df = spark.sql('select type, is_fraud, count(1) as count from gold_transactions group by type, is_fraud').toPandas()

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=df[df['is_fraud']]['type'], values=df[df['is_fraud']]['count'], title="Fraud Transactions", hole=0.6), 1, 1)
fig.add_trace(go.Pie(labels=df[~df['is_fraud']]['type'], values=df[~df['is_fraud']]['count'], title="Normal Transactions", hole=0.6) ,1, 2)

# COMMAND ----------

# DBTITLE 1,Create the final features using pandas API
# Convert to koalas
dataset = spark.table('gold_transactions').dropDuplicates(['id']).pandas_api()
# Drop columns we don't want to use in our model
# Typical DS project would include more transformations / cleanup here
dataset = dataset.drop(columns=['address', 'email', 'firstname', 'lastname', 'creation_date', 'last_activity_date', 'customer_id'])

# Drop missing values
dataset.dropna()
dataset.describe()

# COMMAND ----------

# DBTITLE 1,Save them to our feature store
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

try:
  #drop table if exists
  fs.drop_table(f'{catalog}.{db}.transactions_features')
except:
  pass

fs.create_table(
  name=f'{catalog}.{db}.transactions_features',
  primary_keys='id',
  schema=dataset.spark.schema(),
  description='These features are derived from the gold_transactions table in the lakehouse.  We created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the transaction is a fraud or not.  No aggregations were performed.')

fs.write_table(df=dataset.to_spark(), name=f'{catalog}.{db}.transactions_features', mode='overwrite')
features = fs.read_table(f'{catalog}.{db}.transactions_features')
display(features)

# COMMAND ----------

# DBTITLE 1,Starting AutoML run usingDtabricks API
from databricks import automl

xp_path = "/Shared/dbdemos/experiments/lakehouse-fsi-fraud-detection"
xp_name = f"automl_fraud_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
automl_run = automl.classify(
    experiment_name = xp_name,
    experiment_dir = xp_path,
    dataset = features.sample(0.05), #drastically reduce the training size to speedup the demo
    target_col = "is_fraud",
    timeout_minutes = 15
)
#Make sure all users can access dbdemos shared experiment
DBDemos.set_experiment_permission(f"{xp_path}/{xp_name}")