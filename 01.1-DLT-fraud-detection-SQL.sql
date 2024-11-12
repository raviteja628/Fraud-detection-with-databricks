-- Databricks notebook source
-- DBTITLE 1,Let's explore our raw incoming data: transactions (json)
-- MAGIC %python
-- MAGIC display(spark.read.json('/Volumes/main/dbdemos_fsi_fraud_detection/fraud_raw_data/transactions'))

-- COMMAND ----------

-- DBTITLE 1,Raw incoming customers (json)
-- MAGIC %python
-- MAGIC display(spark.read.csv('/Volumes/main/dbdemos_fsi_fraud_detection/fraud_raw_data/customers', header=True, multiLine=True))

-- COMMAND ----------

-- DBTITLE 1,Raw incoming country
-- MAGIC %python
-- MAGIC display(spark.read.csv('/Volumes/main/dbdemos_fsi_fraud_detection/fraud_raw_data/country_code', header=True))

-- COMMAND ----------

-- DBTITLE 1,Ingest transactions
CREATE STREAMING LIVE TABLE bronze_transactions 
  COMMENT "Historical banking transaction to be trained on fraud detection"
AS 
  SELECT * FROM cloud_files("/Volumes/main/dbdemos_fsi_fraud_detection/fraud_raw_data/transactions", "json", map("cloudFiles.maxFilesPerTrigger", "1", "cloudFiles.inferColumnTypes", "true"))

-- COMMAND ----------

-- DBTITLE 1,Customers
CREATE STREAMING LIVE TABLE banking_customers (
  CONSTRAINT correct_schema EXPECT (_rescued_data IS NULL)
)
COMMENT "Customer data coming from csv files ingested in incremental with Auto Loader to support schema inference and evolution"
AS 
  SELECT * FROM cloud_files("/Volumes/main/dbdemos_fsi_fraud_detection/fraud_raw_data/customers", "csv", map("cloudFiles.inferColumnTypes", "true", "multiLine", "true"))

-- COMMAND ----------

-- DBTITLE 1,Reference table
CREATE STREAMING LIVE TABLE country_coordinates
AS 
  SELECT * FROM cloud_files("/Volumes/main/dbdemos_fsi_fraud_detection/fraud_raw_data/country_code", "csv")

-- COMMAND ----------

-- DBTITLE 1,Fraud report (labels for ML training)
CREATE STREAMING LIVE TABLE fraud_reports
AS 
  SELECT * FROM cloud_files("/Volumes/main/dbdemos_fsi_fraud_detection/fraud_raw_data/fraud_report", "csv")

-- COMMAND ----------

-- DBTITLE 1,Silver
CREATE STREAMING LIVE TABLE silver_transactions (
  CONSTRAINT correct_data EXPECT (id IS NOT NULL),
  CONSTRAINT correct_customer_id EXPECT (customer_id IS NOT NULL)
)
AS 
  SELECT * EXCEPT(countryOrig, countryDest, t._rescued_data, f._rescued_data), 
          regexp_replace(countryOrig, "\-\-", "") as countryOrig, 
          regexp_replace(countryDest, "\-\-", "") as countryDest, 
          newBalanceOrig - oldBalanceOrig as diffOrig, 
          newBalanceDest - oldBalanceDest as diffDest
FROM STREAM(live.bronze_transactions) t
  LEFT JOIN live.fraud_reports f using(id)

-- COMMAND ----------

-- DBTITLE 1,Gold, ready for Data Scientists to consume
CREATE LIVE TABLE gold_transactions (
  CONSTRAINT amount_decent EXPECT (amount > 10)
)
AS 
  SELECT t.* EXCEPT(countryOrig, countryDest, is_fraud), c.* EXCEPT(id, _rescued_data),
          boolean(coalesce(is_fraud, 0)) as is_fraud,
          o.alpha3_code as countryOrig, o.country as countryOrig_name, o.long_avg as countryLongOrig_long, o.lat_avg as countryLatOrig_lat,
          d.alpha3_code as countryDest, d.country as countryDest_name, d.long_avg as countryLongDest_long, d.lat_avg as countryLatDest_lat
FROM live.silver_transactions t
  INNER JOIN live.country_coordinates o ON t.countryOrig=o.alpha3_code 
  INNER JOIN live.country_coordinates d ON t.countryDest=d.alpha3_code 
  INNER JOIN live.banking_customers c ON c.id=t.customer_id 