-- Databricks notebook source
-- MAGIC %run ../_resources/00-setup $reset_all_data=false

-- COMMAND ----------

-- the catalog has been created for your user and is defined as default. 
-- make sure you run the 00-setup cell above to init the catalog to your user. 
SELECT CURRENT_CATALOG();

-- COMMAND ----------

-- DBTITLE 1,As you can see, our tables are available under our catalog.
SHOW TABLES;

-- COMMAND ----------

-- DBTITLE 1,Granting access to Analysts & Data Engineers:
-- Let's grant our ANALYSTS a SELECT permission:
-- Note: make sure you created an analysts and dataengineers group first.
GRANT SELECT ON TABLE main.dbdemos_fsi_fraud_detection.gold_transactions TO `analysts`;
GRANT SELECT ON TABLE main.dbdemos_fsi_fraud_detection.gold_transactions TO `analysts`;
GRANT SELECT ON TABLE main.dbdemos_fsi_fraud_detection.gold_transactions TO `analysts`;

-- We'll grant an extra MODIFY to our Data Engineer
GRANT SELECT, MODIFY ON SCHEMA main.dbdemos_fsi_fraud_detection TO `dataengineers`;