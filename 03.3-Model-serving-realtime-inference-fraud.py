# Databricks notebook source
# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

model_name = "dbdemos_fsi_fraud"
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# DBTITLE 1,Starting the model inference REST endpoint using Databricks API
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput
from mlflow import MlflowClient

model_name = f"{catalog}.{db}.dbdemos_fsi_fraud"
serving_endpoint_name = "dbdemos_fsi_fraud_endpoint"
w = WorkspaceClient()

mlflow_client = MlflowClient(registry_uri="databricks-uc")
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=model_name,
            entity_version=mlflow_client.get_model_version_by_alias(model_name, "prod").version,
            scale_to_zero_enabled=True,
            workload_size="Small"
        )
    ],
    auto_capture_config = AutoCaptureConfigInput(catalog_name=catalog, schema_name=db, enabled=True, table_name_prefix="fraud_ep_inference_table" )
)

force_update = False #Set this to True to release a newer version (the demo won't update the endpoint to a newer model version by default)
try:
  existing_endpoint = w.serving_endpoints.get(serving_endpoint_name)
  print(f"endpoint {serving_endpoint_name} already exist...")
  if force_update:
    w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=serving_endpoint_name)
except:
    print(f"Creating the endpoint {serving_endpoint_name}, this will take a few minutes to package and deploy the endpoint...")
    spark.sql('drop table if exists fraud_ep_inference_table_payload')
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)

# COMMAND ----------

# DBTITLE 1,Running HTTP REST inferences in realtime !
p = ModelsArtifactRepository(f"models:/{model_name}@prod").download_artifacts("") 
dataset =  {"dataframe_split": Model.load(p).load_input_example(p).to_dict(orient='split')}

# COMMAND ----------

import mlflow
from mlflow import deployments
client = mlflow.deployments.get_deploy_client("databricks")
predictions = client.predict(endpoint=serving_endpoint_name, inputs=dataset)

print(predictions)