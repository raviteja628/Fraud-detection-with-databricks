# Databricks notebook source
# MAGIC %run ../_resources/00-setup $reset_all_data=false

# COMMAND ----------

from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput, TrafficConfig, Route
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient
from datetime import timedelta

model_name = f"{catalog}.{db}.dbdemos_fsi_fraud"
serving_endpoint_name = "dbdemos_fsi_fraud_endpoint"

w = WorkspaceClient()
mlflow_client = MlflowClient(registry_uri="databricks-uc")
served_entities=[
        ServedEntityInput(
            name="prod_model",
            entity_name=model_name,
            entity_version=mlflow_client.get_model_version_by_alias(model_name, "prod").version,
            scale_to_zero_enabled=True,
            workload_size="Small"
        ),
        ServedEntityInput(
            name="candidate_model",
            entity_name=model_name,
            entity_version=mlflow_client.get_model_version_by_alias(model_name, "candidate").version,
            scale_to_zero_enabled=True,
            workload_size="Small"
        )
    ]
traffic_config=TrafficConfig(routes=[
        Route(
            served_model_name="prod_model",
            traffic_percentage=90
        ),
        Route(
            served_model_name="candidate_model",
            traffic_percentage=10
        )
    ])

print('Updating the endpoint, this will take a few sec, please wait...')
w.serving_endpoints.update_config_and_wait(name=serving_endpoint_name, served_entities=served_entities, traffic_config=traffic_config, timeout=timedelta(minutes=30))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Our new model is now serving 10% of our requests
# MAGIC
# MAGIC Open your <a href="#mlflow/endpoints/dbdemos_fsi_fraud_endpoint" target="_blank"> Model Serving Endpoint</a> to view the changes and track the 2 models performance

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc') 
p = ModelsArtifactRepository(f"models:/{model_name}@prod").download_artifacts("") 
dataset =  {"dataframe_split": Model.load(p).load_input_example(p).to_dict(orient='split')}

# COMMAND ----------

# DBTITLE 1,Trying our new Model Serving setup
from mlflow import deployments
client = mlflow.deployments.get_deploy_client("databricks")
#Let's do multiple call to track the results in the model endpoint inference table
for i in range(10):
    predictions = client.predict(endpoint=serving_endpoint_name, inputs=dataset)
    print(predictions)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Conclusion: the power of the Lakehouse
# MAGIC
# MAGIC In this demo, we've seen an end 2 end flow with the Lakehouse:
# MAGIC
# MAGIC - Data ingestion made simple with Delta Live Table
# MAGIC - Leveraging Databricks warehouse to Analyze existing Fraud
# MAGIC - Model Training with AutoML for citizen Data Scientist
# MAGIC - Ability to tune our model for better results, improving our revenue
# MAGIC - Ultimately, the ability to Deploy and track our models in real time, made possible with the full lakehouse capabilities.
# MAGIC
# MAGIC [Go back to the introduction]($../00-FSI-fraud-detection-introduction-lakehouse) or discover how to use Databricks Workflow to orchestrate this tasks: [05-Workflow-orchestration-fsi-fraud]($../05-Workflow-orchestration/05-Workflow-orchestration-fsi-fraud)