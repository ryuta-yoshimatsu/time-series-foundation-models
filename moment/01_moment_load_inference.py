# Databricks notebook source
# MAGIC %md
# MAGIC This is an example notebook that shows how to use [moment](https://github.com/moment-timeseries-foundation-model/moment) model on Databricks. The notebook loads the model, distributes the inference, registers the model, deploys the model and makes online forecasts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster setup
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or above. The cluster can be single-node or multi-node with one or more GPU instances on each worker: e.g. [g5.12xlarge [A10G]](https://aws.amazon.com/ec2/instance-types/g5/) on AWS or [Standard_NV72ads_A10_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nva10v5-series) on Azure. This notebook will leverage [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html) for distributing the inference tasks and utilizing all the available resource.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install package

# COMMAND ----------

# MAGIC %pip install git+https://github.com/moment-timeseries-foundation-model/moment.git --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data 
# MAGIC We use [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast/tree/main/) package to download M4 data. M4 dataset contains a set of time series which we use for testing. See the `data_preparation` notebook for a number of custom functions we wrote to convert M4 time series to an expected format.
# MAGIC
# MAGIC Make sure that the catalog and the schema already exist.

# COMMAND ----------

catalog = "tsfm"  # Name of the catalog we use to manage our assets
db = "m4"  # Name of the schema we use to manage our assets (e.g. datasets)
n = 100  # Number of time series to sample

# COMMAND ----------

# This cell will create tables: 
# 1. {catalog}.{db}.m4_daily_train
# 2. {catalog}.{db}.m4_monthly_train
dbutils.notebook.run("../data_preparation", timeout_seconds=0, arguments={"catalog": catalog, "db": db, "n": n})

# COMMAND ----------

from pyspark.sql.functions import collect_list

# Make sure that the data exists
df = spark.table(f'{catalog}.{db}.m4_daily_train')
df = df.groupBy('unique_id').agg(collect_list('ds').alias('ds'), collect_list('y').alias('y'))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribute Inference
# MAGIC We use [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html#iterator-of-series-to-iterator-of-series-udf) to distribute the inference.

# COMMAND ----------

import pandas as pd
import numpy as np
import torch
from typing import Iterator
from pyspark.sql.functions import pandas_udf

# Function to create a UDF for generating horizon timestamps for a given frequency and prediction length
def create_get_horizon_timestamps(freq, prediction_length):

  # Define a Pandas UDF to generate horizon timestamps
  @pandas_udf('array<timestamp>')
  def get_horizon_timestamps(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
      # Define the offset based on the frequency
      one_ts_offset = pd.offsets.MonthEnd(1) if freq == "M" else pd.DateOffset(days=1)
      barch_horizon_timestamps = []  # Initialize a list to store horizon timestamps for each batch
      
      # Iterate over batches of time series
      for batch in batch_iterator:
          for series in batch:
              timestamp = last = series.max()  # Get the latest timestamp in the series
              horizon_timestamps = []  # Initialize a list to store horizon timestamps for the series
              for i in range(prediction_length):
                  timestamp = timestamp + one_ts_offset  # Increment the timestamp by the offset
                  horizon_timestamps.append(timestamp.to_numpy())  # Convert timestamp to numpy format and add to list
              barch_horizon_timestamps.append(np.array(horizon_timestamps))  # Add the list of horizon timestamps to the batch list
      yield pd.Series(barch_horizon_timestamps)  # Yield the batch of horizon timestamps as a Pandas Series

  return get_horizon_timestamps  # Return the UDF


# Function to create a UDF for generating forecasts using a pre-trained model
def create_forecast_udf(repository, prediction_length):

  # Define a Pandas UDF to generate forecasts
  @pandas_udf('array<double>')
  def forecast_udf(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    ## Initialization step
    from momentfm import MOMENTPipeline  # Import the MOMENTPipeline class from the momentfm library
    
    # Load the pre-trained model from the repository
    model = MOMENTPipeline.from_pretrained(
      repository, 
      model_kwargs={
        "task_name": "forecasting",
        "forecast_horizon": prediction_length},
      )
    model.init()  # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device to GPU if available, otherwise CPU
    model = model.to(device)  # Move the model to the selected device

    ## Inference step
    for batch in batch_iterator:
      batch_forecast = []  # Initialize a list to store forecasts for each batch
      for series in batch:
        # Prepare the input context and mask
        context = list(series)
        if len(context) < 512:
          input_mask = [1] * len(context) + [0] * (512 - len(context))  # Create an input mask with padding
          context = context + [0] * (512 - len(context))  # Pad the context to the required length
        else:
          input_mask = [1] * 512  # Create an input mask without padding
          context = context[-512:]  # Truncate the context to the required length
        
        # Convert context and input mask to PyTorch tensors and move them to the selected device
        input_mask = torch.reshape(torch.tensor(input_mask), (1, 512)).to(device)
        context = torch.reshape(torch.tensor(context), (1, 1, 512)).to(dtype=torch.float32).to(device)
        
        # Generate the forecast using the model
        output = model(context, input_mask=input_mask)
        forecast = output.forecast.squeeze().tolist()  # Squeeze the output tensor and convert to a list
        batch_forecast.append(forecast)  # Add the forecast to the batch list

    yield pd.Series(batch_forecast)  # Yield the batch of forecasts as a Pandas Series

  return forecast_udf  # Return the UDF

# COMMAND ----------

# MAGIC %md
# MAGIC We specify the requirements of our forecasts. 

# COMMAND ----------

moment_model = "MOMENT-1-large"
prediction_length = 10  # Time horizon for forecasting
freq = "D" # Frequency of the time series
device_count = torch.cuda.device_count()  # Number of GPUs available

# COMMAND ----------

# MAGIC %md
# MAGIC Let's generate the forecasts.

# COMMAND ----------

# Create a UDF for generating horizon timestamps using the specified frequency and prediction length
get_horizon_timestamps = create_get_horizon_timestamps(freq=freq, prediction_length=prediction_length)

# Create a UDF for generating forecasts using the specified model repository and prediction length
forecast_udf = create_forecast_udf(
  repository=f"AutonLab/{moment_model}",  # Repository where the pre-trained model is stored
  prediction_length=prediction_length,  # Length of the forecast horizon
)

# Apply the UDFs to the DataFrame
forecasts = df.repartition(device_count).select(
  df.unique_id,  # Select the unique_id column from the DataFrame
  get_horizon_timestamps(df.ds).alias("ds"),  # Apply the horizon timestamps UDF to the ds column and alias the result as "ds"
  forecast_udf(df.y).alias("forecast"),  # Apply the forecast UDF to the y column and alias the result as "forecast"
)

# Display the resulting DataFrame with the forecasts
display(forecasts)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Register Model
# MAGIC We will package our model using [`mlflow.pyfunc.PythonModel`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) and register this in Unity Catalog.

# COMMAND ----------

import mlflow
import torch
import numpy as np
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, TensorSpec

# Set the MLflow registry URI to use Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Define a custom MLflow Python model class for MomentModel
class MomentModel(mlflow.pyfunc.PythonModel):
  def __init__(self, repository):
    from momentfm import MOMENTPipeline  # Import the MOMENTPipeline class from the momentfm library
    # Load the pre-trained model from the specified repository with the given task and forecast horizon
    self.model = MOMENTPipeline.from_pretrained(
      repository, 
      model_kwargs={
        "task_name": "forecasting",
        "forecast_horizon": 10},
      )
    self.model.init()  # Initialize the model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device to GPU if available, otherwise CPU
    self.model = self.model.to(self.device)  # Move the model to the selected device

  def predict(self, context, input_data, params=None):
    series = list(input_data)  # Convert input data to a list
    if len(series) < 512:
      # If the series is shorter than 512, pad with zeros
      input_mask = [1] * len(series) + [0] * (512 - len(series))
      series = series + [0] * (512 - len(series))
    else:
      # If the series is longer than or equal to 512, truncate to the last 512 values
      input_mask = [1] * 512
      series = series[-512:]
    # Convert input mask and series to PyTorch tensors and move them to the selected device
    input_mask = torch.reshape(torch.tensor(input_mask), (1, 512)).to(self.device)
    series = torch.reshape(torch.tensor(series), (1, 1, 512)).to(dtype=torch.float32).to(self.device)
    # Generate the forecast using the model
    output = self.model(series, input_mask=input_mask)
    forecast = output.forecast.squeeze().tolist()  # Squeeze the output tensor and convert to a list
    return forecast  # Return the forecast

# Initialize the custom MomentModel with the specified repository ID
pipeline = MomentModel(f"AutonLab/{moment_model}")
# Define the input and output schema for the model
input_schema = Schema([TensorSpec(np.dtype(np.double), (-1,))])
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1,))])
# Create a ModelSignature object to represent the input and output schema
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
# Create an example input to log with the model
input_example = np.random.rand(52)

# Define the registered model name using variables for catalog, database, and model
registered_model_name = f"{catalog}.{db}.{moment_model}"

# Log and register the model with MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
      "model",  # The artifact path where the model is logged
      python_model=pipeline,  # The custom Python model to log
      registered_model_name=registered_model_name,  # The name to register the model under
      signature=signature,  # The model signature
      input_example=input_example,  # An example input to log with the model
      pip_requirements=[
        "git+https://github.com/moment-timeseries-foundation-model/moment.git",  # Python package requirements
      ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ##Reload Model
# MAGIC Once the registration is complete, we will reload the model and generate forecasts.

# COMMAND ----------

from mlflow import MlflowClient
mlflow_client = MlflowClient()

# Define a function to get the latest version number of a registered model
def get_latest_model_version(mlflow_client, registered_model_name):
    latest_version = 1  # Initialize the latest version number to 1
    # Iterate through all model versions of the specified registered model
    for mv in mlflow_client.search_model_versions(f"name='{registered_model_name}'"):
        version_int = int(mv.version)  # Convert the version number to an integer
        if version_int > latest_version:  # Check if the current version is greater than the latest version
            latest_version = version_int  # Update the latest version number
    return latest_version  # Return the latest version number

# Get the latest version number of the specified registered model
model_version = get_latest_model_version(mlflow_client, registered_model_name)
# Construct the model URI using the registered model name and the latest version number
logged_model = f"models:/{registered_model_name}/{model_version}"

# Load the model as a PyFuncModel using the constructed model URI
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Create random input data (52 data points)
input_data = np.random.rand(52)

# Generate forecasts using the loaded model
loaded_model.predict(input_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Model
# MAGIC We will deploy our model behind a real-time endpoint of [Databricks Mosaic AI Model Serving](https://www.databricks.com/product/model-serving).

# COMMAND ----------

# With the token, you can create our authorization header for our subsequent REST calls
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Next you need an endpoint at which to execute your request which you can get from the notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()

# This object comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)

# Lastly, extract the Databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

# COMMAND ----------

import requests

model_serving_endpoint_name = moment_model

my_json = {
    "name": model_serving_endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": registered_model_name,
                "model_version": model_version,
                "workload_type": "GPU_SMALL",
                "workload_size": "Small",
                "scale_to_zero_enabled": "true",
            }
        ],
        "auto_capture_config": {
            "catalog_name": catalog,
            "schema_name": db,
            "table_name_prefix": model_serving_endpoint_name,
        },
    },
}

# Make sure to drop the inference table of it exists
_ = spark.sql(
    f"DROP TABLE IF EXISTS {catalog}.{db}.`{model_serving_endpoint_name}_payload`"
)

# COMMAND ----------

# Function to create an endpoint in Model Serving and deploy the model behind it
def func_create_endpoint(model_serving_endpoint_name):
    # get endpoint status
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    url = f"{endpoint_url}/{model_serving_endpoint_name}"
    r = requests.get(url, headers=headers)
    if "RESOURCE_DOES_NOT_EXIST" in r.text:
        print(
            "Creating this new endpoint: ",
            f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations",
        )
        re = requests.post(endpoint_url, headers=headers, json=my_json)
    else:
        new_model_version = (my_json["config"])["served_models"][0]["model_version"]
        print(
            "This endpoint existed previously! We are updating it to a new config with new model version: ",
            new_model_version,
        )
        # update config
        url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
        re = requests.put(url, headers=headers, json=my_json["config"])
        # wait till new config file in place
        import time, json

        # get endpoint status
        url = f"https://{instance}/api/2.0/serving-endpoints/{model_serving_endpoint_name}"
        retry = True
        total_wait = 0
        while retry:
            r = requests.get(url, headers=headers)
            assert (
                r.status_code == 200
            ), f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
            endpoint = json.loads(r.text)
            if "pending_config" in endpoint.keys():
                seconds = 10
                print("New config still pending")
                if total_wait < 6000:
                    # if less the 10 mins waiting, keep waiting
                    print(f"Wait for {seconds} seconds")
                    print(f"Total waiting time so far: {total_wait} seconds")
                    time.sleep(10)
                    total_wait += seconds
                else:
                    print(f"Stopping,  waited for {total_wait} seconds")
                    retry = False
            else:
                print("New config in place now!")
                retry = False

    assert (
        re.status_code == 200
    ), f"Expected an HTTP 200 response, received {re.status_code}"

# Function to delete the endpoint from Model Serving
def func_delete_model_serving_endpoint(model_serving_endpoint_name):
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    url = f"{endpoint_url}/{model_serving_endpoint_name}"
    response = requests.delete(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    else:
        print(model_serving_endpoint_name, "endpoint is deleted!")
    return response.json()

# COMMAND ----------

# Create an endpoint. This may take some time.
func_create_endpoint(model_serving_endpoint_name)

# COMMAND ----------

import time, mlflow

# Define a function to wait for a serving endpoint to be ready
def wait_for_endpoint():
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"  # Construct the base URL for the serving endpoints API
    while True:  # Infinite loop to repeatedly check the status of the endpoint
        url = f"{endpoint_url}/{model_serving_endpoint_name}"  # Construct the URL for the specific model serving endpoint
        response = requests.get(url, headers=headers)  # Send a GET request to the endpoint URL with the necessary headers
        
        # Ensure the response status code is 200 (OK)
        assert (
            response.status_code == 200
        ), f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        # Extract the status of the endpoint from the response JSON
        status = response.json().get("state", {}).get("ready", {})
        # print("status",status)  # Optional: Print the status for debugging purposes
        
        # Check if the endpoint status is "READY"
        if status == "READY":
            print(status)  # Print the status if the endpoint is ready
            print("-" * 80)  # Print a separator line for clarity
            return  # Exit the function when the endpoint is ready
        else:
            # Print a message indicating the endpoint is not ready and wait for 5 minutes
            print(f"Endpoint not ready ({status}), waiting 5 minutes")
            time.sleep(300)  # Wait for 300 seconds before checking again

# Get the Databricks web application URL using an MLflow utility function
api_url = mlflow.utils.databricks_utils.get_webapp_url()

# Call the wait_for_endpoint function to wait for the serving endpoint to be ready
wait_for_endpoint()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Online Forecast
# MAGIC Once the endpoint is ready, let's send a request to the model and generate an online forecast.

# COMMAND ----------

import os
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt

# Replace URL with the end point invocation url you get from Model Seriving page.
endpoint_url = f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
def forecast(input_data, url=endpoint_url, databricks_token=token):
    headers = {
        "Authorization": f"Bearer {databricks_token}",
        "Content-Type": "application/json",
    }
    body = {"inputs": input_data.tolist()}
    data = json.dumps(body)
    response = requests.request(method="POST", headers=headers, url=url, data=data)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    return response.json()

# COMMAND ----------

# Send request to the endpoint
input_data = np.random.rand(52)
forecast(input_data)

# COMMAND ----------

# Delete the serving endpoint
func_delete_model_serving_endpoint(model_serving_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. 
# MAGIC
# MAGIC The sources in all notebooks in this directory and the sub-directories are provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | datasetsforecast | Datasets for Time series forecasting | MIT | https://pypi.org/project/datasetsforecast/
# MAGIC | moment | A Family of Open Time-series Foundation Models | MIT | https://github.com/moment-timeseries-foundation-model/moment
