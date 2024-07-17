# Databricks notebook source
# MAGIC %md
# MAGIC This is an example notebook that shows how to use [chronos](https://github.com/amazon-science/chronos-forecasting/tree/main) models on Databricks. The notebook loads, fine-tunes, and registers the model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster setup
# MAGIC **As of June 17, 2024, Chronos finetuning script works on DBR ML 14.3 and below (do not use DBR ML 15 or above).**
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html). The cluster can be single-node or multi-node with one or more GPU instances on each worker: e.g. [g5.12xlarge [A10G]](https://aws.amazon.com/ec2/instance-types/g5/) on AWS or [Standard_NV72ads_A10_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nva10v5-series) on Azure.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install package

# COMMAND ----------

# MAGIC %pip install "chronos[training] @ git+https://github.com/amazon-science/chronos-forecasting.git" --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data 
# MAGIC We use [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast/tree/main/) package to download M4 data. M4 dataset contains a set of time series which we use for testing MMF. Below we have written a number of custome functions to convert M4 time series to an expected format.
# MAGIC
# MAGIC Make sure that the catalog and the schema already exist.

# COMMAND ----------

catalog = "mmf"  # Name of the catalog we use to manage our assets
db = "m4"  # Name of the schema we use to manage our assets (e.g. datasets)
volume = "chronos_fine_tune" # Name of the volume we store the data and the weigts
model = "chronos-t5-tiny" # Chronos model to finetune. Alternatives: -mini, -small, -base, -large
n = 1000  # Number of time series to sample

# COMMAND ----------

# This cell runs the notebook ../data_preparation and creates the following tables with M4 data: 
# 1. {catalog}.{db}.m4_daily_train
# 2. {catalog}.{db}.m4_monthly_train
dbutils.notebook.run("../data_preparation", timeout_seconds=0, arguments={"catalog": catalog, "db": db, "n": n})

# COMMAND ----------

from pyspark.sql.functions import collect_list

# Make sure that the data exists
df = spark.table(f'{catalog}.{db}.m4_daily_train')
df = df.groupBy('unique_id').agg(collect_list('ds').alias('ds'), collect_list('y').alias('y')).toPandas()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC We need to convert our time series dataset into a GluonTS-compatible file dataset.

# COMMAND ----------

import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from gluonts.dataset.arrow import ArrowWriter

def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
    compression: str = "lz4",
):
    """
    This function converts time series data into the Apache Arrow format and saves it to a file.
    
    Parameters:
    - path (Union[str, Path]): The file path where the Arrow file will be saved.
    - time_series (Union[List[np.ndarray], np.ndarray]): The time series data to be converted.
    - start_times (Optional[Union[List[np.datetime64], np.ndarray]]): The start times for each time series. If None, a default start time is used.
    - compression (str): The compression algorithm to use for the Arrow file. Default is 'lz4'.
    """
    
    # If start_times is not provided, set all start times to '2000-01-01 00:00:00'
    if start_times is None:
        start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

    # Ensure there is a start time for each time series
    assert len(time_series) == len(start_times)

    # Create a list of dictionaries where each dictionary represents a time series
    # Each dictionary contains the start time and the corresponding time series data
    dataset = [
        {"start": start, "target": ts} for ts, start in zip(time_series, start_times)
    ]
    
    # Use ArrowWriter to write the dataset to a file in Arrow format with the specified compression
    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


# COMMAND ----------

# MAGIC %md
# MAGIC Convert the Pandas dataframe to an arrow file and write it to UC Volume.  

# COMMAND ----------

time_series = list(df["y"])
start_times = list(df["ds"].apply(lambda x: x.min().to_numpy()))

# Make sure that the volume exists. We stored the fine-tuned weights here.
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{db}.{volume}")

# Convert to GluonTS arrow format and save it in UC Volume
convert_to_arrow(
    f"/Volumes/{catalog}/{db}/{volume}/data.arrow", 
    time_series=time_series, 
    start_times=start_times,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ##Run Fine-tuning
# MAGIC
# MAGIC In this example, we wil fine-tune `amazon/chronos-t5-tiny` for 1000 steps with initial learning rate of 1e-3. 
# MAGIC
# MAGIC Make sure that you have the configuration yaml files placed inside the `configs` folder and the `train.py` script in the same directory. These two assets are taken directly from [chronos-forecasting/scripts/training](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/training). They are subject to change as the Chronos' team develops the framework further. Keep your eyes on the latest changes (we will try too) and use the latest versions as needed. We have made a small change to our `train.py` script and set the frequency of the time series to daily ("D"). 
# MAGIC
# MAGIC Inside the configuration yaml (for this example, `configs/chronos-t5-tiny.yaml`), make sure to set the parameters: 
# MAGIC - `training_data_paths` to `/Volumes/mmf/m4/chronos_fine_tune/data.arrow`, where your arrow converted file is stored
# MAGIC - `probability` to `1.0` if there is only one data source
# MAGIC - `prediction_length` to your use case's forecasting horizon (in this example `10`)
# MAGIC - `num_samples` to how many sample you want to generate  
# MAGIC - `output_dir` to `/Volumes/mmf/m4/chronos_fine_tune/`, where you want to store your fine-tuned weights
# MAGIC
# MAGIC And other parameters if needed. 
# MAGIC
# MAGIC `CUDA_VISIBLE_DEVICES` tell the script about the avalaible GPU resources. In this example, we are using a single node cluster with g5.12xlarge on AWS, which comes with 4 A10G GPU isntances, hence `CUDA_VISIBLE_DEVICES=0,1,2,3`. See Chronos' training [README](https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/README.md) for more information on multinode multigpu setup.

# COMMAND ----------

# MAGIC %sh CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
# MAGIC     --config configs/chronos-t5-tiny.yaml \
# MAGIC     --model-id amazon/chronos-t5-tiny \
# MAGIC     --no-random-init \
# MAGIC     --max-steps 1000 \
# MAGIC     --learning-rate 0.001

# COMMAND ----------

# MAGIC %md
# MAGIC ##Register Model
# MAGIC We get the fine-tuned weights from the latest run from UC volume, wrap the pipeline with [`mlflow.pyfunc.PythonModel`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) and register this on Unity Catalog.

# COMMAND ----------

import os
import glob
import mlflow
import torch
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, TensorSpec

# Set the registry URI for MLflow to Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

class FineTunedChronosModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load the model context including the pretrained weights.
        The model is loaded onto a GPU if available, otherwise on CPU.
        """
        import torch
        from chronos import ChronosPipeline
        
        # Load the pretrained model pipeline from provided weights
        self.pipeline = ChronosPipeline.from_pretrained(
            context.artifacts["weights"],
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )

    def predict(self, context, input_data, params=None):
        """
        Make predictions using the loaded model.
        
        Parameters:
        - context: The context in which the model is being run.
        - input_data: The input data for prediction, expected to be a list of series.
        - params: Additional parameters for prediction (not used here).
        
        Returns:
        - forecast: The predicted results as a NumPy array.
        """
        # Convert input data to a list of torch tensors
        history = [torch.tensor(list(series)) for series in input_data]
        
        # Make predictions using the model pipeline
        forecast = self.pipeline.predict(
            context=history,
            prediction_length=10,
            num_samples=10,
        )
        
        # Convert the forecast to a NumPy array and return
        return forecast.numpy()

# Directory path components for locating the latest run
files = os.listdir(f"/Volumes/{catalog}/{db}/{volume}/")

# Extract run numbers from the directory names
runs = [int(file[4:]) for file in files if "run-" in file]

# Identify the latest run based on the highest run number
latest_run = max(runs)

# Construct the registered model name and weights path
registered_model_name = f"{catalog}.{db}.{model}_finetuned"
weights = f"/Volumes/{catalog}/{db}/{volume}/run-{latest_run}/checkpoint-final/"

# Define the model input and output schema for registration
input_schema = Schema([TensorSpec(np.dtype(np.double), (-1, -1))])
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, -1, -1))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Example input data for model registration
input_example = np.random.rand(1, 52)

# Register the fine-tuned model with MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=FineTunedChronosModel(),
        artifacts={"weights": weights},
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "chronos[training] @ git+https://github.com/amazon-science/chronos-forecasting.git",
        ],
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ##Reload Model
# MAGIC We reload the model from the registry and perform forecasting on the in-training time series (for testing purpose). You can also go ahead and deploy this model behind a Model Serving's real-time endpoint. See the previous notebook: [`01_chronos_load_inference`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation-model-examples/chronos/01_chronos_load_inference.py) for more information.

# COMMAND ----------

from mlflow import MlflowClient

# Create an instance of the MlflowClient to interact with the MLflow tracking server
client = MlflowClient()

def get_latest_model_version(client, registered_model_name):
    """
    Retrieve the latest version number of a registered model.
    
    Parameters:
    - client (MlflowClient): The MLflow client instance.
    - registered_model_name (str): The name of the registered model.
    
    Returns:
    - latest_version (int): The latest version number of the registered model.
    """
    # Initialize the latest version to 1 (assuming at least one version exists)
    latest_version = 1
    
    # Iterate over all model versions for the given registered model
    for mv in client.search_model_versions(f"name='{registered_model_name}'"):
        # Convert the version to an integer
        version_int = int(mv.version)
        
        # Update the latest version if a higher version is found
        if version_int > latest_version:
            latest_version = version_int
            
    # Return the latest version number
    return latest_version

# Get the latest version of the registered model
model_version = get_latest_model_version(client, registered_model_name)

# Construct the URI for the logged model using the registered model name and latest version
logged_model = f"models:/{registered_model_name}/{model_version}"

# Load the model as a PyFuncModel from the logged model URI
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Create input data for prediction from the first 100 elements of the 'y' column in a DataFrame
input_data = df["y"][:100].to_numpy() # Shape should be (batch, series)

# Generate forecasts using the loaded model
loaded_model.predict(input_data)


# COMMAND ----------


