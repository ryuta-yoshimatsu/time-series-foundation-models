# Databricks notebook source
# MAGIC %pip install datasetsforecast==0.0.8 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pathlib
import pandas as pd
from datasetsforecast.m4 import M4
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("db", "")
dbutils.widgets.text("n", "")

catalog = dbutils.widgets.get("catalog")  # Name of the catalog we use to manage our assets
db = dbutils.widgets.get("db")  # Name of the schema we use to store assets
n = int(dbutils.widgets.get("n"))  # Number of time series to sample

#  Make sure the catalog, schema and volume exist
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Daily M4 Data

# COMMAND ----------

def create_m4_daily():
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Daily")
    _ids = [f"D{i}" for i in range(1, n)]
    y_df = (
        y_df.groupby("unique_id")
        .filter(lambda x: x.unique_id.iloc[0] in _ids)
        .groupby("unique_id")
        .apply(transform_group_daily)
        .reset_index(drop=True)
    )
    return y_df


def transform_group_daily(df):
    unique_id = df.unique_id.iloc[0]
    if len(df) > 1020:
        df = df.iloc[-1020:]
    _start = pd.Timestamp("2020-01-01")
    _end = _start + pd.DateOffset(days=int(df.count()[0]) - 1)
    date_idx = pd.date_range(start=_start, end=_end, freq="D", name="ds")
    res_df = pd.DataFrame(data=[], index=date_idx).reset_index()
    res_df["unique_id"] = unique_id
    res_df["y"] = df.y.values
    return res_df

 
(
    spark.createDataFrame(create_m4_daily())
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.m4_daily_train")
)

print(f"Saved data to {catalog}.{db}.m4_daily_train")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monthly M4 Data

# COMMAND ----------

def create_m4_monthly():
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Monthly")
    _ids = [f"M{i}" for i in range(1, n + 1)]
    y_df = (
        y_df.groupby("unique_id")
        .filter(lambda x: x.unique_id.iloc[0] in _ids)
        .groupby("unique_id")
        .apply(transform_group_monthly)
        .reset_index(drop=True)
    )
    return y_df


def transform_group_monthly(df):
    unique_id = df.unique_id.iloc[0]
    _cnt = 60  # df.count()[0]
    _start = pd.Timestamp("2018-01-01")
    _end = _start + pd.DateOffset(months=_cnt)
    date_idx = pd.date_range(start=_start, end=_end, freq="M", name="date")
    _df = (
        pd.DataFrame(data=[], index=date_idx)
        .reset_index()
        .rename(columns={"index": "date"})
    )
    _df["unique_id"] = unique_id
    _df["y"] = df[:60].y.values
    return _df


(
    spark.createDataFrame(create_m4_monthly())
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.m4_monthly_train")
)

print(f"Saved data to {catalog}.{db}.m4_monthly_train")

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. 
# MAGIC
# MAGIC The sources in all notebooks in this directory and the sub-directories are provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | datasetsforecast | Datasets for Time series forecasting | MIT | https://pypi.org/project/datasetsforecast/

# COMMAND ----------


