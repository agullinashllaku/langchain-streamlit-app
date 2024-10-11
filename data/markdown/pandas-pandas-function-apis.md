  * __[![Databricks](../_static/small-scale-lockup-full-color-rgb.svg)](https://www.databricks.com/)

  * __[![Databricks](../_static/small-scale-lockup-full-color-rgb.svg)](https://www.databricks.com/)
  * [Help Center](https://help.databricks.com/s/)
  * [Documentation](https://docs.databricks.com/en/index.html)
  * [Knowledge Base](https://kb.databricks.com/)

  * [Community](https://community.databricks.com)
  * [Support](https://help.databricks.com)
  * [Feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback)
  * [Try Databricks](https://databricks.com/try-databricks)

[![](../_static/icons/globe.png)English](javascript:void\(0\))

  * [English](../../en/pandas/pandas-function-apis.html)
  * [æ¥æ¬èª](../../ja/pandas/pandas-function-apis.html)
  * [PortuguÃªs](../../pt/pandas/pandas-function-apis.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/pandas/pandas-function-apis)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/pandas/pandas-function-apis.html)

[Databricks on AWS](../index.html)

Get started

  * [Get started](../getting-started/index.html)
  * [What is Databricks?](../introduction/index.html)
  * [DatabricksIQ](../databricksiq/index.html)
  * [Release notes](../release-notes/index.html)

Load & manage data

  * [Work with database objects](../database-objects/index.html)
  * [Connect to data sources](../connect/index.html)
  * [Connect to compute](../compute/index.html)
  * [Discover data](../discover/index.html)
  * [Query data](../query/index.html)
  * [Ingest data](../ingestion/index.html)
  * [Work with files](../files/index.html)
  * [Transform data](../transform/index.html)
  * [Schedule and orchestrate workflows](../jobs/index.html)
  * [Monitor data and AI assets](../lakehouse-monitoring/index.html)
  * [Share data securely](../data-sharing/index.html)

Work with data

  * [Data engineering](../workspace-index.html)
  * [AI and machine learning](../machine-learning/index.html)
  * [Generative AI tutorial](../generative-ai/tutorials/ai-cookbook/index.html)
  * [Business intelligence](../ai-bi/index.html)
  * [Data warehousing](../sql/index.html)
  * [Notebooks](../notebooks/index.html)
  * [Delta Lake](../delta/index.html)
  * [Developers](../languages/index.html)
    * [Python](../languages/python.html)
      * [PySpark](../pyspark/index.html)
      * [Pandas API on Spark](pandas-on-spark.html)
        * [pandas on Databricks](index.html)
        * [Convert between PySpark and pandas DataFrames](pyspark-pandas-conversion.html)
        * pandas function APIs
    * [R](../sparkr/index.html)
    * [Scala](../languages/scala.html)
    * [SQL](../sql/language-manual/index.html)
    * [User-defined functions (UDFs)](../udf/index.html)
    * [Databricks Apps](../dev-tools/databricks-apps/index.html)
    * [Tools](../dev-tools/index.html)
  * [Technology partners](../integrations/index.html)

Administration

  * [Account and workspace administration](../admin/index.html)
  * [Security and compliance](../security/index.html)
  * [Data governance (Unity Catalog)](../data-governance/index.html)
  * [Lakehouse architecture](../lakehouse-architecture/index.html)

Reference & resources

  * [Reference](../reference/api.html)
  * [Resources](../resources/index.html)
  * [Whatâs coming?](../whats-coming.html)
  * [Documentation archive](../archive/index.html)

Updated Oct 11, 2024

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation
Feedback)

  * [Documentation](../index.html)
  * [Develop on Databricks](../languages/index.html)
  * [Databricks for Python developers](../languages/python.html)
  * [Pandas API on Spark](pandas-on-spark.html)
  * pandas function APIs
  * 

# pandas function APIs

pandas function APIs enable you to directly apply a Python native function
that takes and outputs pandas instances to a PySpark DataFrame. Similar to
[pandas user-defined functions](../udf/pandas.html), function APIs also use
[Apache Arrow](https://arrow.apache.org/) to transfer data and pandas to work
with the data; however, Python type hints are optional in pandas function
APIs.

There are three types of pandas function APIs:

  * Grouped map

  * Map

  * Cogrouped map

pandas function APIs leverage the same internal logic that pandas UDF
execution uses. They share characteristics such as PyArrow, supported SQL
types, and the configurations.

For more information, see the blog post [New Pandas UDFs and Python Type Hints
in the Upcoming Release of Apache Spark
3.0](https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-
hints-in-the-upcoming-release-of-apache-spark-3-0.html).

## Grouped map

You transform your grouped data using `groupBy().applyInPandas()` to implement
the âsplit-apply-combineâ pattern. Split-apply-combine consists of three
steps:

  * Split the data into groups by using `DataFrame.groupBy`.

  * Apply a function on each group. The input and output of the function are both `pandas.DataFrame`. The input data contains all the rows and columns for each group.

  * Combine the results into a new `DataFrame`.

To use `groupBy().applyInPandas()`, you must define the following:

  * A Python function that defines the computation for each group

  * A `StructType` object or a string that defines the schema of the output `DataFrame`

The column labels of the returned `pandas.DataFrame` must either match the
field names in the defined output schema if specified as strings, or match the
field data types by position if not strings, for example, integer indices. See
[pandas.DataFrame](https://pandas.pydata.org/pandas-
docs/stable/reference/api/pandas.DataFrame.html) for how to label columns when
constructing a `pandas.DataFrame`.

All data for a group is loaded into memory before the function is applied.
This can lead to out of memory exceptions, especially if the group sizes are
skewed. The configuration for
[maxRecordsPerBatch](https://spark.apache.org/docs/latest/sql-pyspark-pandas-
with-arrow.html#setting-arrow-batch-size) is not applied on groups and it is
up to you to ensure that the grouped data fits into the available memory.

The following example shows how to use `groupby().apply()` to subtract the
mean from each value in the group.

    
    
    df = spark.createDataFrame(
        [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
        ("id", "v"))
    
    def subtract_mean(pdf):
        # pdf is a pandas.DataFrame
        v = pdf.v
        return pdf.assign(v=v - v.mean())
    
    df.groupby("id").applyInPandas(subtract_mean, schema="id long, v double").show()
    # +---+----+
    # | id|   v|
    # +---+----+
    # |  1|-0.5|
    # |  1| 0.5|
    # |  2|-3.0|
    # |  2|-1.0|
    # |  2| 4.0|
    # +---+----+
    

For detailed usage, see [pyspark.sql.GroupedData.applyInPandas](https://api-
docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.GroupedData.applyInPandas.html#pyspark-
sql-groupeddata-applyinpandas).

## Map

You perform map operations with pandas instances by `DataFrame.mapInPandas()`
in order to transform an iterator of `pandas.DataFrame` to another iterator of
`pandas.DataFrame` that represents the current PySpark DataFrame and returns
the result as a PySpark DataFrame.

The underlying function takes and outputs an iterator of `pandas.DataFrame`.
It can return output of arbitrary length in contrast to some pandas UDFs such
as Series to Series.

The following example shows how to use `mapInPandas()`:

    
    
    df = spark.createDataFrame([(1, 21), (2, 30)], ("id", "age"))
    
    def filter_func(iterator):
        for pdf in iterator:
            yield pdf[pdf.id == 1]
    
    df.mapInPandas(filter_func, schema=df.schema).show()
    # +---+---+
    # | id|age|
    # +---+---+
    # |  1| 21|
    # +---+---+
    

For detailed usage, see [pyspark.sql.DataFrame.mapInPandas](https://api-
docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.DataFrame.mapInPandas.html#pyspark-
sql-dataframe-mapinpandas).

## Cogrouped map

For cogrouped map operations with pandas instances, use
`DataFrame.groupby().cogroup().applyInPandas()` to cogroup two PySpark
`DataFrame`s by a common key and then apply a Python function to each cogroup
as shown:

  * Shuffle the data such that the groups of each DataFrame which share a key are cogrouped together.

  * Apply a function to each cogroup. The input of the function is two `pandas.DataFrame` (with an optional tuple representing the key). The output of the function is a `pandas.DataFrame`.

  * Combine the `pandas.DataFrame`s from all groups into a new PySpark `DataFrame`.

To use `groupBy().cogroup().applyInPandas()`, you must define the following:

  * A Python function that defines the computation for each cogroup.

  * A `StructType` object or a string that defines the schema of the output PySpark `DataFrame`.

The column labels of the returned `pandas.DataFrame` must either match the
field names in the defined output schema if specified as strings, or match the
field data types by position if not strings, for example, integer indices. See
[pandas.DataFrame](https://pandas.pydata.org/pandas-
docs/stable/reference/api/pandas.DataFrame.html) for how to label columns when
constructing a `pandas.DataFrame`.

All data for a cogroup is loaded into memory before the function is applied.
This can lead to out of memory exceptions, especially if the group sizes are
skewed. The configuration for
[maxRecordsPerBatch](https://spark.apache.org/docs/latest/sql-pyspark-pandas-
with-arrow.html#setting-arrow-batch-size) is not applied and it is up to you
to ensure that the cogrouped data fits into the available memory.

The following example shows how to use `groupby().cogroup().applyInPandas()`
to perform an `asof join` between two datasets.

    
    
    import pandas as pd
    
    df1 = spark.createDataFrame(
        [(20000101, 1, 1.0), (20000101, 2, 2.0), (20000102, 1, 3.0), (20000102, 2, 4.0)],
        ("time", "id", "v1"))
    
    df2 = spark.createDataFrame(
        [(20000101, 1, "x"), (20000101, 2, "y")],
        ("time", "id", "v2"))
    
    def asof_join(l, r):
        return pd.merge_asof(l, r, on="time", by="id")
    
    df1.groupby("id").cogroup(df2.groupby("id")).applyInPandas(
        asof_join, schema="time int, id int, v1 double, v2 string").show()
    # +--------+---+---+---+
    # |    time| id| v1| v2|
    # +--------+---+---+---+
    # |20000101|  1|1.0|  x|
    # |20000102|  1|3.0|  x|
    # |20000101|  2|2.0|  y|
    # |20000102|  2|4.0|  y|
    # +--------+---+---+---+
    

For detailed usage, see
[pyspark.sql.PandasCogroupedOps.applyInPandas](https://api-
docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.PandasCogroupedOps.applyInPandas.html#pyspark-
sql-pandascogroupedops-applyinpandas).

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

