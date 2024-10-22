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

  * [English](../../en/pandas/pyspark-pandas-conversion.html)
  * [æ¥æ¬èª](../../ja/pandas/pyspark-pandas-conversion.html)
  * [PortuguÃªs](../../pt/pandas/pyspark-pandas-conversion.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/pandas/pyspark-pandas-conversion)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/pandas/pyspark-pandas-conversion.html)

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
        * Convert between PySpark and pandas DataFrames
        * [pandas function APIs](pandas-function-apis.html)
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

Updated Oct 21, 2024

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation
Feedback)

  * [Documentation](../index.html)
  * [Develop on Databricks](../languages/index.html)
  * [Databricks for Python developers](../languages/python.html)
  * [Pandas API on Spark](pandas-on-spark.html)
  * Convert between PySpark and pandas DataFrames
  * 

# Convert between PySpark and pandas DataFrames

Learn how to convert Apache Spark DataFrames to and from pandas DataFrames
using Apache Arrow in Databricks.

## Apache Arrow and PyArrow

[Apache Arrow](https://arrow.apache.org/) is an in-memory columnar data format
used in Apache Spark to efficiently transfer data between JVM and Python
processes. This is beneficial to Python developers who work with pandas and
NumPy data. However, its usage requires some minor configuration or code
changes to ensure compatibility and gain the most benefit.

PyArrow is a Python binding for Apache Arrow and is installed in Databricks
Runtime. For information on the version of PyArrow available in each
Databricks Runtime version, see the [Databricks Runtime release notes versions
and compatibility](../release-notes/runtime/index.html).

## Supported SQL types

All Spark SQL data types are supported by Arrow-based conversion except
`ArrayType` of `TimestampType`. `MapType` and `ArrayType` of nested
`StructType` are only supported when using PyArrow 2.0.0 and above.
`StructType` is represented as a `pandas.DataFrame` instead of
`pandas.Series`.

## Convert PySpark DataFrames to and from pandas DataFrames

Arrow is available as an optimization when converting a PySpark DataFrame to a
pandas DataFrame with `toPandas()` and when creating a PySpark DataFrame from
a pandas DataFrame with `createDataFrame(pandas_df)`.

To use Arrow for these methods, set the [Spark
configuration](../compute/configure.html#spark-configuration)
`spark.sql.execution.arrow.pyspark.enabled` to `true`. This configuration is
enabled by default except for High Concurrency clusters as well as user
isolation clusters in workspaces that are Unity Catalog enabled.

In addition, optimizations enabled by
`spark.sql.execution.arrow.pyspark.enabled` could fall back to a non-Arrow
implementation if an error occurs before the computation within Spark. You can
control this behavior using the Spark configuration
`spark.sql.execution.arrow.pyspark.fallback.enabled`.

### Example

    
    
    import numpy as np
    import pandas as pd
    
    # Enable Arrow-based columnar data transfers
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    # Generate a pandas DataFrame
    pdf = pd.DataFrame(np.random.rand(100, 3))
    
    # Create a Spark DataFrame from a pandas DataFrame using Arrow
    df = spark.createDataFrame(pdf)
    
    # Convert the Spark DataFrame back to a pandas DataFrame using Arrow
    result_pdf = df.select("*").toPandas()
    

Using the Arrow optimizations produces the same results as when Arrow is not
enabled. Even with Arrow, `toPandas()` results in the collection of all
records in the DataFrame to the driver program and should be done on a small
subset of the data.

In addition, not all Spark data types are supported and an error can be raised
if a column has an unsupported type. If an error occurs during
`createDataFrame()`, Spark creates the DataFrame without Arrow.

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

