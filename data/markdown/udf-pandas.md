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

  * [English](../../en/udf/pandas.html)
  * [æ¥æ¬èª](../../ja/udf/pandas.html)
  * [PortuguÃªs](../../pt/udf/pandas.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/udf/pandas)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/udf/pandas.html)

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
    * [R](../sparkr/index.html)
    * [Scala](../languages/scala.html)
    * [SQL](../sql/language-manual/index.html)
    * [User-defined functions (UDFs)](index.html)
      * [UDFs in Unity Catalog](unity-catalog.html)
      * pandas UDFs
      * [Python scalar UDFs](python.html)
      * [Python UDTFs (user-defined table functions)](python-udtf.html)
      * [Scala UDFs](scala.html)
      * [Scala UDAFs (user-defined aggregate functions)](aggregate-scala.html)
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
  * [What are user-defined functions (UDFs)?](index.html)
  * pandas user-defined functions
  * 

# pandas user-defined functions

A pandas user-defined function (UDF)âalso known as vectorized UDFâis a
user-defined function that uses [Apache Arrow](https://arrow.apache.org/) to
transfer data and pandas to work with the data. pandas UDFs allow vectorized
operations that can increase performance up to 100x compared to row-at-a-time
[Python UDFs](python.html).

For background information, see the blog post [New Pandas UDFs and Python Type
Hints in the Upcoming Release of Apache Spark
3.0](https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-
hints-in-the-upcoming-release-of-apache-spark-3-0.html).

You define a pandas UDF using the keyword `pandas_udf` as a decorator and wrap
the function with a [Python type
hint](https://www.python.org/dev/peps/pep-0484/). This article describes the
different types of pandas UDFs and shows how to use pandas UDFs with type
hints.

## Series to Series UDF

You use a Series to Series pandas UDF to vectorize scalar operations. You can
use them with APIs such as `select` and `withColumn`.

The Python function should take a pandas Series as an input and return a
pandas Series of the same length, and you should specify these in the Python
type hints. Spark runs a pandas UDF by splitting columns into batches, calling
the function for each batch as a subset of the data, then concatenating the
results.

The following example shows how to create a pandas UDF that computes the
product of 2 columns.

    
    
    import pandas as pd
    from pyspark.sql.functions import col, pandas_udf
    from pyspark.sql.types import LongType
    
    # Declare the function and create the UDF
    def multiply_func(a: pd.Series, b: pd.Series) -> pd.Series:
        return a * b
    
    multiply = pandas_udf(multiply_func, returnType=LongType())
    
    # The function for a pandas_udf should be able to execute with local pandas data
    x = pd.Series([1, 2, 3])
    print(multiply_func(x, x))
    # 0    1
    # 1    4
    # 2    9
    # dtype: int64
    
    # Create a Spark DataFrame, 'spark' is an existing SparkSession
    df = spark.createDataFrame(pd.DataFrame(x, columns=["x"]))
    
    # Execute function as a Spark vectorized UDF
    df.select(multiply(col("x"), col("x"))).show()
    # +-------------------+
    # |multiply_func(x, x)|
    # +-------------------+
    # |                  1|
    # |                  4|
    # |                  9|
    # +-------------------+
    

## Iterator of Series to Iterator of Series UDF

An iterator UDF is the same as a scalar pandas UDF except:

  * The Python function

    * Takes an iterator of batches instead of a single input batch as input.

    * Returns an iterator of output batches instead of a single output batch.

  * The length of the entire output in the iterator should be the same as the length of the entire input.

  * The wrapped pandas UDF takes a single Spark column as an input.

You should specify the Python type hint as `Iterator[pandas.Series]` ->
`Iterator[pandas.Series]`.

This pandas UDF is useful when the UDF execution requires initializing some
state, for example, loading a machine learning model file to apply inference
to every input batch.

The following example shows how to create a pandas UDF with iterator support.

    
    
    import pandas as pd
    from typing import Iterator
    from pyspark.sql.functions import col, pandas_udf, struct
    
    pdf = pd.DataFrame([1, 2, 3], columns=["x"])
    df = spark.createDataFrame(pdf)
    
    # When the UDF is called with the column,
    # the input to the underlying function is an iterator of pd.Series.
    @pandas_udf("long")
    def plus_one(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for x in batch_iter:
            yield x + 1
    
    df.select(plus_one(col("x"))).show()
    # +-----------+
    # |plus_one(x)|
    # +-----------+
    # |          2|
    # |          3|
    # |          4|
    # +-----------+
    
    # In the UDF, you can initialize some state before processing batches.
    # Wrap your code with try/finally or use context managers to ensure
    # the release of resources at the end.
    y_bc = spark.sparkContext.broadcast(1)
    
    @pandas_udf("long")
    def plus_y(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        y = y_bc.value  # initialize states
        try:
            for x in batch_iter:
                yield x + y
        finally:
            pass  # release resources here, if any
    
    df.select(plus_y(col("x"))).show()
    # +---------+
    # |plus_y(x)|
    # +---------+
    # |        2|
    # |        3|
    # |        4|
    # +---------+
    

## Iterator of multiple Series to Iterator of Series UDF

An Iterator of multiple Series to Iterator of Series UDF has similar
characteristics and restrictions as Iterator of Series to Iterator of Series
UDF. The specified function takes an iterator of batches and outputs an
iterator of batches. It is also useful when the UDF execution requires
initializing some state.

The differences are:

  * The underlying Python function takes an iterator of a _tuple_ of pandas Series.

  * The wrapped pandas UDF takes _multiple_ Spark columns as an input.

You specify the type hints as `Iterator[Tuple[pandas.Series, ...]]` ->
`Iterator[pandas.Series]`.

    
    
    from typing import Iterator, Tuple
    import pandas as pd
    
    from pyspark.sql.functions import col, pandas_udf, struct
    
    pdf = pd.DataFrame([1, 2, 3], columns=["x"])
    df = spark.createDataFrame(pdf)
    
    @pandas_udf("long")
    def multiply_two_cols(
            iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
        for a, b in iterator:
            yield a * b
    
    df.select(multiply_two_cols("x", "x")).show()
    # +-----------------------+
    # |multiply_two_cols(x, x)|
    # +-----------------------+
    # |                      1|
    # |                      4|
    # |                      9|
    # +-----------------------+
    

## Series to scalar UDF

Series to scalar pandas UDFs are similar to Spark aggregate functions. A
Series to scalar pandas UDF defines an aggregation from one or more pandas
Series to a scalar value, where each pandas Series represents a Spark column.
You use a Series to scalar pandas UDF with APIs such as `select`,
`withColumn`, `groupBy.agg`, and [pyspark.sql.Window](https://api-
docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.Window.html).

You express the type hint as `pandas.Series, ...` -> `Any`. The return type
should be a primitive data type, and the returned scalar can be either a
Python primitive type, for example, `int` or `float` or a NumPy data type such
as `numpy.int64` or `numpy.float64`. `Any` should ideally be a specific scalar
type.

This type of UDF _does not_ support partial aggregation and all data for each
group is loaded into memory.

The following example shows how to use this type of UDF to compute mean with
`select`, `groupBy`, and `window` operations:

    
    
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql import Window
    
    df = spark.createDataFrame(
        [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
        ("id", "v"))
    
    # Declare the function and create the UDF
    @pandas_udf("double")
    def mean_udf(v: pd.Series) -> float:
        return v.mean()
    
    df.select(mean_udf(df['v'])).show()
    # +-----------+
    # |mean_udf(v)|
    # +-----------+
    # |        4.2|
    # +-----------+
    
    df.groupby("id").agg(mean_udf(df['v'])).show()
    # +---+-----------+
    # | id|mean_udf(v)|
    # +---+-----------+
    # |  1|        1.5|
    # |  2|        6.0|
    # +---+-----------+
    
    w = Window \
        .partitionBy('id') \
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    df.withColumn('mean_v', mean_udf(df['v']).over(w)).show()
    # +---+----+------+
    # | id|   v|mean_v|
    # +---+----+------+
    # |  1| 1.0|   1.5|
    # |  1| 2.0|   1.5|
    # |  2| 3.0|   6.0|
    # |  2| 5.0|   6.0|
    # |  2|10.0|   6.0|
    # +---+----+------+
    

For detailed usage, see [pyspark.sql.functions.pandas_udf](https://api-
docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.functions.pandas_udf.html?highlight=pandas%20udf#pyspark-
sql-functions-pandas-udf).

## Usage

### Setting Arrow batch size

Note

This configuration has no impact on compute configured with shared access mode
and Databricks Runtime 13.3 LTS through 14.2.

Data partitions in Spark are converted into Arrow record batches, which can
temporarily lead to high memory usage in the JVM. To avoid possible out of
memory exceptions, you can adjust the size of the Arrow record batches by
setting the `spark.sql.execution.arrow.maxRecordsPerBatch` configuration to an
integer that determines the maximum number of rows for each batch. The default
value is 10,000 records per batch. If the number of columns is large, the
value should be adjusted accordingly. Using this limit, each data partition is
divided into 1 or more record batches for processing.

### Timestamp with time zone semantics

Spark internally stores timestamps as UTC values, and timestamp data brought
in without a specified time zone is converted as local time to UTC with
microsecond resolution.

When timestamp data is exported or displayed in Spark, the session time zone
is used to localize the timestamp values. The session time zone is set with
the `spark.sql.session.timeZone` configuration and defaults to the JVM system
local time zone. pandas uses a `datetime64` type with nanosecond resolution,
`datetime64[ns]`, with optional time zone on a per-column basis.

When timestamp data is transferred from Spark to pandas it is converted to
nanoseconds and each column is converted to the Spark session time zone then
localized to that time zone, which removes the time zone and displays values
as local time. This occurs when calling `toPandas()` or `pandas_udf` with
timestamp columns.

When timestamp data is transferred from pandas to Spark, it is converted to
UTC microseconds. This occurs when calling `createDataFrame` with a pandas
DataFrame or when returning a timestamp from a pandas UDF. These conversions
are done automatically to ensure Spark has data in the expected format, so it
is not necessary to do any of these conversions yourself. Any nanosecond
values are truncated.

A standard UDF loads timestamp data as Python datetime objects, which is
different than a pandas timestamp. To get the best performance, we recommend
that you use pandas time series functionality when working with timestamps in
a pandas UDF. For details, see [Time Series / Date
functionality](https://pandas.pydata.org/pandas-
docs/stable/user_guide/timeseries.html).

## Example notebook

The following notebook illustrates the performance improvements you can
achieve with pandas UDFs:

### pandas UDFs benchmark notebook

[Open notebook in new tab](/_extras/notebooks/source/pandas-udfs-
benchmark.html) ![Copy to clipboard](/_static/clippy.svg) Copy link for import

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

