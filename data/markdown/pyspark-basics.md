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

  * [English](../../en/pyspark/basics.html)
  * [æ¥æ¬èª](../../ja/pyspark/basics.html)
  * [PortuguÃªs](../../pt/pyspark/basics.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/pyspark/basics)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/pyspark/basics.html)

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
      * [PySpark](index.html)
        * PySpark basics
        * [PySpark custom data sources](datasources.html)
      * [Pandas API on Spark](../pandas/pandas-on-spark.html)
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
  * [PySpark on Databricks](index.html)
  * PySpark basics
  * 

# PySpark basics

This article walks through simple examples to illustrate usage of PySpark. It
assumes you understand fundamental [Apache Spark concepts](index.html#spark-
concepts) and are running commands in a Databricks
[notebook](../notebooks/notebooks-code.html) connected to compute. You create
DataFrames using sample data, perform basic transformations including row and
column operations on this data, combine multiple DataFrames and aggregate this
data, visualize this data, and then save it to a table or file.

## Upload data

Some examples in this article use Databricks-provided sample data to
demonstrate using DataFrames to load, transform, and save data. If you want to
use your own data that is not yet in Databricks, you can upload it first and
create a DataFrame from it. See [Create or modify a table using file
upload](../ingestion/file-upload/upload-data.html) and [Upload files to a
Unity Catalog volume](../ingestion/file-upload/upload-to-volume.html).

### About Databricks sample data

Databricks provides sample data in the `samples` catalog and in the
`/databricks-datasets` directory.

  * To access the sample data in the `samples` catalog, use the format `samples.<schema-name>.<table-name>`. This article uses tables in the `samples.tpch` schema, which contains data from a fictional business. The `customer` table contains information about customers, and `orders` contains information about orders placed by those customers.

  * Use `dbutils.fs.ls` to explore data in `/databricks-datasets`. Use Spark SQL or DataFrames to query data in this location using file paths. To learn more about Databricks-provided sample data, see [Sample datasets](../discover/databricks-datasets.html).

## Import data types

Many PySpark operations require that you use SQL functions or interact with
native Spark types. You can either directly import only those functions and
types that you need, or you can import the entire module.

    
    
    # import all
    from pyspark.sql.types import *
    from pyspark.sql.functions import *
    
    # import select functions and types
    from pyspark.sql.types import IntegerType, StringType
    from pyspark.sql.functions import floor, round
    

Because some imported functions might override Python built-in functions, some
users choose to import these modules using an alias. The following examples
show a common alias used in Apache Spark code examples:

    
    
    import pyspark.sql.types as T
    import pyspark.sql.functions as F
    

For a comprehensive list of data types, see [Spark Data Types](https://api-
docs.databricks.com/python/pyspark/latest/pyspark.sql/data_types.html).

For a comprehensive list of PySpark SQL functions, see [Spark
Functions](https://api-
docs.databricks.com/python/pyspark/latest/pyspark.sql/functions.html).

## Create a DataFrame

There are several ways to create a DataFrame. Usually you define a DataFrame
against a data source such as a table or collection of files. Then as
described in the Apache Spark [fundamental concepts section](index.html#spark-
concepts), use an action, such as `display`, to trigger the transformations to
execute. The `display` method outputs DataFrames.

### Create a DataFrame with specified values

To create a DataFrame with specified values, use the `createDataFrame` method,
where rows are expressed as a list of tuples:

    
    
    df_children = spark.createDataFrame(
      data = [("Mikhail", 15), ("Zaky", 13), ("Zoya", 8)],
      schema = ['name', 'age'])
    display(df_children)
    

Notice in the output that the data types of columns of `df_children` are
automatically inferred. You can alternatively specify the types by adding a
schema. Schemas are defined using the `StructType` which is made up of
`StructFields` that specify the name, data type and a boolean flag indicating
whether they contain a null value or not. You must import data types from
`pyspark.sql.types`.

    
    
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType
    
    df_children_with_schema = spark.createDataFrame(
      data = [("Mikhail", 15), ("Zaky", 13), ("Zoya", 8)],
      schema = StructType([
        StructField('name', StringType(), True),
        StructField('age', IntegerType(), True)
      ])
    )
    display(df_children_with_schema)
    

### Create a DataFrame from a table in Unity Catalog

To create a DataFrame from a table in Unity Catalog, use the `table` method
identifying the table using the format `<catalog-name>.<schema-name>.<table-
name>`. Click on **Catalog** on the left navigation bar to use **Catalog
Explorer** to navigate to your table. Click it, then select **Copy table
path** to insert the table path into the notebook.

The following example loads the table `samples.tpch.customer`, but you can
alternatively provide the path to your own table.

    
    
    df_customer = spark.table('samples.tpch.customer')
    display(df_customer)
    

### Create a DataFrame from an uploaded file

To create a DataFrame from a file you uploaded to Unity Catalog volumes, use
the `read` property. This method returns a `DataFrameReader`, which you can
then use to read the appropriate format. Click on the catalog option on the
small sidebar on the left and use the catalog browser to locate your file.
Select it, then click **Copy volume file path**.

The example below reads from a `*.csv` file, but `DataFrameReader` supports
uploading files in many other formats. See [DataFrameReader
methods](https://api-
docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.DataFrameReader.html).

    
    
    # Assign this variable your full volume file path
    volume_file_path = ""
    
    df_csv = (spark.read
      .format("csv")
      .option("header", True)
      .option("inferSchema", True)
      .load(volume_file_path)
    )
    display(df_csv)
    

For more information about Unity Catalog volumes, see [What are Unity Catalog
volumes?](../volumes/index.html).

### Create a DataFrame from a JSON response

To create a DataFrame from a JSON response payload returned by a REST API, use
the Python `requests` package to query and parse the response. You must import
the package to use it. This example uses data from the United States Food and
Drug Administrationâs drug application database.

    
    
    import requests
    
    # Download data from URL
    url = "https://api.fda.gov/drug/drugsfda.json?limit=100"
    response = requests.get(url)
    
    # Create the DataFrame
    df_drugs = spark.createDataFrame(response.json()["results"])
    display(df_drugs)
    

For information on working with JSON and other semi-structured data on
Databricks, see [Model semi-structured data](../semi-structured/index.html).

#### Select a JSON field or object

To select a specific field or object from the converted JSON, use the `[]`
notation. For example, to select the `products` field which itself is an array
of products:

    
    
    display(df_drugs.select(df_drugs["products"]))
    

You can also chain together method calls to traverse multiple fields. For
example, to output the brand name of the first product in a drug application:

    
    
    display(df_drugs.select(df_drugs["products"][0]["brand_name"]))
    

### Create a DataFrame from a file

To demonstrate creating a DataFrame from a file, this example loads CSV data
in the `/databricks-datasets` directory.

To navigate to the sample datasets, you can use the [Databricks
Utilties](../dev-tools/databricks-utils.html) file system commands. The
following example uses `dbutils` to list the datasets available in
`/databricks-datasets`:

    
    
    display(dbutils.fs.ls('/databricks-datasets'))
    

Alternatively, you can use `%fs` to access [Databricks CLI file system
commands](../dev-tools/cli/fs-commands.html), as shown in the following
example:

    
    
    %fs ls '/databricks-datasets'
    

To create a DataFrame from a file or directory of files, specify the path in
the `load` method:

    
    
    df_population = (spark.read
      .format("csv")
      .option("header", True)
      .option("inferSchema", True)
      .load("/databricks-datasets/samples/population-vs-price/data_geo.csv")
    )
    display(df_population)
    

## Transform data with DataFrames

DataFrames make it easy to transform data using built-in methods to sort,
filter and aggregate data. Many transformations are not specified as methods
on DataFrames, but instead are provided in the `spark.sql.functions` package.
See [Databricks Spark SQL Functions](https://api-
docs.databricks.com/python/pyspark/latest/pyspark.sql/functions.html).

  * Column operations

  * Row operations

  * Join DataFrames

  * Aggregate data

  * Chaining calls

### Column operations

Spark provides many basic column operations:

  * Select columns

  * Create columns

  * Rename columns

  * Cast column types

  * Remove columns

Tip

To output all of the columns in a DataFrame, use `columns`, for example
`df_customer.columns`.

#### Select columns

You can select specific columns using `select` and `col`. The `col` function
is in the `pyspark.sql.functions` submodule.

    
    
    from pyspark.sql.functions import col
    
    df_customer.select(
      col("c_custkey"),
      col("c_acctbal")
    )
    

You can also refer to a column using `expr` which takes an expression defined
as a string:

    
    
    from pyspark.sql.functions import expr
    
    df_customer.select(
      expr("c_custkey"),
      expr("c_acctbal")
    )
    

You can also use `selectExpr`, which accepts SQL expressions:

    
    
    df_customer.selectExpr(
      "c_custkey as key",
      "round(c_acctbal) as account_rounded"
    )
    

To select columns using a string literal, do the following:

    
    
    df_customer.select(
      "c_custkey",
      "c_acctbal"
    )
    

To explicitly select a column from a specific DataFrame, you can use the `[]`
operator or the `.` operator. (The `.` operator cannot be used to select
columns starting with an integer, or ones that contain a space or special
character.) This can be especially helpful when you are joining DataFrames
where some columns have the same name.

    
    
    df_customer.select(
      df_customer["c_custkey"],
      df_customer["c_acctbal"]
    )
    
    
    
    df_customer.select(
      df_customer.c_custkey,
      df_customer.c_acctbal
    )
    

#### Create columns

To create a new column, use the `withColumn` method. The following example
creates a new column that contains a boolean value based on whether the
customer account balance `c_acctbal` exceeds `1000`:

    
    
    df_customer_flag = df_customer.withColumn("balance_flag", col("c_acctbal") > 1000)
    

#### Rename columns

To rename a column, use the `withColumnRenamed` method, which accepts the
existing and new column names:

    
    
    df_customer_flag_renamed = df_customer_flag.withColumnRenamed("balance_flag", "balance_flag_renamed")
    

The `alias` method is especially helpful when you want to rename your columns
as part of aggregations:

    
    
    from pyspark.sql.functions import avg
    
    df_segment_balance = df_customer.groupBy("c_mktsegment").agg(
        avg(df_customer["c_acctbal"]).alias("avg_account_balance")
    )
    
    display(df_segment_balance)
    

#### Cast column types

In some cases you may want to change the data type for one or more of the
columns in your DataFrame. To do this, use the `cast` method to convert
between column data types. The following example shows how to convert a column
from an integer to string type, using the `col` method to reference a column:

    
    
    from pyspark.sql.functions import col
    
    df_casted = df_customer.withColumn("c_custkey", col("c_custkey").cast(StringType()))
    print(type(df_casted))
    

#### Remove columns

To remove columns, you can omit columns during a select or `select(*) except`
or you can use the `drop` method:

    
    
    df_customer_flag_renamed.drop("balance_flag_renamed")
    

You can also drop multiple columns at once:

    
    
    df_customer_flag_renamed.drop("c_phone", "balance_flag_renamed")
    

### Row operations

Spark provides many basic row operations:

  * Filter rows

  * Remove duplicate rows

  * Handle null values

  * Append rows

  * Sort rows

  * Filter rows

#### Filter rows

To filter rows, use the `filter` or `where` method on a DataFrame to return
only certain rows. To identify a column to filter on, use the `col` method or
an expression that evaluates to a column.

    
    
    from pyspark.sql.functions import col
    
    df_that_one_customer = df_customer.filter(col("c_custkey") == 412449)
    

To filter on multiple conditions, use logical operators. For example, `&` and
`|` enable you to `AND` and `OR` conditions, respectively. The following
example filters rows where the `c_nationkey` is equal to `20` and `c_acctbal`
is greater than `1000`.

    
    
    df_customer.filter((col("c_nationkey") == 20) & (col("c_acctbal") > 1000))
    
    
    
    df_filtered_customer = df_customer.filter((col("c_custkey") == 412446) | (col("c_custkey") == 412447))
    

#### Remove duplicate rows

To de-duplicate rows, use `distinct`, which returns only the unique rows.

    
    
    df_unique = df_customer.distinct()
    

#### Handle null values

To handle null values, drop rows that contain null values using the `na.drop`
method. This method lets you specify if you want to drop rows containing `any`
null values or `all` null values.

To drop any null values use either of the following examples.

    
    
    df_customer_no_nulls = df_customer.na.drop()
    df_customer_no_nulls = df_customer.na.drop("any")
    

If instead you want to only filter out rows that contain all null values use
the following:

    
    
    df_customer_no_nulls = df_customer.na.drop("all")
    

You can apply this for a subset of columns by specifying this, as shown below:

    
    
    df_customer_no_nulls = df_customer.na.drop("all", subset=["c_acctbal", "c_custkey"])
    

To fill in missing values, use the `fill` method. You can choose to apply this
to all columns or a subset of columns. In the example below account balances
that have a null value for their account balance `c_acctbal` are filled with
`0`.

    
    
    df_customer_filled = df_customer.na.fill("0", subset=["c_acctbal"])
    

To replace strings with other values, use the `replace` method. In the example
below, any empty address strings are replaced with the word `UNKNOWN`:

    
    
    df_customer_phone_filled = df_customer.na.replace([""], ["UNKNOWN"], subset=["c_phone"])
    

#### Append rows

To append rows you need to use the `union` method to create a new DataFrame.
In the following example, the DataFrame `df_that_one_customer` created
previously and `df_filtered_customer` are combined, which returns a DataFrame
with three customers:

    
    
    df_appended_rows = df_that_one_customer.union(df_filtered_customer)
    
    display(df_appended_rows)
    

Note

You can also combine DataFrames by writing them to a table and then appending
new rows. For production workloads, incremental processing of data sources to
a target table can drastically reduce latency and compute costs as data grows
in size. See [Ingest data into a Databricks
lakehouse](../ingestion/index.html).

#### Sort rows

Important

Sorting can be expensive at scale, and if you store sorted data and reload the
data with Spark, order is not guaranteed. Make sure you are intentional in
your use of sorting.

To sort rows by one or more columns, use the `sort` or `orderBy` method. By
default these methods sort in ascending order:

    
    
    df_customer.orderBy(col("c_acctbal"))
    

To filter in descending order, use `desc`:

    
    
    df_customer.sort(col("c_custkey").desc())
    

The following example shows how to sort on two columns:

    
    
    df_sorted = df_customer.orderBy(col("c_acctbal").desc(), col("c_custkey").asc())
    df_sorted = df_customer.sort(col("c_acctbal").desc(), col("c_custkey").asc())
    

To limit the number of rows to return once the DataFrame is sorted, use the
`limit` method. The following example displays only the top `10` results:

    
    
    display(df_sorted.limit(10))
    

### Join DataFrames

To join two or more DataFrames, use the `join` method. You can specify how you
would like the DataFrames to be joined in the `how` (the join type) and `on`
(on which columns to base the join) parameters. Common join types include:

  * `inner`: This is the join type default, which returns a DataFrame that keeps only the rows where there is a match for the `on` parameter across the DataFrames.

  * `left`: This keeps all rows of the first specified DataFrame and only rows from the second specified DataFrame that have a match with the first.

  * `outer`: An outer join keeps all rows from both DataFrames regardless of match.

For detailed information on joins, see [Work with joins on
Databricks](../transform/join.html). For a list of joins supported in PySpark,
see [DataFrame joins](https://api-
docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.DataFrame.join.html?highlight=joins).

The following example returns a single DataFrame where each row of the
`orders` DataFrame is joined with the corresponding row from the `customers`
DataFrame. An inner join is used, as the expectation is that every order
corresponds to exactly one customer.

    
    
    df_customer = spark.table('samples.tpch.customer')
    df_order = spark.table('samples.tpch.orders')
    
    df_joined = df_order.join(
      df_customer,
      on = df_order["o_custkey"] == df_customer["c_custkey"],
      how = "inner"
    )
    
    display(df_joined)
    

To join on multiple conditions, use boolean operators such as `&` and `|` to
specify `AND` and `OR`, respectively. The following example adds an additional
condition, filtering to just the rows that have `o_totalprice` greater than
`500,000`:

    
    
    df_customer = spark.table('samples.tpch.customer')
    df_order = spark.table('samples.tpch.orders')
    
    df_complex_joined = df_order.join(
      df_customer,
      on = ((df_order["o_custkey"] == df_customer["c_custkey"]) & (df_order["o_totalprice"] > 500000)),
      how = "inner"
    )
    
    display(df_complex_joined)
    

### Aggregate data

To aggregate data in a DataFrame, similar to a `GROUP BY` in SQL, use the
`groupBy` method to specify columns to group by and the `agg` method to
specify aggregations. Import common aggregations including `avg`, `sum`,
`max`, and `min` from `pyspark.sql.functions`. The following example shows the
average customer balance by market segment:

    
    
    from pyspark.sql.functions import avg
    
    # group by one column
    df_segment_balance = df_customer.groupBy("c_mktsegment").agg(
        avg(df_customer["c_acctbal"])
    )
    
    display(df_segment_balance)
    
    
    
    from pyspark.sql.functions import avg
    
    # group by two columns
    df_segment_nation_balance = df_customer.groupBy("c_mktsegment", "c_nationkey").agg(
        avg(df_customer["c_acctbal"])
    )
    
    display(df_segment_nation_balance)
    

Some aggregations are actions, which means that they trigger computations. In
this case you do not need to use other actions to output results.

To count rows in a DataFrame, use the `count` method:

    
    
    df_customer.count()
    

### Chaining calls

Methods that transform DataFrames return DataFrames, and Spark does not act on
transformations until actions are called. This [lazy
evaluation](index.html#data-processing) means you can chain multiple methods
for convenience and readability. The following example shows how to chain
filtering, aggregation and ordering:

    
    
    from pyspark.sql.functions import count
    
    df_chained = (
        df_order.filter(col("o_orderstatus") == "F")
        .groupBy(col("o_orderpriority"))
        .agg(count(col("o_orderkey")).alias("n_orders"))
        .sort(col("n_orders").desc())
    )
    
    display(df_chained)
    

## Visualize your DataFrame

To visualize a DataFrame in a notebook, click the **+** sign next to table on
the top left of the DataFrame, then select **Visualization** to add one or
more charts based on your DataFrame. For details on visualizations, see
[Visualizations in Databricks notebooks](../visualizations/index.html).

    
    
    display(df_order)
    

To perform additional visualizations, Databricks recommends using pandas API
for Spark. The `.pandas_api()` allows you to cast to the corresponding pandas
API for a Spark DataFrame. For more information, see [Pandas API on
Spark](../pandas/pandas-on-spark.html).

## Save your data

Once you have transformed your data, you can save it using the
`DataFrameWriter` methods. A complete list of these methods can be found in
[DataFrameWriter](https://api-
docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.DataFrameWriter.html).
The following sections show how to save your DataFrame as a table and as a
collection of data files.

### Save your DataFrame as a table

To save your DataFrame as a table in Unity Catalog, use the
`write.saveAsTable` method and specify the path in the format `<catalog-
name>.<schema-name>.<table-name>`.

    
    
    df_joined.write.saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")
    

### Write your DataFrame as CSV

To write your DataFrame to `*.csv` format, use the `write.csv` method,
specifying the format and options. By default if data exists at the specified
path the write operation fails. You can specify one of the following modes to
take a different action:

  * `overwrite` overwrites all existing data in the target path with the DataFrame contents.

  * `append` appends contents of the DataFrame to data in the target path.

  * `ignore` silently fails the write if data exists in the target path.

The following example demonstrates overwriting data with DataFrame contents as
CSV files:

    
    
    # Assign this variable your file path
    file_path = ""
    
    (df_joined.write
      .format("csv")
      .mode("overwrite")
      .write(file_path)
    )
    

## Next steps

To leverage more Spark capabilities on Databricks, see:

  * [Visualizations](../languages/python.html#visualizations)

  * [Automation with jobs](../languages/python.html#jobs)

  * [IDEs and SDKs](../languages/python.html#ides-tools-sdks)

  * [UDFs (User Defined Functions)](../udf/index.html)

  * [Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#overview)

  * [Machine Learning Library (MLlib)](https://spark.apache.org/docs/latest/ml-guide.html#overview)

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

