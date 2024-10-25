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

  * [English](../../en/pyspark/index.html)
  * [æ¥æ¬èª](../../ja/pyspark/index.html)
  * [PortuguÃªs](../../pt/pyspark/index.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/pyspark/)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/pyspark/index.html)

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
      * PySpark
        * [PySpark basics](basics.html)
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

Updated Oct 24, 2024

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation
Feedback)

  * [Documentation](../index.html)
  * [Develop on Databricks](../languages/index.html)
  * [Databricks for Python developers](../languages/python.html)
  * PySpark on Databricks
  * 

# PySpark on Databricks

Databricks is built on top of [Apache Spark](../spark/index.html), a unified
analytics engine for big data and machine learning. PySpark helps you
interface with Apache Spark using the Python programming language, which is a
flexible language that is easy to learn, implement, and maintain. It also
provides many options for [data visualization](../visualizations/index.html)
in Databricks. PySpark combines the power of Python and Apache Spark.

This article provides an overview of the fundamentals of PySpark on
Databricks.

## Introduction to Spark concepts

It is important to understand key Apache Spark concepts before diving into
using PySpark.

### DataFrames

DataFrames are the primary objects in Apache Spark. A DataFrame is a dataset
organized into named columns. You can think of a DataFrame like a spreadsheet
or a SQL table, a two-dimensional labeled data structure of a series of
records (similar to rows in a table) and columns of different types.
DataFrames provide a rich set of functions (for example, select columns,
filter, join, and aggregate) that allow you to perform common data
manipulation and analysis tasks efficiently.

Some important DataFrame elements include:

  * **Schema** : A schema defines the column names and types of a DataFrame. Data formats have different semantics for schema definition and enforcement. Some data sources provide schema information, while others either rely on manual schema definition or allow schema inference. Users can define schemas manually or schemas can be read from a data source.

  * **Rows** : Spark represents records in a DataFrame as `Row` objects. While underlying data formats such as Delta Lake use columns to store data, for optimization Spark caches and shuffles data using rows.

  * **Columns** : Columns in Spark are similar to columns in a spreadsheet and can represent a simple type such as a string or integer, but also complex types like array, map, or null. You can write queries that select, manipulate, or remove columns from a data source. Possible data sources include tables, views, files, or other DataFrames. Columns are never removed from a dataset or a DataFrame, they are just omitted from results through `.drop` transformations or omission in `select` statements.

### Data processing

Apache Spark uses lazy evaluation to process transformations and actions
defined with DataFrames. These concepts are fundamental to understanding data
processing with Spark.

**Transformations** : In Spark you express processing logic as
transformations, which are instructions for loading and manipulating data
using DataFrames. Common transformations include reading data, joins,
aggregations, and type casting. For information about transformations in
Databricks, see [Transform data](../transform/index.html).

**Lazy Evaluation** : Spark optimizes data processing by identifying the most
efficient physical plan to evaluate logic specified by transformations.
However, Spark does not act on transformations until actions are called.
Rather than evaluating each transformation in the exact order specified, Spark
waits until an action triggers computation on all transformations. This is
known as lazy evaluation, or lazy loading, which allows you to _chain_
multiple operations because Spark handles their execution in a deferred
manner, rather than immediately executing them when they are defined.

Note

Lazy evaluation means that DataFrames store logical queries as a set of
instructions against a data source rather than an in-memory result. This
varies drastically from eager execution, which is the model used by [pandas
DataFrames](../pandas/index.html).

**Actions** : Actions instruct Spark to compute a result from a series of
transformations on one or more DataFrames. Action operations return a value,
and can be any of the following:

  * Actions to output data in the console or your editor, such as `display` or `show`

  * Actions to collect data (returns `Row` objects), such as `take(n)`, and `first` or `head`

  * Actions to write to data sources, such as `saveAsTable`

  * Aggregations that trigger a computation, such as `count`

Important

In production data pipelines, writing data is typically the only action that
should be present. All other actions interrupt query optimization and can lead
to bottlenecks.

#### What does it mean that DataFrames are immutable?

DataFrames are a collection of transformations and actions that are defined
against one or more data sources, but ultimately Apache Spark resolves queries
back to the original data sources, so the data itself is not changed, and no
DataFrames are changed. In other words, DataFrames are **_immutable_**.
Because of this, after performing transformations, a new DataFrame is returned
that has to be saved to a variable in order to access it in subsequent
operations. If you want to evaluate an intermediate step of your
transformation, call an action.

## APIs and libraries

As with all APIs for Spark, PySpark comes equipped with many APIs and
libraries that enable and support powerful functionality, including:

  * Processing of structured data with relational queries with **Spark SQL and DataFrames**. Spark SQL allows you to mix SQL queries with Spark programs. With Spark DataFrames, you can efficiently read, write, transform, and analyze data using Python and SQL, which means you are always leveraging the full power of Spark. See [PySpark Getting Started](https://spark.apache.org/docs/latest/api/python/getting_started/index.html).

  * Scalable processing of streams with **Structured Streaming**. You can express your streaming computation the same way you would express a batch computation on static data and the Spark SQL engine runs it incrementally and continuously as streaming data continues to arrive. See [Structured Streaming Overview](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#overview).

  * Pandas data structures and data analysis tools that work on Apache Spark with **Pandas API on Spark**. Pandas API on Spark allows you to scale your pandas workload to any size by running it distributed across multiple nodes, with a single codebase that works with pandas (tests, smaller datasets) and with Spark (production, distributed datasets). See [Pandas API on Spark Overview](https://spark.apache.org/pandas-on-spark/).

  * Machine learning algorithms with **Machine Learning (MLLib)**. MLlib is a scalable machine learning library built on Spark that provides a uniform set of APIs that help users create and tune practical machine learning pipelines. See [Machine Learning Library Overview](https://spark.apache.org/docs/latest/ml-guide.html#overview).

  * Graphs and graph-parallel computation with **GraphX**. GraphX introduces a new directed multigraph with properties attached to each vertex and edge, and exposes graph computation operators, algorithms, and builders to simplify graph analytics tasks. See [GraphX Overview](https://spark.apache.org/docs/latest/graphx-programming-guide.html#overview).

## Spark tutorials

For PySpark on Databricks usage examples, see the following articles:

  * [DataFrames tutorial](../getting-started/dataframes.html)

  * [PySpark basics](basics.html)

The [Apache Spark documentation](https://spark.apache.org/docs/latest) also
has quickstarts and guides for learning Spark, including the following:

  * [PySpark DataFrames QuickStart](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_df.html)

  * [Spark SQL Getting Started](https://spark.apache.org/docs/latest/sql-getting-started.html)

  * [Structured Streaming Programming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)

  * [Pandas API on Spark QuickStart](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_ps.html)

  * [Machine Learning Library Programming Guide](https://spark.apache.org/docs/latest/ml-guide.html)

## PySpark reference

Databricks maintains its own version of the PySpark APIs and corresponding
reference, which can be found in these sections:

  * [Spark SQL Reference](https://api-docs.databricks.com/python/pyspark/latest/pyspark.sql/index.html)

  * [Pandas API on Spark Reference](https://api-docs.databricks.com/python/pyspark/latest/pyspark.pandas/index.html)

  * [Structured Streaming API Reference](https://api-docs.databricks.com/python/pyspark/latest/pyspark.ss/index.html)

  * [MLlib (DataFrame-based) API Reference](https://api-docs.databricks.com/python/pyspark/latest/pyspark.ml.html)

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

