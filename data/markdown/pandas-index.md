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

  * [English](../../en/pandas/index.html)
  * [æ¥æ¬èª](../../ja/pandas/index.html)
  * [PortuguÃªs](../../pt/pandas/index.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/pandas/)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/pandas/index.html)

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
        * pandas on Databricks
        * [Convert between PySpark and pandas DataFrames](pyspark-pandas-conversion.html)
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
  * Can you use pandas on Databricks?
  * 

# Can you use pandas on Databricks?

Databricks Runtime includes pandas as one of the standard Python packages,
allowing you to create and leverage pandas DataFrames in Databricks notebooks
and jobs.

In Databricks Runtime 10.4 LTS and above, [Pandas API on Spark](pandas-on-
spark.html) provides familiar pandas commands on top of PySpark DataFrames.
You can also [convert DataFrames between pandas and PySpark](pyspark-pandas-
conversion.html).

Apache Spark includes Arrow-optimized execution of Python logic in the form of
[pandas function APIs](pandas-function-apis.html), which allow users to apply
pandas transformations directly to PySpark DataFrames. Apache Spark also
supports [pandas UDFs](../udf/pandas.html), which use similar Arrow-
optimizations for arbitrary user functions defined in Python.

## Where does pandas store data on Databricks?

You can use pandas to store data in many different locations on Databricks.
Your ability to store and load data from some locations depends on
configurations set by workspace administrators.

Note

Databricks recommends storing production data on cloud object storage. See
[Connect to Amazon S3](../connect/storage/amazon-s3.html).

If youâre in a Unity Catalog-enabled workspace, you can access cloud storage
with external locations. See [Create an external location to connect cloud
storage to Databricks](../connect/unity-catalog/external-locations.html).

For quick exploration and data without sensitive information, you can safely
save data using either relative paths or the [DBFS](../dbfs/index.html), as in
the following examples:

    
    
    import pandas as pd
    
    df = pd.DataFrame([["a", 1], ["b", 2], ["c", 3]])
    
    df.to_csv("./relative_path_test.csv")
    df.to_csv("/dbfs/dbfs_test.csv")
    

You can explore files written to the DBFS with the `%fs` magic command, as in
the following example. Note that the `/dbfs` directory is the root path for
these commands.

    
    
    %fs ls
    

When you save to a relative path, the location of your file depends on where
you execute your code. If youâre using a Databricks notebook, your data file
saves to the volume storage attached to the driver of your cluster. Data
stored in this location is permanently deleted when the cluster terminates. If
youâre using [Databricks Git folders](../repos/index.html) with arbitrary
file support enabled, your data saves to the root of your current project. In
either case, you can explore the files written using the `%sh` magic command,
which allows simple bash operations relative to your current root directory,
as in the following example:

    
    
    %sh ls
    

For more information on how Databricks stores various files, see [Work with
files on Databricks](../files/index.html).

## How do you load data with pandas on Databricks?

Databricks provides a number of options to facilitate uploading data to the
workspace for exploration. The preferred method to load data with pandas
varies depending on how you load your data to the workspace.

If you have small data files stored alongside notebooks on your local machine,
you can upload your data and code together with [Git
folders](../files/workspace.html). You can then use relative paths to load
data files.

Databricks provides extensive [UI-based options for data
loading](../ingestion/file-upload/index.html). Most of these options store
your data as Delta tables. You can [read a Delta
table](../delta/tutorial.html#read) to a Spark DataFrame, and then [convert
that to a pandas DataFrame](pyspark-pandas-conversion.html).

If you have saved data files using DBFS or relative paths, you can use DBFS or
relative paths to reload those data files. The following code provides an
example:

    
    
    import pandas as pd
    
    df = pd.read_csv("./relative_path_test.csv")
    df = pd.read_csv("/dbfs/dbfs_test.csv")
    

You can load data directly from [S3](../connect/storage/amazon-s3.html) using
pandas and a fully qualified URL. You need to provide cloud credentials to
access cloud data.

    
    
    df = pd.read_csv(
      f"s3://{bucket_name}/{file_path}",
      storage_options={
        "key": aws_access_key_id,
        "secret": aws_secret_access_key,
        "token": aws_session_token
      }
    )
    

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

