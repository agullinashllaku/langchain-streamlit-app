[ ![logo](../../_static/spark-logo-reverse.png) ](../../index.html)

  * [ Overview ](../../index.html)
  * [ Getting Started ](../../getting_started/index.html)
  * [ User Guides ](../index.html)
  * [ API Reference ](../../reference/index.html)
  * [ Development ](../../development/index.html)
  * [ Migration Guides ](../../migration_guide/index.html)

3.5.3

__

  * [ Python Package Management ](../python_packaging.html)
  * [ Spark SQL ](../sql/index.html) __
    * [ Apache Arrow in PySpark ](../sql/arrow_pandas.html)
    * [ Python User-defined Table Functions (UDTFs) ](../sql/python_udtf.html)
  * Pandas API on Spark  __
    * [ Options and settings ](options.html)
    * [ From/to pandas and PySpark DataFrames ](pandas_pyspark.html)
    * [ Transform and apply a function ](transform_apply.html)
    * [ Type Support in Pandas API on Spark ](types.html)
    * [ Type Hints in Pandas API on Spark ](typehints.html)
    * [ From/to other DBMSes ](from_to_dbms.html)
    * [ Best Practices ](best_practices.html)
    * [ Supported pandas API ](supported_pandas_api.html)
    * [ FAQ ](faq.html)

# Pandas API on SparkÂ¶

  * [Options and settings](options.html)
    * [Getting and setting options](options.html#getting-and-setting-options)
    * [Operations on different DataFrames](options.html#operations-on-different-dataframes)
    * [Default Index type](options.html#default-index-type)
    * [Available options](options.html#available-options)
  * [From/to pandas and PySpark DataFrames](pandas_pyspark.html)
    * [pandas](pandas_pyspark.html#pandas)
    * [PySpark](pandas_pyspark.html#pyspark)
  * [Transform and apply a function](transform_apply.html)
    * [`transform` and `apply`](transform_apply.html#transform-and-apply)
    * [`pandas_on_spark.transform_batch` and `pandas_on_spark.apply_batch`](transform_apply.html#pandas-on-spark-transform-batch-and-pandas-on-spark-apply-batch)
  * [Type Support in Pandas API on Spark](types.html)
    * [Type casting between PySpark and pandas API on Spark](types.html#type-casting-between-pyspark-and-pandas-api-on-spark)
    * [Type casting between pandas and pandas API on Spark](types.html#type-casting-between-pandas-and-pandas-api-on-spark)
    * [Internal type mapping](types.html#internal-type-mapping)
  * [Type Hints in Pandas API on Spark](typehints.html)
    * [pandas-on-Spark DataFrame and Pandas DataFrame](typehints.html#pandas-on-spark-dataframe-and-pandas-dataframe)
    * [Type Hinting with Names](typehints.html#type-hinting-with-names)
    * [Type Hinting with Index](typehints.html#type-hinting-with-index)
  * [From/to other DBMSes](from_to_dbms.html)
    * [Reading and writing DataFrames](from_to_dbms.html#reading-and-writing-dataframes)
  * [Best Practices](best_practices.html)
    * [Leverage PySpark APIs](best_practices.html#leverage-pyspark-apis)
    * [Check execution plans](best_practices.html#check-execution-plans)
    * [Use checkpoint](best_practices.html#use-checkpoint)
    * [Avoid shuffling](best_practices.html#avoid-shuffling)
    * [Avoid computation on single partition](best_practices.html#avoid-computation-on-single-partition)
    * [Avoid reserved column names](best_practices.html#avoid-reserved-column-names)
    * [Do not use duplicated column names](best_practices.html#do-not-use-duplicated-column-names)
    * [Specify the index column in conversion from Spark DataFrame to pandas-on-Spark DataFrame](best_practices.html#specify-the-index-column-in-conversion-from-spark-dataframe-to-pandas-on-spark-dataframe)
    * [Use `distributed` or `distributed-sequence` default index](best_practices.html#use-distributed-or-distributed-sequence-default-index)
    * [Reduce the operations on different DataFrame/Series](best_practices.html#reduce-the-operations-on-different-dataframe-series)
    * [Use pandas API on Spark directly whenever possible](best_practices.html#use-pandas-api-on-spark-directly-whenever-possible)
  * [Supported pandas API](supported_pandas_api.html)
    * [CategoricalIndex API](supported_pandas_api.html#categoricalindex-api)
    * [DataFrame API](supported_pandas_api.html#dataframe-api)
    * [DatetimeIndex API](supported_pandas_api.html#datetimeindex-api)
    * [Index API](supported_pandas_api.html#index-api)
    * [MultiIndex API](supported_pandas_api.html#multiindex-api)
    * [Series API](supported_pandas_api.html#series-api)
    * [TimedeltaIndex API](supported_pandas_api.html#timedeltaindex-api)
    * [General Function API](supported_pandas_api.html#general-function-api)
    * [Expanding API](supported_pandas_api.html#expanding-api)
    * [ExpandingGroupby API](supported_pandas_api.html#expandinggroupby-api)
    * [Rolling API](supported_pandas_api.html#rolling-api)
    * [RollingGroupby API](supported_pandas_api.html#rollinggroupby-api)
    * [Window API](supported_pandas_api.html#window-api)
    * [DataFrameGroupBy API](supported_pandas_api.html#dataframegroupby-api)
    * [GroupBy API](supported_pandas_api.html#groupby-api)
    * [SeriesGroupBy API](supported_pandas_api.html#seriesgroupby-api)
  * [FAQ](faq.html)
    * [Should I use PySparkâs DataFrame API or pandas API on Spark?](faq.html#should-i-use-pyspark-s-dataframe-api-or-pandas-api-on-spark)
    * [Does pandas API on Spark support Structured Streaming?](faq.html#does-pandas-api-on-spark-support-structured-streaming)
    * [How is pandas API on Spark different from Dask?](faq.html#how-is-pandas-api-on-spark-different-from-dask)

[ __ previous Python User-defined Table Functions (UDTFs)
](../sql/python_udtf.html "previous page") [ next Options and settings
__](options.html "next page")

(C) Copyright .  

Created using [Sphinx](http://sphinx-doc.org/) 3.0.4.  

