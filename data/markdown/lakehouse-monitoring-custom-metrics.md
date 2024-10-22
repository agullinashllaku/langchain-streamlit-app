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

  * [English](../../en/lakehouse-monitoring/custom-metrics.html)
  * [æ¥æ¬èª](../../ja/lakehouse-monitoring/custom-metrics.html)
  * [PortuguÃªs](../../pt/lakehouse-monitoring/custom-metrics.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/custom-metrics)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/lakehouse-monitoring/custom-metrics.html)

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
  * [Monitor data and AI assets](index.html)
    * [Create a monitor using the Databricks UI](create-monitor-ui.html)
    * [Create a monitor using the API](create-monitor-api.html)
    * [Monitor metric tables](monitor-output.html)
    * [Use the generated SQL dashboard](monitor-dashboard.html)
    * [Monitor alerts](monitor-alerts.html)
    * Use custom metrics with Databricks Lakehouse Monitoring
    * [Monitor fairness and bias for classification models](fairness-bias.html)
    * [View Lakehouse Monitoring expenses](expense.html)
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
  * [Introduction to Databricks Lakehouse Monitoring](index.html)
  * Use custom metrics with Databricks Lakehouse Monitoring
  * 

# Use custom metrics with Databricks Lakehouse Monitoring

This page describes how to create a custom metric in Databricks Lakehouse
Monitoring. In addition to the analysis and drift statistics that are
automatically calculated, you can create custom metrics. For example, you
might want to track a weighted mean that captures some aspect of business
logic or use a custom model quality score. You can also create custom drift
metrics that track changes to the values in the primary table (compared to the
baseline or the previous time window).

For more details on how to use the `MonitorMetric` API, see the [API
reference](https://databricks-sdk-
py.readthedocs.io/en/latest/dbdataclasses/catalog.html#databricks.sdk.service.catalog.MonitorMetric).

## Types of custom metrics

Databricks Lakehouse Monitoring includes the following types of custom
metrics:

  * Aggregate metrics, which are calculated based on columns in the primary table. Aggregate metrics are stored in the profile metrics table.

  * Derived metrics, which are calculated based on previously computed aggregate metrics and do not directly use data from the primary table. Derived metrics are stored in the profile metrics table.

  * Drift metrics, which compare previously computed aggregate or derived metrics from two different time windows, or between the primary table and the baseline table. Drift metrics are stored in the drift metrics table.

Using derived and drift metrics where possible minimizes recomputation over
the full primary table. Only aggregate metrics access data from the primary
table. Derived and drift metrics can then be computed directly from the
aggregate metric values.

## Custom metrics parameters

To define a custom metric, you create a [Jinja
template](https://jinja.palletsprojects.com/en/3.0.x/templates/#variables) for
a SQL column expression. The tables in this section describe the parameters
that define the metric, and the parameters that are used in the Jinja
template.

Parameter | Description  
---|---  
`type` |  One of `MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE`, `MonitorMetricType.CUSTOM_METRIC_TYPE_DERIVED`, or `MonitorMetricType.CUSTOM_METRIC_TYPE_DRIFT`.  
`name` |  Column name for the custom metric in metric tables.  
`input_columns` |  List of column names in the input table the metric should be computed for. To indicate that more than one column is used in the calculation, use `:table`. See the examples in this article.  
`definition` |  Jinja template for a SQL expression that specifies how to compute the metric. See Create definition.  
`output_data_type` |  Spark datatype of the metric output in a JSON string format.  
  
### Create `definition`

The `definition` parameter must be a single string expression in the form of a
Jinja template. It cannot contain joins or subqueries.

The following table lists the parameters you can use to create a SQL Jinja
Template to specify how to calculate the metric.

Parameter | Description  
---|---  
`{{input_column}}` |  Column used to compute the custom metric.  
`{{prediction_col}}` |  Column holding ML model predictions. Used with `InferenceLog` analysis.  
`{{label_col}}` |  Column holding ML model ground truth labels. Used with `InferenceLog` analysis.  
`{{current_df}}` |  For drift compared to the previous time window. Data from the previous time window.  
`{{base_df}}` |  For drift compared to the baseline table. Baseline data.  
  
## Aggregate metric example

The following example computes the average of the square of the values in a
column, and is applied to columns `f1` and `f2`. The output is saved as a new
column in the profile metrics table and is shown in the analysis rows
corresponding to the columns `f1` and `f2`. The applicable column names are
substituted for the Jinja parameter `{{input_column}}`.

    
    
    from databricks.sdk.service.catalog import MonitorMetric, MonitorMetricType
    from pyspark.sql import types as T
    
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="squared_avg",
        input_columns=["f1", "f2"],
        definition="avg(`{{input_column}}`*`{{input_column}}`)",
        output_data_type=T.StructField("output", T.DoubleType()).json(),
    )
    

The following code defines a custom metric that computes the average of the
difference between columns `f1` and `f2`. This example shows the use of
`[":table"]` in the `input_columns` parameter to indicate that more than one
column from the table is used in the calculation.

    
    
    from databricks.sdk.service.catalog import MonitorMetric, MonitorMetricType
    from pyspark.sql import types as T
    
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="avg_diff_f1_f2",
        input_columns=[":table"],
        definition="avg(f1 - f2)",
        output_data_type=T.StructField("output", T.DoubleType()).json(),
    )
    

This example computes a weighted model quality score. For observations where
the `critical` column is `True`, a heavier penalty is assigned when the
predicted value for that row does not match the ground truth. Because itâs
defined on the raw columns (`prediction` and `label`), itâs defined as an
aggregate metric. The `:table` column indicates that this metric is calculated
from multiple columns. The Jinja parameters `{{prediction_col}}` and
`{{label_col}}` are replaced with the name of the prediction and ground truth
label columns for the monitor.

    
    
    from databricks.sdk.service.catalog import MonitorMetric, MonitorMetricType
    from pyspark.sql import types as T
    
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="weighted_error",
        input_columns=[":table"],
        definition="""avg(CASE
          WHEN {{prediction_col}} = {{label_col}} THEN 0
          WHEN {{prediction_col}} != {{label_col}} AND critical=TRUE THEN 2
          ELSE 1 END)""",
        output_data_type=T.StructField("output", T.DoubleType()).json(),
    )
    

## Derived metric example

The following code defines a custom metric that computes the square root of
the `squared_avg` metric defined earlier in this section. Because this is a
derived metric, it does not reference the primary table data and instead is
defined in terms of the `squared_avg` aggregate metric. The output is saved as
a new column in the profile metrics table.

    
    
    from databricks.sdk.service.catalog import MonitorMetric, MonitorMetricType
    from pyspark.sql import types as T
    
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_DERIVED,
        name="root_mean_square",
        input_columns=["f1", "f2"],
        definition="sqrt(squared_avg)",
        output_data_type=T.StructField("output", T.DoubleType()).json(),
    )
    

## Drift metrics example

The following code defines a drift metric that tracks the change in the
`weighted_error` metric defined earlier in this section. The `{{current_df}}`
and `{{base_df}}` parameters allow the metric to reference the
`weighted_error` values from the current window and the comparison window. The
comparison window can be either the baseline data or the data from the
previous time window. Drift metrics are saved in the drift metrics table.

    
    
    from databricks.sdk.service.catalog import MonitorMetric, MonitorMetricType
    from pyspark.sql import types as T
    
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_DRIFT,
        name="error_rate_delta",
        input_columns=[":table"],
        definition="{{current_df}}.weighted_error - {{base_df}}.weighted_error",
        output_data_type=T.StructField("output", T.DoubleType()).json(),
    )
    

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

