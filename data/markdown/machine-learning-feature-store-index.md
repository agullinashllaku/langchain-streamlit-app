  * __[![Databricks](../../_static/small-scale-lockup-full-color-rgb.svg)](https://www.databricks.com/)

  * __[![Databricks](../../_static/small-scale-lockup-full-color-rgb.svg)](https://www.databricks.com/)
  * [Help Center](https://help.databricks.com/s/)
  * [Documentation](https://docs.databricks.com/en/index.html)
  * [Knowledge Base](https://kb.databricks.com/)

  * [Community](https://community.databricks.com)
  * [Support](https://help.databricks.com)
  * [Feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback)
  * [Try Databricks](https://databricks.com/try-databricks)

[![](../../_static/icons/globe.png)English](javascript:void\(0\))

  * [English](../../../en/machine-learning/feature-store/index.html)
  * [æ¥æ¬èª](../../../ja/machine-learning/feature-store/index.html)
  * [PortuguÃªs](../../../pt/machine-learning/feature-store/index.html)

[![](../../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/)
  * [![](../../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/machine-learning/feature-store/index.html)

[Databricks on AWS](../../index.html)

Get started

  * [Get started](../../getting-started/index.html)
  * [What is Databricks?](../../introduction/index.html)
  * [DatabricksIQ](../../databricksiq/index.html)
  * [Release notes](../../release-notes/index.html)

Load & manage data

  * [Work with database objects](../../database-objects/index.html)
  * [Connect to data sources](../../connect/index.html)
  * [Connect to compute](../../compute/index.html)
  * [Discover data](../../discover/index.html)
  * [Query data](../../query/index.html)
  * [Ingest data](../../ingestion/index.html)
  * [Work with files](../../files/index.html)
  * [Transform data](../../transform/index.html)
  * [Schedule and orchestrate workflows](../../jobs/index.html)
  * [Monitor data and AI assets](../../lakehouse-monitoring/index.html)
  * [Share data securely](../../data-sharing/index.html)

Work with data

  * [Data engineering](../../workspace-index.html)
  * [AI and machine learning](../index.html)
    * [Tutorials](../ml-tutorials.html)
    * [AI playground](../../large-language-models/ai-playground.html)
    * [AI functions in SQL](../../large-language-models/ai-functions.html)
    * [Serve models](../serve-models.html)
    * [Train models](../train-model/index.html)
    * [Serve data for AI](../serve-data-ai.html)
      * Feature management
        * [Concepts](concepts.html)
        * [Feature Engineering and Workspace Feature Store Python API](python-api.html)
        * [Work with feature tables](uc/feature-tables-uc.html)
        * [Point-in-time applications](time-series.html)
        * [Feature governance and lineage](lineage.html)
        * [Use features to train models](train-models-with-feature-store.html)
        * [Use features in online workflows](online-workflows.html)
        * [Use features with structured RAG applications](rag.html)
        * [Workspace feature store (Legacy)](workspace-feature-store/index.html)
        * [Troubleshooting and limitations](troubleshooting-and-limitations.html)
      * [Vector Search](../../generative-ai/vector-search.html)
    * [Evaluate AI](../../generative-ai/agent-evaluation/index.html)
    * [Build gen AI apps](../../generative-ai/build-genai-apps.html)
    * [MLOps and MLflow](../../mlflow/index.html)
    * [Integrations](../integrations.html)
    * [Graph and network analysis](../graph-analysis.html)
    * [Reference solutions](../reference-solutions/index.html)
  * [Generative AI tutorial](../../generative-ai/tutorials/ai-cookbook/index.html)
  * [Business intelligence](../../ai-bi/index.html)
  * [Data warehousing](../../sql/index.html)
  * [Notebooks](../../notebooks/index.html)
  * [Delta Lake](../../delta/index.html)
  * [Developers](../../languages/index.html)
  * [Technology partners](../../integrations/index.html)

Administration

  * [Account and workspace administration](../../admin/index.html)
  * [Security and compliance](../../security/index.html)
  * [Data governance (Unity Catalog)](../../data-governance/index.html)
  * [Lakehouse architecture](../../lakehouse-architecture/index.html)

Reference & resources

  * [Reference](../../reference/api.html)
  * [Resources](../../resources/index.html)
  * [Whatâs coming?](../../whats-coming.html)
  * [Documentation archive](../../archive/index.html)

Updated Oct 24, 2024

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation
Feedback)

  * [Documentation](../../index.html)
  * [AI and machine learning on Databricks](../index.html)
  * [Serve data for ML and AI](../serve-data-ai.html)
  * Feature engineering and serving
  * 

# Feature engineering and serving

This page covers feature engineering and serving capabilities for workspaces
that are enabled for Unity Catalog. If your workspace is not enabled for Unity
Catalog, see [Workspace feature store (Legacy)](workspace-feature-
store/index.html).

## Why use Databricks as your feature store?

With the Databricks Data Intelligence Platform, the entire model training
workflow takes place on a single platform:

  * Data pipelines that ingest raw data, create feature tables, train models, and perform batch inference. When you train and log a model using feature engineering in Unity Catalog, the model is packaged with feature metadata. When you use the model for batch scoring or online inference, it automatically retrieves feature values. The caller does not need to know about them or include logic to look up or join features to score new data.

  * Model and feature serving endpoints that are available with a single click and that provide milliseconds of latency.

  * Data and model monitoring.

In addition, the platform provides the following:

  * Feature discovery. You can browse and search for features in the Databricks UI.

  * Governance. Feature tables, functions, and models are all governed by Unity Catalog. When you train a model, it inherits permissions from the data it was trained on.

  * Lineage. When you create a feature table in Databricks, the data sources used to create the feature table are saved and accessible. For each feature in a feature table, you can also access the models, notebooks, jobs, and endpoints that use the feature.

  * Cross-workspace access. Feature tables, functions, and models are automatically available in any workspace that has access to the catalog.

## Requirements

  * Your workspace must be enabled for Unity Catalog.

  * Feature engineering in Unity Catalog requires Databricks Runtime 13.3 LTS or above.

If your workspace does not meet these requirements, see [Workspace feature
store (Legacy)](workspace-feature-store/index.html) for how to use the
workspace feature store.

## How does feature engineering on Databricks work?

The typical machine learning workflow using feature engineering on Databricks
follows this path:

  1. Write code to convert raw data into features and create a Spark DataFrame containing the desired features.

  2. [Create a Delta table in Unity Catalog](uc/feature-tables-uc.html#create-feature-table). Any Delta table with a primary key is automatically a feature table.

  3. Train and log a model using the feature table. When you do this, the model stores the specifications of features used for training. When the model is used for inference, it automatically joins features from the appropriate feature tables.

  4. Register model in [Model Registry](../manage-model-lifecycle/index.html).

You can now use the model to make predictions on new data. For batch use
cases, the model automatically retrieves the features it needs from Feature
Store.

![Feature Store workflow for batch machine learning use
cases.](../../_images/feature-store-flow-gcp.png)

For real-time serving use cases, publish the features to an [online
table](online-tables.html). Third-party online stores are also supported. See
[Third-party online stores](online-feature-stores.html).

At inference time, the model reads pre-computed features from the online store
and joins them with the data provided in the client request to the model
serving endpoint.

![Feature Store flow for machine learning models that are
served.](../../_images/feature-store-flow-with-online-store.png)

## Start using feature engineering â example notebooks

To get started, try these example notebooks. The basic notebook steps you
through how to create a feature table, use it to train a model, and then
perform batch scoring using automatic feature lookup. It also introduces you
to the Feature Engineering UI and shows how you can use it to search for
features and understand how features are created and used.

### Basic Feature Engineering in Unity Catalog example notebook

[Open notebook in new tab](/_extras/notebooks/source/machine-learning/feature-
store-with-uc-basic-example.html) ![Copy to clipboard](/_static/clippy.svg)
Copy link for import

The taxi example notebook illustrates the process of creating features,
updating them, and using them for model training and batch inference.

### Feature Engineering in Unity Catalog taxi example notebook

[Open notebook in new tab](/_extras/notebooks/source/machine-learning/feature-
store-with-uc-taxi-example.html) ![Copy to clipboard](/_static/clippy.svg)
Copy link for import

## Supported data types

Feature engineering in Unity Catalog and workspace feature store support the
following [PySpark data types](https://spark.apache.org/docs/latest/sql-ref-
datatypes.html):

  * `IntegerType`

  * `FloatType`

  * `BooleanType`

  * `StringType`

  * `DoubleType`

  * `LongType`

  * `TimestampType`

  * `DateType`

  * `ShortType`

  * `ArrayType`

  * `BinaryType` [1]

  * `DecimalType` [1]

  * `MapType` [1]

  * `StructType` [2]

[1] `BinaryType`, `DecimalType`, and `MapType` are supported in all versions
of Feature Engineering in Unity Catalog and in Workspace Feature Store v0.3.5
or above. [2] `StructType` is supported in Feature Engineering v0.6.0 or
above.

The data types listed above support feature types that are common in machine
learning applications. For example:

  * You can store dense vectors, tensors, and embeddings as `ArrayType`.

  * You can store sparse vectors, tensors, and embeddings as `MapType`.

  * You can store text as `StringType`.

When published to online stores, `ArrayType` and `MapType` features are stored
in JSON format.

The Feature Store UI displays metadata on feature data types:

![Complex data types example](../../_images/complex-data-type-example.png)

## More information

For more information on best practices, download [The Comprehensive Guide to
Feature Stores](https://www.databricks.com/p/ebook/the-comprehensive-guide-to-
feature-stores).

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

