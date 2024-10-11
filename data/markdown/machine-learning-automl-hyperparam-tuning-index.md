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

  * [English](../../../en/machine-learning/automl-hyperparam-tuning/index.html)
  * [æ¥æ¬èª](../../../ja/machine-learning/automl-hyperparam-tuning/index.html)
  * [PortuguÃªs](../../../pt/machine-learning/automl-hyperparam-tuning/index.html)

[![](../../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/machine-learning/automl-hyperparam-tuning/)
  * [![](../../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/machine-learning/automl-hyperparam-tuning/index.html)

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
      * [Databricks Runtime ML](../databricks-runtime-ml.html)
      * [Load data for training](../load-data/index.html)
      * [AutoML](../automl/index.html)
      * [Gen AI models](../../large-language-models/foundation-model-training/index.html)
      * [Model training examples](../train-model/training-examples.html)
        * [Use XGBoost on Databricks](../train-model/xgboost.html)
        * Hyperparameter tuning
          * [Optuna hyperparameter tuning](optuna.html)
          * [Hyperopt hyperparameter tuning](hyperopt-distributed-ml.html)
          * [Hyperopt concepts](hyperopt-concepts.html)
          * [Compare model types with Hyperopt and MLflow](hyperopt-model-selection.html)
          * [Hyperopt best practices and troubleshooting](hyperopt-best-practices.html)
          * [Parallelize Hyperopt hyperparameter tuning](hyperopt-spark-mlflow-integration.html)
      * [Deep learning](../train-model/deep-learning.html)
      * [Train recommender models](../train-recommender-models.html)
    * [Serve data for AI](../serve-data-ai.html)
    * [Evaluate AI](../../generative-ai/agent-evaluation/index.html)
    * [Build gen AI apps](../../generative-ai/build-genai-apps.html)
    * [MLOps and MLflow](../../mlflow/index.html)
    * [Integrations](../integrations.html)
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

Updated Oct 11, 2024

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation
Feedback)

  * [Documentation](../../index.html)
  * [AI and machine learning on Databricks](../index.html)
  * [Train AI and ML models](../train-model/index.html)
  * [Model training examples](../train-model/training-examples.html)
  * Hyperparameter tuning
  * 

# Hyperparameter tuning

Python libraries like Optuna, Ray Tune, and Hyperopt simplify and automate
hyperparameter tuning to efficiently find an optimal set of hyperparameters
for machine learning models. These libraries scale across multiple computes to
quickly find hyperparameters with minimal manual orchestration and
configuration requirements.

## Optuna

[Optuna](https://github.com/optuna/optuna) is a light-weight framework that
makes it easy to define a dynamic search space for hyperparameter tuning and
model selection. Optuna includes some of the latest optimization and machine
learning algorithms.

Optuna can be easily parallelized with Joblib to scale workloads, and
integrated with Mlflow to track hyperparameters and metrics across trials.

To get started with Optuna, see [Hyperparameter tuning with
Optuna](optuna.html).

## Ray Tune

Databricks Runtime ML includes Ray, an open-source framework used for parallel
compute processing. Ray Tune is a hyperparameter tuning library that comes
with Ray and uses Ray as a backend for distributed computing.

For details on how to run Ray on Databricks, see [What is Ray on
Databricks?](../ray/index.html). For examples of Ray Tune, see [Ray Tune
documentation](https://docs.ray.io/en/latest/tune/tutorials/tune-
distributed.html).

## Hyperopt

Note

The open-source version of [Hyperopt](https://github.com/hyperopt/hyperopt) is
no longer being maintained.

Hyperopt will no longer be pre-installed on Databricks Runtime ML 17.0 and
above. Databricks recommends using [Optuna](optuna.html) instead for a similar
experience and access to more up-to-date hyperparameter tuning algorithms.

[Hyperopt](https://github.com/hyperopt/hyperopt) is a Python library used for
distributed hyperparameter tuning and model selection. Hyperopt works with
both distributed ML algorithms such as Apache Spark MLlib and Horovod, as well
as with single-machine ML models such as scikit-learn and TensorFlow.

To get started using Hyperopt, see [Use distributed training algorithms with
Hyperopt](hyperopt-distributed-ml.html).

## MLlib automated MLflow tracking

Note

MLlib automated MLflow tracking is deprecated and disabled by default on
clusters that run Databricks Runtime 10.4 LTS ML and above.

Instead, use [MLflow PySpark ML
autologging](https://www.mlflow.org/docs/latest/python_api/mlflow.pyspark.ml.html#mlflow.pyspark.ml.autolog)
by calling `mlflow.pyspark.ml.autolog()`, which is enabled by default with
[Databricks Autologging](../../mlflow/databricks-autologging.html).

With MLlib automated MLflow tracking, when you run tuning code that uses
CrossValidator or TrainValidationSplit. Hyperparameters and evaluation metrics
are automatically logged in MLflow.

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

