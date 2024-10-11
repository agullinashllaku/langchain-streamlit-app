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

  * [English](../../../en/machine-learning/automl-hyperparam-tuning/hyperopt-distributed-ml.html)
  * [æ¥æ¬èª](../../../ja/machine-learning/automl-hyperparam-tuning/hyperopt-distributed-ml.html)
  * [PortuguÃªs](../../../pt/machine-learning/automl-hyperparam-tuning/hyperopt-distributed-ml.html)

[![](../../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/machine-learning/automl-hyperparam-tuning/hyperopt-distributed-ml)
  * [![](../../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/machine-learning/automl-hyperparam-tuning/hyperopt-distributed-ml.html)

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
        * [Hyperparameter tuning](index.html)
          * [Optuna hyperparameter tuning](optuna.html)
          * Hyperopt hyperparameter tuning
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
  * [Hyperparameter tuning](index.html)
  * Use distributed training algorithms with Hyperopt
  * 

# Use distributed training algorithms with Hyperopt

Note

The open-source version of [Hyperopt](https://github.com/hyperopt/hyperopt) is
no longer being maintained.

Hyperopt will no longer be pre-installed on Databricks Runtime ML 17.0 and
above. Databricks recommends using [Optuna](optuna.html) instead for a similar
experience and access to more up-to-date hyperparameter tuning algorithms.

In addition to single-machine training algorithms such as those from scikit-
learn, you can use Hyperopt with distributed training algorithms. In this
scenario, Hyperopt generates trials with different hyperparameter settings on
the driver node. Each trial is executed from the driver node, giving it access
to the full cluster resources. This setup works with any distributed machine
learning algorithms or libraries, including Apache Spark MLlib and
HorovodRunner.

When you use Hyperopt with distributed training algorithms, do not pass a
`trials` argument to `fmin()`, and specifically, do not use the `SparkTrials`
class. `SparkTrials` is designed to distribute trials for algorithms that are
not themselves distributed. With distributed training algorithms, use the
default `Trials` class, which runs on the cluster driver. Hyperopt evaluates
each trial on the driver node so that the ML algorithm itself can initiate
distributed training.

Note

Databricks does not support automatic logging to MLflow with the `Trials`
class. When using distributed training algorithms, you must manually call
MLflow to log trials for Hyperopt.

## Notebook example: Use Hyperopt with MLlib algorithms

The example notebook shows how to use Hyperopt to tune MLlibâs distributed
training algorithms.

### Hyperopt and MLlib distributed training notebook

[Open notebook in new tab](/_extras/notebooks/source/hyperopt-spark-ml.html)
![Copy to clipboard](/_static/clippy.svg) Copy link for import

## Notebook example: Use Hyperopt with HorovodRunner

HorovodRunner is a general API used to run distributed deep learning workloads
on Databricks. HorovodRunner integrates
[Horovod](https://github.com/horovod/horovod) with Sparkâs [barrier
mode](https://issues.apache.org/jira/browse/SPARK-24374) to provide higher
stability for long-running deep learning training jobs on Spark.

The example notebook shows how to use Hyperopt to tune distributed training
for deep learning based on HorovodRunner.

### Hyperopt and HorovodRunner distributed training notebook

[Open notebook in new tab](/_extras/notebooks/source/hyperopt-distributed-ml-
training.html) ![Copy to clipboard](/_static/clippy.svg) Copy link for import

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

