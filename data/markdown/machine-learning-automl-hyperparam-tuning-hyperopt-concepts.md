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

  * [English](../../../en/machine-learning/automl-hyperparam-tuning/hyperopt-concepts.html)
  * [æ¥æ¬èª](../../../ja/machine-learning/automl-hyperparam-tuning/hyperopt-concepts.html)
  * [PortuguÃªs](../../../pt/machine-learning/automl-hyperparam-tuning/hyperopt-concepts.html)

[![](../../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/machine-learning/automl-hyperparam-tuning/hyperopt-concepts)
  * [![](../../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/machine-learning/automl-hyperparam-tuning/hyperopt-concepts.html)

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
          * [Hyperopt hyperparameter tuning](hyperopt-distributed-ml.html)
          * Hyperopt concepts
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
  * Hyperopt concepts
  * 

# Hyperopt concepts

Note

The open-source version of [Hyperopt](https://github.com/hyperopt/hyperopt) is
no longer being maintained.

Hyperopt will no longer be pre-installed on Databricks Runtime ML 17.0 and
above. Databricks recommends using [Optuna](optuna.html) instead for a similar
experience and access to more up-to-date hyperparameter tuning algorithms.

This article describes some of the concepts you need to know to use
distributed Hyperopt.

In this section:

  * `fmin()`

  * The `SparkTrials` class

  * `SparkTrials` and MLflow

For examples illustrating how to use Hyperopt in Databricks, see
[Hyperopt](index.html#hyperopt-overview).

## `fmin()`

You use `fmin()` to execute a Hyperopt run. The arguments for `fmin()` are
shown in the table; see [the Hyperopt
documentation](https://github.com/hyperopt/hyperopt/wiki/FMin) for more
information. For examples of how to use each argument, see [the example
notebooks](index.html#hyperopt-overview).

Argument name | Description  
---|---  
`fn` |  Objective function. Hyperopt calls this function with values generated from the hyperparameter space provided in the space argument. This function can return the loss as a scalar value or in a dictionary (see [Hyperopt docs](https://github.com/hyperopt/hyperopt/wiki/FMin) for details). This function typically contains code for model training and loss calculation.  
`space` |  Defines the hyperparameter space to search. Hyperopt provides great flexibility in how this space is defined. You can choose a categorical option such as algorithm, or probabilistic distribution for numeric values such as uniform and log.  
`algo` |  Hyperopt search algorithm to use to search hyperparameter space. Most commonly used are `hyperopt.rand.suggest` for Random Search and `hyperopt.tpe.suggest` for TPE.  
`max_evals` |  Number of hyperparameter settings to try (the number of models to fit).  
`max_queue_len` |  Number of hyperparameter settings Hyperopt should generate ahead of time. Because the Hyperopt TPE generation algorithm can take some time, it can be helpful to increase this beyond the default value of 1, but generally no larger than the `SparkTrials` setting `parallelism`.  
`trials` |  A `Trials` or `SparkTrials` object. Use `SparkTrials` when you call single-machine algorithms such as scikit-learn methods in the objective function. Use `Trials` when you call distributed training algorithms such as MLlib methods or Horovod in the objective function.  
`early_stop_fn` |  An optional early stopping function to determine if `fmin` should stop before `max_evals` is reached. Default is `None`. The input signature of the function is `Trials, *args` and the output signature is `bool, *args`. The output boolean indicates whether or not to stop. `*args` is any state, where the output of a call to `early_stop_fn` serves as input to the next call. `Trials` can be a `SparkTrials` object. When using `SparkTrials`, the early stopping function is not guaranteed to run after every trial, and is instead polled. [Example of an early stopping function](https://github.com/hyperopt/hyperopt/blob/master/hyperopt/early_stop.py)  
  
## The `SparkTrials` class

`SparkTrials` is an API developed by Databricks that allows you to distribute
a Hyperopt run without making other changes to your Hyperopt code.
`SparkTrials` accelerates single-machine tuning by distributing trials to
Spark workers.

Note

`SparkTrials` is designed to parallelize computations for single-machine ML
models such as scikit-learn. For models created with distributed ML algorithms
such as MLlib or Horovod, do not use `SparkTrials`. In this case the model
building process is automatically parallelized on the cluster and you should
use the default Hyperopt class `Trials`.

This section describes how to configure the arguments you pass to
`SparkTrials` and implementation aspects of `SparkTrials`.

### Arguments

`SparkTrials` takes two optional arguments:

  * `parallelism`: Maximum number of trials to evaluate concurrently. A higher number lets you scale-out testing of more hyperparameter settings. Because Hyperopt proposes new trials based on past results, there is a trade-off between parallelism and adaptivity. For a fixed `max_evals`, greater parallelism speeds up calculations, but lower parallelism may lead to better results since each iteration has access to more past results.

Default: Number of Spark executors available. Maximum: 128. If the value is
greater than the number of concurrent tasks allowed by the cluster
configuration, `SparkTrials` reduces parallelism to this value.

  * `timeout`: Maximum number of seconds an `fmin()` call can take. When this number is exceeded, all runs are terminated and `fmin()` exits. Information about completed runs is saved.

### Implementation

When defining the objective function `fn` passed to `fmin()`, and when
selecting a cluster setup, it is helpful to understand how `SparkTrials`
distributes tuning tasks.

In Hyperopt, a trial generally corresponds to fitting one model on one setting
of hyperparameters. Hyperopt iteratively generates trials, evaluates them, and
repeats.

With `SparkTrials`, the driver node of your cluster generates new trials, and
worker nodes evaluate those trials. Each trial is generated with a Spark job
which has one task, and is evaluated in the task on a worker machine. If your
cluster is set up to run multiple tasks per worker, then multiple trials may
be evaluated at once on that worker.

## `SparkTrials` and MLflow

Databricks Runtime ML supports logging to MLflow from workers. You can add
custom logging code in the objective function you pass to Hyperopt.

`SparkTrials` logs tuning results as nested MLflow runs as follows:

  * Main or parent run: The call to `fmin()` is logged as the main run. If there is an active run, `SparkTrials` logs to this active run and does not end the run when `fmin()` returns. If there is no active run, `SparkTrials` creates a new run, logs to it, and ends the run before `fmin()` returns.

  * Child runs: Each hyperparameter setting tested (a âtrialâ) is logged as a child run under the main run. MLflow log records from workers are also stored under the corresponding child runs.

When calling `fmin()`, Databricks recommends active MLflow run management;
that is, wrap the call to `fmin()` inside a `with mlflow.start_run():`
statement. This ensures that each `fmin()` call is logged to a separate MLflow
main run, and makes it easier to log extra tags, parameters, or metrics to
that run.

Note

When you call `fmin()` multiple times within the same active MLflow run,
MLflow logs those calls to the same main run. To resolve name conflicts for
logged parameters and tags, MLflow appends a UUID to names with conflicts.

When logging from workers, you do not need to manage runs explicitly in the
objective function. Call `mlflow.log_param("param_from_worker", x)` in the
objective function to log a parameter to the child run. You can log
parameters, metrics, tags, and artifacts in the objective function.

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

