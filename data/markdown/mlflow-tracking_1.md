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

  * [English](../../en/mlflow/tracking.html)
  * [æ¥æ¬èª](../../ja/mlflow/tracking.html)
  * [PortuguÃªs](../../pt/mlflow/tracking.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/mlflow/tracking)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/mlflow/tracking.html)

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
    * [Tutorials](../machine-learning/ml-tutorials.html)
    * [AI playground](../large-language-models/ai-playground.html)
    * [AI functions in SQL](../large-language-models/ai-functions.html)
    * [Serve models](../machine-learning/serve-models.html)
    * [Train models](../machine-learning/train-model/index.html)
    * [Serve data for AI](../machine-learning/serve-data-ai.html)
    * [Evaluate AI](../generative-ai/agent-evaluation/index.html)
    * [Build gen AI apps](../generative-ai/build-genai-apps.html)
    * [MLOps and MLflow](index.html)
      * [MLOps workflows on Databricks](../machine-learning/mlops/mlops-workflow.html)
      * [Get started with MLflow experiments](quick-start.html)
      * [MLflow experiment tracking](../machine-learning/track-model-development/index.html)
        * Track ML and deep learning training runs
          * [Organize training runs with MLflow experiments](experiments.html)
          * [Manage training code with MLflow runs](runs.html)
          * [Build dashboards with the MLflow Search API](build-dashboards.html)
          * [Track ML Model training data with Delta Lake](tracking-ex-delta.html)
        * [Databricks Autologging](databricks-autologging.html)
        * [Access the MLflow tracking server from outside Databricks](access-hosted-tracking-server.html)
      * [Log, load, register, and deploy MLflow models](models.html)
      * [Manage model lifecycle](../machine-learning/manage-model-lifecycle/index.html)
      * [Run MLflow Projects on Databricks](projects.html)
      * [Copy MLflow objects between workspaces](migrate-mlflow-objects.html)
    * [Integrations](../machine-learning/integrations.html)
    * [Graph and network analysis](../machine-learning/graph-analysis.html)
    * [Reference solutions](../machine-learning/reference-solutions/index.html)
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

Updated Oct 24, 2024

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation
Feedback)

  * [Documentation](../index.html)
  * [AI and machine learning on Databricks](../machine-learning/index.html)
  * [ML lifecycle management using MLflow](index.html)
  * [Track model development using MLflow](../machine-learning/track-model-development/index.html)
  * Track ML and deep learning training runs
  * 

# Track ML and deep learning training runs

The MLflow tracking component lets you log source properties, parameters,
metrics, tags, and artifacts related to training a machine learning or deep
learning model. To get started with MLflow, try one of the [MLflow quickstart
tutorials](quick-start.html).

## MLflow tracking with experiments and runs

MLflow tracking is based on two concepts, _experiments_ and _runs_ :

Note

Starting March 27, 2024, MLflow imposes a quota limit on the number of total
parameters, tags, and metric steps for all existing and new runs, and the
number of total runs for all existing and new experiments, see [Resource
limits](../resources/limits.html). If you hit the runs per experiment quota,
Databricks recommends you delete runs that you no longer need [using the
delete runs API in Python](runs.html#bulk-delete). If you hit other quota
limits, Databricks recommends adjusting your logging strategy to keep under
the limit. If you require an increase to this limit, reach out to your
Databricks account team with a brief explanation of your use case, why the
suggested mitigation approaches do not work, and the new limit you request.

  * An MLflow _experiment_ is the primary unit of organization and access control for MLflow runs; all MLflow runs belong to an experiment. Experiments let you visualize, search for, and compare runs, as well as download run artifacts and metadata for analysis in other tools.

  * An MLflow _run_ corresponds to a single execution of model code.

  * [Organize training runs with MLflow experiments](experiments.html)
  * [Manage training code with MLflow runs](runs.html)

The [MLflow Tracking API](https://www.mlflow.org/docs/latest/tracking.html)
logs parameters, metrics, tags, and artifacts from a model run. The Tracking
API communicates with an MLflow [tracking
server](https://www.mlflow.org/docs/latest/tracking.html#tracking-server).
When you use Databricks, a Databricks-hosted tracking server logs the data.
The hosted MLflow tracking server has Python, Java, and R APIs.

Note

MLflow is installed on Databricks Runtime ML clusters. To use MLflow on a
Databricks Runtime cluster, you must install the `mlflow` library. For
instructions on installing a library onto a cluster, see [Install a library on
a cluster](../libraries/cluster-libraries.html#install-libraries). The
specific packages to install for MLflow are:

  * For Python, select **Library Source** PyPI and enter `mlflow` in the **Package** field.

  * For R, select **Library Source** CRAN and enter `mlflow` in the **Package** field.

  * For Scala, install these two packages:

    * Select **Library Source** Maven and enter `org.mlflow:mlflow-client:1.11.0` in the **Coordinates** field.

    * Select **Library Source** PyPI and enter `mlflow` in the **Package** field.

## Where MLflow runs are logged

All MLflow runs are logged to the active experiment, which can be set using
any of the following ways:

  * Use the [mlflow.set_experiment() command](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment).

  * Use the `experiment_id` parameter in the [mlflow.start_run() command](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run).

  * Set one of the MLflow environment variables [MLFLOW_EXPERIMENT_NAME or MLFLOW_EXPERIMENT_ID](https://mlflow.org/docs/latest/cli.html#cmdoption-mlflow-run-arg-uri).

If no active experiment is set, runs are logged to the [notebook
experiment](experiments.html#mlflow-notebook-experiments).

To log your experiment results to a remotely hosted MLflow Tracking server in
a workspace other than the one in which you are running your experiment, set
the tracking URI to reference the remote workspace with
`mlflow.set_tracking_uri()`, and set the path to your experiment in the remote
workspace by using `mlflow.set_experiment()`.

    
    
    mlflow.set_tracking_uri(<uri-of-remote-workspace>)
    mlflow.set_experiment("path to experiment in remote workspace")
    

If you are running experiments locally and want to log experiment results to
the Databricks MLflow Tracking server, provide your Databricks workspace
instance (`DATABRICKS_HOST`) and Databricks personal access token
(`DATABRICKS_TOKEN`). Next, you can set the tracking URI to reference the
workspace with `mlflow.set_tracking_uri()`, and set the path to your
experiment by using `mlflow.set_experiment()`. See [Perform Databricks
personal access token authentication](../dev-tools/auth/pat.html#token-auth)
for details on where to find values for the `DATABRICKS_HOST` and
`DATABRICKS_TOKEN` environment variables.

The following code example demonstrates setting these values:

    
    
    os.environ["DATABRICKS_HOST"] = "https://dbc-1234567890123456.cloud.databricks.com" # set to your server URI
    os.environ["DATABRICKS_TOKEN"] = "dapixxxxxxxxxxxxx"
    
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/your-experiment")
    

## Logging example notebook

This notebook shows how to log runs to a notebook experiment and to a
workspace experiment. Only MLflow runs initiated within a notebook can be
logged to the notebook experiment. MLflow runs launched from any notebook or
from the APIs can be logged to a workspace experiment. For information about
viewing logged runs, see [View notebook experiment](experiments.html#view-
notebook-experiment) and [View workspace experiment](experiments.html#view-
workspace-experiment).

### Log MLflow runs notebook

[Open notebook in new tab](/_extras/notebooks/source/mlflow/mlflow-log-
runs.html) ![Copy to clipboard](/_static/clippy.svg) Copy link for import

You can use MLflow Python, Java or Scala, and R APIs to start runs and record
run data. For details, see the [MLflow example notebooks](quick-start.html).

## Access the MLflow tracking server from outside Databricks

You can also write to and read from the tracking server from outside
Databricks, for example using the MLflow CLI. See [Access the MLflow tracking
server from outside Databricks](access-hosted-tracking-server.html).

## Analyze MLflow runs programmatically

You can access MLflow run data programmatically using the following two
DataFrame APIs:

  * The MLflow Python client [search_runs API](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs) returns a pandas DataFrame.

  * The [MLflow experiment](../query/formats/mlflow-experiment.html) data source returns an Apache Spark DataFrame.

This example demonstrates how to use the MLflow Python client to build a
dashboard that visualizes changes in evaluation metrics over time, tracks the
number of runs started by a specific user, and measures the total number of
runs across all users:

  * [Build dashboards with the MLflow Search API](build-dashboards.html)

## Why model training metrics and outputs may vary

Many of the algorithms used in ML have a random element, such as sampling or
random initial conditions within the algorithm itself. When you train a model
using one of these algorithms, the results might not be the same with each
run, even if you start the run with the same conditions. Many libraries offer
a seeding mechanism to fix the initial conditions for these stochastic
elements. However, there may be other sources of variation that are not
controlled by seeds. Some algorithms are sensitive to the order of the data,
and distributed ML algorithms may also be affected by how the data is
partitioned. Generally this variation is not significant and not important in
the model development process.

To control variation caused by differences in ordering and partitioning, use
the PySpark functions
[repartition](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.repartition.html)
and
[sortWithinPartitions](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.sortWithinPartitions.html).

## MLflow tracking examples

The following notebooks demonstrate how to train several types of models and
track the training data in MLflow and how to store tracking data in Delta
Lake.

  * [Track scikit-learn model training with MLflow](tracking-ex-scikit.html)

  * [Track ML Model training data with Delta Lake](tracking-ex-delta.html)

  * [Track Keras model training with MLflow](https://mlflow.org/docs/latest/deep-learning/keras/quickstart/quickstart_keras_core.html_)

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

