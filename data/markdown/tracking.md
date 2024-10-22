  * __[![MLflow](_static/MLflow-logo-final-black.png)](index.html) 2.17.0

[![](_static/icons/nav-home.svg) MLflow](index.html)

  * [MLflow Overview](introduction/index.html)
  * [Getting Started with MLflow](getting-started/index.html)
  * [New Features](new-features/index.html)
  * [LLMs](llms/index.html)
  * [MLflow Tracing](llms/tracing/index.html)
  * [Model Evaluation](model-evaluation/index.html)
  * [Deep Learning](deep-learning/index.html)
  * [Traditional ML](traditional-ml/index.html)
  * [Deployment](deployment/index.html)
  * MLflow Tracking
    * Quickstart
    * Concepts
      * Runs
      * Experiments
      * Tracing
    * Tracking Runs
      * [MLflow Tracking APIs](tracking/tracking-api.html)
    * Tracking Datasets
      * [MLflow Dataset Tracking Tutorial](tracking/data-api.html)
    * Explore Runs and Results
      * Tracking UI
      * Querying Runs Programmatically
    * Set up the MLflow Tracking Environment
      * Components
        * MLflow Tracking APIs
        * Backend Store
        * Artifact Store
        * MLflow Tracking Server (Optional)
      * Common Setups
        * [MLflow Tracking Quickstart](getting-started/intro-quickstart/index.html)
        * [Tracking Experiments with a Local Database](tracking/tutorials/local-database.html)
        * [Remote Experiment Tracking with MLflow Tracking Server](tracking/tutorials/remote-server.html)
      * Other Configuration with MLflow Tracking Server
    * FAQ
      * Can I launch multiple runs in parallel?
      * How can I organize many MLflow Runs neatly?
      * Can I directly access remote storage without running the Tracking Server?
      * How to integrate MLflow Tracking with Model Registry?
      * How to include additional decription texts about the run?
  * [System Metrics](system-metrics/index.html)
  * [MLflow Projects](projects.html)
  * [MLflow Models](models.html)
  * [MLflow Model Registry](model-registry.html)
  * [MLflow Recipes](recipes.html)
  * [MLflow Plugins](plugins.html)
  * [MLflow Authentication](auth/index.html)
  * [Command-Line Interface](cli.html)
  * [Search Runs](search-runs.html)
  * [Search Experiments](search-experiments.html)
  * [Python API](python_api/index.html)
  * [R API](R-api.html)
  * [Java API](java_api/index.html)
  * [REST API](rest-api.html)
  * [Official MLflow Docker Image](docker.html)
  * [Community Model Flavors](community-model-flavors.html)
  * [Tutorials and Examples](tutorials-and-examples/index.html)

[Contribute](https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md)

  * [Documentation](index.html)
  * MLflow Tracking

# MLflow Tracking

The MLflow Tracking is an API and UI for logging parameters, code versions,
metrics, and output files when running your machine learning code and for
later visualizing the results. MLflow Tracking provides
[Python](python_api/index.html#python-api), [REST](rest-api.html#rest-api),
[R](R-api.html#r-api), and [Java](java_api/index.html#java-api) APIs.

![_images/tracking-metrics-ui-temp.png](_images/tracking-metrics-ui-temp.png)

A screenshot of the MLflow Tracking UI, showing a plot of validation loss
metrics during model training.

## Quickstart

If you havenât used MLflow Tracking before, we strongly recommend going
through the following quickstart tutorial.

[ MLflow Tracking Quickstart  A great place to start to learn the fundamentals
of MLflow Tracking! Learn in 5 minutes how to log, register, and load a model
for inference.  ](getting-started/intro-quickstart/index.html)

## Concepts

### Runs

MLflow Tracking is organized around the concept of **runs** , which are
executions of some piece of data science code, for example, a single `python
train.py` execution. Each run records metadata (various information about your
run such as metrics, parameters, start and end times) and artifacts (output
files from the run such as model weights, images, etc).

### Experiments

An experiment groups together runs for a specific task. You can create an
experiment using the CLI, API, or UI. The MLflow API and UI also let you
create and search for experiments. See [Organizing Runs into
Experiments](tracking/tracking-api.html#organizing-runs-in-experiments) for
more details on how to organize your runs into experiments.

### Tracing

MLflow Tracing is an integrated part of the MLflow Tracking API that allows
you to instrument your GenAI applications. Whether youâre using the
`LangChain` integration with MLflow Tracing, the Fluent APIs for tracing, or
the lower-level Client APIs, MLflow tracking will record your trace data for
future review, debugging, or analysis. See the [MLflow Tracing
Guide](llms/tracing/index.html) for more details.

## Tracking Runs

[MLflow Tracking APIs](tracking/tracking-api.html) provide a set of functions
to track your runs. For example, you can call
[`mlflow.start_run()`](python_api/mlflow.html#mlflow.start_run
"mlflow.start_run") to start a new run, then call [Logging
Functions](tracking/tracking-api.html#tracking-logging-functions) such as
[`mlflow.log_param()`](python_api/mlflow.html#mlflow.log_param
"mlflow.log_param") and
[`mlflow.log_metric()`](python_api/mlflow.html#mlflow.log_metric
"mlflow.log_metric") to log a parameters and metrics respectively. Please
visit the [Tracking API documentation](tracking/tracking-api.html) for more
details about using these APIs.

    
    
    import mlflow
    
    with mlflow.start_run():
        mlflow.log_param("lr", 0.001)
        # Your ml code
        ...
        mlflow.log_metric("val_loss", val_loss)
    

Alternatively, [Auto-logging](tracking/autolog.html) offers the ultra-quick
setup for starting MLflow tracking. This powerful feature allows you to log
metrics, parameters, and models without the need for explicit log statements -
all you need to do is call
[`mlflow.autolog()`](python_api/mlflow.html#mlflow.autolog "mlflow.autolog")
before your training code. Auto-logging supports popular libraries such as
[Scikit-learn](tracking/autolog.html#autolog-sklearn),
[XGBoost](tracking/autolog.html#autolog-xgboost),
[PyTorch](tracking/autolog.html#autolog-pytorch),
[Keras](tracking/autolog.html#autolog-keras),
[Spark](tracking/autolog.html#autolog-spark), and more. See [Automatic Logging
Documentation](tracking/autolog.html#automatic-logging) for supported
libraries and how to use auto-logging APIs with each of them.

    
    
    import mlflow
    
    mlflow.autolog()
    
    # Your training code...
    

Note

By default, without any particular server/database configuration, MLflow
Tracking logs data to the local mlruns directory. If you want to log your runs
to a different location, such as a remote database and cloud storage, to share
your results with your team, follow the instructions in the Set up MLflow
Tracking Environment section.

## Tracking Datasets

MLflow offers the ability to track datasets that are associated with model
training events. These metadata associated with the Dataset can be stored
through the use of the
[`mlflow.log_input()`](python_api/mlflow.html#mlflow.log_input
"mlflow.log_input") API. To learn more, please visit the [MLflow data
documentation](tracking/data-api.html) to see the features available in this
API.

## Explore Runs and Results

### Tracking UI

The Tracking UI lets you visually explore your experiments and runs, as shown
on top of this page.

  * Experiment-based run listing and comparison (including run comparison across multiple experiments)

  * Searching for runs by parameter or metric value

  * Visualizing run metrics

  * Downloading run results (artifacts and metadata)

If you log runs to a local `mlruns` directory, run the following command in
the directory above it, then access <http://127.0.0.1:5000> in your browser.

    
    
    mlflow ui --port 5000
    

Alternatively, the MLflow Tracking Server serves the same UI and enables
remote storage of run artifacts. In that case, you can view the UI at
`http://<IP address of your MLflow tracking server>:5000` from any machine
that can connect to your tracking server.

### Querying Runs Programmatically

You can also access all of the functions in the Tracking UI programmatically
with [`MlflowClient`](python_api/mlflow.client.html#mlflow.client.MlflowClient
"mlflow.client.MlflowClient").

For example, the following code snippet search for runs that has the best
validation loss among all runs in the experiment.

    
    
    client = mlflow.tracking.MlflowClient()
    experiment_id = "0"
    best_run = client.search_runs(
        experiment_id, order_by=["metrics.val_loss ASC"], max_results=1
    )[0]
    print(best_run.info)
    # {'run_id': '...', 'metrics': {'val_loss': 0.123}, ...}
    

## Set up the MLflow Tracking Environment

Note

If you just want to log your experiment data and models to local files, you
can skip this section.

MLflow Tracking supports many different scenarios for your development
workflow. This section will guide you through how to set up the MLflow
Tracking environment for your particular use case. From a birdâs-eye view,
the MLflow Tracking environment consists of the following components.

### Components

#### [MLflow Tracking APIs](tracking/tracking-api.html)

You can call MLflow Tracking APIs in your ML code to log runs and communicate
with the MLflow Tracking Server if necessary.

#### [Backend Store](tracking/backend-stores.html)

The backend store persists various metadata for each Run, such as run ID,
start and end times, parameters, metrics, etc. MLflow supports two types of
storage for the backend: **file-system-based** like local files and
**database-based** like PostgreSQL.

#### [Artifact Store](tracking/artifacts-stores.html)

Artifact store persists (typicaly large) artifacts for each run, such as model
weights (e.g. a pickled scikit-learn model), images (e.g. PNGs), model and
data files (e.g. [Parquet](https://parquet.apache.org/) file). MLflow stores
artifacts ina a local file (mlruns) by default, but also supports different
storage options such as Amazon S3 and Azure Blob Storage.

#### [MLflow Tracking Server](tracking/server.html) (Optional)

MLflow Tracking Server is a stand-alone HTTP server that provides REST APIs
for accessing backend and/or artifact store. Tracking server also offers
flexibility to configure what data to server, govern access control,
versioning, and etc. Read [MLflow Tracking Server
documentation](tracking/server.html) for more details.

### Common Setups

By configuring these components properly, you can create an MLflow Tracking
environment suitable for your teamâs development workflow. The following
diagram and table show a few common setups for the MLflow Tracking
environment.

![_images/tracking-setup-overview.png](_images/tracking-setup-overview.png)

| **1\. Localhost (default)** | **2\. Local Tracking with Local Database** | **3\. Remote Tracking with** MLflow Tracking Server  
---|---|---|---  
**Scenario** | Solo development | Solo development | Team development  
**Use Case** | By default, MLflow records metadata and artifacts for each run to a local directory, `mlruns`. This is the simplest way to get started with MLflow Tracking, without setting up any external server, database, and storage. | The MLflow client can interface with a SQLAlchemy-compatible database (e.g., SQLite, PostgreSQL, MySQL) for the [backend](tracking/backend-stores.html). Saving metadata to a database allows you cleaner management of your experiment data while skipping the effort of setting up a server. | MLflow Tracking Server can be configured with an artifacts HTTP proxy, passing artifact requests through the tracking server to store and retrieve artifacts without having to interact with underlying object store services. This is particularly useful for team development scenarios where you want to store artifacts and experiment metadata in a shared location with proper access control.  
**Tutorial** | [QuickStart](getting-started/intro-quickstart/index.html) | [Tracking Experiments using a Local Database](tracking/tutorials/local-database.html) | [Remote Experiment Tracking with MLflow Tracking Server](tracking/tutorials/remote-server.html)  
  
### Other Configuration with MLflow Tracking Server

MLflow Tracking Server provides customizability for other special use cases.
Please follow [Remote Experiment Tracking with MLflow Tracking
Server](tracking/tutorials/remote-server.html) for learning the basic setup
and continue to the following materials for advanced configurations to meet
your needs.

Local Tracking ServerArtifacts-only ModeDirect Access to Artifacts

#### Using MLflow Tracking Server Locally

You can of course run MLflow Tracking Server locally. While this doesn't
provide much additional benefit over directly using the local files or
database, might useful for testing your team development workflow locally or
running your machine learning code on a container environment.

![](_static/images/tracking/tracking-setup-local-server.png)

#### Running MLflow Tracking Server in Artifacts-only Mode

MLflow Tracking Server has `--artifacts-only` option, which lets the server to
serve (proxy) only artifacts and not metadata. This is particularly useful
when you are in a large organization or training huge models, you might have
high artifact transfer volumes and want to split out the traffic for serving
artifacts to not impact tracking functionality. Please read [Optionally using
a Tracking Server instance exclusively for artifact
handling](tracking/server.html#optionally-using-a-tracking-server-instance-
exclusively-for-artifact-handling) for more details on how to use this mode.

![](_static/images/tracking/tracking-setup-artifacts-only.png)

####  Disable Artifact Proxying to Allow Direct Access to Artifacts

MLflow Tracking Server, by default, serves both artifacts and only metadata.
However, in some cases, you may want to allow direct access to the remote
artifacts storage to avoid the overhead of a proxy while preserving the
functionality of metadata tracking. This can be done by disabling artifact
proxying by starting server with `--no-serve-artifacts` option. Refer to [Use
Tracking Server without Proxying Artifacts Access](tracking/server.html#use-
tracking-server-w-o-proxying-artifacts-access) for how to set this up.

![](_static/images/tracking/tracking-setup-no-serve-artifacts.png)

## FAQ

### Can I launch multiple runs in parallel?

Yes, MLflow supports launching multiple runs in parallel e.g. multi processing
/ threading. See [Launching Multiple Runs in One Program](tracking/tracking-
api.html#launching-multiple-runs) for more details.

### How can I organize many MLflow Runs neatly?

MLflow provides a few ways to organize your runs:

  * [Organize runs into experiments](tracking/tracking-api.html#organizing-runs-in-experiments) \- Experiments are logical containers for your runs. You can create an experiment using the CLI, API, or UI.

  * [Create child runs](tracking/tracking-api.html#child-runs) \- You can create child runs under a single parent run to group them together. For example, you can create a child run for each fold in a cross-validation experiment.

  * [Add tags to runs](tracking/tracking-api.html#add-tags-to-runs) \- You can associate arbitrary tags with each run, which allows you to filter and search runs based on tags.

### Can I directly access remote storage without running the Tracking Server?

Yes, while it is best practice to have the MLflow Tracking Server as a proxy
for artifacts access for team development workflows, you may not need that if
you are using it for personal projects or testing. You can achieve this by
following the workaround below:

  1. Set up artifacts configuration such as credentials and endpoints, just like you would for the MLflow Tracking Server. See [configure artifact storage](tracking/artifacts-stores.html#artifacts-store-supported-storages) for more details.

  2. Create an experiment with an explicit artifact location,

    
    
    experiment_name = "your_experiment_name"
    mlflow.create_experiment(experiment_name, artifact_location="s3://your-bucket")
    mlflow.set_experiment(experiment_name)
    

Your runs under this experiment will log artifacts to the remote storage
directly.

### How to integrate MLflow Tracking with [Model Registry](model-
registry.html#registry)?

To use the Model Registry functionality with MLflow tracking, you **must use
database backed store** such as PostgresQL and log a model using the
`log_model` methods of the corresponding model flavors. Once a model has been
logged, you can add, modify, update, or delete the model in the Model Registry
through the UI or the API. See [Backend Stores](tracking/backend-stores.html)
and Common Setups for how to configures backend store properly for your
workflow.

### How to include additional decription texts about the run?

A system tag `mlflow.note.content` can be used to add descriptive note about
this run. While the other [system tags](tracking/tracking-api.html#system-
tags) are set automatically, this tag is **not set by default** and users can
override it to include additional information about the run. The content will
be displayed on the runâs page under the Notes section.

[ Previous](deployment/deploy-model-to-kubernetes/tutorial.html "Develop ML
model with MLflow and deploy to Kubernetes") [Next ](tracking/tracking-
api.html "MLflow Tracking APIs")

* * *

(C) MLflow Project, a Series of LF Projects, LLC. All rights reserved.

