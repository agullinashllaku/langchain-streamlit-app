  * __[![MLflow](../_static/MLflow-logo-final-black.png)](../index.html) 2.17.0rc0

[![](../_static/icons/nav-home.svg) MLflow](../index.html)

  * [MLflow Overview](../introduction/index.html)
  * [Getting Started with MLflow](../getting-started/index.html)
  * [New Features](../new-features/index.html)
  * [LLMs](../llms/index.html)
  * [MLflow Tracing](../llms/tracing/index.html)
  * [Model Evaluation](../model-evaluation/index.html)
  * [Deep Learning](../deep-learning/index.html)
  * [Traditional ML](../traditional-ml/index.html)
  * [Deployment](../deployment/index.html)
  * [MLflow Tracking](../tracking.html)
  * [System Metrics](../system-metrics/index.html)
  * [MLflow Projects](../projects.html)
  * [MLflow Models](../models.html)
  * [MLflow Model Registry](../model-registry.html)
  * [MLflow Recipes](../recipes.html)
  * [MLflow Plugins](../plugins.html)
  * [MLflow Authentication](../auth/index.html)
  * [Command-Line Interface](../cli.html)
  * [Search Runs](../search-runs.html)
  * [Search Experiments](../search-experiments.html)
  * [Python API](index.html)
    * mlflow
    * MLflow Tracing APIs
    * [mlflow.artifacts](mlflow.artifacts.html)
    * [mlflow.autogen](mlflow.autogen.html)
    * [mlflow.catboost](mlflow.catboost.html)
    * [mlflow.client](mlflow.client.html)
    * [mlflow.config](mlflow.config.html)
    * [mlflow.data](mlflow.data.html)
    * [mlflow.deployments](mlflow.deployments.html)
    * [mlflow.diviner](mlflow.diviner.html)
    * [mlflow.entities](mlflow.entities.html)
    * [mlflow.environment_variables](mlflow.environment_variables.html)
    * [mlflow.fastai](mlflow.fastai.html)
    * [mlflow.gateway](mlflow.gateway.html)
    * [mlflow.gluon](mlflow.gluon.html)
    * [mlflow.h2o](mlflow.h2o.html)
    * [mlflow.johnsnowlabs](mlflow.johnsnowlabs.html)
    * [mlflow.keras](mlflow.keras.html)
    * [mlflow.langchain](mlflow.langchain.html)
    * [mlflow.lightgbm](mlflow.lightgbm.html)
    * [mlflow.llama_index](mlflow.llama_index.html)
    * [mlflow.metrics](mlflow.metrics.html)
    * [mlflow.mleap](mlflow.mleap.html)
    * [mlflow.models](mlflow.models.html)
    * [mlflow.onnx](mlflow.onnx.html)
    * [mlflow.paddle](mlflow.paddle.html)
    * [mlflow.pmdarima](mlflow.pmdarima.html)
    * [mlflow.projects](mlflow.projects.html)
    * [mlflow.promptflow](mlflow.promptflow.html)
    * [mlflow.prophet](mlflow.prophet.html)
    * [mlflow.pyfunc](mlflow.pyfunc.html)
    * [mlflow.pyspark.ml](mlflow.pyspark.ml.html)
    * [mlflow.pytorch](mlflow.pytorch.html)
    * [mlflow.recipes](mlflow.recipes.html)
    * [mlflow.sagemaker](mlflow.sagemaker.html)
    * [mlflow.sentence_transformers](mlflow.sentence_transformers.html)
    * [mlflow.server](mlflow.server.html)
    * [mlflow.shap](mlflow.shap.html)
    * [mlflow.sklearn](mlflow.sklearn.html)
    * [mlflow.spacy](mlflow.spacy.html)
    * [mlflow.spark](mlflow.spark.html)
    * [mlflow.statsmodels](mlflow.statsmodels.html)
    * [mlflow.system_metrics](mlflow.system_metrics.html)
    * [mlflow.tensorflow](mlflow.tensorflow.html)
    * [mlflow.tracing](mlflow.tracing.html)
    * [mlflow.transformers](mlflow.transformers.html)
    * [mlflow.types](mlflow.types.html)
    * [mlflow.utils](mlflow.utils.html)
    * [mlflow.xgboost](mlflow.xgboost.html)
    * [mlflow.openai](openai/index.html)
    * [Log Levels](index.html#log-levels)
  * [R API](../R-api.html)
  * [Java API](../java_api/index.html)
  * [REST API](../rest-api.html)
  * [Official MLflow Docker Image](../docker.html)
  * [Community Model Flavors](../community-model-flavors.html)
  * [Tutorials and Examples](../tutorials-and-examples/index.html)

[Contribute](https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md)

  * [Documentation](../index.html)
  * [Python API](index.html)
  * mlflow

# mlflow

The `mlflow` module provides a high-level âfluentâ API for starting and
managing MLflow runs. For example:

    
    
    import mlflow
    
    mlflow.start_run()
    mlflow.log_param("my", "param")
    mlflow.log_metric("score", 100)
    mlflow.end_run()
    

You can also use the context manager syntax like this:

    
    
    with mlflow.start_run() as run:
        mlflow.log_param("my", "param")
        mlflow.log_metric("score", 100)
    

which automatically terminates the run at the end of the `with` block.

The fluent tracking API is not currently threadsafe. Any concurrent callers to
the tracking API must implement mutual exclusion manually.

For a lower level API, see the [`mlflow.client`](mlflow.client.html#module-
mlflow.client "mlflow.client") module.

_class
_`mlflow.``ActiveRun`(_run_)[[source]](../_modules/mlflow/tracking/fluent.html#ActiveRun)

    

Wrapper around
[`mlflow.entities.Run`](mlflow.entities.html#mlflow.entities.Run
"mlflow.entities.Run") to enable using Python `with` syntax.

_class _`mlflow.``Image`(_image : Union[numpy.ndarray, PIL.Image.Image, str,
list]_)[[source]](../_modules/mlflow/tracking/multimedia.html#Image)

    

mlflow.Image is an image media object that provides a lightweight option for
handling images in MLflow. The image can be a numpy array, a PIL image, or a
file path to an image. The image is stored as a PIL image and can be logged to
MLflow using mlflow.log_image or mlflow.log_table.

Parameters

    

**image** â Image can be a numpy array, a PIL image, or a file path to an
image.

Example

    
    
    import mlflow
    import numpy as np
    from PIL import Image
    
    # Create an image as a numpy array
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :50] = [255, 128, 0]
    # Create an Image object
    image_obj = mlflow.Image(image)
    # Convert the Image object to a list of pixel values
    pixel_values = image_obj.to_list()
    

`resize`(_size : Tuple[int,
int]_)[[source]](../_modules/mlflow/tracking/multimedia.html#Image.resize)

    

Resize the image to the specified size.

Parameters

    

**size** â Size to resize the image to.

Returns

    

A copy of the resized image object.

`save`(_path :
str_)[[source]](../_modules/mlflow/tracking/multimedia.html#Image.save)

    

Save the image to a file.

Parameters

    

**path** â File path to save the image.

`to_array`()[[source]](../_modules/mlflow/tracking/multimedia.html#Image.to_array)

    

Convert the image to a numpy array.

Returns

    

Numpy array of pixel values.

`to_list`()[[source]](../_modules/mlflow/tracking/multimedia.html#Image.to_list)

    

Convert the image to a list of pixel values.

Returns

    

List of pixel values.

`to_pil`()[[source]](../_modules/mlflow/tracking/multimedia.html#Image.to_pil)

    

Convert the image to a PIL image.

Returns

    

PIL image.

_exception _`mlflow.``MlflowException`(_message_ , _error_code =1_, _**
kwargs_)[[source]](../_modules/mlflow/exceptions.html#MlflowException)

    

Generic exception thrown to surface failure information about external-facing
operations. The error message associated with this exception may be exposed to
clients in HTTP responses for debugging purposes. If the error text is
sensitive, raise a generic Exception object instead.

`get_http_status_code`()[[source]](../_modules/mlflow/exceptions.html#MlflowException.get_http_status_code)

    

_classmethod _`invalid_parameter_value`(_message_ , _**
kwargs_)[[source]](../_modules/mlflow/exceptions.html#MlflowException.invalid_parameter_value)

    

Constructs an MlflowException object with the INVALID_PARAMETER_VALUE error
code.

Parameters

    

  * **message** â The message describing the error that occurred. This will be included in the exceptionâs serialized JSON representation.

  * **kwargs** â Additional key-value pairs to include in the serialized JSON representation of the MlflowException.

`serialize_as_json`()[[source]](../_modules/mlflow/exceptions.html#MlflowException.serialize_as_json)

    

`mlflow.``active_run`() ->
Optional[ActiveRun][[source]](../_modules/mlflow/tracking/fluent.html#active_run)

    

Get the currently active `Run`, or None if no such run exists.

**Note** : You cannot access currently-active run attributes (parameters,
metrics, etc.) through the run returned by `mlflow.active_run`. In order to
access such attributes, use the
[`mlflow.client.MlflowClient`](mlflow.client.html#mlflow.client.MlflowClient
"mlflow.client.MlflowClient") as follows:

Example

    
    
    import mlflow
    
    mlflow.start_run()
    run = mlflow.active_run()
    print(f"Active run_id: {run.info.run_id}")
    mlflow.end_run()
    

Output

    
    
    Active run_id: 6f252757005748708cd3aad75d1ff462
    

`mlflow.``add_trace`(_trace :
Union[[Trace](mlflow.entities.html#mlflow.entities.Trace
"mlflow.entities.Trace"), Dict[str, Any]]_, _target :
Optional[[LiveSpan](mlflow.entities.html#mlflow.entities.LiveSpan
"mlflow.entities.LiveSpan")] =
None_)[[source]](../_modules/mlflow/tracing/fluent.html#add_trace)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Add a completed trace object into another trace.

This is particularly useful when you call a remote service instrumented by
MLflow Tracing. By using this function, you can merge the trace from the
remote service into the current active local trace, so that you can see the
full trace including what happens inside the remote service call.

The following example demonstrates how to use this function to merge a trace
from a remote service to the current active trace in the function.

    
    
    @mlflow.trace(name="predict")
    def predict(input):
        # Call a remote service that returns a trace in the response
        resp = requests.get("https://your-service-endpoint", ...)
    
        # Extract the trace from the response
        trace_json = resp.json().get("trace")
    
        # Use the remote trace as a part of the current active trace.
        # It will be merged under the span "predict" and exported together when it is ended.
        mlflow.add_trace(trace_json)
    

If you have a specific target span to merge the trace under, you can pass the
target span

    
    
    def predict(input):
        # Create a local span
        span = MlflowClient().start_span(name="predict")
    
        resp = requests.get("https://your-service-endpoint", ...)
        trace_json = resp.json().get("trace")
    
        # Merge the remote trace under the span created above
        mlflow.add_trace(trace_json, target=span)
    

Parameters

    

  * **trace** â 

A [`Trace`](mlflow.entities.html#mlflow.entities.Trace
"mlflow.entities.Trace") object or a dictionary representation of the trace.
The trace **must** be already completed i.e. no further updates should be made
to it. Otherwise, this function will raise an exception.

  * **target** â 

The target span to merge the given trace.

    * If provided, the trace will be merged under the target span.

    * If not provided, the trace will be merged under the current active span.

    * If not provided and there is no active span, a new span named âRemote Trace <â¦>â will be created and the trace will be merged under it.

`mlflow.``autolog`(_log_input_examples : bool = False_, _log_model_signatures
: bool = True_, _log_models : bool = True_, _log_datasets : bool = True_,
_disable : bool = False_, _exclusive : bool = False_,
_disable_for_unsupported_versions : bool = False_, _silent : bool = False_,
_extra_tags : Optional[Dict[str, str]] = None_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#autolog)

    

Enables (or disables) and configures autologging for all supported
integrations.

The parameters are passed to any autologging integrations that support them.

See the [tracking docs](../tracking/autolog.html#automatic-logging) for a list
of supported autologging integrations.

Note that framework-specific configurations set at any point will take
precedence over any configurations set by this function. For example:

    
    
    import mlflow
    
    mlflow.autolog(log_models=False, exclusive=True)
    import sklearn
    

would enable autologging for sklearn with log_models=False and exclusive=True,
but

    
    
    import mlflow
    
    mlflow.autolog(log_models=False, exclusive=True)
    
    import sklearn
    
    mlflow.sklearn.autolog(log_models=True)
    

would enable autologging for sklearn with log_models=True and exclusive=False,
the latter resulting from the default value for exclusive in
mlflow.sklearn.autolog; other framework autolog functions (e.g.
mlflow.tensorflow.autolog) would use the configurations set by mlflow.autolog
(in this instance, log_models=False, exclusive=True), until they are
explicitly called by the user.

Parameters

    

  * **log_input_examples** â If `True`, input examples from training datasets are collected and logged along with model artifacts during training. If `False`, input examples are not logged. Note: Input examples are MLflow model attributes and are only collected if `log_models` is also `True`.

  * **log_model_signatures** â If `True`, [`ModelSignatures`](mlflow.models.html#mlflow.models.ModelSignature "mlflow.models.ModelSignature") describing model inputs and outputs are collected and logged along with model artifacts during training. If `False`, signatures are not logged. Note: Model signatures are MLflow model attributes and are only collected if `log_models` is also `True`.

  * **log_models** â If `True`, trained models are logged as MLflow model artifacts. If `False`, trained models are not logged. Input examples and model signatures, which are attributes of MLflow models, are also omitted when `log_models` is `False`.

  * **log_datasets** â If `True`, dataset information is logged to MLflow Tracking. If `False`, dataset information is not logged.

  * **disable** â If `True`, disables all supported autologging integrations. If `False`, enables all supported autologging integrations.

  * **exclusive** â If `True`, autologged content is not logged to user-created fluent runs. If `False`, autologged content is logged to the active fluent run, which may be user-created.

  * **disable_for_unsupported_versions** â If `True`, disable autologging for versions of all integration libraries that have not been tested against this version of the MLflow client or are incompatible.

  * **silent** â If `True`, suppress all event logs and warnings from MLflow during autologging setup and training execution. If `False`, show all events and warnings during autologging setup and training execution.

  * **extra_tags** â A dictionary of extra tags to set on each managed run created by autologging.

Example

    
    
    import numpy as np
    import mlflow.sklearn
    from mlflow import MlflowClient
    from sklearn.linear_model import LinearRegression
    
    
    def print_auto_logged_info(r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print(f"run_id: {r.info.run_id}")
        print(f"artifacts: {artifacts}")
        print(f"params: {r.data.params}")
        print(f"metrics: {r.data.metrics}")
        print(f"tags: {tags}")
    
    
    # prepare training data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    
    # Auto log all the parameters, metrics, and artifacts
    mlflow.autolog()
    model = LinearRegression()
    with mlflow.start_run() as run:
        model.fit(X, y)
    
    # fetch the auto logged parameters and metrics for ended run
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    

Output

    
    
    run_id: fd10a17d028c47399a55ab8741721ef7
    artifacts: ['model/MLmodel', 'model/conda.yaml', 'model/model.pkl']
    params: {'copy_X': 'True',
             'normalize': 'False',
             'fit_intercept': 'True',
             'n_jobs': 'None'}
    metrics: {'training_score': 1.0,
              'training_root_mean_squared_error': 4.440892098500626e-16,
              'training_r2_score': 1.0,
              'training_mean_absolute_error': 2.220446049250313e-16,
              'training_mean_squared_error': 1.9721522630525295e-31}
    tags: {'estimator_class': 'sklearn.linear_model._base.LinearRegression',
           'estimator_name': 'LinearRegression'}
    

`mlflow.``create_experiment`(_name : str_, _artifact_location : Optional[str]
= None_, _tags : Optional[Dict[str, Any]] = None_) ->
str[[source]](../_modules/mlflow/tracking/fluent.html#create_experiment)

    

Create an experiment.

Parameters

    

  * **name** â The experiment name, which must be a unique string.

  * **artifact_location** â The location to store run artifacts. If not provided, the server picks an appropriate default.

  * **tags** â An optional dictionary of string keys and values to set as tags on the experiment.

Returns

    

String ID of the created experiment.

Example

    
    
    import mlflow
    from pathlib import Path
    
    # Create an experiment name, which must be unique and case sensitive
    experiment_id = mlflow.create_experiment(
        "Social NLP Experiments",
        artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
        tags={"version": "v1", "priority": "P1"},
    )
    experiment = mlflow.get_experiment(experiment_id)
    print(f"Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")
    

Output

    
    
    Name: Social NLP Experiments
    Experiment_id: 1
    Artifact Location: file:///.../mlruns
    Tags: {'version': 'v1', 'priority': 'P1'}
    Lifecycle_stage: active
    Creation timestamp: 1662004217511
    

`mlflow.``delete_experiment`(_experiment_id : str_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#delete_experiment)

    

Delete an experiment from the backend store.

Parameters

    

**experiment_id** â The string-ified experiment ID returned from
`create_experiment`.

Example

    
    
    import mlflow
    
    experiment_id = mlflow.create_experiment("New Experiment")
    mlflow.delete_experiment(experiment_id)
    
    # Examine the deleted experiment details.
    experiment = mlflow.get_experiment(experiment_id)
    print(f"Name: {experiment.name}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Last Updated timestamp: {experiment.last_update_time}")
    

Output

    
    
    Name: New Experiment
    Artifact Location: file:///.../mlruns/2
    Lifecycle_stage: deleted
    Last Updated timestamp: 1662004217511
    

`mlflow.``delete_run`(_run_id : str_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#delete_run)

    

Deletes a run with the given ID.

Parameters

    

**run_id** â Unique identifier for the run to delete.

Example

    
    
    import mlflow
    
    with mlflow.start_run() as run:
        mlflow.log_param("p", 0)
    
    run_id = run.info.run_id
    mlflow.delete_run(run_id)
    
    lifecycle_stage = mlflow.get_run(run_id).info.lifecycle_stage
    print(f"run_id: {run_id}; lifecycle_stage: {lifecycle_stage}")
    

Output

    
    
    run_id: 45f4af3e6fd349e58579b27fcb0b8277; lifecycle_stage: deleted
    

`mlflow.``delete_tag`(_key : str_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#delete_tag)

    

Delete a tag from a run. This is irreversible. If no run is active, this
method will create a new active run.

Parameters

    

**key** â Name of the tag

Example

    
    
    import mlflow
    
    tags = {"engineering": "ML Platform", "engineering_remote": "ML Platform"}
    
    with mlflow.start_run() as run:
        mlflow.set_tags(tags)
    
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.delete_tag("engineering_remote")
    

`mlflow.``disable_system_metrics_logging`()[[source]](../_modules/mlflow/system_metrics.html#disable_system_metrics_logging)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Disable system metrics logging globally.

> Calling this function will disable system metrics logging globally, but
> users can still opt in system metrics logging for individual runs by
> mlflow.start_run(log_system_metrics=True).

`mlflow.``doctor`(_mask_envs
=False_)[[source]](../_modules/mlflow/utils/doctor.html#doctor)

    

Prints out useful information for debugging issues with MLflow.

Parameters

    

**mask_envs** â If True, mask the MLflow environment variable values (e.g.
âMLFLOW_ENV_VARâ: â***â) in the output to prevent leaking sensitive
information.

Warning

  * This API should only be used for debugging purposes.

  * The output may contain sensitive information such as a database URI containing a password.

Example

    
    
    import mlflow
    
    with mlflow.start_run():
        mlflow.doctor()
    

Output

    
    
    System information: Linux #58~20.04.1-Ubuntu SMP Thu Oct 13 13:09:46 UTC 2022
    Python version: 3.8.13
    MLflow version: 2.0.1
    MLflow module location: /usr/local/lib/python3.8/site-packages/mlflow/__init__.py
    Tracking URI: sqlite:///mlflow.db
    Registry URI: sqlite:///mlflow.db
    MLflow environment variables:
      MLFLOW_TRACKING_URI: sqlite:///mlflow.db
    MLflow dependencies:
      Flask: 2.2.2
      Jinja2: 3.0.3
      alembic: 1.8.1
      click: 8.1.3
      cloudpickle: 2.2.0
      databricks-cli: 0.17.4.dev0
      docker: 6.0.0
      entrypoints: 0.4
      gitpython: 3.1.29
      gunicorn: 20.1.0
      importlib-metadata: 5.0.0
      markdown: 3.4.1
      matplotlib: 3.6.1
      numpy: 1.23.4
      packaging: 21.3
      pandas: 1.5.1
      protobuf: 3.19.6
      pyarrow: 9.0.0
      pytz: 2022.6
      pyyaml: 6.0
      querystring-parser: 1.2.4
      requests: 2.28.1
      scikit-learn: 1.1.3
      scipy: 1.9.3
      shap: 0.41.0
      sqlalchemy: 1.4.42
      sqlparse: 0.4.3
    

`mlflow.``enable_system_metrics_logging`()[[source]](../_modules/mlflow/system_metrics.html#enable_system_metrics_logging)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Enable system metrics logging globally.

> Calling this function will enable system metrics logging globally, but users
> can still opt out system metrics logging for individual runs by
> mlflow.start_run(log_system_metrics=False).

`mlflow.``end_run`(_status : str = 'FINISHED'_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#end_run)

    

End an active MLflow run (if there is one).

Example

    
    
    import mlflow
    
    # Start run and get status
    mlflow.start_run()
    run = mlflow.active_run()
    print(f"run_id: {run.info.run_id}; status: {run.info.status}")
    
    # End run and get status
    mlflow.end_run()
    run = mlflow.get_run(run.info.run_id)
    print(f"run_id: {run.info.run_id}; status: {run.info.status}")
    print("--")
    
    # Check for any active runs
    print(f"Active run: {mlflow.active_run()}")
    

Output

    
    
    run_id: b47ee4563368419880b44ad8535f6371; status: RUNNING
    run_id: b47ee4563368419880b44ad8535f6371; status: FINISHED
    --
    Active run: None
    

`mlflow.``evaluate`(_model =None_, _data =None_, _*_ , _model_type =None_,
_targets =None_, _predictions =None_, _dataset_path =None_, _feature_names
=None_, _evaluators =None_, _evaluator_config =None_, _custom_metrics =None_,
_extra_metrics =None_, _custom_artifacts =None_, _validation_thresholds
=None_, _baseline_model =None_, _env_manager ='local'_, _model_config =None_,
_baseline_config =None_, _inference_params
=None_)[[source]](../_modules/mlflow/models/evaluation/base.html#evaluate)

    

Evaluate the model performance on given data and selected metrics.

This function evaluates a PyFunc model or custom callable on the specified
dataset using specified `evaluators`, and logs resulting metrics & artifacts
to MLflow tracking server. Users can also skip setting `model` and put the
model outputs in `data` directly for evaluation. For detailed information,
please read [the Model Evaluation documentation](../models.html#model-
evaluation).

Default Evaluator behavior:

    

  * The default evaluator, which can be invoked with `evaluators="default"` or `evaluators=None`, supports model types listed below. For each pre-defined model type, the default evaluator evaluates your model on a selected set of metrics and generate artifacts like plots. Please find more details below.

  * For both the `"regressor"` and `"classifier"` model types, the default evaluator generates model summary plots and feature importance plots using [SHAP](https://shap.readthedocs.io/en/latest/index.html).

  * For regressor models, the default evaluator additionally logs:
    
    * **metrics** : example_count, mean_absolute_error, mean_squared_error, root_mean_squared_error, sum_on_target, mean_on_target, r2_score, max_error, mean_absolute_percentage_error.

  * For binary classifiers, the default evaluator additionally logs:
    
    * **metrics** : true_negatives, false_positives, false_negatives, true_positives, recall, precision, f1_score, accuracy_score, example_count, log_loss, roc_auc, precision_recall_auc.

    * **artifacts** : lift curve plot, precision-recall plot, ROC plot.

  * For multiclass classifiers, the default evaluator additionally logs:
    
    * **metrics** : accuracy_score, example_count, f1_score_micro, f1_score_macro, log_loss

    * **artifacts** : A CSV file for âper_class_metricsâ (per-class metrics includes true_negatives/false_positives/false_negatives/true_positives/recall/precision/roc_auc, precision_recall_auc), precision-recall merged curves plot, ROC merged curves plot.

  * For question-answering models, the default evaluator logs:
    
    * **metrics** : `exact_match`, `token_count`, [toxicity](https://huggingface.co/spaces/evaluate-measurement/toxicity) (requires [evaluate](https://pypi.org/project/evaluate), [torch](https://pytorch.org/get-started/locally/), [flesch_kincaid_grade_level](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level) (requires [textstat](https://pypi.org/project/textstat)) and [ari_grade_level](https://en.wikipedia.org/wiki/Automated_readability_index).

    * **artifacts** : A JSON file containing the inputs, outputs, targets (if the `targets` argument is supplied), and per-row metrics of the model in tabular format.

  * For text-summarization models, the default evaluator logs:
    
    * **metrics** : `token_count`, [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) (requires [evaluate](https://pypi.org/project/evaluate), [nltk](https://pypi.org/project/nltk), and [rouge_score](https://pypi.org/project/rouge-score) to be installed), [toxicity](https://huggingface.co/spaces/evaluate-measurement/toxicity) (requires [evaluate](https://pypi.org/project/evaluate), [torch](https://pytorch.org/get-started/locally/), [transformers](https://huggingface.co/docs/transformers/installation)), [ari_grade_level](https://en.wikipedia.org/wiki/Automated_readability_index) (requires [textstat](https://pypi.org/project/textstat)), [flesch_kincaid_grade_level](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level) (requires [textstat](https://pypi.org/project/textstat)).

    * **artifacts** : A JSON file containing the inputs, outputs, targets (if the `targets` argument is supplied), and per-row metrics of the model in the tabular format.

  * For text models, the default evaluator logs:
    
    * **metrics** : `token_count`, [toxicity](https://huggingface.co/spaces/evaluate-measurement/toxicity) (requires [evaluate](https://pypi.org/project/evaluate), [torch](https://pytorch.org/get-started/locally/), [transformers](https://huggingface.co/docs/transformers/installation)), [ari_grade_level](https://en.wikipedia.org/wiki/Automated_readability_index) (requires [textstat](https://pypi.org/project/textstat)), [flesch_kincaid_grade_level](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level) (requires [textstat](https://pypi.org/project/textstat)).

    * **artifacts** : A JSON file containing the inputs, outputs, targets (if the `targets` argument is supplied), and per-row metrics of the model in tabular format.

  * For retriever models, the default evaluator logs:
    
    * **metrics** : [`precision_at_k(k)`](mlflow.metrics.html#mlflow.metrics.precision_at_k "mlflow.metrics.precision_at_k"), [`recall_at_k(k)`](mlflow.metrics.html#mlflow.metrics.recall_at_k "mlflow.metrics.recall_at_k") and [`ndcg_at_k(k)`](mlflow.metrics.html#mlflow.metrics.ndcg_at_k "mlflow.metrics.ndcg_at_k") \- all have a default value of `retriever_k` = 3.

    * **artifacts** : A JSON file containing the inputs, outputs, targets, and per-row metrics of the model in tabular format.

  * For sklearn models, the default evaluator additionally logs the modelâs evaluation criterion (e.g. mean accuracy for a classifier) computed by model.score method.

  * The metrics/artifacts listed above are logged to the active MLflow run. If no active run exists, a new MLflow run is created for logging these metrics and artifacts. Note that no metrics/artifacts are logged for the `baseline_model`.

  * Additionally, information about the specified dataset - hash, name (if specified), path (if specified), and the UUID of the model that evaluated it - is logged to the `mlflow.datasets` tag.

  * The available `evaluator_config` options for the default evaluator include:
    
    * **log_model_explainability** : A boolean value specifying whether or not to log model explainability insights, default value is True.

    * **explainability_algorithm** : A string to specify the SHAP Explainer algorithm for model explainability. Supported algorithm includes: âexactâ, âpermutationâ, âpartitionâ, âkernelâ. If not set, `shap.Explainer` is used with the âautoâ algorithm, which chooses the best Explainer based on the model.

    * **explainability_nsamples** : The number of sample rows to use for computing model explainability insights. Default value is 2000.

    * **explainability_kernel_link** : The kernel link function used by shap kernal explainer. Available values are âidentityâ and âlogitâ. Default value is âidentityâ.

    * **max_classes_for_multiclass_roc_pr** : For multiclass classification tasks, the maximum number of classes for which to log the per-class ROC curve and Precision-Recall curve. If the number of classes is larger than the configured maximum, these curves are not logged.

    * **metric_prefix** : An optional prefix to prepend to the name of each metric and artifact produced during evaluation.

    * **log_metrics_with_dataset_info** : A boolean value specifying whether or not to include information about the evaluation dataset in the name of each metric logged to MLflow Tracking during evaluation, default value is True.

    * **pos_label** : If specified, the positive label to use when computing classification metrics such as precision, recall, f1, etc. for binary classification models. For multiclass classification and regression models, this parameter will be ignored.

    * **average** : The averaging method to use when computing classification metrics such as precision, recall, f1, etc. for multiclass classification models (default: `'weighted'`). For binary classification and regression models, this parameter will be ignored.

    * **sample_weights** : Weights for each sample to apply when computing model performance metrics.

    * **col_mapping** : A dictionary mapping column names in the input dataset or output predictions to column names used when invoking the evaluation functions.

    * **retriever_k** : A parameter used when `model_type="retriever"` as the number of top-ranked retrieved documents to use when computing the built-in metric [`precision_at_k(k)`](mlflow.metrics.html#mlflow.metrics.precision_at_k "mlflow.metrics.precision_at_k"), [`recall_at_k(k)`](mlflow.metrics.html#mlflow.metrics.recall_at_k "mlflow.metrics.recall_at_k") and [`ndcg_at_k(k)`](mlflow.metrics.html#mlflow.metrics.ndcg_at_k "mlflow.metrics.ndcg_at_k"). Default value is 3. For all other model types, this parameter will be ignored.

  * Limitations of evaluation dataset:
    
    * For classification tasks, dataset labels are used to infer the total number of classes.

    * For binary classification tasks, the negative label value must be 0 or -1 or False, and the positive label value must be 1 or True.

  * Limitations of metrics/artifacts computation:
    
    * For classification tasks, some metric and artifact computations require the model to output class probabilities. Currently, for scikit-learn models, the default evaluator calls the `predict_proba` method on the underlying model to obtain probabilities. For other model types, the default evaluator does not compute metrics/artifacts that require probability outputs.

  * Limitations of default evaluator logging model explainability insights:
    
    * The `shap.Explainer` `auto` algorithm uses the `Linear` explainer for linear models and the `Tree` explainer for tree models. Because SHAPâs `Linear` and `Tree` explainers do not support multi-class classification, the default evaluator falls back to using the `Exact` or `Permutation` explainers for multi-class classification tasks.

    * Logging model explainability insights is not currently supported for PySpark models.

    * The evaluation dataset label values must be numeric or boolean, all feature values must be numeric, and each feature column must only contain scalar values.

  * Limitations when environment restoration is enabled:
    
    * When environment restoration is enabled for the evaluated model (i.e. a non-local `env_manager` is specified), the model is loaded as a client that invokes a MLflow Model Scoring Server process in an independent Python environment with the modelâs training time dependencies installed. As such, methods like `predict_proba` (for probability outputs) or `score` (computes the evaluation criterian for sklearn models) of the model become inaccessible and the default evaluator does not compute metrics or artifacts that require those methods.

    * Because the model is an MLflow Model Server process, SHAP explanations are slower to compute. As such, model explainaibility is disabled when a non-local `env_manager` specified, unless the `evaluator_config` option **log_model_explainability** is explicitly set to `True`.

Parameters

    

  * **model** â 

Optional. If specified, it should be one of the following:

    * A pyfunc model instance

    * A URI referring to a pyfunc model

    * A URI referring to an MLflow Deployments endpoint e.g. `"endpoints:/my-chat"`

    * A callable function: This function should be able to take in model input and return predictions. It should follow the signature of the [`predict`](mlflow.pyfunc.html#mlflow.pyfunc.PyFuncModel.predict "mlflow.pyfunc.PyFuncModel.predict") method. Hereâs an example of a valid function:
        
                model = mlflow.pyfunc.load_model(model_uri)
        
        
        def fn(model_input):
            return model.predict(model_input)
        

If omitted, it indicates a static dataset will be used for evaluation instead
of a model. In this case, the `data` argument must be a Pandas DataFrame or an
mlflow PandasDataset that contains model outputs, and the `predictions`
argument must be the name of the column in `data` that contains model outputs.

  * **data** â 

One of the following:

    * A numpy array or list of evaluation features, excluding labels.

    * A Pandas DataFrame containing evaluation features, labels, and optionally model
    

outputs. Model outputs are required to be provided when model is unspecified.
If `feature_names` argument not specified, all columns except for the label
column and model_output column are regarded as feature columns. Otherwise,
only column names present in `feature_names` are regarded as feature columns.

    * A Spark DataFrame containing evaluation features and labels. If
    

`feature_names` argument not specified, all columns except for the label
column are regarded as feature columns. Otherwise, only column names present
in `feature_names` are regarded as feature columns. Only the first 10000 rows
in the Spark DataFrame will be used as evaluation data.

    * A [`mlflow.data.dataset.Dataset`](mlflow.data.html#mlflow.data.dataset.Dataset "mlflow.data.dataset.Dataset") instance containing evaluation
    

features, labels, and optionally model outputs. Model outputs are only
supported with a PandasDataset. Model outputs are required when model is
unspecified, and should be specified via the `predictions` prerty of the
PandasDataset.

  * **targets** â If `data` is a numpy array or list, a numpy array or list of evaluation labels. If `data` is a DataFrame, the string name of a column from `data` that contains evaluation labels. Required for classifier and regressor models, but optional for question-answering, text-summarization, and text models. If `data` is a [`mlflow.data.dataset.Dataset`](mlflow.data.html#mlflow.data.dataset.Dataset "mlflow.data.dataset.Dataset") that defines targets, then `targets` is optional.

  * **predictions** â 

Optional. The name of the column that contains model outputs.

    * When `model` is specified and outputs multiple columns, `predictions` can be used to specify the name of the column that will be used to store model outputs for evaluation.

    * When `model` is not specified and `data` is a pandas dataframe, `predictions` can be used to specify the name of the column in `data` that contains model outputs.

Example usage of predictions

    
        # Evaluate a model that outputs multiple columns
    data = pd.DataFrame({"question": ["foo"]})
    
    
    def model(inputs):
        return pd.DataFrame({"answer": ["bar"], "source": ["baz"]})
    
    
    results = evaluate(model=model, data=data, predictions="answer", ...)
    
    # Evaluate a static dataset
    data = pd.DataFrame({"question": ["foo"], "answer": ["bar"], "source": ["baz"]})
    results = evaluate(data=data, predictions="answer", ...)
    

  * **model_type** â 

(Optional) A string describing the model type. The default evaluator supports
the following model types:

    * `'classifier'`

    * `'regressor'`

    * `'question-answering'`

    * `'text-summarization'`

    * `'text'`

    * `'retriever'`

If no `model_type` is specified, then you must provide a a list of metrics to
compute via the `extra_metrics` param.

Note

`'question-answering'`, `'text-summarization'`, `'text'`, and `'retriever'`
are experimental and may be changed or removed in a future release.

  * **inference_params** â (Optional) A dictionary of inference parameters to be passed to the model when making predictions, such as `{"max_tokens": 100}`. This is only used when the `model` is an MLflow Deployments endpoint URI e.g. `"endpoints:/my-chat"`

  * **dataset_path** â (Optional) The path where the data is stored. Must not contain double quotes (`â`). If specified, the path is logged to the `mlflow.datasets` tag for lineage tracking purposes.

  * **feature_names** â (Optional) A list. If the `data` argument is a numpy array or list, `feature_names` is a list of the feature names for each feature. If `feature_names=None`, then the `feature_names` are generated using the format `feature_{feature_index}`. If the `data` argument is a Pandas DataFrame or a Spark DataFrame, `feature_names` is a list of the names of the feature columns in the DataFrame. If `feature_names=None`, then all columns except the label column and the predictions column are regarded as feature columns.

  * **evaluators** â The name of the evaluator to use for model evaluation, or a list of evaluator names. If unspecified, all evaluators capable of evaluating the specified model on the specified dataset are used. The default evaluator can be referred to by the name `"default"`. To see all available evaluators, call [`mlflow.models.list_evaluators()`](mlflow.models.html#mlflow.models.list_evaluators "mlflow.models.list_evaluators").

  * **evaluator_config** â A dictionary of additional configurations to supply to the evaluator. If multiple evaluators are specified, each configuration should be supplied as a nested dictionary whose key is the evaluator name.

  * **extra_metrics** â 

(Optional) A list of
[`EvaluationMetric`](mlflow.models.html#mlflow.models.EvaluationMetric
"mlflow.models.EvaluationMetric") objects. These metrics are computed in
addition to the default metrics associated with pre-defined model_type, and
setting model_type=None will only compute the metrics specified in
extra_metrics. See the mlflow.metrics module for more information about the
builtin metrics and how to define extra metrics.

Example usage of extra metrics

    
        import mlflow
    import numpy as np
    
    
    def root_mean_squared_error(eval_df, _builtin_metrics):
        return np.sqrt((np.abs(eval_df["prediction"] - eval_df["target"]) ** 2).mean)
    
    
    rmse_metric = mlflow.models.make_metric(
        eval_fn=root_mean_squared_error,
        greater_is_better=False,
    )
    mlflow.evaluate(..., extra_metrics=[rmse_metric])
    

  * **custom_artifacts** â 

(Optional) A list of custom artifact functions with the following signature:

    
        def custom_artifact(
        eval_df: Union[pandas.Dataframe, pyspark.sql.DataFrame],
        builtin_metrics: Dict[str, float],
        artifacts_dir: str,
    ) -> Dict[str, Any]:
        """
        Args:
            eval_df:
                A Pandas or Spark DataFrame containing ``prediction`` and ``target``
                column.  The ``prediction`` column contains the predictions made by the
                model.  The ``target`` column contains the corresponding labels to the
                predictions made on that row.
            builtin_metrics:
                A dictionary containing the metrics calculated by the default evaluator.
                The keys are the names of the metrics and the values are the scalar
                values of the metrics. Refer to the DefaultEvaluator behavior section
                for what metrics will be returned based on the type of model (i.e.
                classifier or regressor).
            artifacts_dir:
                A temporary directory path that can be used by the custom artifacts
                function to temporarily store produced artifacts. The directory will be
                deleted after the artifacts are logged.
    
        Returns:
            A dictionary that maps artifact names to artifact objects
            (e.g. a Matplotlib Figure) or to artifact paths within ``artifacts_dir``.
        """
        ...
    

Object types that artifacts can be represented as:

>     * A string uri representing the file path to the artifact. MLflow will
> infer the type of the artifact based on the file extension.
>
>     * A string representation of a JSON object. This will be saved as a
> .json artifact.
>
>     * Pandas DataFrame. This will be resolved as a CSV artifact.
>
>     * Numpy array. This will be saved as a .npy artifact.
>
>     * Matplotlib Figure. This will be saved as an image artifact. Note that
> `matplotlib.pyplot.savefig` is called behind the scene with default
> configurations. To customize, either save the figure with the desired
> configurations and return its file path or define customizations through
> environment variables in `matplotlib.rcParams`.
>
>     * Other objects will be attempted to be pickled with the default
> protocol.

Example usage of custom artifacts

    
        import mlflow
    import matplotlib.pyplot as plt
    
    
    def scatter_plot(eval_df, builtin_metrics, artifacts_dir):
        plt.scatter(eval_df["prediction"], eval_df["target"])
        plt.xlabel("Targets")
        plt.ylabel("Predictions")
        plt.title("Targets vs. Predictions")
        plt.savefig(os.path.join(artifacts_dir, "example.png"))
        plt.close()
        return {"pred_target_scatter": os.path.join(artifacts_dir, "example.png")}
    
    
    def pred_sample(eval_df, _builtin_metrics, _artifacts_dir):
        return {"pred_sample": pred_sample.head(10)}
    
    
    mlflow.evaluate(..., custom_artifacts=[scatter_plot, pred_sample])
    

  * **validation_thresholds** â 

(Optional) A dictionary of metric name to
[`mlflow.models.MetricThreshold`](mlflow.models.html#mlflow.models.MetricThreshold
"mlflow.models.MetricThreshold") used for model validation. Each metric name
must either be the name of a builtin metric or the name of a metric defined in
the `extra_metrics` parameter.

Example of Model Validation

    
        from mlflow.models import MetricThreshold
    
    thresholds = {
        "accuracy_score": MetricThreshold(
            # accuracy should be >=0.8
            threshold=0.8,
            # accuracy should be at least 5 percent greater than baseline model accuracy
            min_absolute_change=0.05,
            # accuracy should be at least 0.05 greater than baseline model accuracy
            min_relative_change=0.05,
            greater_is_better=True,
        ),
    }
    
    with mlflow.start_run():
        mlflow.evaluate(
            model=your_candidate_model,
            data,
            targets,
            model_type,
            dataset_name,
            evaluators,
            validation_thresholds=thresholds,
            baseline_model=your_baseline_model,
        )
    

See [the Model Validation documentation](../models.html#model-validation) for
more details.

  * **baseline_model** â (Optional) A string URI referring to an MLflow model with the pyfunc flavor. If specified, the candidate `model` is compared to this baseline for model validation purposes.

  * **env_manager** â 

Specify an environment manager to load the candidate `model` and
`baseline_model` in isolated Python environments and restore their
dependencies. Default value is `local`, and the following values are
supported:

    * `virtualenv`: (Recommended) Use virtualenv to restore the python environment that was used to train the model.

    * `conda`: Use Conda to restore the software environment that was used to train the model.

    * `local`: Use the current Python environment for model inference, which may differ from the environment used to train the model and may lead to errors or invalid predictions.

  * **model_config** â the model configuration to use for loading the model with pyfunc. Inspect the modelâs pyfunc flavor to know which keys are supported for your specific model. If not indicated, the default model configuration from the model is used (if any).

  * **baseline_config** â the model configuration to use for loading the baseline model. If not indicated, the default model configuration from the baseline model is used (if any).

Returns

    

An
[`mlflow.models.EvaluationResult`](mlflow.models.html#mlflow.models.EvaluationResult
"mlflow.models.EvaluationResult") instance containing metrics of candidate
model and baseline model, and artifacts of candidate model.

`mlflow.``flush_artifact_async_logging`() ->
None[[source]](../_modules/mlflow/tracking/fluent.html#flush_artifact_async_logging)

    

Flush all pending artifact async logging.

`mlflow.``flush_async_logging`() ->
None[[source]](../_modules/mlflow/tracking/fluent.html#flush_async_logging)

    

Flush all pending async logging.

`mlflow.``flush_trace_async_logging`(_terminate =False_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#flush_trace_async_logging)

    

Flush all pending trace async logging.

Parameters

    

**terminate** â If True, shut down the logging threads after flushing.

`mlflow.``get_artifact_uri`(_artifact_path : Optional[str] = None_) ->
str[[source]](../_modules/mlflow/tracking/fluent.html#get_artifact_uri)

    

Get the absolute URI of the specified artifact in the currently active run.

If path is not specified, the artifact root URI of the currently active run
will be returned; calls to `log_artifact` and `log_artifacts` write
artifact(s) to subdirectories of the artifact root URI.

If no run is active, this method will create a new active run.

Parameters

    

**artifact_path** â The run-relative artifact path for which to obtain an
absolute URI. For example, âpath/to/artifactâ. If unspecified, the
artifact root URI for the currently active run will be returned.

Returns

    

An _absolute_ URI referring to the specified artifact or the currently active
runâs artifact root. For example, if an artifact path is provided and the
currently active run uses an S3-backed store, this may be a uri of the form
`s3://<bucket_name>/path/to/artifact/root/path/to/artifact`. If an artifact
path is not provided and the currently active run uses an S3-backed store,
this may be a URI of the form `s3://<bucket_name>/path/to/artifact/root`.

Example

    
    
    import mlflow
    
    features = "rooms, zipcode, median_price, school_rating, transport"
    with open("features.txt", "w") as f:
        f.write(features)
    
    # Log the artifact in a directory "features" under the root artifact_uri/features
    with mlflow.start_run():
        mlflow.log_artifact("features.txt", artifact_path="features")
    
        # Fetch the artifact uri root directory
        artifact_uri = mlflow.get_artifact_uri()
        print(f"Artifact uri: {artifact_uri}")
    
        # Fetch a specific artifact uri
        artifact_uri = mlflow.get_artifact_uri(artifact_path="features/features.txt")
        print(f"Artifact uri: {artifact_uri}")
    

Output

    
    
    Artifact uri: file:///.../0/a46a80f1c9644bd8f4e5dd5553fffce/artifacts
    Artifact uri: file:///.../0/a46a80f1c9644bd8f4e5dd5553fffce/artifacts/features/features.txt
    

`mlflow.``get_experiment`(_experiment_id : str_) ->
[Experiment](mlflow.entities.html#mlflow.entities.Experiment
"mlflow.entities.Experiment")[[source]](../_modules/mlflow/tracking/fluent.html#get_experiment)

    

Retrieve an experiment by experiment_id from the backend store

Parameters

    

**experiment_id** â The string-ified experiment ID returned from
`create_experiment`.

Returns

    

[`mlflow.entities.Experiment`](mlflow.entities.html#mlflow.entities.Experiment
"mlflow.entities.Experiment")

Example

    
    
    import mlflow
    
    experiment = mlflow.get_experiment("0")
    print(f"Name: {experiment.name}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")
    

Output

    
    
    Name: Default
    Artifact Location: file:///.../mlruns/0
    Tags: {}
    Lifecycle_stage: active
    Creation timestamp: 1662004217511
    

`mlflow.``get_experiment_by_name`(_name : str_) ->
Optional[[Experiment](mlflow.entities.html#mlflow.entities.Experiment
"mlflow.entities.Experiment")][[source]](../_modules/mlflow/tracking/fluent.html#get_experiment_by_name)

    

Retrieve an experiment by experiment name from the backend store

Parameters

    

**name** â The case sensitive experiment name.

Returns

    

An instance of
[`mlflow.entities.Experiment`](mlflow.entities.html#mlflow.entities.Experiment
"mlflow.entities.Experiment") if an experiment with the specified name exists,
otherwise None.

Example

    
    
    import mlflow
    
    # Case sensitive name
    experiment = mlflow.get_experiment_by_name("Default")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")
    

Output

    
    
    Experiment_id: 0
    Artifact Location: file:///.../mlruns/0
    Tags: {}
    Lifecycle_stage: active
    Creation timestamp: 1662004217511
    

`mlflow.``get_parent_run`(_run_id : str_) ->
Optional[[Run](mlflow.entities.html#mlflow.entities.Run
"mlflow.entities.Run")][[source]](../_modules/mlflow/tracking/fluent.html#get_parent_run)

    

Gets the parent run for the given run id if one exists.

Parameters

    

**run_id** â Unique identifier for the child run.

Returns

    

A single [`mlflow.entities.Run`](mlflow.entities.html#mlflow.entities.Run
"mlflow.entities.Run") object, if the parent run exists. Otherwise, returns
None.

Example

    
    
    import mlflow
    
    # Create nested runs
    with mlflow.start_run():
        with mlflow.start_run(nested=True) as child_run:
            child_run_id = child_run.info.run_id
    
    parent_run = mlflow.get_parent_run(child_run_id)
    
    print(f"child_run_id: {child_run_id}")
    print(f"parent_run_id: {parent_run.info.run_id}")
    

Output

    
    
    child_run_id: 7d175204675e40328e46d9a6a5a7ee6a
    parent_run_id: 8979459433a24a52ab3be87a229a9cdf
    

`mlflow.``get_registry_uri`() ->
str[[source]](../_modules/mlflow/tracking/_model_registry/utils.html#get_registry_uri)

    

Get the current registry URI. If none has been specified, defaults to the
tracking URI.

Returns

    

The registry URI.

    
    
    # Get the current model registry uri
    mr_uri = mlflow.get_registry_uri()
    print(f"Current model registry uri: {mr_uri}")
    
    # Get the current tracking uri
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")
    
    # They should be the same
    assert mr_uri == tracking_uri
    
    
    
    Current model registry uri: file:///.../mlruns
    Current tracking uri: file:///.../mlruns
    

`mlflow.``get_run`(_run_id : str_) ->
[Run](mlflow.entities.html#mlflow.entities.Run
"mlflow.entities.Run")[[source]](../_modules/mlflow/tracking/fluent.html#get_run)

    

Fetch the run from backend store. The resulting Run contains a collection of
run metadata â RunInfo as well as a collection of run parameters, tags, and
metrics â RunData. It also contains a collection of run inputs
(experimental), including information about datasets used by the run â
RunInputs. In the case where multiple metrics with the same key are logged for
the run, the RunData contains the most recently logged value at the largest
step for each metric.

Parameters

    

**run_id** â Unique identifier for the run.

Returns

    

A single Run object, if the run exists. Otherwise, raises an exception.

Example

    
    
    import mlflow
    
    with mlflow.start_run() as run:
        mlflow.log_param("p", 0)
    run_id = run.info.run_id
    print(
        f"run_id: {run_id}; lifecycle_stage: {mlflow.get_run(run_id).info.lifecycle_stage}"
    )
    

Output

    
    
    run_id: 7472befefc754e388e8e922824a0cca5; lifecycle_stage: active
    

`mlflow.``get_tracking_uri`() ->
str[[source]](../_modules/mlflow/tracking/_tracking_service/utils.html#get_tracking_uri)

    

Get the current tracking URI. This may not correspond to the tracking URI of
the currently active run, since the tracking URI can be updated via
`set_tracking_uri`.

Returns

    

The tracking URI.

    
    
    import mlflow
    
    # Get the current tracking uri
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")
    
    
    
    Current tracking uri: file:///.../mlruns
    

`mlflow.``is_tracking_uri_set`()[[source]](../_modules/mlflow/tracking/_tracking_service/utils.html#is_tracking_uri_set)

    

Returns True if the tracking URI has been set, False otherwise.

`mlflow.``last_active_run`() ->
Optional[[Run](mlflow.entities.html#mlflow.entities.Run
"mlflow.entities.Run")][[source]](../_modules/mlflow/tracking/fluent.html#last_active_run)

    

Gets the most recent active run.

Examples:

To retrieve the most recent autologged run:

    
    
    import mlflow
    
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    
    mlflow.autolog()
    
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
    
    # Create and train models.
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)
    
    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    autolog_run = mlflow.last_active_run()
    

To get the most recently active run that ended:

    
    
    import mlflow
    
    mlflow.start_run()
    mlflow.end_run()
    run = mlflow.last_active_run()
    

To retrieve the currently active run:

    
    
    import mlflow
    
    mlflow.start_run()
    run = mlflow.last_active_run()
    mlflow.end_run()
    

Returns

    

The active run (this is equivalent to `mlflow.active_run()`) if one exists.
Otherwise, the last run started from the current Python process that reached a
terminal status (i.e. FINISHED, FAILED, or KILLED).

`mlflow.``load_table`(_artifact_file : str_, _run_ids : Optional[List[str]] =
None_, _extra_columns : Optional[List[str]] = None_) ->
pandas.DataFrame[[source]](../_modules/mlflow/tracking/fluent.html#load_table)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Load a table from MLflow Tracking as a pandas.DataFrame. The table is loaded
from the specified artifact_file in the specified run_ids. The extra_columns
are columns that are not in the table but are augmented with run information
and added to the DataFrame.

Parameters

    

  * **artifact_file** â The run-relative artifact file path in posixpath format to which table to load (e.g. âdir/file.jsonâ).

  * **run_ids** â Optional list of run_ids to load the table from. If no run_ids are specified, the table is loaded from all runs in the current experiment.

  * **extra_columns** â Optional list of extra columns to add to the returned DataFrame For example, if extra_columns=[ârun_idâ], then the returned DataFrame will have a column named run_id.

Returns

    

pandas.DataFrame containing the loaded table if the artifact exists or else
throw a MlflowException.

Example with passing run_ids

    
    
    import mlflow
    
    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }
    
    with mlflow.start_run() as run:
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file="qabot_eval_results.json")
        run_id = run.info.run_id
    
    loaded_table = mlflow.load_table(
        artifact_file="qabot_eval_results.json",
        run_ids=[run_id],
        # Append a column containing the associated run ID for each row
        extra_columns=["run_id"],
    )
    

Example with passing no run_ids

    
    
    # Loads the table with the specified name for all runs in the given
    # experiment and joins them together
    import mlflow
    
    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }
    
    with mlflow.start_run():
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file="qabot_eval_results.json")
    
    loaded_table = mlflow.load_table(
        "qabot_eval_results.json",
        # Append the run ID and the parent run ID to the table
        extra_columns=["run_id"],
    )
    

`mlflow.``log_artifact`(_local_path : str_, _artifact_path : Optional[str] =
None_, _run_id : Optional[str] = None_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#log_artifact)

    

Log a local file or directory as an artifact of the currently active run. If
no run is active, this method will create a new active run.

Parameters

    

  * **local_path** â Path to the file to write.

  * **artifact_path** â If provided, the directory in `artifact_uri` to write to.

  * **run_id** â If specified, log the artifact to the specified run. If not specified, log the artifact to the currently active run.

Example

    
    
    import tempfile
    from pathlib import Path
    
    import mlflow
    
    # Create a features.txt artifact file
    features = "rooms, zipcode, median_price, school_rating, transport"
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir, "features.txt")
        path.write_text(features)
        # With artifact_path=None write features.txt under
        # root artifact_uri/artifacts directory
        with mlflow.start_run():
            mlflow.log_artifact(path)
    

`mlflow.``log_artifacts`(_local_dir : str_, _artifact_path : Optional[str] =
None_, _run_id : Optional[str] = None_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#log_artifacts)

    

Log all the contents of a local directory as artifacts of the run. If no run
is active, this method will create a new active run.

Parameters

    

  * **local_dir** â Path to the directory of files to write.

  * **artifact_path** â If provided, the directory in `artifact_uri` to write to.

  * **run_id** â If specified, log the artifacts to the specified run. If not specified, log the artifacts to the currently active run.

Example

    
    
    import json
    import tempfile
    from pathlib import Path
    
    import mlflow
    
    # Create some files to preserve as artifacts
    features = "rooms, zipcode, median_price, school_rating, transport"
    data = {"state": "TX", "Available": 25, "Type": "Detached"}
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        with (tmp_dir / "data.json").open("w") as f:
            json.dump(data, f, indent=2)
        with (tmp_dir / "features.json").open("w") as f:
            f.write(features)
        # Write all files in `tmp_dir` to root artifact_uri/states
        with mlflow.start_run():
            mlflow.log_artifacts(tmp_dir, artifact_path="states")
    

`mlflow.``log_dict`(_dictionary : Dict[str, Any]_, _artifact_file : str_,
_run_id : Optional[str] = None_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#log_dict)

    

Log a JSON/YAML-serializable object (e.g. dict) as an artifact. The
serialization format (JSON or YAML) is automatically inferred from the
extension of artifact_file. If the file extension doesnât exist or match any
of [â.jsonâ, â.ymlâ, â.yamlâ], JSON format is used.

Parameters

    

  * **dictionary** â Dictionary to log.

  * **artifact_file** â The run-relative artifact file path in posixpath format to which the dictionary is saved (e.g. âdir/data.jsonâ).

  * **run_id** â If specified, log the dictionary to the specified run. If not specified, log the dictionary to the currently active run.

Example

    
    
    import mlflow
    
    dictionary = {"k": "v"}
    
    with mlflow.start_run():
        # Log a dictionary as a JSON file under the run's root artifact directory
        mlflow.log_dict(dictionary, "data.json")
    
        # Log a dictionary as a YAML file in a subdirectory of the run's root artifact directory
        mlflow.log_dict(dictionary, "dir/data.yml")
    
        # If the file extension doesn't exist or match any of [".json", ".yaml", ".yml"],
        # JSON format is used.
        mlflow.log_dict(dictionary, "data")
        mlflow.log_dict(dictionary, "data.txt")
    

`mlflow.``log_figure`(_figure : Union[matplotlib.figure.Figure,
plotly.graph_objects.Figure]_, _artifact_file : str_, _*_ , _save_kwargs :
Optional[Dict[str, Any]] = None_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#log_figure)

    

Log a figure as an artifact. The following figure objects are supported:

  * [matplotlib.figure.Figure](https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html)

  * [plotly.graph_objects.Figure](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html)

Parameters

    

  * **figure** â Figure to log.

  * **artifact_file** â The run-relative artifact file path in posixpath format to which the figure is saved (e.g. âdir/file.pngâ).

  * **save_kwargs** â Additional keyword arguments passed to the method that saves the figure.

Matplotlib Example

    
    
    import mlflow
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.plot([0, 1], [2, 3])
    
    with mlflow.start_run():
        mlflow.log_figure(fig, "figure.png")
    

Plotly Example

    
    
    import mlflow
    from plotly import graph_objects as go
    
    fig = go.Figure(go.Scatter(x=[0, 1], y=[2, 3]))
    
    with mlflow.start_run():
        mlflow.log_figure(fig, "figure.html")
    

`mlflow.``log_image`(_image : Union[numpy.ndarray, PIL.Image.Image,
mlflow.Image]_, _artifact_file : Optional[str] = None_, _key : Optional[str] =
None_, _step : Optional[int] = None_, _timestamp : Optional[int] = None_,
_synchronous : Optional[bool] = False_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#log_image)

    

Logs an image in MLflow, supporting two use cases:

  1. Time-stepped image logging:
    

Ideal for tracking changes or progressions through iterative processes (e.g.,
during model training phases).

     * Usage: `log_image(image, key=key, step=step, timestamp=timestamp)`

  2. Artifact file image logging:
    

Best suited for static image logging where the image is saved directly as a
file artifact.

     * Usage: `log_image(image, artifact_file)`

The following image formats are supported:

    

  * [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)

  * [PIL.Image.Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image)

  * `mlflow.Image`: An MLflow wrapper around PIL image for convenient image logging.

Numpy array support

    

  * data types:

>     * bool (useful for logging image masks)
>
>     * integer [0, 255]
>
>     * unsigned integer [0, 255]
>
>     * float [0.0, 1.0]
>
> Warning
>
>     * Out-of-range integer values will raise ValueError.
>
>     * Out-of-range float values will auto-scale with min/max and warn.

  * shape (H: height, W: width):

>     * H x W (Grayscale)
>
>     * H x W x 1 (Grayscale)
>
>     * H x W x 3 (an RGB channel order is assumed)
>
>     * H x W x 4 (an RGBA channel order is assumed)

Parameters

    

  * **run_id** â String ID of run.

  * **image** â The image object to be logged.

  * **artifact_file** â Specifies the path, in POSIX format, where the image will be stored as an artifact relative to the runâs root directory (for example, âdir/image.pngâ). This parameter is kept for backward compatibility and should not be used together with key, step, or timestamp.

  * **key** â Image name for time-stepped image logging. This string may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/).

  * **step** â Integer training step (iteration) at which the image was saved. Defaults to 0.

  * **timestamp** â Time when this image was saved. Defaults to the current system time.

  * **synchronous** â _Experimental_ If True, blocks until the image is logged successfully.

Time-stepped image logging numpy example

    
    
    import mlflow
    import numpy as np
    
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    
    with mlflow.start_run():
        mlflow.log_image(image, key="dogs", step=3)
    

Time-stepped image logging pillow example

    
    
    import mlflow
    from PIL import Image
    
    image = Image.new("RGB", (100, 100))
    
    with mlflow.start_run():
        mlflow.log_image(image, key="dogs", step=3)
    

Time-stepped image logging with mlflow.Image example

    
    
    import mlflow
    from PIL import Image
    
    # If you have a preexisting saved image
    Image.new("RGB", (100, 100)).save("image.png")
    
    image = mlflow.Image("image.png")
    with mlflow.start_run() as run:
        mlflow.log_image(run.info.run_id, image, key="dogs", step=3)
    

Legacy artifact file image logging numpy example

    
    
    import mlflow
    import numpy as np
    
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    
    with mlflow.start_run():
        mlflow.log_image(image, "image.png")
    

Legacy artifact file image logging pillow example

    
    
    import mlflow
    from PIL import Image
    
    image = Image.new("RGB", (100, 100))
    
    with mlflow.start_run():
        mlflow.log_image(image, "image.png")
    

`mlflow.``log_input`(_dataset :
[mlflow.data.dataset.Dataset](mlflow.data.html#mlflow.data.dataset.Dataset
"mlflow.data.dataset.Dataset")_, _context : Optional[str] = None_, _tags :
Optional[Dict[str, str]] = None_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#log_input)

    

Log a dataset used in the current run.

Parameters

    

  * **dataset** â [`mlflow.data.dataset.Dataset`](mlflow.data.html#mlflow.data.dataset.Dataset "mlflow.data.dataset.Dataset") object to be logged.

  * **context** â Context in which the dataset is used. For example: âtrainingâ, âtestingâ. This will be set as an input tag with key mlflow.data.context.

  * **tags** â Tags to be associated with the dataset. Dictionary of tag_key -> tag_value.

Example

    
    
    import numpy as np
    import mlflow
    
    array = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = mlflow.data.from_numpy(array, source="data.csv")
    
    # Log an input dataset used for training
    with mlflow.start_run():
        mlflow.log_input(dataset, context="training")
    

`mlflow.``log_metric`(_key : str_, _value : float_, _step : Optional[int] =
None_, _synchronous : Optional[bool] = None_, _timestamp : Optional[int] =
None_, _run_id : Optional[str] = None_) ->
Optional[[mlflow.utils.async_logging.run_operations.RunOperations](mlflow.utils.html#mlflow.utils.async_logging.run_operations.RunOperations
"mlflow.utils.async_logging.run_operations.RunOperations")][[source]](../_modules/mlflow/tracking/fluent.html#log_metric)

    

Log a metric under the current run. If no run is active, this method will
create a new active run.

Parameters

    

  * **key** â Metric name. This string may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/). All backend stores will support keys up to length 250, but some may support larger keys.

  * **value** â Metric value. Note that some special values such as +/- Infinity may be replaced by other values depending on the store. For example, the SQLAlchemy store replaces +/- Infinity with max / min float values. All backend stores will support values up to length 5000, but some may support larger values.

  * **step** â Metric step. Defaults to zero if unspecified.

  * **timestamp** â Time when this metric was calculated. Defaults to the current system time.

  * **synchronous** â _Experimental_ If True, blocks until the metric is logged successfully. If False, logs the metric asynchronously and returns a future representing the logging operation. If None, read from environment variable MLFLOW_ENABLE_ASYNC_LOGGING, which defaults to False if not set.

  * **run_id** â If specified, log the metric to the specified run. If not specified, log the metric to the currently active run.

Returns

    

When synchronous=True, returns None. When synchronous=False, returns
RunOperations that represents future for logging operation.

Example

    
    
    import mlflow
    
    # Log a metric
    with mlflow.start_run():
        mlflow.log_metric("mse", 2500.00)
    
    # Log a metric in async fashion.
    with mlflow.start_run():
        mlflow.log_metric("mse", 2500.00, synchronous=False)
    

`mlflow.``log_metrics`(_metrics : Dict[str, float]_, _step : Optional[int] =
None_, _synchronous : Optional[bool] = None_, _run_id : Optional[str] = None_,
_timestamp : Optional[int] = None_) ->
Optional[[mlflow.utils.async_logging.run_operations.RunOperations](mlflow.utils.html#mlflow.utils.async_logging.run_operations.RunOperations
"mlflow.utils.async_logging.run_operations.RunOperations")][[source]](../_modules/mlflow/tracking/fluent.html#log_metrics)

    

Log multiple metrics for the current run. If no run is active, this method
will create a new active run.

Parameters

    

  * **metrics** â Dictionary of metric_name: String -> value: Float. Note that some special values such as +/- Infinity may be replaced by other values depending on the store. For example, sql based store may replace +/- Infinity with max / min float values.

  * **step** â A single integer step at which to log the specified Metrics. If unspecified, each metric is logged at step zero.

  * **synchronous** â _Experimental_ If True, blocks until the metrics are logged successfully. If False, logs the metrics asynchronously and returns a future representing the logging operation. If None, read from environment variable MLFLOW_ENABLE_ASYNC_LOGGING, which defaults to False if not set.

  * **run_id** â Run ID. If specified, log metrics to the specified run. If not specified, log metrics to the currently active run.

  * **timestamp** â Time when these metrics were calculated. Defaults to the current system time.

Returns

    

When synchronous=True, returns None. When synchronous=False, returns an
[`mlflow.utils.async_logging.run_operations.RunOperations`](mlflow.utils.html#mlflow.utils.async_logging.run_operations.RunOperations
"mlflow.utils.async_logging.run_operations.RunOperations") instance that
represents future for logging operation.

Example

    
    
    import mlflow
    
    metrics = {"mse": 2500.00, "rmse": 50.00}
    
    # Log a batch of metrics
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
    
    # Log a batch of metrics in async fashion.
    with mlflow.start_run():
        mlflow.log_metrics(metrics, synchronous=False)
    

`mlflow.``log_param`(_key : str_, _value : Any_, _synchronous : Optional[bool]
= None_) -> Any[[source]](../_modules/mlflow/tracking/fluent.html#log_param)

    

Log a parameter (e.g. model hyperparameter) under the current run. If no run
is active, this method will create a new active run.

Parameters

    

  * **key** â Parameter name. This string may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/). All backend stores support keys up to length 250, but some may support larger keys.

  * **value** â Parameter value, but will be string-ified if not. All built-in backend stores support values up to length 6000, but some may support larger values.

  * **synchronous** â _Experimental_ If True, blocks until the parameter is logged successfully. If False, logs the parameter asynchronously and returns a future representing the logging operation. If None, read from environment variable MLFLOW_ENABLE_ASYNC_LOGGING, which defaults to False if not set.

Returns

    

When synchronous=True, returns parameter value. When synchronous=False,
returns an
[`mlflow.utils.async_logging.run_operations.RunOperations`](mlflow.utils.html#mlflow.utils.async_logging.run_operations.RunOperations
"mlflow.utils.async_logging.run_operations.RunOperations") instance that
represents future for logging operation.

Example

    
    
    import mlflow
    
    with mlflow.start_run():
        value = mlflow.log_param("learning_rate", 0.01)
        assert value == 0.01
        value = mlflow.log_param("learning_rate", 0.02, synchronous=False)
    

`mlflow.``log_params`(_params : Dict[str, Any]_, _synchronous : Optional[bool]
= None_, _run_id : Optional[str] = None_) ->
Optional[[mlflow.utils.async_logging.run_operations.RunOperations](mlflow.utils.html#mlflow.utils.async_logging.run_operations.RunOperations
"mlflow.utils.async_logging.run_operations.RunOperations")][[source]](../_modules/mlflow/tracking/fluent.html#log_params)

    

Log a batch of params for the current run. If no run is active, this method
will create a new active run.

Parameters

    

  * **params** â Dictionary of param_name: String -> value: (String, but will be string-ified if not)

  * **synchronous** â _Experimental_ If True, blocks until the parameters are logged successfully. If False, logs the parameters asynchronously and returns a future representing the logging operation. If None, read from environment variable MLFLOW_ENABLE_ASYNC_LOGGING, which defaults to False if not set.

  * **run_id** â Run ID. If specified, log params to the specified run. If not specified, log params to the currently active run.

Returns

    

When synchronous=True, returns None. When synchronous=False, returns an
[`mlflow.utils.async_logging.run_operations.RunOperations`](mlflow.utils.html#mlflow.utils.async_logging.run_operations.RunOperations
"mlflow.utils.async_logging.run_operations.RunOperations") instance that
represents future for logging operation.

Example

    
    
    import mlflow
    
    params = {"learning_rate": 0.01, "n_estimators": 10}
    
    # Log a batch of parameters
    with mlflow.start_run():
        mlflow.log_params(params)
    
    # Log a batch of parameters in async fashion.
    with mlflow.start_run():
        mlflow.log_params(params, synchronous=False)
    

`mlflow.``log_table`(_data : Union[Dict[str, Any], pandas.DataFrame]_,
_artifact_file : str_, _run_id : Optional[str] = None_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#log_table)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Log a table to MLflow Tracking as a JSON artifact. If the artifact_file
already exists in the run, the data would be appended to the existing
artifact_file.

Parameters

    

  * **data** â Dictionary or pandas.DataFrame to log.

  * **artifact_file** â The run-relative artifact file path in posixpath format to which the table is saved (e.g. âdir/file.jsonâ).

  * **run_id** â If specified, log the table to the specified run. If not specified, log the table to the currently active run.

Dictionary Example

    
    
    import mlflow
    
    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }
    with mlflow.start_run():
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file="qabot_eval_results.json")
    

Pandas DF Example

    
    
    import mlflow
    import pandas as pd
    
    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }
    df = pd.DataFrame.from_dict(table_dict)
    with mlflow.start_run():
        # Log the df as a table
        mlflow.log_table(data=df, artifact_file="qabot_eval_results.json")
    

`mlflow.``log_text`(_text : str_, _artifact_file : str_, _run_id :
Optional[str] = None_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#log_text)

    

Log text as an artifact.

Parameters

    

  * **text** â String containing text to log.

  * **artifact_file** â The run-relative artifact file path in posixpath format to which the text is saved (e.g. âdir/file.txtâ).

  * **run_id** â If specified, log the artifact to the specified run. If not specified, log the artifact to the currently active run.

Example

    
    
    import mlflow
    
    with mlflow.start_run():
        # Log text to a file under the run's root artifact directory
        mlflow.log_text("text1", "file1.txt")
    
        # Log text in a subdirectory of the run's root artifact directory
        mlflow.log_text("text2", "dir/file2.txt")
    
        # Log HTML text
        mlflow.log_text("<h1>header</h1>", "index.html")
    

`mlflow.``login`(_backend : str = 'databricks'_, _interactive : bool = True_)
-> None[[source]](../_modules/mlflow/utils/credentials.html#login)

    

Configure MLflow server authentication and connect MLflow to tracking server.

This method provides a simple way to connect MLflow to its tracking server.
Currently only Databricks tracking server is supported. Users will be prompted
to enter the credentials if no existing Databricks profile is found, and the
credentials will be saved to ~/.databrickscfg.

Parameters

    

  * **backend** â string, the backend of the tracking server. Currently only âdatabricksâ is supported.

  * **interactive** â bool, controls request for user input on missing credentials. If true, user input will be requested if no credentials are found, otherwise an exception will be raised if no credentials are found.

Example

    
    
    import mlflow
    
    mlflow.login()
    with mlflow.start_run():
        mlflow.log_param("p", 0)
    

`mlflow.``register_model`(_model_uri_ , _name_ , _await_registration_for
=300_, _*_ , _tags : Optional[Dict[str, Any]] = None_) ->
[ModelVersion](mlflow.entities.html#mlflow.entities.model_registry.ModelVersion
"mlflow.entities.model_registry.ModelVersion")[[source]](../_modules/mlflow/tracking/_model_registry/fluent.html#register_model)

    

Create a new model version in model registry for the model files specified by
`model_uri`.

Note that this method assumes the model registry backend URI is the same as
that of the tracking backend.

Parameters

    

  * **model_uri** â URI referring to the MLmodel directory. Use a `runs:/` URI if you want to record the run ID with the model in model registry (recommended), or pass the local filesystem path of the model if registering a locally-persisted MLflow model that was previously saved using `save_model`. `models:/` URIs are currently not supported.

  * **name** â Name of the registered model under which to create a new model version. If a registered model with the given name does not exist, it will be created automatically.

  * **await_registration_for** â Number of seconds to wait for the model version to finish being created and is in `READY` status. By default, the function waits for five minutes. Specify 0 or None to skip waiting.

  * **tags** â A dictionary of key-value pairs that are converted into [`mlflow.entities.model_registry.ModelVersionTag`](mlflow.entities.html#mlflow.entities.model_registry.ModelVersionTag "mlflow.entities.model_registry.ModelVersionTag") objects.

Returns

    

Single
[`mlflow.entities.model_registry.ModelVersion`](mlflow.entities.html#mlflow.entities.model_registry.ModelVersion
"mlflow.entities.model_registry.ModelVersion") object created by backend.

Example

    
    
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    
    mlflow.set_tracking_uri("sqlite:////tmp/mlruns.db")
    params = {"n_estimators": 3, "random_state": 42}
    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    # Log MLflow entities
    with mlflow.start_run() as run:
        rfr = RandomForestRegressor(**params).fit(X, y)
        signature = infer_signature(X, rfr.predict(X))
        mlflow.log_params(params)
        mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model", signature=signature)
    model_uri = f"runs:/{run.info.run_id}/sklearn-model"
    mv = mlflow.register_model(model_uri, "RandomForestRegressionModel")
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")
    

Output

    
    
    Name: RandomForestRegressionModel
    Version: 1
    

`mlflow.``run`(_uri_ , _entry_point ='main'_, _version =None_, _parameters
=None_, _docker_args =None_, _experiment_name =None_, _experiment_id =None_,
_backend ='local'_, _backend_config =None_, _storage_dir =None_, _synchronous
=True_, _run_id =None_, _run_name =None_, _env_manager =None_, _build_image
=False_, _docker_auth =None_)[[source]](../_modules/mlflow/projects.html#run)

    

Run an MLflow project. The project can be local or stored at a Git URI.

MLflow provides built-in support for running projects locally or remotely on a
Databricks or Kubernetes cluster. You can also run projects against other
targets by installing an appropriate third-party plugin. See [Community
Plugins](../plugins.html#community-plugins) for more information.

For information on using this method in chained workflows, see [Building
Multistep Workflows](../projects.html#building-multistep-workflows).

Raises

    

[**mlflow.exceptions.ExecutionException**](exceptions/mlflow.exceptions.html#mlflow.exceptions.ExecutionException
"mlflow.exceptions.ExecutionException") â is unsuccessful.

Parameters

    

  * **uri** â URI of project to run. A local filesystem path or a Git repository URI (e.g. <https://github.com/mlflow/mlflow-example>) pointing to a project directory containing an MLproject file.

  * **entry_point** â Entry point to run within the project. If no entry point with the specified name is found, runs the project file `entry_point` as a script, using âpythonâ to run `.py` files and the default shell (specified by environment variable `$SHELL`) to run `.sh` files.

  * **version** â For Git-based projects, either a commit hash or a branch name.

  * **parameters** â Parameters (dictionary) for the entry point command.

  * **docker_args** â Arguments (dictionary) for the docker command.

  * **experiment_name** â Name of experiment under which to launch the run.

  * **experiment_id** â ID of experiment under which to launch the run.

  * **backend** â Execution backend for the run: MLflow provides built-in support for âlocalâ, âdatabricksâ, and âkubernetesâ (experimental) backends. If running against Databricks, will run against a Databricks workspace determined as follows: if a Databricks tracking URI of the form `databricks://profile` has been set (e.g. by setting the MLFLOW_TRACKING_URI environment variable), will run against the workspace specified by <profile>. Otherwise, runs against the workspace specified by the default Databricks CLI profile.

  * **backend_config** â A dictionary, or a path to a JSON file (must end in â.jsonâ), which will be passed as config to the backend. The exact content which should be provided is different for each execution backend and is documented at <https://www.mlflow.org/docs/latest/projects.html>.

  * **storage_dir** â Used only if `backend` is âlocalâ. MLflow downloads artifacts from distributed URIs passed to parameters of type `path` to subdirectories of `storage_dir`.

  * **synchronous** â Whether to block while waiting for a run to complete. Defaults to True. Note that if `synchronous` is False and `backend` is âlocalâ, this method will return, but the current process will block when exiting until the local run completes. If the current process is interrupted, any asynchronous runs launched via this method will be terminated. If `synchronous` is True and the run fails, the current process will error out as well.

  * **run_id** â Note: this argument is used internally by the MLflow project APIs and should not be specified. If specified, the run ID will be used instead of creating a new run.

  * **run_name** â The name to give the MLflow Run associated with the project execution. If `None`, the MLflow Run name is left unset.

  * **env_manager** â 

Specify an environment manager to create a new environment for the run and
install project dependencies within that environment. The following values are
supported:

    * local: use the local environment

    * virtualenv: use virtualenv (and pyenv for Python version management)

    * conda: use conda

If unspecified, MLflow automatically determines the environment manager to use
by inspecting files in the project directory. For example, if
`python_env.yaml` is present, virtualenv will be used.

  * **build_image** â Whether to build a new docker image of the project or to reuse an existing image. Default: False (reuse an existing image)

  * **docker_auth** â A dictionary representing information to authenticate with a Docker registry. See [docker.client.DockerClient.login](https://docker-py.readthedocs.io/en/stable/client.html#docker.client.DockerClient.login) for available options.

Returns

    

[`mlflow.projects.SubmittedRun`](mlflow.projects.html#mlflow.projects.SubmittedRun
"mlflow.projects.SubmittedRun") exposing information (e.g. run ID) about the
launched run.

Example

    
    
    import mlflow
    
    project_uri = "https://github.com/mlflow/mlflow-example"
    params = {"alpha": 0.5, "l1_ratio": 0.01}
    
    # Run MLflow project and create a reproducible conda environment
    # on a local host
    mlflow.run(project_uri, parameters=params)
    

Output

    
    
    ...
    ...
    Elasticnet model (alpha=0.500000, l1_ratio=0.010000):
    RMSE: 0.788347345611717
    MAE: 0.6155576449938276
    R2: 0.19729662005412607
    ... mlflow.projects: === Run (ID '6a5109febe5e4a549461e149590d0a7c') succeeded ===
    

`mlflow.``search_experiments`(_view_type : int = 1_, _max_results :
Optional[int] = None_, _filter_string : Optional[str] = None_, _order_by :
Optional[List[str]] = None_) ->
List[[Experiment](mlflow.entities.html#mlflow.entities.Experiment
"mlflow.entities.Experiment")][[source]](../_modules/mlflow/tracking/fluent.html#search_experiments)

    

Search for experiments that match the specified search query.

Parameters

    

  * **view_type** â One of enum values `ACTIVE_ONLY`, `DELETED_ONLY`, or `ALL` defined in [`mlflow.entities.ViewType`](mlflow.entities.html#mlflow.entities.ViewType "mlflow.entities.ViewType").

  * **max_results** â If passed, specifies the maximum number of experiments desired. If not passed, all experiments will be returned.

  * **filter_string** â 

Filter query string (e.g., `"name = 'my_experiment'"`), defaults to searching
for all experiments. The following identifiers, comparators, and logical
operators are supported.

Identifiers

    
    * `name`: Experiment name

    * `creation_time`: Experiment creation time

    * `last_update_time`: Experiment last update time

    * `tags.<tag_key>`: Experiment tag. If `tag_key` contains spaces, it must be wrapped with backticks (e.g., `"tags.`extra key`"`).

Comparators for string attributes and tags

    
    * `=`: Equal to

    * `!=`: Not equal to

    * `LIKE`: Case-sensitive pattern match

    * `ILIKE`: Case-insensitive pattern match

Comparators for numeric attributes

    
    * `=`: Equal to

    * `!=`: Not equal to

    * `<`: Less than

    * `<=`: Less than or equal to

    * `>`: Greater than

    * `>=`: Greater than or equal to

Logical operators

    
    * `AND`: Combines two sub-queries and returns True if both of them are True.

  * **order_by** â 

List of columns to order by. The `order_by` column can contain an optional
`DESC` or `ASC` value (e.g., `"name DESC"`). The default ordering is `ASC`, so
`"name"` is equivalent to `"name ASC"`. If unspecified, defaults to
`["last_update_time DESC"]`, which lists experiments updated most recently
first. The following fields are supported:

>     * `experiment_id`: Experiment ID
>
>     * `name`: Experiment name
>
>     * `creation_time`: Experiment creation time
>
>     * `last_update_time`: Experiment last update time

Returns

    

A list of [`Experiment`](mlflow.entities.html#mlflow.entities.Experiment
"mlflow.entities.Experiment") objects.

Example

    
    
    import mlflow
    
    
    def assert_experiment_names_equal(experiments, expected_names):
        actual_names = [e.name for e in experiments if e.name != "Default"]
        assert actual_names == expected_names, (actual_names, expected_names)
    
    
    mlflow.set_tracking_uri("sqlite:///:memory:")
    # Create experiments
    for name, tags in [
        ("a", None),
        ("b", None),
        ("ab", {"k": "v"}),
        ("bb", {"k": "V"}),
    ]:
        mlflow.create_experiment(name, tags=tags)
    
    # Search for experiments with name "a"
    experiments = mlflow.search_experiments(filter_string="name = 'a'")
    assert_experiment_names_equal(experiments, ["a"])
    # Search for experiments with name starting with "a"
    experiments = mlflow.search_experiments(filter_string="name LIKE 'a%'")
    assert_experiment_names_equal(experiments, ["ab", "a"])
    # Search for experiments with tag key "k" and value ending with "v" or "V"
    experiments = mlflow.search_experiments(filter_string="tags.k ILIKE '%v'")
    assert_experiment_names_equal(experiments, ["bb", "ab"])
    # Search for experiments with name ending with "b" and tag {"k": "v"}
    experiments = mlflow.search_experiments(filter_string="name LIKE '%b' AND tags.k = 'v'")
    assert_experiment_names_equal(experiments, ["ab"])
    # Sort experiments by name in ascending order
    experiments = mlflow.search_experiments(order_by=["name"])
    assert_experiment_names_equal(experiments, ["a", "ab", "b", "bb"])
    # Sort experiments by ID in descending order
    experiments = mlflow.search_experiments(order_by=["experiment_id DESC"])
    assert_experiment_names_equal(experiments, ["bb", "ab", "b", "a"])
    

`mlflow.``search_model_versions`(_max_results : Optional[int] = None_,
_filter_string : Optional[str] = None_, _order_by : Optional[List[str]] =
None_) ->
List[[ModelVersion](mlflow.entities.html#mlflow.entities.model_registry.ModelVersion
"mlflow.entities.model_registry.ModelVersion")][[source]](../_modules/mlflow/tracking/_model_registry/fluent.html#search_model_versions)

    

Search for model versions that satisfy the filter criteria.

Parameters

    

  * **filter_string** â 

Filter query string (e.g., `"name = 'a_model_name' and tag.key = 'value1'"`),
defaults to searching for all model versions. The following identifiers,
comparators, and logical operators are supported.

Identifiers

    
    * `name`: model name.

    * `source_path`: model version source path.

    * `run_id`: The id of the mlflow run that generates the model version.

    * `tags.<tag_key>`: model version tag. If `tag_key` contains spaces, it must be wrapped with backticks (e.g., `"tags.`extra key`"`).

Comparators

    
    * `=`: Equal to.

    * `!=`: Not equal to.

    * `LIKE`: Case-sensitive pattern match.

    * `ILIKE`: Case-insensitive pattern match.

    * `IN`: In a value list. Only `run_id` identifier supports `IN` comparator.

Logical operators

    
    * `AND`: Combines two sub-queries and returns True if both of them are True.

  * **max_results** â If passed, specifies the maximum number of models desired. If not passed, all models will be returned.

  * **order_by** â List of column names with ASC|DESC annotation, to be used for ordering matching search results.

Returns

    

A list of
[`mlflow.entities.model_registry.ModelVersion`](mlflow.entities.html#mlflow.entities.model_registry.ModelVersion
"mlflow.entities.model_registry.ModelVersion") objects

    

that satisfy the search expressions.

Example

    
    
    import mlflow
    from sklearn.linear_model import LogisticRegression
    
    for _ in range(2):
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                LogisticRegression(),
                "Cordoba",
                registered_model_name="CordobaWeatherForecastModel",
            )
    
    # Get all versions of the model filtered by name
    filter_string = "name = 'CordobaWeatherForecastModel'"
    results = mlflow.search_model_versions(filter_string=filter_string)
    print("-" * 80)
    for res in results:
        print(f"name={res.name}; run_id={res.run_id}; version={res.version}")
    
    # Get the version of the model filtered by run_id
    filter_string = "run_id = 'ae9a606a12834c04a8ef1006d0cff779'"
    results = mlflow.search_model_versions(filter_string=filter_string)
    print("-" * 80)
    for res in results:
        print(f"name={res.name}; run_id={res.run_id}; version={res.version}")
    

Output

    
    
    --------------------------------------------------------------------------------
    name=CordobaWeatherForecastModel; run_id=ae9a606a12834c04a8ef1006d0cff779; version=2
    name=CordobaWeatherForecastModel; run_id=d8f028b5fedf4faf8e458f7693dfa7ce; version=1
    --------------------------------------------------------------------------------
    name=CordobaWeatherForecastModel; run_id=ae9a606a12834c04a8ef1006d0cff779; version=2
    

`mlflow.``search_registered_models`(_max_results : Optional[int] = None_,
_filter_string : Optional[str] = None_, _order_by : Optional[List[str]] =
None_) ->
List[[RegisteredModel](mlflow.entities.html#mlflow.entities.model_registry.RegisteredModel
"mlflow.entities.model_registry.RegisteredModel")][[source]](../_modules/mlflow/tracking/_model_registry/fluent.html#search_registered_models)

    

Search for registered models that satisfy the filter criteria.

Parameters

    

  * **filter_string** â 

Filter query string (e.g., âname = âa_model_nameâ and tag.key =
âvalue1ââ), defaults to searching for all registered models. The
following identifiers, comparators, and logical operators are supported.

Identifiers

    
    * ânameâ: registered model name.

    * âtags.<tag_key>â: registered model tag. If âtag_keyâ contains spaces, it must be wrapped with backticks (e.g., âtags.`extra key`â).

Comparators

    
    * â=â: Equal to.

    * â!=â: Not equal to.

    * âLIKEâ: Case-sensitive pattern match.

    * âILIKEâ: Case-insensitive pattern match.

Logical operators

    
    * âANDâ: Combines two sub-queries and returns True if both of them are True.

  * **max_results** â If passed, specifies the maximum number of models desired. If not passed, all models will be returned.

  * **order_by** â List of column names with ASC|DESC annotation, to be used for ordering matching search results.

Returns

    

A list of
[`mlflow.entities.model_registry.RegisteredModel`](mlflow.entities.html#mlflow.entities.model_registry.RegisteredModel
"mlflow.entities.model_registry.RegisteredModel") objects that satisfy the
search expressions.

Example

    
    
    import mlflow
    from sklearn.linear_model import LogisticRegression
    
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            LogisticRegression(),
            "Cordoba",
            registered_model_name="CordobaWeatherForecastModel",
        )
        mlflow.sklearn.log_model(
            LogisticRegression(),
            "Boston",
            registered_model_name="BostonWeatherForecastModel",
        )
    
    # Get search results filtered by the registered model name
    filter_string = "name = 'CordobaWeatherForecastModel'"
    results = mlflow.search_registered_models(filter_string=filter_string)
    print("-" * 80)
    for res in results:
        for mv in res.latest_versions:
            print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")
    
    # Get search results filtered by the registered model name that matches
    # prefix pattern
    filter_string = "name LIKE 'Boston%'"
    results = mlflow.search_registered_models(filter_string=filter_string)
    print("-" * 80)
    for res in results:
        for mv in res.latest_versions:
            print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")
    
    # Get all registered models and order them by ascending order of the names
    results = mlflow.search_registered_models(order_by=["name ASC"])
    print("-" * 80)
    for res in results:
        for mv in res.latest_versions:
            print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")
    

Output

    
    
    --------------------------------------------------------------------------------
    name=CordobaWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
    --------------------------------------------------------------------------------
    name=BostonWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
    --------------------------------------------------------------------------------
    name=BostonWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
    name=CordobaWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
    

`mlflow.``search_runs`(_experiment_ids : Optional[List[str]] = None_,
_filter_string : str = ''_, _run_view_type : int = 1_, _max_results : int =
100000_, _order_by : Optional[List[str]] = None_, _output_format : str =
'pandas'_, _search_all_experiments : bool = False_, _experiment_names :
Optional[List[str]] = None_) ->
Union[List[[Run](mlflow.entities.html#mlflow.entities.Run
"mlflow.entities.Run")],
pandas.DataFrame][[source]](../_modules/mlflow/tracking/fluent.html#search_runs)

    

Search for Runs that fit the specified criteria.

Parameters

    

  * **experiment_ids** â List of experiment IDs. Search can work with experiment IDs or experiment names, but not both in the same call. Values other than `None` or `[]` will result in error if `experiment_names` is also not `None` or `[]`. `None` will default to the active experiment if `experiment_names` is `None` or `[]`.

  * **filter_string** â Filter query string, defaults to searching all runs.

  * **run_view_type** â one of enum values `ACTIVE_ONLY`, `DELETED_ONLY`, or `ALL` runs defined in [`mlflow.entities.ViewType`](mlflow.entities.html#mlflow.entities.ViewType "mlflow.entities.ViewType").

  * **max_results** â The maximum number of runs to put in the dataframe. Default is 100,000 to avoid causing out-of-memory issues on the userâs machine.

  * **order_by** â List of columns to order by (e.g., âmetrics.rmseâ). The `order_by` column can contain an optional `DESC` or `ASC` value. The default is `ASC`. The default ordering is to sort by `start_time DESC`, then `run_id`.

  * **output_format** â The output format to be returned. If `pandas`, a `pandas.DataFrame` is returned and, if `list`, a list of [`mlflow.entities.Run`](mlflow.entities.html#mlflow.entities.Run "mlflow.entities.Run") is returned.

  * **search_all_experiments** â Boolean specifying whether all experiments should be searched. Only honored if `experiment_ids` is `[]` or `None`.

  * **experiment_names** â List of experiment names. Search can work with experiment IDs or experiment names, but not both in the same call. Values other than `None` or `[]` will result in error if `experiment_ids` is also not `None` or `[]`. `None` will default to the active experiment if `experiment_ids` is `None` or `[]`.

Returns

    

a list of [`mlflow.entities.Run`](mlflow.entities.html#mlflow.entities.Run
"mlflow.entities.Run"). If output_format is `pandas`: `pandas.DataFrame` of
runs, where each metric, parameter, and tag is expanded into its own column
named metrics.*, params.*, or tags.* respectively. For runs that donât have
a particular metric, parameter, or tag, the value for the corresponding column
is (NumPy) `Nan`, `None`, or `None` respectively.

Return type

    

If output_format is `list`

Example

    
    
    import mlflow
    
    # Create an experiment and log two runs under it
    experiment_name = "Social NLP Experiments"
    experiment_id = mlflow.create_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metric("m", 1.55)
        mlflow.set_tag("s.release", "1.1.0-RC")
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metric("m", 2.50)
        mlflow.set_tag("s.release", "1.2.0-GA")
    # Search for all the runs in the experiment with the given experiment ID
    df = mlflow.search_runs([experiment_id], order_by=["metrics.m DESC"])
    print(df[["metrics.m", "tags.s.release", "run_id"]])
    print("--")
    # Search the experiment_id using a filter_string with tag
    # that has a case insensitive pattern
    filter_string = "tags.s.release ILIKE '%rc%'"
    df = mlflow.search_runs([experiment_id], filter_string=filter_string)
    print(df[["metrics.m", "tags.s.release", "run_id"]])
    print("--")
    # Search for all the runs in the experiment with the given experiment name
    df = mlflow.search_runs(experiment_names=[experiment_name], order_by=["metrics.m DESC"])
    print(df[["metrics.m", "tags.s.release", "run_id"]])
    

Output

    
    
       metrics.m tags.s.release                            run_id
    0       2.50       1.2.0-GA  147eed886ab44633902cc8e19b2267e2
    1       1.55       1.1.0-RC  5cc7feaf532f496f885ad7750809c4d4
    --
       metrics.m tags.s.release                            run_id
    0       1.55       1.1.0-RC  5cc7feaf532f496f885ad7750809c4d4
    --
       metrics.m tags.s.release                            run_id
    0       2.50       1.2.0-GA  147eed886ab44633902cc8e19b2267e2
    1       1.55       1.1.0-RC  5cc7feaf532f496f885ad7750809c4d4
    

`mlflow.``set_experiment`(_experiment_name : Optional[str] = None_,
_experiment_id : Optional[str] = None_) ->
[Experiment](mlflow.entities.html#mlflow.entities.Experiment
"mlflow.entities.Experiment")[[source]](../_modules/mlflow/tracking/fluent.html#set_experiment)

    

Set the given experiment as the active experiment. The experiment must either
be specified by name via experiment_name or by ID via experiment_id. The
experiment name and ID cannot both be specified.

Note

If the experiment being set by name does not exist, a new experiment will be
created with the given name. After the experiment has been created, it will be
set as the active experiment. On certain platforms, such as Databricks, the
experiment name must be an absolute path, e.g. `"/Users/<username>/my-
experiment"`.

Parameters

    

  * **experiment_name** â Case sensitive name of the experiment to be activated.

  * **experiment_id** â ID of the experiment to be activated. If an experiment with this ID does not exist, an exception is thrown.

Returns

    

An instance of
[`mlflow.entities.Experiment`](mlflow.entities.html#mlflow.entities.Experiment
"mlflow.entities.Experiment") representing the new active experiment.

Example

    
    
    import mlflow
    
    # Set an experiment name, which must be unique and case-sensitive.
    experiment = mlflow.set_experiment("Social NLP Experiments")
    # Get Experiment Details
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    

Output

    
    
    Experiment_id: 1
    Artifact Location: file:///.../mlruns/1
    Tags: {}
    Lifecycle_stage: active
    

`mlflow.``set_experiment_tag`(_key : str_, _value : Any_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#set_experiment_tag)

    

Set a tag on the current experiment. Value is converted to a string.

Parameters

    

  * **key** â Tag name. This string may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/). All backend stores will support keys up to length 250, but some may support larger keys.

  * **value** â Tag value, but will be string-ified if not. All backend stores will support values up to length 5000, but some may support larger values.

Example

    
    
    import mlflow
    
    with mlflow.start_run():
        mlflow.set_experiment_tag("release.version", "2.2.0")
    

`mlflow.``set_experiment_tags`(_tags : Dict[str, Any]_) ->
None[[source]](../_modules/mlflow/tracking/fluent.html#set_experiment_tags)

    

Set tags for the current active experiment.

Parameters

    

**tags** â Dictionary containing tag names and corresponding values.

Example

    
    
    import mlflow
    
    tags = {
        "engineering": "ML Platform",
        "release.candidate": "RC1",
        "release.version": "2.2.0",
    }
    
    # Set a batch of tags
    with mlflow.start_run():
        mlflow.set_experiment_tags(tags)
    

`mlflow.``set_registry_uri`(_uri : str_) ->
None[[source]](../_modules/mlflow/tracking/_model_registry/utils.html#set_registry_uri)

    

Set the registry server URI. This method is especially useful if you have a
registry server thatâs different from the tracking server.

Parameters

    

**uri** â An empty string, or a local file path, prefixed with `file:/`.
Data is stored locally at the provided file (or `./mlruns` if empty). An HTTP
URI like `https://my-tracking-server:5000` or `http://my-oss-uc-server:8080`.
A Databricks workspace, provided as the string âdatabricksâ or, to use a
Databricks CLI [profile](https://github.com/databricks/databricks-
cli#installation), âdatabricks://<profileName>â.

Example

    
    
    import mflow
    
    # Set model registry uri, fetch the set uri, and compare
    # it with the tracking uri. They should be different
    mlflow.set_registry_uri("sqlite:////tmp/registry.db")
    mr_uri = mlflow.get_registry_uri()
    print(f"Current registry uri: {mr_uri}")
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")
    
    # They should be different
    assert tracking_uri != mr_uri
    

Output

    
    
    Current registry uri: sqlite:////tmp/registry.db
    Current tracking uri: file:///.../mlruns
    

`mlflow.``set_system_metrics_node_id`(_node_id_)[[source]](../_modules/mlflow/system_metrics.html#set_system_metrics_node_id)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Set the system metrics node id.

> node_id is the identifier of the machine where the metrics are collected.
> This is useful in multi-node (distributed training) setup.

`mlflow.``set_system_metrics_samples_before_logging`(_samples_)[[source]](../_modules/mlflow/system_metrics.html#set_system_metrics_samples_before_logging)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Set the number of samples before logging system metrics.

> Every time samples samples have been collected, the system metrics will be
> logged to mlflow. By default samples=1.

`mlflow.``set_system_metrics_sampling_interval`(_interval_)[[source]](../_modules/mlflow/system_metrics.html#set_system_metrics_sampling_interval)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Set the system metrics sampling interval.

> Every interval seconds, the system metrics will be collected. By default
> interval=10.

`mlflow.``set_tag`(_key : str_, _value : Any_, _synchronous : Optional[bool] =
None_) ->
Optional[[mlflow.utils.async_logging.run_operations.RunOperations](mlflow.utils.html#mlflow.utils.async_logging.run_operations.RunOperations
"mlflow.utils.async_logging.run_operations.RunOperations")][[source]](../_modules/mlflow/tracking/fluent.html#set_tag)

    

Set a tag under the current run. If no run is active, this method will create
a new active run.

Parameters

    

  * **key** â Tag name. This string may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/). All backend stores will support keys up to length 250, but some may support larger keys.

  * **value** â Tag value, but will be string-ified if not. All backend stores will support values up to length 5000, but some may support larger values.

  * **synchronous** â _Experimental_ If True, blocks until the tag is logged successfully. If False, logs the tag asynchronously and returns a future representing the logging operation. If None, read from environment variable MLFLOW_ENABLE_ASYNC_LOGGING, which defaults to False if not set.

Returns

    

When synchronous=True, returns None. When synchronous=False, returns an
[`mlflow.utils.async_logging.run_operations.RunOperations`](mlflow.utils.html#mlflow.utils.async_logging.run_operations.RunOperations
"mlflow.utils.async_logging.run_operations.RunOperations") instance that
represents future for logging operation.

Example

    
    
    import mlflow
    
    # Set a tag.
    with mlflow.start_run():
        mlflow.set_tag("release.version", "2.2.0")
    
    # Set a tag in async fashion.
    with mlflow.start_run():
        mlflow.set_tag("release.version", "2.2.1", synchronous=False)
    

`mlflow.``set_tags`(_tags : Dict[str, Any]_, _synchronous : Optional[bool] =
None_) ->
Optional[[mlflow.utils.async_logging.run_operations.RunOperations](mlflow.utils.html#mlflow.utils.async_logging.run_operations.RunOperations
"mlflow.utils.async_logging.run_operations.RunOperations")][[source]](../_modules/mlflow/tracking/fluent.html#set_tags)

    

Log a batch of tags for the current run. If no run is active, this method will
create a new active run.

Parameters

    

  * **tags** â Dictionary of tag_name: String -> value: (String, but will be string-ified if not)

  * **synchronous** â _Experimental_ If True, blocks until tags are logged successfully. If False, logs tags asynchronously and returns a future representing the logging operation. If None, read from environment variable MLFLOW_ENABLE_ASYNC_LOGGING, which defaults to False if not set.

Returns

    

When synchronous=True, returns None. When synchronous=False, returns an
[`mlflow.utils.async_logging.run_operations.RunOperations`](mlflow.utils.html#mlflow.utils.async_logging.run_operations.RunOperations
"mlflow.utils.async_logging.run_operations.RunOperations") instance that
represents future for logging operation.

Example

    
    
    import mlflow
    
    tags = {
        "engineering": "ML Platform",
        "release.candidate": "RC1",
        "release.version": "2.2.0",
    }
    
    # Set a batch of tags
    with mlflow.start_run():
        mlflow.set_tags(tags)
    
    # Set a batch of tags in async fashion.
    with mlflow.start_run():
        mlflow.set_tags(tags, synchronous=False)
    

`mlflow.``set_tracking_uri`(_uri : Union[str, pathlib.Path]_) ->
None[[source]](../_modules/mlflow/tracking/_tracking_service/utils.html#set_tracking_uri)

    

Set the tracking server URI. This does not affect the currently active run (if
one exists), but takes effect for successive runs.

Parameters

    

**uri** â

  * An empty string, or a local file path, prefixed with `file:/`. Data is stored locally at the provided file (or `./mlruns` if empty).

  * An HTTP URI like `https://my-tracking-server:5000`.

  * A Databricks workspace, provided as the string âdatabricksâ or, to use a Databricks CLI [profile](https://github.com/databricks/databricks-cli#installation), âdatabricks://<profileName>â.

  * A `pathlib.Path` instance

Example

    
    
    import mlflow
    
    mlflow.set_tracking_uri("file:///tmp/my_tracking")
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")
    

Output

    
    
    Current tracking uri: file:///tmp/my_tracking
    

`mlflow.``start_run`(_run_id : Optional[str] = None_, _experiment_id :
Optional[str] = None_, _run_name : Optional[str] = None_, _nested : bool =
False_, _parent_run_id : Optional[str] = None_, _tags : Optional[Dict[str,
Any]] = None_, _description : Optional[str] = None_, _log_system_metrics :
Optional[bool] = None_) ->
ActiveRun[[source]](../_modules/mlflow/tracking/fluent.html#start_run)

    

Start a new MLflow run, setting it as the active run under which metrics and
parameters will be logged. The return value can be used as a context manager
within a `with` block; otherwise, you must call `end_run()` to terminate the
current run.

If you pass a `run_id` or the `MLFLOW_RUN_ID` environment variable is set,
`start_run` attempts to resume a run with the specified run ID and other
parameters are ignored. `run_id` takes precedence over `MLFLOW_RUN_ID`.

If resuming an existing run, the run status is set to `RunStatus.RUNNING`.

MLflow sets a variety of default tags on the run, as defined in [MLflow system
tags](../tracking/tracking-api.html#system-tags).

Parameters

    

  * **run_id** â If specified, get the run with the specified UUID and log parameters and metrics under that run. The runâs end time is unset and its status is set to running, but the runâs other attributes (`source_version`, `source_type`, etc.) are not changed.

  * **experiment_id** â ID of the experiment under which to create the current run (applicable only when `run_id` is not specified). If `experiment_id` argument is unspecified, will look for valid experiment in the following order: activated using `set_experiment`, `MLFLOW_EXPERIMENT_NAME` environment variable, `MLFLOW_EXPERIMENT_ID` environment variable, or the default experiment as defined by the tracking server.

  * **run_name** â Name of new run. Used only when `run_id` is unspecified. If a new run is created and `run_name` is not specified, a random name will be generated for the run.

  * **nested** â Controls whether run is nested in parent run. `True` creates a nested run.

  * **parent_run_id** â If specified, the current run will be nested under the the run with the specified UUID. The parent run must be in the ACTIVE state.

  * **tags** â An optional dictionary of string keys and values to set as tags on the run. If a run is being resumed, these tags are set on the resumed run. If a new run is being created, these tags are set on the new run.

  * **description** â An optional string that populates the description box of the run. If a run is being resumed, the description is set on the resumed run. If a new run is being created, the description is set on the new run.

  * **log_system_metrics** â bool, defaults to None. If True, system metrics will be logged to MLflow, e.g., cpu/gpu utilization. If None, we will check environment variable MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING to determine whether to log system metrics. System metrics logging is an experimental feature in MLflow 2.8 and subject to change.

Returns

    

`mlflow.ActiveRun` object that acts as a context manager wrapping the runâs
state.

Example

    
    
    import mlflow
    
    # Create nested runs
    experiment_id = mlflow.create_experiment("experiment1")
    with mlflow.start_run(
        run_name="PARENT_RUN",
        experiment_id=experiment_id,
        tags={"version": "v1", "priority": "P1"},
        description="parent",
    ) as parent_run:
        mlflow.log_param("parent", "yes")
        with mlflow.start_run(
            run_name="CHILD_RUN",
            experiment_id=experiment_id,
            description="child",
            nested=True,
        ) as child_run:
            mlflow.log_param("child", "yes")
    print("parent run:")
    print(f"run_id: {parent_run.info.run_id}")
    print("description: {}".format(parent_run.data.tags.get("mlflow.note.content")))
    print("version tag value: {}".format(parent_run.data.tags.get("version")))
    print("priority tag value: {}".format(parent_run.data.tags.get("priority")))
    print("--")
    
    # Search all child runs with a parent id
    query = f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'"
    results = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=query)
    print("child runs:")
    print(results[["run_id", "params.child", "tags.mlflow.runName"]])
    
    # Create a nested run under the existing parent run
    with mlflow.start_run(
        run_name="NEW_CHILD_RUN",
        experiment_id=experiment_id,
        description="new child",
        parent_run_id=parent_run.info.run_id,
    ) as child_run:
        mlflow.log_param("new-child", "yes")
    

Output

    
    
    parent run:
    run_id: 8979459433a24a52ab3be87a229a9cdf
    description: starting a parent for experiment 7
    version tag value: v1
    priority tag value: P1
    --
    child runs:
                                 run_id params.child tags.mlflow.runName
    0  7d175204675e40328e46d9a6a5a7ee6a          yes           CHILD_RUN
    

# MLflow Tracing APIs

The `mlflow` module provides a set of high-level APIs for [MLflow
Tracing](../llms/tracing/index.html). For the detailed guidance on how to use
these tracing APIs, please refer to the [Tracing Fluent APIs
Guide](../llms/tracing/index.html#tracing-fluent-apis).

For some advanced use cases such as multi-threaded application,
instrumentation via callbacks, you may need to use the low-level tracing APIs
[`MlflowClient`](mlflow.client.html#mlflow.client.MlflowClient
"mlflow.client.MlflowClient") provides. For detailed guidance on how to use
the low-level tracing APIs, please refer to the [Tracing Client APIs
Guide](../llms/tracing/index.html#tracing-client-apis).

`mlflow.``trace`(_func : Optional[Callable] = None_, _name : Optional[str] =
None_, _span_type : str = 'UNKNOWN'_, _attributes : Optional[Dict[str, Any]] =
None_) -> Callable[[source]](../_modules/mlflow/tracing/fluent.html#trace)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

A decorator that creates a new span for the decorated function.

When you decorate a function with this `@mlflow.trace()` decorator, a span
will be created for the scope of the decorated function. The span will
automatically capture the input and output of the function. When it is applied
to a method, it doesnât capture the self argument. Any exception raised
within the function will set the span status to `ERROR` and detailed
information such as exception message and stacktrace will be recorded to the
`attributes` field of the span.

For example, the following code will yield a span with the name
`"my_function"`, capturing the input arguments `x` and `y`, and the output of
the function.

    
    
    import mlflow
    
    
    @mlflow.trace
    def my_function(x, y):
        return x + y
    

This is equivalent to doing the following using the `mlflow.start_span()`
context manager, but requires less boilerplate code.

    
    
    import mlflow
    
    
    def my_function(x, y):
        return x + y
    
    
    with mlflow.start_span("my_function") as span:
        x = 1
        y = 2
        span.set_inputs({"x": x, "y": y})
        result = my_function(x, y)
        span.set_outputs({"output": result})
    

Tip

The @mlflow.trace decorator is useful when you want to trace a function
defined by yourself. However, you may also want to trace a function in
external libraries. In such case, you can use this `mlflow.trace()` function
to directly wrap the function, instead of using as the decorator. This will
create the exact same span as the one created by the decorator i.e. captures
information from the function call.

    
    
    import math
    
    import mlflow
    
    mlflow.trace(math.factorial)(5)
    

Parameters

    

  * **func** â The function to be decorated. Must **not** be provided when using as a decorator.

  * **name** â The name of the span. If not provided, the name of the function will be used.

  * **span_type** â The type of the span. Can be either a string or a [`SpanType`](mlflow.entities.html#mlflow.entities.SpanType "mlflow.entities.SpanType") enum value.

  * **attributes** â A dictionary of attributes to set on the span.

`mlflow.``start_span`(_name : str = 'span'_, _span_type : Optional[str] =
'UNKNOWN'_, _attributes : Optional[Dict[str, Any]] = None_) ->
Generator[[LiveSpan](mlflow.entities.html#mlflow.entities.LiveSpan
"mlflow.entities.LiveSpan"), None,
None][[source]](../_modules/mlflow/tracing/fluent.html#start_span)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Context manager to create a new span and start it as the current span in the
context.

This context manager automatically manages the span lifecycle and parent-child
relationships. The span will be ended when the context manager exits. Any
exception raised within the context manager will set the span status to
`ERROR`, and detailed information such as exception message and stacktrace
will be recorded to the `attributes` field of the span. New spans can be
created within the context manager, then they will be assigned as child spans.

    
    
    import mlflow
    
    with mlflow.start_span("my_span") as span:
        x = 1
        y = 2
        span.set_inputs({"x": x, "y": y})
    
        z = x + y
    
        span.set_outputs(z)
        span.set_attribute("key", "value")
        # do something
    

When this context manager is used in the top-level scope, i.e. not within
another span context, the span will be treated as a root span. The root span
doesnât have a parent reference and **the entire trace will be logged when
the root span is ended**.

Tip

If you want more explicit control over the trace lifecycle, you can use
[`MLflow Client
APIs`](mlflow.client.html#mlflow.client.MlflowClient.start_trace
"mlflow.client.MlflowClient.start_trace"). It provides lower level to start
and end traces manually, as well as setting the parent spans explicitly.
However, it is generally recommended to use this context manager as long as it
satisfies your requirements, because it requires less boilerplate code and is
less error-prone.

Note

The context manager doesnât propagate the span context across threads. If
you want to create a child span in a different thread, you should use [`MLflow
Client APIs`](mlflow.client.html#mlflow.client.MlflowClient.start_trace
"mlflow.client.MlflowClient.start_trace") and pass the parent span ID
explicitly.

Note

All spans created under the root span (i.e. a single trace) are buffered in
memory and not exported until the root span is ended. The buffer has a default
size of 1000 traces and TTL of 1 hour. You can configure the buffer size and
TTL using the environment variables `MLFLOW_TRACE_BUFFER_MAX_SIZE` and
`MLFLOW_TRACE_BUFFER_TTL_SECONDS` respectively.

Parameters

    

  * **name** â The name of the span.

  * **span_type** â The type of the span. Can be either a string or a [`SpanType`](mlflow.entities.html#mlflow.entities.SpanType "mlflow.entities.SpanType") enum value

  * **attributes** â A dictionary of attributes to set on the span.

Returns

    

Yields an [`mlflow.entities.Span`](mlflow.entities.html#mlflow.entities.Span
"mlflow.entities.Span") that represents the created span.

`mlflow.``get_trace`(_request_id : str_) ->
Optional[[Trace](mlflow.entities.html#mlflow.entities.Trace
"mlflow.entities.Trace")][[source]](../_modules/mlflow/tracing/fluent.html#get_trace)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Get a trace by the given request ID if it exists.

This function retrieves the trace from the in-memory buffer first, and if it
doesnât exist, it fetches the trace from the tracking store. If the trace is
not found in the tracking store, it returns None.

Parameters

    

**request_id** â The request ID of the trace.

    
    
    import mlflow
    
    
    with mlflow.start_span(name="span") as span:
        span.set_attribute("key", "value")
    
    trace = mlflow.get_trace(span.request_id)
    print(trace)
    

Returns

    

A [`mlflow.entities.Trace`](mlflow.entities.html#mlflow.entities.Trace
"mlflow.entities.Trace") objects with the given request ID.

`mlflow.``search_traces`(_experiment_ids : Optional[List[str]] = None_,
_filter_string : Optional[str] = None_, _max_results : Optional[int] = None_,
_order_by : Optional[List[str]] = None_, _extract_fields : Optional[List[str]]
= None_) ->
pandas.DataFrame[[source]](../_modules/mlflow/tracing/fluent.html#search_traces)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Return traces that match the given list of search expressions within the
experiments.

Tip

This API returns a **Pandas DataFrame** that contains the traces as rows. To
retrieve a list of the original
[`Trace`](mlflow.entities.html#mlflow.entities.Trace "mlflow.entities.Trace")
objects, you can use the
[`MlflowClient().search_traces`](mlflow.client.html#mlflow.client.MlflowClient.search_traces
"mlflow.client.MlflowClient.search_traces") method instead.

Parameters

    

  * **experiment_ids** â List of experiment ids to scope the search. If not provided, the search will be performed across the current active experiment.

  * **filter_string** â A search filter string.

  * **max_results** â Maximum number of traces desired. If None, all traces matching the search expressions will be returned.

  * **order_by** â List of order_by clauses.

  * **extract_fields** â 

Specify fields to extract from traces using the format
`"span_name.[inputs|outputs].field_name"` or `"span_name.[inputs|outputs]"`.
For instance, `"predict.outputs.result"` retrieves the output `"result"` field
from a span named `"predict"`, while `"predict.outputs"` fetches the entire
outputs dictionary, including keys `"result"` and `"explanation"`.

By default, no fields are extracted into the DataFrame columns. When multiple
fields are specified, each is extracted as its own column. If an invalid field
string is provided, the function silently returns without adding that
fieldâs column. The supported fields are limited to `"inputs"` and
`"outputs"` of spans. If the span name or field name contains a dot it must be
enclosed in backticks. For example:

    
        # span name contains a dot
    extract_fields = ["`span.name`.inputs.field"]
    
    # field name contains a dot
    extract_fields = ["span.inputs.`field.name`"]
    
    # span name and field name contain a dot
    extract_fields = ["`span.name`.inputs.`field.name`"]
    

Returns

    

A Pandas DataFrame containing information about traces that satisfy the search
expressions.

Search traces with extract_fields

    
    
    import mlflow
    
    with mlflow.start_span(name="span1") as span:
        span.set_inputs({"a": 1, "b": 2})
        span.set_outputs({"c": 3, "d": 4})
    
    mlflow.search_traces(
        extract_fields=["span1.inputs", "span1.outputs", "span1.outputs.c"]
    )
    

Search traces with extract_fields and non-dictionary span inputs and outputs

    
    
    import mlflow
    
    with mlflow.start_span(name="non_dict_span") as span:
        span.set_inputs(["a", "b"])
        span.set_outputs([1, 2, 3])
    
    mlflow.search_traces(
        extract_fields=["non_dict_span.inputs", "non_dict_span.outputs"],
    )
    

`mlflow.``get_current_active_span`() ->
Optional[[LiveSpan](mlflow.entities.html#mlflow.entities.LiveSpan
"mlflow.entities.LiveSpan")][[source]](../_modules/mlflow/tracing/fluent.html#get_current_active_span)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Get the current active span in the global context.

Attention

This only works when the span is created with fluent APIs like @mlflow.trace
or with mlflow.start_span. If a span is created with MlflowClient APIs, it
wonât be attached to the global context so this function will not return it.

    
    
    import mlflow
    
    
    @mlflow.trace
    def f():
        span = mlflow.get_current_active_span()
        span.set_attribute("key", "value")
        return 0
    
    
    f()
    

Returns

    

The current active span if exists, otherwise None.

`mlflow.``get_last_active_trace`() ->
Optional[[Trace](mlflow.entities.html#mlflow.entities.Trace
"mlflow.entities.Trace")][[source]](../_modules/mlflow/tracing/fluent.html#get_last_active_trace)

    

Note

Experimental: This function may change or be removed in a future release
without warning.

Get the last active trace in the same process if exists.

Warning

This function DOES NOT work in the model deployed in Databricks model serving.

Note

The last active trace is only stored in-memory for the time defined by the TTL
(Time To Live) configuration. By default, the TTL is 1 hour and can be
configured using the environment variable `MLFLOW_TRACE_BUFFER_TTL_SECONDS`.

Note

This function returns an immutable copy of the original trace that is logged
in the tracking store. Any changes made to the returned object will not be
reflected in the original trace. To modify the already ended trace (while most
of the data is immutable after the trace is ended, you can still edit some
fields such as tags), please use the respective MlflowClient APIs with the
request ID of the trace, as shown in the example below.

    
    
    import mlflow
    
    
    @mlflow.trace
    def f():
        pass
    
    
    f()
    
    trace = mlflow.get_last_active_trace()
    
    
    # Use MlflowClient APIs to mutate the ended trace
    mlflow.MlflowClient().set_trace_tag(trace.info.request_id, "key", "value")
    

Returns

    

The last active trace if exists, otherwise None.

`mlflow.tracing.``disable`()[[source]](../_modules/mlflow/tracing/provider.html#disable)

    

Disable tracing.

Note

This function sets up OpenTelemetry to use
[NoOpTracerProvider](https://github.com/open-telemetry/opentelemetry-
python/blob/4febd337b019ea013ccaab74893bd9883eb59000/opentelemetry-
api/src/opentelemetry/trace/__init__.py#L222) and effectively disables all
tracing operations.

Example:

    
    
    import mlflow
    
    
    @mlflow.trace
    def f():
        return 0
    
    
    # Tracing is enabled by default
    f()
    assert len(mlflow.search_traces()) == 1
    
    # Disable tracing
    mlflow.tracing.disable()
    f()
    assert len(mlflow.search_traces()) == 1
    

`mlflow.tracing.``enable`()[[source]](../_modules/mlflow/tracing/provider.html#enable)

    

Enable tracing.

Example:

    
    
    import mlflow
    
    
    @mlflow.trace
    def f():
        return 0
    
    
    # Tracing is enabled by default
    f()
    assert len(mlflow.search_traces()) == 1
    
    # Disable tracing
    mlflow.tracing.disable()
    f()
    assert len(mlflow.search_traces()) == 1
    
    # Re-enable tracing
    mlflow.tracing.enable()
    f()
    assert len(mlflow.search_traces()) == 2
    

[ Previous](index.html "Python API") [Next ](mlflow.artifacts.html
"mlflow.artifacts")

* * *

(C) MLflow Project, a Series of LF Projects, LLC. All rights reserved.

