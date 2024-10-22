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

  * [English](../../../en/machine-learning/manage-model-lifecycle/index.html)
  * [æ¥æ¬èª](../../../ja/machine-learning/manage-model-lifecycle/index.html)
  * [PortuguÃªs](../../../pt/machine-learning/manage-model-lifecycle/index.html)

[![](../../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/machine-learning/manage-model-lifecycle/)
  * [![](../../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/machine-learning/manage-model-lifecycle/index.html)

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
    * [Evaluate AI](../../generative-ai/agent-evaluation/index.html)
    * [Build gen AI apps](../../generative-ai/build-genai-apps.html)
    * [MLOps and MLflow](../../mlflow/index.html)
      * [MLOps workflows on Databricks](../mlops/mlops-workflow.html)
      * [Get started with MLflow experiments](../../mlflow/quick-start.html)
      * [MLflow experiment tracking](../track-model-development/index.html)
      * [Log, load, register, and deploy MLflow models](../../mlflow/models.html)
      * Manage model lifecycle
        * [Upgrade ML workflows to target models in Unity Catalog](upgrade-workflows.html)
        * [Upgrade models to Unity Catalog](upgrade-models.html)
        * [Models in Unity Catalog example](../../mlflow/models-in-uc-example.html)
        * [Manage model lifecycle using the Workspace Model Registry (legacy)](workspace-model-registry.html)
        * [Workspace Model Registry example](../../mlflow/workspace-model-registry-example.html)
      * [Run MLflow Projects on Databricks](../../mlflow/projects.html)
      * [Copy MLflow objects between workspaces](../../mlflow/migrate-mlflow-objects.html)
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

Updated Oct 21, 2024

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation
Feedback)

  * [Documentation](../../index.html)
  * [AI and machine learning on Databricks](../index.html)
  * [ML lifecycle management using MLflow](../../mlflow/index.html)
  * Manage model lifecycle in Unity Catalog
  * 

# Manage model lifecycle in Unity Catalog

Important

  * This article documents Models in Unity Catalog, which Databricks recommends for governing and deploying models. If your workspace is not enabled for Unity Catalog, the functionality on this page is not available. Instead, see [Manage model lifecycle using the Workspace Model Registry (legacy)](workspace-model-registry.html). For guidance on how to upgrade from the Workspace Model Registry to Unity Catalog, see Migrate workflows and models to Unity Catalog.

  * Models in Unity Catalog isnât available in AWS GovCloud regions.

This article describes how to use Models in Unity Catalog as part of your
machine learning workflow to manage the full lifecycle of ML models.
Databricks provides a hosted version of MLflow Model Registry in [Unity
Catalog](../../data-governance/unity-catalog/index.html). Models in Unity
Catalog extends the benefits of Unity Catalog to ML models, including
centralized access control, auditing, lineage, and model discovery across
workspaces. Models in Unity Catalog is compatible with the open-source MLflow
Python client.

For an overview of Model Registry concepts, see [ML lifecycle management using
MLflow](../../mlflow/index.html).

## Requirements

  1. Unity Catalog must be enabled in your workspace. See [Get started using Unity Catalog](../../data-governance/unity-catalog/get-started.html) to create a Unity Catalog Metastore, enable it in a workspace, and create a catalog. If Unity Catalog is not enabled, use the [workspace model registry](workspace-model-registry.html).

  2. You must use a compute resource that has access to Unity Catalog. For ML workloads, this means that the access mode for the compute must be **Single user**. For more information, see [Access modes](../../compute/configure.html#access-mode).

  3. To create new registered models, you need the following privileges:

     * `USE SCHEMA` and `USE CATALOG` privileges on the schema and its enclosing catalog.

     * `CREATE_MODEL` privilege on the schema. To grant this privilege, use the Catalog Explorer UI or the following [SQL GRANT command](../../sql/language-manual/security-grant.html):
    
        GRANT CREATE_MODEL ON SCHEMA <schema-name> TO <principal>
    

Note

Your workspace must be attached to a Unity Catalog metastore that supports
privilege inheritance. This is true for all metastores created after August
25, 2022. If running on an older metastore, [follow docs](../../data-
governance/unity-catalog/manage-privileges/upgrade-privilege-model.html) to
upgrade.

## Install and configure MLflow client for Unity Catalog

This section includes instructions for installing and configuring the MLflow
client for Unity Catalog.

### Install MLflow Python client

Support for models in Unity Catalog is included in Databricks Runtime 13.2 ML
and above.

You can also use models in Unity Catalog on Databricks Runtime 11.3 LTS and
above by installing the latest version of the MLflow Python client in your
notebook, using the following code.

    
    
    %pip install --upgrade "mlflow-skinny[databricks]"
    dbutils.library.restartPython()
    

### Configure MLflow client to access models in Unity Catalog

If your workspaceâs [default catalog](../../catalogs/default.html) is in
Unity Catalog (rather than `hive_metastore`) and you are running a cluster
using Databricks Runtime 13.3 LTS or above, models are automatically created
in and loaded from the default catalog. You do not have to perform this step.

For other workspaces, the MLflow Python client creates models in the
Databricks workspace model registry. To upgrade to models in Unity Catalog,
use the following code in your notebooks to configure the MLflow client:

    
    
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    

For a small number of workspaces where both the default catalog was configured
to a catalog in Unity Catalog prior to January 2024 and the workspace model
registry was used prior to January 2024, you must manually set the default
catalog to Unity Catalog using the command shown above.

## Train and register Unity Catalog-compatible models

**Permissions required** : To create a new registered model, you need the
`CREATE_MODEL` and `USE SCHEMA` privileges on the enclosing schema, and `USE
CATALOG` privilege on the enclosing catalog. To create new model versions
under a registered model, you must be the owner of the registered model and
have `USE SCHEMA` and `USE CATALOG` privileges on the schema and catalog
containing the model.

ML model versions in UC must have a [model
signature](https://mlflow.org/docs/latest/models.html#model-signature). If
youâre not already logging MLflow models with signatures in your model
training workloads, you can either:

  * Use [Databricks autologging](../../mlflow/databricks-autologging.html), which automatically logs models with signatures for many popular ML frameworks. See supported frameworks in the [MLflow docs](https://mlflow.org/docs/latest/tracking.html#automatic-logging).

  * With MLflow 2.5.0 and above, you can specify an input example in your `mlflow.<flavor>.log_model` call, and the model signature is automatically inferred. For further information, refer to [the MLflow documentation](https://mlflow.org/docs/latest/models.html#how-to-log-models-with-signatures).

Then, pass the three-level name of the model to MLflow APIs, in the form
`<catalog>.<schema>.<model>`.

The examples in this section create and access models in the `ml_team` schema
under the `prod` catalog.

The model training examples in this section create a new model version and
register it in the `prod` catalog. Using the `prod` catalog doesnât
necessarily mean that the model version serves production traffic. The model
versionâs enclosing catalog, schema, and registered model reflect its
environment (`prod`) and associated governance rules (for example, privileges
can be set up so that only admins can delete from the `prod` catalog), but not
its deployment status. To manage the deployment status, use model aliases.

### Register a model to Unity Catalog using autologging

To register a model, use MLflow Client API `register_model()` method. See
[mlflow.register_model](https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=register_model#mlflow.register_model).

    
    
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier
    
    # Train a sklearn model on the iris dataset
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    clf = RandomForestClassifier(max_depth=7)
    clf.fit(X, y)
    
    # Note that the UC model name follows the pattern
    # <catalog_name>.<schema_name>.<model_name>, corresponding to
    # the catalog, schema, and registered model name
    # in Unity Catalog under which to create the version
    # The registered model will be created if it doesn't already exist
    autolog_run = mlflow.last_active_run()
    model_uri = "runs:/{}/model".format(autolog_run.info.run_id)
    mlflow.register_model(model_uri, "prod.ml_team.iris_model")
    

### Register a model using the API

    
    
    mlflow.register_model(
      "runs:/<run_uuid>/model", "prod.ml_team.iris_model"
    )
    

### Register a model to Unity Catalog with automatically inferred signature

Support for automatically inferred signatures is available in MLflow version
2.5.0 and above, and is supported in Databricks Runtime 11.3 LTS ML and above.
To use automatically inferred signatures, use the following code to install
the latest MLflow Python client in your notebook:

    
    
    %pip install --upgrade "mlflow-skinny[databricks]"
    dbutils.library.restartPython()
    

The following code shows an example of an automatically inferred signature.

    
    
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier
    
    with mlflow.start_run():
        # Train a sklearn model on the iris dataset
        X, y = datasets.load_iris(return_X_y=True, as_frame=True)
        clf = RandomForestClassifier(max_depth=7)
        clf.fit(X, y)
        # Take the first row of the training dataset as the model input example.
        input_example = X.iloc[[0]]
        # Log the model and register it as a new version in UC.
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            # The signature is automatically inferred from the input example and its predicted output.
            input_example=input_example,
            registered_model_name="prod.ml_team.iris_model",
        )
    

### Register a model using the UI

Follow these steps:

  1. From the experiment run page, click **Register model** in the upper-right corner of the UI.

  2. In the dialog, select **Unity Catalog** , and select a destination model from the drop down list.

![Register model version dialog with dropdown menu](../../_images/uc-register-
model-dialog.png)

  3. Click **Register**.

![Register model version dialog with button](../../_images/uc-register-model-
button.png)

Registering a model can take time. To monitor progress, navigate to the
destination model in Unity Catalog and refresh periodically.

## Deploy models using aliases

Model aliases allow you to assign a mutable, named reference to a particular
version of a registered model. You can use aliases to indicate the deployment
status of a model version. For example, you could allocate a âChampionâ
alias to the model version currently in production and target this alias in
workloads that use the production model. You can then update the production
model by reassigning the âChampionâ alias to a different model version.

### Set and delete aliases on models

**Permissions required** : Owner of the registered model, plus `USE SCHEMA`
and `USE CATALOG` privileges on the schema and catalog containing the model.

You can set, update, and remove aliases for models in Unity Catalog by using
[Catalog Explorer](../../catalog-explorer/explore-models.html). You can manage
aliases across a registered model in the [model details page](../../catalog-
explorer/explore-models.html#view-model-information) and configure aliases for
a specific model version in the [model version details page](../../catalog-
explorer/explore-models.html#view-model-version-information).

To set, update, and delete aliases using the MLflow Client API, see the
examples below:

    
    
    from mlflow import MlflowClient
    client = MlflowClient()
    
    # create "Champion" alias for version 1 of model "prod.ml_team.iris_model"
    client.set_registered_model_alias("prod.ml_team.iris_model", "Champion", 1)
    
    # reassign the "Champion" alias to version 2
    client.set_registered_model_alias("prod.ml_team.iris_model", "Champion", 2)
    
    # get a model version by alias
    client.get_model_version_by_alias("prod.ml_team.iris_model", "Champion")
    
    # delete the alias
    client.delete_registered_model_alias("prod.ml_team.iris_model", "Champion")
    

For more details on alias client APIs, see the [MLflow API
documentation](https://mlflow.org/docs/latest/python_api/mlflow.client.html).

### Load model version by alias for inference workloads

**Permissions required** : `EXECUTE` privilege on the registered model, plus
`USE SCHEMA` and `USE CATALOG` privileges on the schema and catalog containing
the model.

Batch inference workloads can reference a model version by alias. The snippet
below loads and applies the âChampionâ model version for batch inference.
If the âChampionâ version is updated to reference a new model version, the
batch inference workload automatically picks it up on its next execution. This
allows you to decouple model deployments from your batch inference workloads.

    
    
    import mlflow.pyfunc
    model_version_uri = "models:/prod.ml_team.iris_model@Champion"
    champion_version = mlflow.pyfunc.load_model(model_version_uri)
    champion_version.predict(test_x)
    

Model serving endpoints can also reference a model version by alias. You can
write deployment workflows to get a model version by alias and update a model
serving endpoint to serve that version, using the [model serving REST
API](../model-serving/create-manage-serving-endpoints.html#endpoint-config).
For example:

    
    
    import mlflow
    import requests
    client = mlflow.tracking.MlflowClient()
    champion_version = client.get_model_version_by_alias("prod.ml_team.iris_model", "Champion")
    # Invoke the model serving REST API to update endpoint to serve the current "Champion" version
    model_name = champion_version.name
    model_version = champion_version.version
    requests.request(...)
    

### Load model version by version number for inference workloads

You can also load model versions by version number:

    
    
    import mlflow.pyfunc
    # Load version 1 of the model "prod.ml_team.iris_model"
    model_version_uri = "models:/prod.ml_team.iris_model/1"
    first_version = mlflow.pyfunc.load_model(model_version_uri)
    first_version.predict(test_x)
    

## Share models across workspaces

### Share models with users in the same region

As long as you have the appropriate privileges, you can access models in Unity
Catalog from any workspace that is attached to the metastore containing the
model. For example, you can access models from the `prod` catalog in a dev
workspace, to facilitate comparing newly-developed models to the production
baseline.

To collaborate with other users (share write privileges) on a registered model
you created, you must grant ownership of the model to a group containing
yourself and the users youâd like to collaborate with. Collaborators must
also have the `USE CATALOG` and `USE SCHEMA` privileges on the catalog and
schema containing the model. See [Unity Catalog privileges and securable
objects](../../data-governance/unity-catalog/manage-
privileges/privileges.html) for details.

### Share models with users in another region or account

To share models with users in other regions or accounts, use the Delta Sharing
[Databricks-to-Databricks sharing flow](../../delta-sharing/index.html#delta-
sharing). See [Add models to a share](../../delta-sharing/create-
share.html#models) (for providers) and [Get access in the Databricks-to-
Databricks model](../../delta-sharing/recipient.html#get-access-db-to-db) (for
recipients). As a recipient, after you create a catalog from a share, you
access models in that shared catalog the same way as any other model in Unity
Catalog.

## Track the data lineage of a model in Unity Catalog

Note

Support for table to model lineage in Unity Catalog is available in MLflow
2.11.0 and above.

When you train a model on a table in Unity Catalog, you can track the lineage
of the model to the upstream dataset(s) it was trained and evaluated on. To do
this, use
[mlflow.log_input](https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=log_input#mlflow.log_input).
This saves the input table information with the MLflow run that generated the
model. Data lineage is also automatically captured for models logged using
feature store APIs. See [Feature governance and lineage](../feature-
store/lineage.html).

When you register the model to Unity Catalog, lineage information is
automatically saved and is visible in the **Lineage** tab of the [model
version UI in Catalog Explorer](../../catalog-explorer/explore-
models.html#view-model-version-information).

The following code shows an example.

    
    
    import mlflow
    import pandas as pd
    import pyspark.pandas as ps
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestRegressor
    
    # Write a table to Unity Catalog
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df.rename(
      columns = {
        'sepal length (cm)':'sepal_length',
        'sepal width (cm)':'sepal_width',
        'petal length (cm)':'petal_length',
        'petal width (cm)':'petal_width'},
      inplace = True
    )
    iris_df['species'] = iris.target
    ps.from_pandas(iris_df).to_table("prod.ml_team.iris", mode="overwrite")
    
    # Load a Unity Catalog table, train a model, and log the input table
    dataset = mlflow.data.load_delta(table_name="prod.ml_team.iris", version="0")
    pd_df = dataset.df.toPandas()
    X = pd_df.drop("species", axis=1)
    y = pd_df["species"]
    with mlflow.start_run():
        clf = RandomForestRegressor(n_estimators=100)
        clf.fit(X, y)
        mlflow.log_input(dataset, "training")
        # Take the first row of the training dataset as the model input example.
        input_example = X.iloc[[0]]
        # Log the model and register it as a new version in UC.
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            # The signature is automatically inferred from the input example and its predicted output.
            input_example=input_example,
            registered_model_name="prod.ml_team.iris_classifier",
        )
    

## Control access to models

In Unity Catalog, registered models are a subtype of the `FUNCTION` securable
object. To grant access to a model registered in Unity Catalog, you use `GRANT
ON FUNCTION`. For details, see [Unity Catalog privileges and securable
objects](../../data-governance/unity-catalog/manage-
privileges/privileges.html). For best practices on organizing models across
catalogs and schemas, see [Organize your data](../../data-governance/unity-
catalog/best-practices.html#organize-data).

You can configure model permissions programmatically using the [Grants REST
API](https://docs.databricks.com/api/workspace/grants). When you configure
model permissions, set `securable_type` to `"FUNCTION"` in REST API requests.
For example, use `PATCH /api/2.1/unity-
catalog/permissions/function/{full_name}` to update registered model
permissions.

## View models in the UI

**Permissions required** : To view a registered model and its model versions
in the UI, you need `EXECUTE` privilege on the registered model, plus `USE
SCHEMA` and `USE CATALOG` privileges on the schema and catalog containing the
model

You can view and manage registered models and model versions in Unity Catalog
using the [Catalog Explorer](../../catalog-explorer/index.html).

## Rename a model

**Permissions required** : Owner of the registered model, `CREATE_MODEL`
privilege on the schema containing the registered model, and `USE SCHEMA` and
`USE CATALOG` privileges on the schema and catalog containing the model.

To rename a registered model, use the MLflow Client API
`rename_registered_model()` method:

    
    
    client=MlflowClient()
    client.rename_registered_model("<full-model-name>", "<new-model-name>")
    

## Copy a model version

You can copy a model version from one model to another in Unity Catalog.

### Copy a model version using the UI

Follow these steps:

  1. From the model version page, click **Copy this version** in the upper-right corner of the UI.

  2. Select a destination model from the drop down list and click **Copy**.

![Copy model version dialog](../../_images/uc-copy-model-dialog.png)

Copying a model can take time. To monitor progress, navigate to the
destination model in Unity Catalog and refresh periodically.

### Copy a model version using the API

To copy a model version, use the MLflowâs
[copy_model_version()](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.copy_model_version)
Python API:

    
    
    client = MlflowClient()
    client.copy_model_version(
      "models:/<source-model-name>/<source-model-version>",
      "<destination-model-name>",
    )
    

## Delete a model or model version

**Permissions required** : Owner of the registered model, plus `USE SCHEMA`
and `USE CATALOG` privileges on the schema and catalog containing the model.

You can delete a registered model or a model version within a registered model
using the [Catalog Explorer UI](../../catalog-explorer/explore-models.html) or
the API.

Warning

You cannot undo this action. When you delete a model, all model artifacts
stored by Unity Catalog and all the metadata associated with the registered
model are deleted.

### Delete a model version or model using the UI

To delete a model or model version in Unity Catalog, follow these steps.

  1. On the model page or model version page, click the kebab menu ![Kebab menu](../../_images/kebab-menu.png) in the upper-right corner.

From the model page:

![model page kebab menu with delete](../../_images/uc-delete-model-dialog.png)

From the model version page:

![model version page kebab menu with delete](../../_images/uc-delete-model-
version-dialog.png)

  2. Select **Delete**.

  3. A confirmation dialog appears. Click **Delete** to confirm.

### Delete a model version or model using the API

To delete a model version, use the MLflow Client API `delete_model_version()`
method:

    
    
    # Delete versions 1,2, and 3 of the model
    client = MlflowClient()
    versions=[1, 2, 3]
    for version in versions:
      client.delete_model_version(name="<model-name>", version=version)
    

To delete a model, use the MLflow Client API `delete_registered_model()`
method:

    
    
    client = MlflowClient()
    client.delete_registered_model(name="<model-name>")
    

## Use tags on models

[Tags](../../database-objects/tags.html) are key-value pairs that you
associate with registered models and model versions, allowing you to label and
categorize them by function or status. For example, you could apply a tag with
key `"task"` and value `"question-answering"` (displayed in the UI as
`task:question-answering`) to registered models intended for question
answering tasks. At the model version level, you could tag versions undergoing
pre-deployment validation with `validation_status:pending` and those cleared
for deployment with `validation_status:approved`.

**Permissions required** : Owner of or have `APPLY_TAG` privilege on the
registered model, plus `USE SCHEMA` and `USE CATALOG` privileges on the schema
and catalog containing the model.

See [Add and update tags using Catalog Explorer](../../database-
objects/tags.html#catalog-explorer) on how to set and delete tags using the
UI.

To set and delete tags using the MLflow Client API, see the examples below:

    
    
    from mlflow import MlflowClient
    client = MlflowClient()
    
    # Set registered model tag
    client.set_registered_model_tag("prod.ml_team.iris_model", "task", "classification")
    
    # Delete registered model tag
    client.delete_registered_model_tag("prod.ml_team.iris_model", "task")
    
    # Set model version tag
    client.set_model_version_tag("prod.ml_team.iris_model", "1", "validation_status", "approved")
    
    # Delete model version tag
    client.delete_model_version_tag("prod.ml_team.iris_model", "1", "validation_status")
    

Both registered model and model version tags must meet the [platform-wide
constraints](../../database-objects/tags.html#constraint).

For more details on tag client APIs, see the [MLflow API
documentation](https://mlflow.org/docs/latest/python_api/mlflow.client.html).

## Add a description (comments) to a model or model version

**Permissions required** : Owner of the registered model, plus `USE SCHEMA`
and `USE CATALOG` privileges on the schema and catalog containing the model.

You can include a text description for any model or model version in Unity
Catalog. For example, you can provide an overview of the problem or
information about the methodology and algorithm used.

For models, you also have the option of using AI-generated comments. See [Add
AI-generated comments to Unity Catalog objects](../../comments/ai-
comments.html).

### Add a description to a model using the UI

To add a description for a model, you can use AI-generated comments, or you
can enter your own comments. You can edit AI-generated comments as necessary.

  * To add automatically generated comments, click the **AI generate** button.

  * To add your own comments, click **Add**. Enter your comments in the dialog, and click **Save**.

![uc model description buttons](../../_images/uc-model-description.png)

### Add a description to a model version using the UI

To add a description to a model version in Unity Catalog, follow these steps:

  1. On the model version page, click the pencil icon under **Description**.

![pencil icon to add comments to a model version](../../_images/uc-model-
version-description.png)

  2. Enter your comments in the dialog, and click **Save**.

### Add a description to a model or model version using the API

To update a registered model description, use the MLflow Client API
`update_registered_model()` method:

    
    
    client = MlflowClient()
    client.update_registered_model(
      name="<model-name>",
      description="<description>"
    )
    

To update a model version description, use the MLflow Client API
`update_model_version()` method:

    
    
    client = MlflowClient()
    client.update_model_version(
      name="<model-name>",
      version=<model-version>,
      description="<description>"
    )
    

## List and search models

To get a list of registered models in Unity Catalog, use MLflowâs
[search_registered_models()](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_registered_models)
Python API:

    
    
    client=MlflowClient()
    client.search_registered_models()
    

To search for a specific model name and get information about that modelâs
versions, use `search_model_versions()`:

    
    
    from pprint import pprint
    
    client=MlflowClient()
    [pprint(mv) for mv in client.search_model_versions("name='<model-name>'")]
    

Note

Not all search API fields and operators are supported for models in Unity
Catalog. See Limitations for details.

## Download model files (advanced use case)

In most cases, to load models, you should use MLflow APIs like
`mlflow.pyfunc.load_model` or `mlflow.<flavor>.load_model` (for example,
`mlflow.transformers.load_model` for HuggingFace models).

In some cases you may need to download model files to debug model behavior or
model loading issues. You can download model files using
`mlflow.artifacts.download_artifacts`, as follows:

    
    
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    model_uri = f"models:/{model_name}/{version}" # reference model by version or alias
    destination_path = "/local_disk0/model"
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=destination_path)
    

## Promote a model across environments

Databricks recommends that you deploy ML pipelines as code. This eliminates
the need to promote models across environments, as all production models can
be produced through automated training workflows in a production environment.

However, in some cases, it may be too expensive to retrain models across
environments. Instead, you can copy model versions across registered models in
Unity Catalog to promote them across environments.

You need the following privileges to execute the example code below:

  * `USE CATALOG` on the `staging` and `prod` catalogs.

  * `USE SCHEMA` on the `staging.ml_team` and `prod.ml_team` schemas.

  * `EXECUTE` on `staging.ml_team.fraud_detection`.

In addition, you must be the owner of the registered model
`prod.ml_team.fraud_detection`.

The following code snippet uses the `copy_model_version` [MLflow Client
API](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.copy_model_version),
available in MLflow version 2.8.0 and above.

    
    
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    
    client = mlflow.tracking.MlflowClient()
    src_model_name = "staging.ml_team.fraud_detection"
    src_model_version = "1"
    src_model_uri = f"models:/{src_model_name}/{src_model_version}"
    dst_model_name = "prod.ml_team.fraud_detection"
    copied_model_version = client.copy_model_version(src_model_uri, dst_model_name)
    

After the model version is in the production environment, you can perform any
necessary pre-deployment validation. Then, you can mark the model version for
deployment using aliases.

    
    
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(name="prod.ml_team.fraud_detection", alias="Champion", version=copied_model_version.version)
    

In the example above, only users who can read from the
`staging.ml_team.fraud_detection` registered model and write to the
`prod.ml_team.fraud_detection` registered model can promote staging models to
the production environment. The same users can also use aliases to manage
which model versions are deployed within the production environment. You
donât need to configure any other rules or policies to govern model
promotion and deployment.

You can customize this flow to promote the model version across multiple
environments that match your setup, such as `dev`, `qa`, and `prod`. Access
control is enforced as configured in each environment.

## Example

This example illustrates how to use Models in Unity Catalog to build a machine
learning application.

[Models in Unity Catalog example](../../mlflow/models-in-uc-example.html)

## Migrate workflows and models to Unity Catalog

Databricks recommends using Models in Unity Catalog for improved governance,
easy sharing across workspaces and environments, and more flexible MLOps
workflows. The table compares the capabilities of the Workspace Model Registry
and Unity Catalog.

Capability | Workspace Model Registry (legacy) | Models in Unity Catalog (recommended)  
---|---|---  
Reference model versions by named aliases | Model Registry Stages: Move model versions into one of four fixed stages to reference them by that stage. Cannot rename or add stages. | Model Registry Aliases: Create up to 10 custom and reassignable named references to model versions for each registered model.  
Create access-controlled environments for models | Model Registry Stages: Use stages within one registered model to denote the environment of its model versions, with access controls for only two of the four fixed stages (`Staging` and `Production`). | Registered Models: Create a registered model for each environment in your MLOps workflow, utilizing three-level namespaces and permissions of Unity Catalog to express governance.  
Promote models across environments (deploy model) | Use the `transition_model_version_stage()` MLflow Client API to move a model version to a different stage, potentially breaking workflows that reference the previous stage. | Use the `copy_model_version()` MLflow Client API to copy a model version from one registered model to another.  
Access and share models across workspaces | Manually export and import models across workspaces, or configure connections to remote model registries using personal access tokens and workspace secret scopes. | Out of the box access to models across workspaces in the same account. No configuration required.  
Configure permissions | Set permissions at the workspace-level. | Set permissions at the account-level, which applies consistent governance across workspaces.  
Access models in the Databricks markplace | Unavailable. | Load models from the Databricks marketplace into your Unity Catalog metastore and access them across workspaces.  
  
The articles linked below describe how to migrate workflows (model training
and batch inference jobs) and models from the Workspace Model Registry to
Unity Catalog.

  * [Upgrade ML workflows to target models in Unity Catalog](upgrade-workflows.html)

  * [Upgrade models to Unity Catalog](upgrade-models.html)

## Limitations

  * Stages are not supported for models in Unity Catalog. Databricks recommends using the three-level namespace in Unity Catalog to express the environment a model is in, and using aliases to promote models for deployment. See Promote a model across environments for details.

  * Webhooks are not supported for models in Unity Catalog. See suggested alternatives in [the upgrade guide](upgrade-workflows.html#manual-approval).

  * Some search API fields and operators are not supported for models in Unity Catalog. This can be mitigated by calling the search APIs using supported filters and scanning the results. Following are some examples:

    * The `order_by` parameter is not supported in the [search_model_versions](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_model_versions) or [search_registered_models](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_registered_models) client APIs.

    * Tag-based filters (`tags.mykey = 'myvalue'`) are not supported for `search_model_versions` or `search_registered_models`.

    * Operators other than exact equality (for example, `LIKE`, `ILIKE`, `!=`) are not supported for `search_model_versions` or `search_registered_models`.

    * Searching registered models by name (for example, `MlflowClient().search_registered_models(filter_string="name='main.default.mymodel'")` is not supported. To fetch a particular registered model by name, use [get_registered_model](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.get_registered_model).

  * Email notifications and comment discussion threads on registered models and model versions are not supported in Unity Catalog.

  * The activity log is not supported for models in Unity Catalog. To track activity on models in Unity Catalog, use [audit logs](../../admin/account-settings/audit-logs.html#uc).

  * `search_registered_models` might return stale results for models shared through Delta Sharing. To ensure the most recent results, use the Databricks CLI or [SDK](https://databricks-sdk-py.readthedocs.io/en/latest/workspace/catalog/registered_models.html#databricks.sdk.service.catalog.RegisteredModelsAPI.list) to list the models in a schema.

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

