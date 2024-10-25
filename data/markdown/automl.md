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

  * [English](../../../en/machine-learning/automl/index.html)
  * [æ¥æ¬èª](../../../ja/machine-learning/automl/index.html)
  * [PortuguÃªs](../../../pt/machine-learning/automl/index.html)

[![](../../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/machine-learning/automl/)
  * [![](../../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/machine-learning/automl/index.html)

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
      * AutoML
        * [Train with AutoML UI](train-ml-model-automl-ui.html)
        * [Train with AutoML API](train-ml-model-automl-api.html)
        * [Data preparation settings](automl-data-preparation.html)
        * [AutoML Python API reference](automl-api-reference.html)
      * [Gen AI models](../../large-language-models/foundation-model-training/index.html)
      * [Model training examples](../train-model/training-examples.html)
      * [Deep learning](../train-model/deep-learning.html)
      * [Train recommender models](../train-recommender-models.html)
    * [Serve data for AI](../serve-data-ai.html)
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
  * [Train AI and ML models](../train-model/index.html)
  * What is AutoML?
  * 

# What is AutoML?

Databricks AutoML simplifies the process of applying machine learning to your
datasets by automatically finding the best algorithm and hyperparameter
configuration for you.

Provide your dataset and specify the type of machine learning problem, then
AutoML does the following:

  1. Cleans and [prepares your data](automl-data-preparation.html).

  2. Orchestrates distributed model training and hyperparameter tuning across multiple algorithms.

  3. Finds the best model using open source evaluation algorithms from scikit-learn, xgboost, LightGBM, Prophet, and ARIMA.

  4. Presents the results. AutoML also generates source code notebooks for each trial, allowing you to review, reproduce, and modify the code as needed.

Get started with AutoML experiments through a [low-code UI](train-ml-model-
automl-ui.html) or the [Python API](train-ml-model-automl-api.html).

## Requirements

  * Databricks Runtime 9.1 ML or above. For the general availability (GA) version, Databricks Runtime 10.4 LTS ML or above.

    * For time series forecasting, Databricks Runtime 10.0 ML or above.

    * With Databricks Runtime 9.1 LTS ML and above, AutoML depends on the `databricks-automl-runtime` package, which contains components that are useful outside of AutoML and also helps simplify the notebooks generated by AutoML training. `databricks-automl-runtime` is available on [PyPI](https://pypi.org/project/databricks-automl-runtime/).

  * No additional libraries other than those preinstalled in Databricks Runtime for Machine Learning should be installed on the cluster.

    * Any modification (removal, upgrades, or downgrades) to existing library versions results in run failures due to incompatibility.

  * To access files in your workspace, you must have network ports 1017 and 1021 open for AutoML experiments. To open these ports or confirm they are open, review your cloud VPN firewall configuration and security group rules or contact your local cloud administrator. For additional information on workspace configuration and deployment, see [Create a workspace](../../admin/workspace/index.html).

  * Use a compute resource with a supported [compute access mode](../../compute/configure.html#access-mode). Not all compute access modes have access to the Unity Catalog:

Compute access mode | AutoML support | Unity Catalog support  
---|---|---  
**single user** | Supported (must be the designated single user for the cluster) | Supported  
**Shared access mode** | Unsupported | Unsupported  
**No isolation shared** | Supported | Unsupported  

## AutoML algorithms

Databricks AutoML trains and evaluates models based on the algorithms in the
following table.

Note

For classification and regression models, the decision tree, random forests,
logistic regression, and linear regression with stochastic gradient descent
algorithms are based on scikit-learn.

Classification models | Regression models | Forecasting models  
---|---|---  
[Decision trees](https://scikit-learn.org/stable/modules/tree.html#classification) |  [Decision trees](https://scikit-learn.org/stable/modules/tree.html#regression) | [Prophet](https://facebook.github.io/prophet/docs/quick_start.html#python-api)  
[Random forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) |  [Random forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor) |  [Auto-ARIMA](https://pypi.org/project/pmdarima/) (Available in Databricks Runtime 10.3 ML and above.)  
[Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) |  [Linear regression with stochastic gradient descent](https://scikit-learn.org/stable/modules/sgd.html#regression) |   
[XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier) |  [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor) |   
[LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) |  [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html) |   
  
## Trial notebook generation

AutoML generates notebooks of the source code behind trials so you can review,
reproduce, and modify the code as needed.

For forecasting experiments, AutoML-generated notebooks are automatically
imported to your workspace for all trials of your experiment.

For classification and regression experiments, AutoML-generated notebooks for
data exploration and the best trial in your experiment are automatically
imported to your workspace. Generated notebooks for other experiment trials
are saved as MLflow artifacts on DBFS instead of auto-imported into your
workspace. For all trials besides the best trial, the `notebook_path` and
`notebook_url` in the `TrialInfo` Python API are not set. If you need to use
these notebooks, you can manually import them into your workspace with the
AutoML experiment UI or the `databricks.automl.import_notebook` [Python
API](automl-api-reference.html#import-notebook).

If you only use the data exploration notebook or best trial notebook generated
by AutoML, the **Source** column in the AutoML experiment UI contains the link
to the generated notebook for the best trial.

If you use other generated notebooks in the AutoML experiment UI, these are
not automatically imported into the workspace. You can find the notebooks by
clicking into each MLflow run. The IPython notebook is saved in the
**Artifacts** section of the run page. You can download this notebook and
import it into the workspace, if downloading artifacts is enabled by your
workspace administrators.

## Shapley values (SHAP) for model explainability

Note

For MLR 11.1 and below, SHAP plots are not generated if the dataset contains a
`datetime` column.

The notebooks produced by AutoML regression and classification runs include
code to calculate [Shapley
values](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html#).
Shapley values are based in game theory and estimate the importance of each
feature to a modelâs predictions.

AutoML notebooks calculate Shapley values using the [SHAP
package](https://shap.readthedocs.io/en/latest/overviews.html). Because these
calculations are highly memory-intensive, the calculations are not performed
by default.

To calculate and display Shapley values:

  1. Go to the **Feature importance** section in an AutoML-generated trial notebook.

  2. Set `shap_enabled = True`.

  3. Re-run the notebook.

## Next steps

  * [Train with AutoML UI](train-ml-model-automl-ui.html)
  * [Train with AutoML API](train-ml-model-automl-api.html)
  * [Data preparation settings](automl-data-preparation.html)
  * [AutoML Python API reference](automl-api-reference.html)

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

