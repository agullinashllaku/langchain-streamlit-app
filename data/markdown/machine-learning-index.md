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

  * [English](../../en/machine-learning/index.html)
  * [æ¥æ¬èª](../../ja/machine-learning/index.html)
  * [PortuguÃªs](../../pt/machine-learning/index.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/machine-learning/)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/machine-learning/index.html)

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
  * AI and machine learning
    * [Tutorials](ml-tutorials.html)
    * [AI playground](../large-language-models/ai-playground.html)
    * [AI functions in SQL](../large-language-models/ai-functions.html)
    * [Serve models](serve-models.html)
    * [Train models](train-model/index.html)
    * [Serve data for AI](serve-data-ai.html)
    * [Evaluate AI](../generative-ai/agent-evaluation/index.html)
    * [Build gen AI apps](../generative-ai/build-genai-apps.html)
    * [MLOps and MLflow](../mlflow/index.html)
    * [Integrations](integrations.html)
    * [Graph and network analysis](graph-analysis.html)
    * [Reference solutions](reference-solutions/index.html)
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
  * AI and machine learning on Databricks
  * 

# AI and machine learning on Databricks

This article describes the tools that Mosaic AI (formerly Databricks Machine
Learning) provides to help you build AI and ML systems. The diagram shows how
various products on Databricks platform help you implement your end to end
workflows to build and deploy AI and ML systems

![Machine learning diagram: Model development and deployment on
Databricks](../_images/ml-diagram-model-development-deployment.png)

## Generative AI on Databricks

Mosaic AI unifies the AI lifecycle from data collection and preparation, to
model development and LLMOps, to serving and monitoring. The following
features are specifically optimized to facilitate the development of
generative AI applications:

  * [Unity Catalog](../data-governance/unity-catalog/index.html) for governance, discovery, versioning, and access control for data, features, models, and functions.

  * [MLflow](../mlflow/tracking.html) for model development tracking.

  * [Mosaic AI Model Serving](model-serving/index.html) for deploying LLMs. You can configure a model serving endpoint specifically for accessing generative AI models:

    * State-of-the-art open LLMs using [Foundation Model APIs](foundation-models/index.html).

    * Third-party models hosted outside of Databricks. See [External models in Mosaic AI Model Serving](../generative-ai/external-models/index.html).

  * [Mosaic AI Vector Search](../generative-ai/vector-search.html) provides a queryable vector database that stores embedding vectors and can be configured to automatically sync to your knowledge base.

  * [Lakehouse Monitoring](../lakehouse-monitoring/index.html) for data monitoring and tracking model prediction quality and drift using [automatic payload logging with inference tables](model-serving/inference-tables.html).

  * [AI Playground](../large-language-models/ai-playground.html) for testing generative AI models from your Databricks workspace. You can prompt, compare and adjust settings such as system prompt and inference parameters.

  * [Mosaic AI Model Training](../large-language-models/foundation-model-training/index.html) (formerly Foundation Model Training) for customizing a foundation model using your own data to optimize its performance for your specific application.

  * [Mosaic AI Agent Framework](../generative-ai/retrieval-augmented-generation.html) for building and deploying production-quality agents like Retrieval Augmented Generation (RAG) applications.

  * [Mosaic AI Agent Evaluation](../generative-ai/agent-evaluation/index.html) for evaluating the quality, cost, and latency of generative AI applications, including RAG applications and chains.

### What is generative AI?

Generative AI is a type of artificial intelligence focused on the ability of
computers to use models to create content like images, text, code, and
synthetic data.

Generative AI applications are built on top of generative AI models: large
language models (LLMs) and foundation models.

  * **LLMs** are deep learning models that consume and train on massive datasets to excel in language processing tasks. They create new combinations of text that mimic natural language based on their training data.

  * **Foundation models** are large ML models pre-trained with the intention that they are to be fine-tuned for more specific language understanding and generation tasks. These models are used to discern patterns within the input data.

After these models have completed their learning processes, together they
generate statistically probable outputs when prompted and they can be employed
to accomplish various tasks, including:

  * Image generation based on existing ones or utilizing the style of one image to modify or create a new one.

  * Speech tasks such as transcription, translation, question/answer generation, and interpretation of the intent or meaning of text.

Important

While many LLMs or other generative AI models have safeguards, they can still
generate harmful or inaccurate information.

Generative AI has the following design patterns:

  * Prompt Engineering: Crafting specialized prompts to guide LLM behavior

  * Retrieval Augmented Generation (RAG): Combining an LLM with external knowledge retrieval

  * Fine-tuning: Adapting a pre-trained LLM to specific data sets of domains

  * Pre-training: Training an LLM from scratch

## Machine learning on Databricks

With Mosaic AI, a single platform serves every step of ML development and
deployment, from raw data to inference tables that save every request and
response for a served model. Data scientists, data engineers, ML engineers and
DevOps can do their jobs using the same set of tools and a single source of
truth for the data.

Mosaic AI unifies the data layer and ML platform. All data assets and
artifacts, such as models and functions, are discoverable and governed in a
single catalog. Using a single platform for data and models makes it possible
to track lineage from the raw data to the production model. Built-in data and
model monitoring saves quality metrics to tables that are also stored in the
platform, making it easier to identify the root cause of model performance
problems. For more information about how Databricks supports the full ML
lifecycle and MLOps, see [MLOps workflows on Databricks](mlops/mlops-
workflow.html) and [MLOps Stacks: model development process as
code](mlops/mlops-stacks.html).

Some of the key components of the data intelligence platform are:

Tasks | Component  
---|---  
Govern and manage data, features, models, and functions. Also discovery, versioning, and lineage. |  [Unity Catalog](../data-governance/unity-catalog/index.html)  
Track changes to data, data quality, and model prediction quality |  [Lakehouse Monitoring](../lakehouse-monitoring/index.html), [Inference tables](model-serving/inference-tables.html)  
Feature development and management |  [Feature engineering and serving](feature-store/feature-function-serving.html).  
Train models |  [Databricks AutoML](automl/index.html), [Databricks notebooks](../notebooks/index.html)  
Track model development |  [MLflow tracking](../mlflow/tracking.html)  
Serve custom models |  [Mosaic AI Model Serving](model-serving/index.html).  
Build automated workflows and production-ready ETL pipelines |  [Databricks Jobs](../jobs/index.html)  
Git integration |  [Databricks Git folders](../repos/index.html)  
  
## Deep learning on Databricks

Configuring infrastructure for deep learning applications can be difficult.
[Databricks Runtime for Machine Learning](databricks-runtime-ml.html) takes
care of that for you, with clusters that have built-in compatible versions of
the most common deep learning libraries like TensorFlow, PyTorch, and Keras.

Databricks Runtime ML clusters also include pre-configured GPU support with
drivers and supporting libraries. It also supports libraries like
[Ray](ray/index.html) to parallelize compute processing for scaling ML
workflows and ML applications.

Databricks Runtime ML clusters also include pre-configured GPU support with
drivers and supporting libraries. [Mosaic AI Model Serving](model-
serving/index.html) enables creation of scalable GPU endpoints for deep
learning models with no extra configuration.

For machine learning applications, Databricks recommends using a cluster
running Databricks Runtime for Machine Learning. See [Create a cluster using
Databricks Runtime ML](databricks-runtime-ml.html#create-ml-cluster).

To get started with deep learning on Databricks, see:

  * [Best practices for deep learning on Databricks](train-model/dl-best-practices.html)

  * [Deep learning on Databricks](train-model/deep-learning.html)

  * [Reference solutions for deep learning](reference-solutions/index.html)

## Next steps

To get started, see:

  * [Tutorials: Get started with AI and machine learning](ml-tutorials.html)

For a recommended MLOps workflow on Databricks Mosaic AI, see:

  * [MLOps workflows on Databricks](mlops/mlops-workflow.html)

To learn about key Databricks Mosaic AI features, see:

  * [What is AutoML?](automl/index.html)

  * [Feature engineering and serving](feature-store/index.html)

  * [Model serving with Databricks](model-serving/index.html)

  * [Lakehouse Monitoring](../lakehouse-monitoring/index.html)

  * [Manage model lifecycle](manage-model-lifecycle/index.html)

  * [MLflow experiment tracking](../mlflow/tracking.html)

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

