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

  * [English](../../../en/machine-learning/model-serving/index.html)
  * [æ¥æ¬èª](../../../ja/machine-learning/model-serving/index.html)
  * [PortuguÃªs](../../../pt/machine-learning/model-serving/index.html)

[![](../../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/)
  * [![](../../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/machine-learning/model-serving/index.html)

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
      * [Pre-trained models](../../generative-ai/pretrained-models.html)
      * [AI Gateway](../../ai-gateway/index.html)
      * Model serving
        * [Manage model serving endpoints](manage-serving-endpoints.html)
        * [Serve multiple models to a model serving endpoint](serve-multiple-models-to-serving-endpoint.html)
        * [Monitor model quality and endpoint health](monitor-diagnose-endpoints.html)
        * [Debugging guide for Model Serving](model-serving-debug.html)
        * [Model Serving limits and regions](model-serving-limits.html)
        * [Migrate to Model Serving](migrate-model-serving.html)
      * [Custom Python models](custom-models.html)
      * [Foundation Model APIs](../foundation-models/index.html)
      * [Query foundation models and external models](score-foundation-models.html)
      * [Batch inference](../model-inference/index.html)
    * [Train models](../train-model/index.html)
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
  * [Serve models with Databricks](../serve-models.html)
  * Model serving with Databricks
  * 

# Model serving with Databricks

This article describes Mosaic AI Model Serving, including its advantages and
limitations.

## What is Mosaic AI Model Serving?

Mosaic AI Model Serving provides a unified interface to deploy, govern, and
query AI models for real-time and batch inference. Each model you serve is
available as a REST API that you can integrate into your web or client
application.

Model Serving provides a highly available and low-latency service for
deploying models. The service automatically scales up or down to meet demand
changes, saving infrastructure costs while optimizing latency performance.
This functionality uses [serverless compute](../../getting-
started/overview.html#serverless). See the [Model Serving pricing
page](https://www.databricks.com/product/pricing/model-serving) for more
details.

Model serving supports serving:

  * [Custom models](custom-models.html). These are Python models packaged in the MLflow format. They can be registered either in Unity Catalog or in the workspace model registry. Examples include scikit-learn, XGBoost, PyTorch, and Hugging Face transformer models.

    * Agent serving is supported as a custom model. See [Deploy an agent for generative AI application](../../generative-ai/deploy-agent.html)

  * State-of-the-art open models made available by [Foundation Model APIs](../foundation-models/index.html). These models are curated foundation model architectures that support optimized inference. Base models, like Llama-2-70B-chat, BGE-Large, and Mistral-7B are available for immediate use with **pay-per-token** pricing, and workloads that require performance guarantees and fine-tuned model variants can be deployed with **provisioned throughput**.

    * Databricks recommends using `ai_query` with Model Serving for batch inference. For quick experimentation, `ai_query` can be used with [pay-per-token endpoints](../foundation-models/index.html#token-foundation-apis). When you are ready to run batch inference on large or production data, Databricks recommends using provisioned throughput endpoints for faster performance. See [Provisioned throughput Foundation Model APIs](../foundation-models/deploy-prov-throughput-foundation-model-apis.html) for how to create a provisioned throughput endpoint.

      * See [Perform batch inference using ai_query](../../large-language-models/ai-query-batch-inference.html).

      * To get started with batch inference with LLMs on Unity Catalog> tables, see the notebook examples in [Batch inference using Foundation Model APIs provisioned throughput](../model-inference/batch-inference-throughput.html).

  * [External models](../../generative-ai/external-models/index.html). These are generative AI models that are hosted outside of Databricks. Examples include models like, OpenAIâs GPT-4, Anthropicâs Claude, and others. Endpoints that serve external models can be centrally governed and customers can establish rate limits and access control for them.

Note

You can interact with supported large language models using the [AI
Playground](../../large-language-models/ai-playground.html). The AI Playground
is a chat-like environment where you can test, prompt, and compare LLMs. This
functionality is available in your Databricks workspace.

Model serving offers a unified REST API and MLflow Deployment API for CRUD and
querying tasks. In addition, it provides a single UI to manage all your models
and their respective serving endpoints. You can also access models directly
from SQL using [AI functions](../../large-language-models/ai-functions.html)
for easy integration into analytics workflows.

For an introductory tutorial on how to serve custom models on Databricks, see
[Tutorial: Deploy and query a custom model](model-serving-intro.html).

For a getting started tutorial on how to query a foundation model on
Databricks, see [Get started querying LLMs on Databricks](../../large-
language-models/llm-serving-intro.html).

## Why use Model Serving?

  * **Deploy and query any models** : Model Serving provides a unified interface that so you can manage all models in one location and query them with a single API, regardless of whether they are hosted on Databricks or externally. This approach simplifies the process of experimenting with, customizing, and deploying models in production across various clouds and providers.

  * **Securely customize models with your private data** : Built on a Data Intelligence Platform, Model Serving simplifies the integration of features and embeddings into models through native integration with the [Databricks Feature Store](../feature-store/automatic-feature-lookup.html) and [Mosaic AI Vector Search](../../generative-ai/vector-search.html). For even more improved accuracy and contextual understanding, models can be fine-tuned with proprietary data and deployed effortlessly on Model Serving.

  * **Govern and monitor models** : The Serving UI allows you to centrally manage all model endpoints in one place, including those that are externally hosted. You can manage permissions, track and set usage limits, and monitor the [quality of all types of models](inference-tables.html). This enables you to democratize access to SaaS and open LLMs within your organization while ensuring appropriate guardrails are in place.

  * **Reduce cost with optimized inference and fast scaling** : Databricks has implemented a range of optimizations to ensure you get the best throughput and latency for large models. The endpoints automatically scale up or down to meet demand changes, saving infrastructure costs while optimizing latency performance. [Monitor model serving costs](../../admin/system-tables/model-serving-cost.html).

Note

For workloads that are latency sensitive or involve a high number of queries
per second, Databricks recommends using [route optimization](route-
optimization.html) on custom model serving endpoints. Reach out to your
Databricks account team to ensure your workspace is enabled for high
scalability.

  * **Bring reliability and security to Model Serving** : Model Serving is designed for high-availability, low-latency production use and can support over 25K queries per second with an overhead latency of less than 50 ms. The serving workloads are protected by multiple layers of security, ensuring a secure and reliable environment for even the most sensitive tasks.

Note

Model Serving does not provide security patches to existing model images
because of the risk of destabilization to production deployments. A new model
image created from a new model version will contain the latest patches. Reach
out to your Databricks account team for more information.

## Requirements

  * Registered model in [Unity Catalog](../../data-governance/unity-catalog/index.html) or the [Workspace Model Registry](../manage-model-lifecycle/workspace-model-registry.html).

  * Permissions on the registered models as described in [Serving endpoint ACLs](../../security/auth/access-control/index.html#serving-endpoints).

  * MLflow 1.29 or higher

## Enable Model Serving for your workspace

To use Model Serving, your account admin must read and accept the terms and
conditions for enabling serverless compute in the account console.

Note

If your account was created after March 28, 2022, serverless compute is
enabled by default for your workspaces.

If you are not an account admin, you cannot perform these steps. Contact an
account admin if your workspace needs access to serverless compute.

  1. As an account admin, go to the [feature enablement tab of the account console settings page](https://accounts.cloud.databricks.com/settings/feature-enablement).

  2. A banner at the top of the page prompts you to accept the additional terms. Once you read the terms, click **Accept**. If you do not see the banner asking you to accept the terms, this step has been completed already.

After youâve accepted the terms, your account is enabled for serverless.

No additional steps are required to enable Model Serving in your workspace.

## Limitations and region availability

Mosaic AI Model Serving imposes default limits to ensure reliable performance.
See [Model Serving limits and regions](model-serving-limits.html). If you have
feedback on these limits or an endpoint in an unsupported region, reach out to
your Databricks account team.

## Data protection in Model Serving

Databricks takes data security seriously. Databricks understands the
importance of the data you analyze using Mosaic AI Model Serving, and
implements the following security controls to protect your data.

  * Every customer request to Model Serving is logically isolated, authenticated, and authorized.

  * Mosaic AI Model Serving encrypts all data at rest (AES-256) and in transit (TLS 1.2+).

For all paid accounts, Mosaic AI Model Serving does not use user inputs
submitted to the service or outputs from the service to train any models or
improve any Databricks services.

For Databricks Foundation Model APIs, as part of providing the service,
Databricks may temporarily process and store inputs and outputs for the
purposes of preventing, detecting, and mitigating abuse or harmful uses. Your
inputs and outputs are isolated from those of other customers, stored in the
same region as your workspace for up to thirty (30) days, and only accessible
for detecting and responding to security or abuse concerns.

## Additional resources

  * [Get started querying LLMs on Databricks](../../large-language-models/llm-serving-intro.html).

  * [Tutorial: Deploy and query a custom model](model-serving-intro.html)

  * [Introduction to building gen AI apps on Databricks](../../generative-ai/build-genai-apps.html)

  * [Deploy custom models](custom-models.html).

  * [Migrate to Model Serving](migrate-model-serving.html)

  * [Migrate optimized LLM serving endpoints to provisioned throughput](migrate-provisioned-throughput.html)

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

