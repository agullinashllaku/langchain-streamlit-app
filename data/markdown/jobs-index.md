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

  * [English](../../en/jobs/index.html)
  * [æ¥æ¬èª](../../ja/jobs/index.html)
  * [PortuguÃªs](../../pt/jobs/index.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/jobs/)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/jobs/index.html)

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
  * Schedule and orchestrate workflows
    * [Tutorial](jobs-quickstart.html)
    * [Configure jobs](configure-job.html)
    * [Configure tasks](configure-task.html)
    * [Schedules & triggers](triggers.html)
    * [Parameters](parameters.html)
    * [Identities and privileges](privileges.html)
    * [Configure compute](compute.html)
    * [Monitor jobs](monitor.html)
    * [Troubleshoot and repair job failures](repair-job-failures.html)
    * [Examples](how-to/index.html)
  * [Monitor data and AI assets](../lakehouse-monitoring/index.html)
  * [Share data securely](../data-sharing/index.html)

Work with data

  * [Data engineering](../workspace-index.html)
  * [AI and machine learning](../machine-learning/index.html)
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

Updated Oct 21, 2024

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation
Feedback)

  * [Documentation](../index.html)
  * Schedule and orchestrate workflows
  * 

# Schedule and orchestrate workflows

Databricks Workflows has tools that allow you to schedule and orchestrate data
processing tasks on Databricks. You use Databricks Workflows to configure
Databricks Jobs.

This article introduces concepts and choices related to managing production
workloads using Databricks Jobs.

## What are Databricks Jobs?

A job is the primary unit for scheduling and orchestrating production
workloads on Databricks. Jobs consist of one or more tasks. Together, tasks
and jobs allow you to configure and deploy the following:

  * Custom logic, including Spark, SQL, OSS Python, ML, and arbitrary code.

  * Compute resources with custom environments and libraries.

  * Schedules and triggers for running workloads.

  * Conditional logic for control flow between tasks.

Jobs provide a procedural approach to defining relationships between _tasks_.
Delta Live Tables pipelines provide a declarative approach to defining
relationships between _datasets_ and _transformations_. You can include Delta
Live Tables pipelines as a task in a job. See [Delta Live Tables pipeline task
for jobs](pipeline.html).

Jobs can vary in complexity from a single task running a Databricks notebook
to thousands of tasks running with conditional logic and dependencies.

## How can I configure and run Jobs?

You can create and run a job using the Jobs UI, the Databricks CLI, or by
invoking the Jobs API. Using the UI or API, you can repair and re-run a failed
or canceled job. You can monitor job run results using the UI, CLI, API, and
notifications (for example, email, webhook destination, or Slack
notifications).

If you prefer an infrastructure-as-code (IaC) approach to configuring and
orchestrating your Jobs, use Databricks Asset Bundles (DABs). Bundles can
contain YAML definitions of jobs and tasks, are managed using the Databricks
CLI, and can be shared and run in different target workspaces (such as
development, staging, and production). To learn about using DABs to configure
and orchestrate your jobs, see [Databricks Asset Bundles](../dev-
tools/bundles/index.html).

To learn about using the Databricks CLI, see [What is the Databricks
CLI?](../dev-tools/cli/index.html). To learn about using the Jobs API, see the
[Jobs API](https://docs.databricks.com/api/workspace/jobs).

## What is the minimum configuration needed for a job?

All jobs on Databricks require the following:

  * Source code (such as a Databricks notebook) that contains logic to be run.

  * A compute resource to run the logic. The compute resource can be serverless compute, classic jobs compute, or all-purpose compute. See [Configure compute for jobs](compute.html).

  * A specified schedule for when the job should be run. Optionally, you can omit setting a schedule and trigger the job manually.

  * A unique name.

Note

If you develop your code in Databricks notebooks, you can use the **Schedule**
button to configure that notebook as a job. See [Create and manage scheduled
notebook jobs](../notebooks/schedule-notebook-jobs.html).

## What is a task?

A task represents a unit of logic to be run as a step in a job. Tasks can
range in complexity and can include the following:

  * A notebook

  * A JAR

  * SQL queries

  * A DLT pipeline

  * Another job

  * Control flow tasks

You can control the execution order of tasks by specifying dependencies
between them. You can configure tasks to run in sequence or parallel.

Jobs interact with state information and metadata of tasks, but task scope is
isolated. You can use task values to share context between scheduled tasks.
See [Use task values to pass information between tasks](task-values.html).

## What control flow options are available for jobs?

When configuring jobs and tasks in jobs, you can customize settings that
control how the entire job and individual tasks run. These options are:

  * Triggers

  * Retries

  * Run if conditional tasks

  * If/else conditional tasks

  * For each tasks

  * Duration thresholds

  * Concurrency settings

### Trigger types

You must specify a trigger type when you configure a job. You can choose from
the following trigger types:

  * [Scheduled](scheduled.html)

  * [File arrival](file-arrival-triggers.html)

  * [Continuous](continuous.html)

You can also choose to trigger your job manually, but this is mainly reserved
for specific use cases such as:

  * You use an external orchestration tool to trigger jobs using REST API calls.

  * You have a job that runs rarely and requires manual intervention for validation or resolving data quality issues.

  * You are running a workload that only needs to be run once or a few times, such as a migration.

See [Trigger types for Databricks Jobs](triggers.html).

### Retries

Retries specify how many times a particular task should be re-run if the task
fails with an error message. Errors are often transient and resolved through
restart. Some features on Databricks, such as schema evolution with Structured
Streaming, assume that you run jobs with retries to reset the environment and
allow a workflow to proceed.

If you specify retries for a task, the task restarts up to the specified
number of times if it encounters an error. Not all job configurations support
task retries. See [Set a retry policy](configure-task.html#retries).

When running in continuous trigger mode, Databricks automatically retries with
exponential backoff. See [How are failures handled for continuous
jobs?](continuous.html#exponential-backoff).

### Run if conditional tasks

You can use the **Run if** task type to specify conditionals for later tasks
based on the outcome of other tasks. You add tasks to your job and specify
upstream-dependent tasks. Based on the status of those tasks, you can
configure one or more downstream tasks to run. Jobs support the following
dependencies:

  * All succeeded

  * At least one succeeded

  * None failed

  * All done

  * At least one failed

  * All failed

See [Configure task dependencies](run-if.html)

### If/else conditional tasks

You can use the **If/else** task type to specify conditionals based on some
value. See [Add branching logic to a job with the If/else task](if-else.html).

Jobs support `taskValues` that you define in your logic and allow you to
return the results of some computation or state from a task to the jobs
environment. You can define **If/else** conditions against `taskValues`, job
parameters, or dynamic values.

Databricks supports the following operands for conditionals:

  * `==`

  * `!=`

  * `>`

  * `>=`

  * `<`

  * `<=`

See also:

  * [Use task values to pass information between tasks](task-values.html)

  * [What is a dynamic value reference?](dynamic-value-references.html)

  * [Parameterize jobs](parameters.html)

### For each tasks

Use the `For each` task to run another task in a loop, passing a different set
of parameters to each iteration of the task.

Adding the `For each` task to a job requires defining two tasks: The `For
each` task and a _nested task_. The nested task is the task to run for each
iteration of the `For each` task and is one of the standard Databricks Jobs
task types. Multiple methods are supported for passing parameters to the
nested task.

See [Run a parameterized Databricks job task in a loop](for-each.html).

### Duration threshold

You can specify a duration threshold to send a warning or stop a task or job
if a specified duration is exceeded. Examples of when you might want to
configure this setting include the following:

  * You have tasks prone to getting stuck in a hung state.

  * You must warn an engineer if an SLA for a workflow is exceeded.

  * To avoid unexpected costs, you want to fail a job configured with a large cluster.

See [Configure an expected completion time or a timeout for a job](configure-
job.html#timeout-setting-job) and [Configure an expected completion time or a
timeout for a task](configure-task.html#timeout-setting-task).

### Concurrency

Most jobs are configured with the default concurrency of 1 concurrent job.
This means that if a previous job run has not completed by the time a new job
should be triggered, the next job run is skipped.

Some use cases exist for increased concurrency, but most workloads do not
require altering this setting.

For more information about configuring concurrency, see [Databricks Jobs
queueing and concurrency settings](advanced.html).

## How can I monitor jobs?

The jobs UI lets you see job runs, including runs in progress. See [Monitoring
and observability for Databricks Jobs](monitor.html).

You can receive notifications when a job or task starts, completes, or fails.
You can send notifications to one or more email addresses or system
destinations. See [Add email and system notifications for job
events](notifications.html).

System tables include a `lakeflow` schema where you can view records related
to job activity in your account. See [Jobs system table
reference](../admin/system-tables/jobs.html).

You can also join the jobs system tables with billing tables to monitor the
cost of jobs across your account. See [Monitor job costs with system
tables](../admin/system-tables/jobs-cost.html).

## Limitations

The following limitations exist:

  * A workspace is limited to 2000 concurrent task runs. A `429 Too Many Requests` response is returned when you request a run that cannot start immediately.

  * The number of jobs a workspace can create in an hour is limited to 10000 (includes âruns submitâ). This limit also affects jobs created by the REST API and notebook workflows.

  * A workspace can contain up to 12000 saved jobs.

  * A job can contain up to 100 tasks.

## Can I manage workflows programmatically?

Databricks has tools and APIs that allow you to schedule and orchestrate your
workflows programmatically, including the following:

  * [Databricks CLI](../dev-tools/cli/index.html)

  * [Databricks Asset Bundles](../dev-tools/bundles/index.html)

  * [Databricks extension for Visual Studio Code](../dev-tools/vscode-ext/index.html)

  * [Databricks SDKs](../dev-tools/sdks.html)

  * [Jobs REST API](https://docs.databricks.com/api/workspace/jobs)

For more information about developer tools, see [Developer tools and
guidance](../dev-tools/index.html).

### Workflow orchestration with Apache AirFlow

You can use [Apache Airflow](https://airflow.apache.org/) to manage and
schedule your data workflows. With Airflow, you define your workflow in a
Python file, and Airflow manages scheduling and running the workflow. See
[Orchestrate Databricks jobs with Apache Airflow](how-to/use-airflow-with-
jobs.html).

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

