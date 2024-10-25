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

  * [English](../../en/repos/index.html)
  * [æ¥æ¬èª](../../ja/repos/index.html)
  * [PortuguÃªs](../../pt/repos/index.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/repos/)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/repos/index.html)

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
  * [Generative AI tutorial](../generative-ai/tutorials/ai-cookbook/index.html)
  * [Business intelligence](../ai-bi/index.html)
  * [Data warehousing](../sql/index.html)
  * [Notebooks](../notebooks/index.html)
  * [Delta Lake](../delta/index.html)
  * [Developers](../languages/index.html)
    * [Python](../languages/python.html)
    * [R](../sparkr/index.html)
    * [Scala](../languages/scala.html)
    * [SQL](../sql/language-manual/index.html)
    * [User-defined functions (UDFs)](../udf/index.html)
    * [Databricks Apps](../dev-tools/databricks-apps/index.html)
    * [Tools](../dev-tools/index.html)
      * [Authentication](../dev-tools/auth/index.html)
      * [Databricks Connect](../dev-tools/databricks-connect/index.html)
      * [Visual Studio Code](../dev-tools/visual-studio-code.html)
      * [Databricks extension for Visual Studio Code](../dev-tools/vscode-ext/index.html)
      * [PyCharm overview](../dev-tools/pycharm.html)
      * [IntelliJ IDEA](../dev-tools/intellij-idea.html)
      * [Eclipse](../dev-tools/eclipse.html)
      * [RStudio Desktop](../dev-tools/rstudio.html)
      * [SDKs](../dev-tools/sdks.html)
      * [SQL drivers and tools](../dev-tools/sql-drivers-tools.html)
      * [Databricks CLI](../dev-tools/cli/index.html)
      * [Databricks Asset Bundles](../dev-tools/bundles/index.html)
      * [Utilities](../dev-tools/databricks-utils.html)
      * [IaC](../dev-tools/iac.html)
      * [CI/CD](../dev-tools/ci-cd.html)
      * Git folders
        * [What happened to Repos?](what-happened-repos.html)
        * [Set up Git folders](repos-setup.html)
        * [Enable or disable Git](enable-disable-repos-with-api.html)
        * [Set up private Git connectivity](git-proxy.html)
        * [Configure and connect to Git folders](get-access-tokens-from-git-provider.html)
        * [Swap Git credentials](swap-git-credentials.html)
        * [Git operations with Git folders](git-operations-with-repos.html)
        * [CI/CD techniques with Git folders](ci-cd-techniques-with-repos.html)
        * [Limitations and FAQ](limits.html)
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
  * [Develop on Databricks](../languages/index.html)
  * [Developer tools and guidance](../dev-tools/index.html)
  * Git integration for Databricks Git folders
  * 

# Git integration for Databricks Git folders

Databricks Git folders is a visual Git client and API in Databricks. It
supports common Git operations such as cloning a repository, committing and
pushing, pulling, branch management, and visual comparison of diffs when
committing.

Within Git folders you can develop code in notebooks or other files and follow
data science and engineering code development best practices using Git for
version control, collaboration, and CI/CD.

Note

Git folders (Repos) are primarily designed for authoring and collaborative
workflows.

## What can you do with Databricks Git folders?

Databricks Git folders provides source control for data and AI projects by
integrating with Git providers.

In Databricks Git folders, you can use Git functionality to:

  * Clone, push to, and pull from a remote Git repository.

  * Create and manage branches for development work, including merging, rebasing, and resolving conflicts.

  * Create notebooks (including IPYNB notebooks) and edit them and other files.

  * Visually compare differences upon commit and resolve merge conflicts.

For step-by-step instructions, see [Run Git operations on Databricks Git
folders (Repos)](git-operations-with-repos.html).

Note

Databricks Git folders also has an
[API](https://docs.databricks.com/api/workspace/repos) that you can integrate
with your CI/CD pipeline. For example, you can programmatically update a
Databricks repo so that it always has the most recent version of the code. For
information about best practices for code development using Databricks Git
folders, see [CI/CD techniques with Git and Databricks Git folders
(Repos)](ci-cd-techniques-with-repos.html).

For information on the kinds of notebooks supported in Databricks, see [Export
and import Databricks notebooks](../notebooks/notebook-export-import.html).

## Supported Git providers

Databricks Git folders are backed by an integrated Git repository. The
repository can be hosted by any of the cloud and enterprise Git providers
listed in the following section.

Note

**What is a âGit providerâ?**

A âGit providerâ is the specific (named) service that hosts a source
control model based on Git. Git-based source control platforms are hosted in
two ways: as a cloud service hosted by the developing company, or as an on-
premises service installed and managed by your own company on its own
hardware. Many Git providers such as GitHub, Microsoft, GitLab, and Atlassian
provide both cloud-based SaaS and on-premises (sometimes called âself-
managedâ) Git services.

When choosing your Git provider during configuration, you must be aware of the
differences between cloud (SaaS) and on-premises Git providers. On-premises
solutions are typically hosted behind a company VPN and might not be
accessible from the internet. Usually, the on-premises Git providers have a
name ending in âServerâ or âSelf-Managedâ, but if you are uncertain,
contact your company admins or review the Git providerâs documentation.

If your Git provider is cloud-based and not listed as a supported provider,
selecting âGitHubâ as your provider may work but is not guaranteed.

Note

If you are using âGitHubâ as a provider and are still uncertain if you are
using the cloud or on-premises version, see [About GitHub Enterprise
Server](https://docs.github.com/en/github-ae@latest/get-started/using-github-
docs/about-versions-of-github-docs#github-enterprise-server) in the GitHub
docs.

### Cloud Git providers supported by Databricks

  * GitHub, GitHub AE, and GitHub Enterprise Cloud

  * Atlassian BitBucket Cloud

  * GitLab and GitLab EE

  * Microsoft Azure DevOps (Azure Repos)

  * AWS CodeCommit

### On-premises Git providers supported by Databricks

  * GitHub Enterprise Server

  * Atlassian BitBucket Server and Data Center

  * GitLab Self-Managed

  * Microsoft Azure DevOps Server: A workspace admin must explicitly allowlist the URL domain prefixes for your Microsoft Azure DevOps Server if the URL does not match `dev.azure.com/*` or `visualstudio.com/*`. For more details, see [Restrict usage to URLs in an allow list](repos-setup.html#allow-lists)

If you are integrating an on-premises Git repo that is not accessible from the
internet, a proxy for Git authentication requests must also be installed
within your companyâs VPN. For more details, see [Set up private Git
connectivity for Databricks Git folders (Repos)](git-proxy.html).

To learn how to use access tokens with your Git provider, see [Configure Git
credentials & connect a remote repo to Databricks](get-access-tokens-from-git-
provider.html).

## Resources for Git integration

Use the Databricks CLI 2.0 for Git integration with Databricks:

  * [Download the latest CLI version](https://github.com/databricks/cli/releases)

  * [Set up the CLI](../dev-tools/cli/install.html)

Read the following reference docs:

  * [Databricks CLI global flags](../dev-tools/cli/commands.html#global-flags) and [commands](../dev-tools/cli/commands.html)

## Next steps

  * [Set up Databricks Git folders (Repos)](repos-setup.html)

  * [Configure Git credentials & connect a remote repo to Databricks](get-access-tokens-from-git-provider.html)

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

