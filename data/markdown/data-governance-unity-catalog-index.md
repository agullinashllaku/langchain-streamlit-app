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

  * [English](../../../en/data-governance/unity-catalog/index.html)
  * [æ¥æ¬èª](../../../ja/data-governance/unity-catalog/index.html)
  * [PortuguÃªs](../../../pt/data-governance/unity-catalog/index.html)

[![](../../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/data-governance/unity-catalog/)
  * [![](../../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/data-governance/unity-catalog/index.html)

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
  * [AI and machine learning](../../machine-learning/index.html)
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
  * [Data governance (Unity Catalog)](../index.html)
    * Unity Catalog
      * [Unity Catalog best practices](best-practices.html)
      * [Set up and manage Unity Catalog](get-started.html)
      * [Create a Unity Catalog metastore](create-metastore.html)
      * [Enable a workspace for Unity Catalog](enable-workspaces.html)
      * [Manage privileges](manage-privileges/index.html)
      * [Capture and view data lineage using Unity Catalog](data-lineage.html)
      * [Control external access to data in Unity Catalog](access-open-api.html)
      * [Paths for Unity Catalog data](paths.html)
      * [Connect BI tools](business-intelligence.html)
      * [Audit Unity Catalog events](audit.html)
      * [Monitor resource quotas](resource-quotas.html)
      * [Configure Unity Catalog storage account for CORS](storage-cors.html)
      * [Work with Unity Catalog and the legacy Hive metastore](hive-metastore.html)
      * [Upgrade Hive tables and views to Unity Catalog](migrate.html)
      * [Migrate to Unity Catalog using UCX](ucx.html)
      * [Automate Unity Catalog setup using Terraform](automate.html)
    * [What is Catalog Explorer?](../../catalog-explorer/index.html)
    * [Hive metastore table access control (legacy)](../table-acls/index.html)
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
  * [Data governance with Unity Catalog](../index.html)
  * What is Unity Catalog?
  * 

# What is Unity Catalog?

This article introduces Unity Catalog, a unified governance solution for data
and AI assets on Databricks.

Note

Unity Catalog is also available as an open-source implementation. See [the
announcement blog](https://www.databricks.com/blog/open-sourcing-unity-
catalog) and the public [Unity Catalog GitHub
repo](https://github.com/unitycatalog/unitycatalog/blob/main/README.md).

## Overview of Unity Catalog

Unity Catalog provides centralized access control, auditing, lineage, and data
discovery capabilities across Databricks workspaces.

![Unity Catalog diagram](../../_images/with-unity-catalog.png)

Key features of Unity Catalog include:

  * **Define once, secure everywhere** : Unity Catalog offers a single place to administer data access policies that apply across all workspaces.

  * **Standards-compliant security model** : Unity Catalogâs security model is based on standard ANSI SQL and allows administrators to grant permissions in their existing data lake using familiar syntax, at the level of catalogs, schemas (also called databases), tables, and views.

  * **Built-in auditing and lineage** : Unity Catalog automatically captures user-level audit logs that record access to your data. Unity Catalog also captures lineage data that tracks how data assets are created and used across all languages.

  * **Data discovery** : Unity Catalog lets you tag and document data assets, and provides a search interface to help data consumers find data.

  * **System tables (Public Preview)** : Unity Catalog lets you easily access and query your accountâs operational data, including audit logs, billable usage, and lineage.

## The Unity Catalog object model

In Unity Catalog, all metadata is registered in a metastore. The hierarchy of
database objects in any Unity Catalog metastore is divided into three levels,
represented as a three-level namespace (`catalog.schema.table-etc`) when you
reference tables, views, volumes, models, and functions.

![Unity Catalog object model diagram](../../_images/object-model.png)

### Metastores

The metastore is the top-level container for metadata in Unity Catalog. It
registers metadata about data and AI assets and the permissions that govern
access to them. For a workspace to use Unity Catalog, it must have a Unity
Catalog metastore attached.

You should have one metastore for each region in which you have workspaces.
Typically, a metastore is created automatically when you create a Databricks
workspace in a region for the first time. For some older accounts, an account
admin must create the metastore and assign the workspaces in that region to
the metastore.

See [Create a Unity Catalog metastore](create-metastore.html).

### Object hierarchy in the metastore

In a Unity Catalog metastore, the three-level database object hierarchy
consists of catalogs that contain schemas, which in turn contain data and AI
objects, like tables and models.

**Level one:**

  * **Catalogs** are used to organize your data assets and are typically used as the top level in your data isolation scheme. Catalogs often mirror organizational units or software development lifecycle scopes. See [What are catalogs in Databricks?](../../catalogs/index.html).

  * **Non-data securable objects** , such as storage credentials and external locations, are used to managed your data governance model in Unity Catalog. These also live directly under the metastore. They are described in more detail in Other securable objects.

**Level two:**

  * **Schemas** (also known as databases) contain tables, views, volumes, AI models, and functions. Schemas organize data and AI assets into logical categories that are more granular than catalogs. Typically a schema represents a single use case, project, or team sandbox. See [What are schemas in Databricks?](../../schemas/index.html).

**Level three:**

  * **Volumes** are logical volumes of unstructured, non-tabular data in cloud object storage. Volumes can be either _managed_ , with Unity Catalog managing the full lifecycle and layout of the data in storage, or _external_ , with Unity Catalog managing access to the data from within Databricks, but not managing access to the data in cloud storage from other clients. See [What are Unity Catalog volumes?](../../volumes/index.html) and Managed versus external tables and volumes.

  * **Tables** are collections of data organized by rows and columns. Tables can be either _managed_ , with Unity Catalog managing the full lifecycle of the table, or _external_ , with Unity Catalog managing access to the data from within Databricks, but not managing access to the data in cloud storage from other clients. See [What is a table?](../../tables/index.html) and Managed versus external tables and volumes.

  * **Views** are saved queries against one or more tables. See [What is a view?](../../views/index.html).

  * **Functions** are units of saved logic that return a scalar value or set of rows. See [User-defined functions (UDFs) in Unity Catalog](../../udf/unity-catalog.html).

  * **Models** are AI models packaged with MLflow and registered in Unity Catalog as functions. See [Manage model lifecycle in Unity Catalog](../../machine-learning/manage-model-lifecycle/index.html).

## Working with database objects in Unity Catalog

Working with database objects in Unity Catalog is very similar to working with
database objects that are registered in a Hive metastore, with the exception
that a Hive metastore doesnât include catalogs in the object namespace. You
can use familiar ANSI syntax to create database objects, manage database
objects, manage permissions, and work with data in Unity Catalog. You can also
create database objects, manage database objects, and manage permissions on
database objects using the Catalog Explorer UI.

For more information, see [Database objects in Databricks](../../database-
objects/index.html) and [Work with Unity Catalog and the legacy Hive
metastore](hive-metastore.html).

## Other securable objects

In addition to the database objects and AI assets that are contained in
schemas, Unity Catalog also governs access to data using the following
securable objects:

  * **Storage credentials** , which encapsulate a long-term cloud credential that provides access to cloud storage. See [Create a storage credential for connecting to AWS S3](../../connect/unity-catalog/storage-credentials.html).

  * **External locations** , which contain a reference to a storage credential and a cloud storage path. External locations can be used to create external tables or to assign a **managed storage location** for managed tables and volumes. See [Create an external location to connect cloud storage to Databricks](../../connect/unity-catalog/external-locations.html), Data isolation using managed storage, and [Specify a managed storage location in Unity Catalog](../../connect/unity-catalog/managed-storage.html).

  * **Connections** , which represent credentials that give read-only access to an external database in a database system like MySQL using Lakehouse Federation. See Lakehouse Federation and Unity Catalog and [What is Lakehouse Federation?](../../query-federation/index.html).

  * **Clean rooms** , which represent a Databricks-managed environment where multiple participants can collaborate on projects without sharing underlying data with each other. See [What is Databricks Clean Rooms?](../../clean-rooms/index.html).

  * **Shares** , which are Delta Sharing objects that represent a read-only collection of data and AI assets that a data provider shares with one or more recipients.

  * **Recipients** , which are Delta Sharing objects that represent an entity that receives shares from a data provider.

  * **Providers** , which are Delta Sharing objects that represent an entity that shares data with a recipient.

For more information about the Delta Sharing securable objects, see [What is
Delta Sharing?](../../delta-sharing/index.html).

## Granting and revoking access to database objects and other securable
objects in Unity Catalog

You can grant and revoke access to securable objects at any level in the
hierarchy, including the metastore itself. Access to an object implicitly
grants the same access to all children of that object, unless access is
revoked.

You can use typical ANSI SQL commands to grant and revoke access to objects in
Unity Catalog. For example:

    
    
    GRANT CREATE TABLE ON SCHEMA mycatalog.myschema TO `finance-team`;
    

You can also use Catalog Explorer, the Databricks CLI, and REST APIs to manage
object permissions.

![Grant privilege using Catalog Explorer](../../_images/catalog-explorer-
grant.png)

To learn how to manage privileges in Unity Catalog, see [Manage privileges in
Unity Catalog](manage-privileges/index.html).

### Default access to database objects in Unity Catalog

Unity Catalog operates on the principle of least privilege, where users have
the minimum access they need to perform their required tasks. When a workspace
is created, non-admin users have access only to the automatically-provisioned
**Workspace catalog** , which makes this catalog a convenient place for users
to try out the process of creating and accessing database objects in Unity
Catalog. See [Workspace catalog privileges](manage-
privileges/index.html#workspace-catalog).

### Admin roles

Workspace admins and account admins have additional privileges by default.
_Metastore admin_ is an optional role, required if you want to manage table
and volume storage at the metastore level, and convenient if you want to
manage data centrally across multiple workspaces in a region. For more
information, see [Admin privileges in Unity Catalog](manage-privileges/admin-
privileges.html) and [(Optional) Assign the metastore admin role](get-
started.html#metastore-admin).

## Managed versus external tables and volumes

Tables and volumes can be managed or external.

  * **Managed tables** are fully managed by Unity Catalog, which means that Unity Catalog manages both the governance and the underlying data files for each managed table. Managed tables are stored in a Unity Catalog-managed location in your cloud storage. Managed tables always use the Delta Lake format. You can store managed tables at the metastore, catalog, or schema levels.

  * **External tables** are tables whose access from Databricks is managed by Unity Catalog, but whose data lifecycle and file layout are managed using your cloud provider and other data platforms. Typically you use external tables to register large amounts of your existing data in Databricks, or if you also require write access to the data using tools outside of Databricks. External tables are supported in multiple data formats. Once an external table is registered in a Unity Catalog metastore, you can manage and audit Databricks access to itâand work with itâjust like you can with managed tables.

  * **Managed volumes** are fully managed by Unity Catalog, which means that Unity Catalog manages access to the volumeâs storage location in your cloud provider account. When you create a managed volume, it is automatically stored in the _managed storage location_ assigned to the containing schema.

  * **External volumes** represent existing data in storage locations that are managed outside of Databricks, but registered in Unity Catalog to control and audit access from within Databricks. When you create an external volume in Databricks, you specify its location, which must be on a path that is defined in a Unity Catalog _external location_.

Databricks recommends managed tables and volumes to take full advantage of
Unity Catalog governance capabilities and performance optimizations.

See [Work with managed tables](../../tables/managed.html), [Work with external
tables](../../tables/external.html), and [Managed vs. external
volumes](../../volumes/managed-vs-external.html).

## Data isolation using managed storage

Your organization may require that data of certain types be stored within
specific accounts or buckets in your cloud tenant.

Unity Catalog gives the ability to configure storage locations at the
metastore, catalog, or schema level to satisfy such requirements. The system
evaluates the hierarchy of storage locations from schema to catalog to
metastore.

For example, letâs say your organization has a company compliance policy
that requires production data relating to human resources to reside in the
bucket s3://mycompany-hr-prod. In Unity Catalog, you can achieve this
requirement by setting a location on a catalog level, creating a catalog
called, for example `hr_prod`, and assigning the location s3://mycompany-hr-
prod/unity-catalog to it. This means that managed tables or volumes created in
the `hr_prod` catalog (for example, using `CREATE TABLE hr_prod.default.table
â¦`) store their data in s3://mycompany-hr-prod/unity-catalog. Optionally,
you can choose to provide schema-level locations to organize data within the
`hr_prod catalog` at a more granular level.

If storage isolation is not required for some catalogs, you can optionally set
a storage location at the metastore level. This location serves as a default
location for managed tables and volumes in catalogs and schemas that donât
have assigned storage. Typically, however, Databricks recommends that you
assign separate managed storage locations for each catalog.

For more information, see [Specify a managed storage location in Unity
Catalog](../../connect/unity-catalog/managed-storage.html) and [Data is
physically separated in storage](best-practices.html#physically-separate).

### Workspace-catalog binding

By default, catalog owners (and metastore admins, if they are defined for the
account) can make a catalog accessible to users in multiple workspaces
attached to the same Unity Catalog metastore. If you use workspaces to isolate
user data access, however, you might want to limit catalog access to specific
workspaces in your account, to ensure that certain kinds of data are processed
only in those workspaces. You might want separate production and development
workspaces, for example, or a separate workspace for processing personal data.
This is known as _workspace-catalog binding_. See [Limit catalog access to
specific workspaces](../../catalogs/binding.html).

Note

For increased data isolation, you can also bind cloud storage access to
specific workspaces. See [(Optional) Assign a storage credential to specific
workspaces](../../connect/unity-catalog/storage-credentials.html#workspace-
binding) and [(Optional) Assign an external location to specific
workspaces](../../connect/unity-catalog/external-locations.html#workspace-
binding).

## Auditing data access

Unity Catalog captures an audit log of actions performed against the
metastore, enabling admins to access fine-grained details about who accessed a
given dataset and the actions they performed.

You can access your accountâs audit logs using system tables managed by
Unity Catalog.

See [Audit Unity Catalog events](audit.html), [Unity Catalog
events](../../admin/account-settings/audit-logs.html#uc), and [Monitor usage
with system tables](../../admin/system-tables/index.html).

## Tracking data lineage

You can use Unity Catalog to capture runtime data lineage across queries in
any language executed on a Databricks cluster or SQL warehouse. Lineage is
captured down to the column level, and includes notebooks, jobs and dashboards
related to the query. To learn more, see [Capture and view data lineage using
Unity Catalog](data-lineage.html).

## Lakehouse Federation and Unity Catalog

Lakehouse Federation is the query federation platform for Databricks. The term
_query federation_ describes a collection of features that enable users and
systems to run queries against multiple siloed data sources without needing to
migrate all data to a unified system.

Databricks uses Unity Catalog to manage query federation. You use Unity
Catalog to configure read-only _connections_ to popular external database
systems and create _foreign catalogs_ that mirror external databases. Unity
Catalogâs data governance and data lineage tools ensure that data access is
managed and audited for all federated queries made by the users in your
Databricks workspaces.

See [What is Lakehouse Federation?](../../query-federation/index.html).

## Delta Sharing, Databricks Marketplace, and Unity Catalog

Delta Sharing is a secure data sharing platform that lets you share data and
AI assets with users outside your organization, whether or not those users use
Databricks. Although Delta Sharing is available as an open-source
implementation, in Databricks it requires Unity Catalog to take full advantage
of extended functionality. See [What is Delta Sharing?](../../delta-
sharing/index.html).

Databricks Marketplace, an open forum for exchanging data products, is built
on top of Delta Sharing, and as such, you must have a Unity Catalog-enabled
workspace to be a Marketplace provider. See [What is Databricks
Marketplace?](../../marketplace/index.html).

## How do I set up Unity Catalog for my organization?

To use Unity Catalog, your Databricks workspace must be enabled for Unity
Catalog, which means that the workspace is attached to a Unity Catalog
metastore. All new workspaces are enabled for Unity Catalog automatically upon
creation, but older workspaces might require that an account admin enable
Unity Catalog manually. Whether or not your workspace was enabled for Unity
Catalog automatically, the following steps are also required to get started
with Unity Catalog:

  * Create catalogs and schemas to contain database objects like tables and volumes.

  * Create managed storage locations to store the managed tables and volumes in these catalogs and schemas.

  * Grant user access to catalogs, schemas, and database objects.

Workspaces that are automatically enabled for Unity Catalog provision a
_workspace catalog_ with broad privileges granted to all workspace users. This
catalog is a convenient starting point for trying out Unity Catalog.

For detailed setup instructions, see [Set up and manage Unity Catalog](get-
started.html).

## Migrating an existing workspace to Unity Catalog

If you have an older workspace that you recently enabled for Unity Catalog,
you probably have data managed by the legacy Hive metastore. You can work with
that data alongside data that is registered in Unity Catalog, but the legacy
Hive metastore is deprecated, and you should migrate the data in your Hive
metastore to Unity Catalog as soon as possible to take advantage of Unity
Catalogâs superior governance capabilities and performance.

Migration involves the following:

  1. Converting any workspace-local groups to account-level groups. Unity Catalog centralizes identity management at the account level.

  2. Migrating tables and views managed in Hive metastore to Unity Catalog.

  3. Update queries and jobs to reference the new Unity Catalog tables instead of the old Hive metastore tables.

The following can help you manage a migration:

  * UCX, a Databricks Labs project, provides tools that help you upgrade your non-Unity-Catalog workspace to Unity Catalog. UCX is a good choice for larger-scale migrations. See [Use the UCX utilities to upgrade your workspace to Unity Catalog](ucx.html).

  * If you have a smaller number of tables to migrate, Databricks provides a UI wizard and SQL commands that you can use. See [Upgrade Hive tables and views to Unity Catalog](migrate.html).

  * To learn how to use tables in the Hive metastore alongside database objects in Unity Catalog in the same workspace, see [Work with Unity Catalog and the legacy Hive metastore](hive-metastore.html).

## Unity Catalog requirements and restrictions

Unity Catalog requires specific types of compute and file formats, described
below. Also listed below are some Databricks features that are not fully
supported in Unity Catalog on all Databricks Runtime versions.

### Region support

All regions support Unity Catalog. For details, see [Databricks clouds and
regions](../../resources/supported-regions.html).

### Compute requirements

Unity Catalog is supported on clusters that run Databricks Runtime 11.3 LTS or
above. Unity Catalog is supported by default on all [SQL
warehouse](../../compute/sql-warehouse/index.html) compute versions.

Clusters running on earlier versions of Databricks Runtime do not provide
support for all Unity Catalog GA features and functionality.

To access data in Unity Catalog, clusters must be configured with the correct
_access mode_. Unity Catalog is secure by default. If a cluster is not
configured with shared or single user access mode, the cluster canât access
data in Unity Catalog. See [Access modes](../../compute/configure.html#access-
mode).

For detailed information about Unity Catalog functionality changes in each
Databricks Runtime version, see the [release notes](../../release-
notes/runtime/index.html).

Limitations for Unity Catalog vary by access mode and Databricks Runtime
version. See [Compute access mode limitations for Unity
Catalog](../../compute/access-mode-limitations.html).

### File format support

Unity Catalog supports the following table formats:

  * [Managed tables](../../tables/managed.html) must use the `delta` table format.

  * [External tables](../../tables/external.html) can use `delta`, `CSV`, `JSON`, `avro`, `parquet`, `ORC`, or `text`.

### Securable object naming requirements

The following limitations apply for all object names in Unity Catalog:

  * Object names cannot exceed 255 characters.

  * The following special characters are not allowed:

    * Period (`.`)

    * Space (` `)

    * Forward slash (`/`)

    * All ASCII control characters (00-1F hex)

    * The DELETE character (7F hex)

  * Unity Catalog stores all object names as lowercase.

  * When referencing UC names in SQL, you must use backticks to escape names that contain special characters such as hyphens (`-`).

Note

Column names can use special characters, but the name must be escaped with
backticks in all SQL statements if special characters are used. Unity Catalog
preserves column name casing, but queries against Unity Catalog tables are
case-insensitive.

### Limitations

Unity Catalog has the following limitations. Some of these are specific to
older Databricks Runtime versions and compute access modes.

Structured Streaming workloads have additional limitations, depending on
Databricks Runtime and access mode. See [Compute access mode limitations for
Unity Catalog](../../compute/access-mode-limitations.html).

Databricks releases new functionality that shrinks this list regularly.

  * Groups that were previously created in a workspace (that is, workspace-level groups) cannot be used in Unity Catalog `GRANT` statements. This is to ensure a consistent view of groups that can span across workspaces. To use groups in `GRAN`T statements, create your groups at the account level and update any automation for principal or group management (such as SCIM, Okta and Microsoft Entra ID connectors, and Terraform) to reference account endpoints instead of workspace endpoints. See [Difference between account groups and workspace-local groups](../../admin/users-groups/groups.html#account-vs-workspace-group).

  * Workloads in R do not support the use of dynamic views for row-level or column-level security on compute running Databricks Runtime 15.3 and below.

Use a single user compute resource running Databricks Runtime 15.4 LTS or
above for workloads in R that query dynamic views. Such workloads also require
a workspace that is enabled for serverless compute. For details, see [Fine-
grained access control on single user compute](../../compute/single-user-
fgac.html).

  * Shallow clones are unsupported in Unity Catalog on compute running Databricks Runtime 12.2 LTS and below. You can use shallow clones to create managed tables on Databricks Runtime 13.3 LTS and above. You cannot use them to create external tables, regardless of Databricks Runtime version. See [Shallow clone for Unity Catalog tables](../../delta/clone-unity-catalog.html).

  * Bucketing is not supported for Unity Catalog tables. If you run commands that try to create a bucketed table in Unity Catalog, it will throw an exception.

  * Writing to the same path or Delta Lake table from workspaces in multiple regions can lead to unreliable performance if some clusters access Unity Catalog and others do not.

  * Custom partition schemes created using commands like `ALTER TABLE ADD PARTITION` are not supported for tables in Unity Catalog. Unity Catalog can access tables that use directory-style partitioning.

  * Overwrite mode for DataFrame write operations into Unity Catalog is supported only for Delta tables, not for other file formats. The user must have the `CREATE` privilege on the parent schema and must be the owner of the existing object or have the `MODIFY` privilege on the object. 

  * Python UDFs are not supported in Databricks Runtime 12.2 LTS and below. This includes UDAFs, UDTFs, and Pandas on Spark (`applyInPandas` and `mapInPandas`). Python scalar UDFs are supported in Databricks Runtime 13.3 LTS and above.

  * Scala UDFs are not supported in Databricks Runtime 14.1 and below on shared clusters. Scala scalar UDFs are supported in Databricks Runtime 14.2 and above on shared clusters.

  * Standard Scala thread pools are not supported. Instead, use the special thread pools in `org.apache.spark.util.ThreadUtils`, for example, `org.apache.spark.util.ThreadUtils.newDaemonFixedThreadPool`. However, the following thread pools in `ThreadUtils` are not supported: `ThreadUtils.newForkJoinPool` and any `ScheduledExecutorService` thread pool.

Models registered in Unity Catalog have additional limitations. See
[Limitations](../../machine-learning/manage-model-
lifecycle/index.html#limitations).

### Resource quotas

Unity Catalog enforces resource quotas on all securable objects. These quotas
are listed in [Resource limits](../../resources/limits.html). If you expect to
exceed these resource limits, contact your Databricks account team.

You can monitor your quota usage using the Unity Catalog resource quotas APIs.
See [Monitor your usage of Unity Catalog resource quotas](resource-
quotas.html).

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

