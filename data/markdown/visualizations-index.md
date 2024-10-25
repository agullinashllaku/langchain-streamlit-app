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

  * [English](../../en/visualizations/index.html)
  * [æ¥æ¬èª](../../ja/visualizations/index.html)
  * [PortuguÃªs](../../pt/visualizations/index.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/visualizations/)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/visualizations/index.html)

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
    * [Explore storage and find data files](../discover/files.html)
    * [Explore database objects](../discover/database-objects.html)
    * [View frequent queries and users of a table](../discover/table-insights.html)
    * [EDA](../exploratory-data-analysis/index.html)
      * [Visualizations in Databricks SQL](../sql/user/visualizations/index.html)
      * Visualizations in notebooks
      * [Visualization types](visualization-types.html)
      * [Preview chart visualizations](preview-chart-visualizations.html)
      * [No-code EDA with bamboolib](../notebooks/bamboolib.html)
    * [Sample datasets](../discover/databricks-datasets.html)
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
  * [Discover data](../discover/index.html)
  * [Exploratory data analysis on Databricks: Tools and techniques](../exploratory-data-analysis/index.html)
  * Visualizations in Databricks notebooks
  * 

# Visualizations in Databricks notebooks

Databricks has built-in support for charts and visualizations in both
Databricks SQL and in notebooks. This page describes how to work with
visualizations in a Databricks notebook. For information about using
visualizations in Databricks SQL, see [Visualization in Databricks
SQL](../sql/user/visualizations/index.html).

To view the types of visualizations, see [visualization types](visualization-
types.html).

Important

For information about a preview version of Databricks charts, see [preview
chart visualizations](preview-chart-visualizations.html).

## Create a new visualization

To recreate the example in this section, use the following code:

    
    
    sparkDF = spark.read.csv("/databricks-datasets/bikeSharing/data-001/day.csv", header="true", inferSchema="true")
    display(sparkDF)
    

To create a visualization, click **+** above a result and select
**Visualization**. The visualization editor appears.

![New visualization menu](../_images/new-visualization-menu.png)

  1. In the **Visualization Type** drop-down, choose a type.

![Visualization editor](../_images/visualization-editor.png)

  2. Select the data to appear in the visualization. The fields available depend on the selected type.

  3. Click **Save**.

### Visualization tools

If you hover over the top right of a chart in the visualization editor, a
Plotly toolbar appears where you can perform operations such as select, zoom,
and pan.

![Notebook visualization editor toolbar](../_images/viz-plotly-bar.png)

If you hover over the top right of a chart outside the visualization editor a
smaller subset of tools appears:

![Notebook chart toolbar](../_images/nb-plotly-bar.png)

## Create a new data profile

Note

Available in Databricks Runtime 9.1 LTS and above.

Data profiles display summary statistics of an Apache Spark DataFrame, a
pandas DataFrame, or a SQL table in tabular and graphic format. To create a
data profile from a results cell, click **+** and select **Data Profile**.

Databricks calculates and displays the summary statistics.

![Data Profile](../_images/data-profile.png)

  * Numeric and categorical features are shown in separate tables.

  * At the top of the tab, you can sort or search for features.

  * At the top of the chart column, you can choose to display a histogram (**Standard**) or quantiles.

  * Check **expand** to enlarge the charts.

  * Check **log** to display the charts on a log scale.

  * You can hover your cursor over the charts for more detailed information, such as the boundaries of a histogram column and the number of rows in it, or the quantile value.

You can also generate data profiles programmatically; see [summarize command
(dbutils.data.summarize)](../dev-tools/databricks-utils.html#summarize-
command-dbutilsdatasummarize).

## Work with visualizations and data profiles

Note

Data profiles are available in Databricks Runtime 9.1 LTS and above.

### Rename, duplicate, or remove a visualization or data profile

To rename, duplicate, or remove a visualization or data profile, click the
downward pointing arrow at the right of the tab name.

![Notebook visualization drop down menu](../_images/nb-viz-work-with-menu.png)

You can also change the name by clicking directly on it and editing the name
in place.

### Edit a visualization

Click ![Edit visualization button](../_images/edit-visualization-button.png)
beneath the visualization to open the visualization editor. When you have
finished making changes, click **Save**.

#### Edit colors

You can customize a visualizationâs colors when you create the visualization
or by editing it.

  1. Create or edit a visualization.

  2. Click **Colors**.

  3. To modify a color, click the square and select the new color by doing one of the following:

     * Click it in the color selector.

     * Enter a hex value.

  4. Click anywhere outside the color selector to close it and save changes.

#### Temporarily hide or show a series

To hide a series in a visualization, click the series in the legend. To show
the series again, click it again in the legend.

To show only a single series, double-click the series in the legend. To show
other series, click each one.

### Download a visualization

To download a visualization in .png format, click the camera icon ![camera
icon](../_images/viz-camera-icon.png)in the notebook cell or in the
visualization editor.

  * In a result cell, the camera icon appears at the upper right when you move the cursor over the cell.

![camera in notebook cell](../_images/camera-in-nb-cell.png)

  * In the visualization editor, the camera icon appears when you move the cursor over the chart. See Visualization tools.

### Add a visualization or data profile to a dashboard

  1. Click the downward pointing arrow at the right of the tab name.

  2. Select **Add to dashboard**. A list of available dashboard views appears, along with a menu option **Add to new dashboard**.

  3. Select a dashboard or select **Add to new dashboard**. The dashboard appears, including the newly added visualization or data profile.

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

