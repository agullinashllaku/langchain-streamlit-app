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

  * [English](../../en/pyspark/datasources.html)
  * [æ¥æ¬èª](../../ja/pyspark/datasources.html)
  * [PortuguÃªs](../../pt/pyspark/datasources.html)

[![](../_static/icons/aws.svg)Amazon Web Services](javascript:void\(0\))

  * [![](../_static/icons/azure.svg)Microsoft Azure](https://learn.microsoft.com/azure/databricks/pyspark/datasources)
  * [![](../_static/icons/gcp.svg)Google Cloud Platform](https://docs.gcp.databricks.com/pyspark/datasources.html)

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
      * [PySpark](index.html)
        * [PySpark basics](basics.html)
        * PySpark custom data sources
      * [Pandas API on Spark](../pandas/pandas-on-spark.html)
    * [R](../sparkr/index.html)
    * [Scala](../languages/scala.html)
    * [SQL](../sql/language-manual/index.html)
    * [User-defined functions (UDFs)](../udf/index.html)
    * [Databricks Apps](../dev-tools/databricks-apps/index.html)
    * [Tools](../dev-tools/index.html)
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

Updated Oct 11, 2024

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation
Feedback)

  * [Documentation](../index.html)
  * [Develop on Databricks](../languages/index.html)
  * [Databricks for Python developers](../languages/python.html)
  * [PySpark on Databricks](index.html)
  * PySpark custom data sources
  * 

# PySpark custom data sources

Preview

PySpark custom data sources are in [Public Preview](../release-notes/release-
types.html) in Databricks Runtime 15.2 and above. Streaming support is
available in Databricks Runtime 15.3 and above.

A PySpark DataSource is created by the [Python (PySpark) DataSource
API](https://github.com/apache/spark/blob/c88fabfee41df1ca4729058450ec6f798641c936/python/docs/source/user_guide/sql/python_data_source.rst),
which enables reading from custom data sources and writing to custom data
sinks in Apache Spark using Python. You can use PySpark custom data sources to
define custom connections to data systems and implement additional
functionality, to build out reusable data sources.

## DataSource class

The PySpark
[DataSource](https://github.com/apache/spark/blob/0d7c07047a628bd42eb53eb49935f5e3f81ea1a1/python/pyspark/sql/datasource.py)
is a base class that provides methods to create data readers and writers.

### Implement the data source subclass

Depending on your use case, the following must be implemented by any subclass
to make a data source either readable, writable, or both:

Property or Method | Description  
---|---  
`name` | Required. The name of the data source  
`schema` | Required. The schema of the data source to be read or written  
`reader()` | Must return a `DataSourceReader` to make the data source readable (batch)  
`writer()` | Must return a `DataSourceWriter` to make the data sink writeable (batch)  
`streamReader()` or `simpleStreamReader()` | Must return a `DataSourceStreamReader` to make the data stream readable (streaming)  
`streamWriter()` | Must return a `DataSourceStreamWriter` to make the data stream writeable (streaming)  
  
Note

The user-defined `DataSource`, `DataSourceReader`, `DataSourceWriter`,
`DataSourceStreamReader`, `DataSourceStreamWriter`, and their methods must be
able to be serialized. In other words, they must be a dictionary or nested
dictionary that contains a primitive type.

### Register the data source

After implementing the interface, you must register it, then you can load or
otherwise use it as shown in the following example:

    
    
    # Register the data source
    spark.dataSource.register(MyDataSourceClass)
    
    # Read from a custom data source
    spark.read.format("my_datasource_name").load().show()
    

## Example 1: Create a PySpark DataSource for batch query

To demonstrate PySpark DataSource reader capabilities, create a data source
that generates example data using the `faker` Python package. For more
information about `faker`, see the [Faker
documentation](https://faker.readthedocs.io/en/master/).

Install the `faker` package using the following command:

    
    
    %pip install faker
    

### Step 1: Define the example DataSource

First, define your new PySpark DataSource as a subclass of `DataSource` with a
name, schema, and reader. The `reader()` method must be defined to read from a
data source in a batch query.

    
    
    from pyspark.sql.datasource import DataSource, DataSourceReader
    from pyspark.sql.types import StructType
    
    class FakeDataSource(DataSource):
        """
        An example data source for batch query using the `faker` library.
        """
    
        @classmethod
        def name(cls):
            return "fake"
    
        def schema(self):
            return "name string, date string, zipcode string, state string"
    
        def reader(self, schema: StructType):
            return FakeDataSourceReader(schema, self.options)
    

### Step 2: Implement the reader for a batch query

Next, implement the reader logic to generate example data. Use the installed
`faker` library to populate each field in the schema.

    
    
    class FakeDataSourceReader(DataSourceReader):
    
        def __init__(self, schema, options):
            self.schema: StructType = schema
            self.options = options
    
        def read(self, partition):
            # Library imports must be within the method.
            from faker import Faker
            fake = Faker()
    
            # Every value in this `self.options` dictionary is a string.
            num_rows = int(self.options.get("numRows", 3))
            for _ in range(num_rows):
                row = []
                for field in self.schema.fields:
                    value = getattr(fake, field.name)()
                    row.append(value)
                yield tuple(row)
    

### Step 3: Register and use the example data source

To use the data source, register it. By default, the `FakeDataSource` has
three rows, and the schema includes these `string` fields: `name`, `date`,
`zipcode`, `state`. The following example registers, loads, and outputs the
example data source with the defaults:

    
    
    spark.dataSource.register(FakeDataSource)
    spark.read.format("fake").load().show()
    
    
    
    +-----------------+----------+-------+----------+
    |             name|      date|zipcode|     state|
    +-----------------+----------+-------+----------+
    |Christine Sampson|1979-04-24|  79766|  Colorado|
    |       Shelby Cox|2011-08-05|  24596|   Florida|
    |  Amanda Robinson|2019-01-06|  57395|Washington|
    +-----------------+----------+-------+----------+
    

Only `string` fields are supported, but you can specify a schema with any
fields that correspond to `faker` package providersâ fields to generate
random data for testing and development. The following example loads the data
source with `name` and `company` fields:

    
    
    spark.read.format("fake").schema("name string, company string").load().show()
    
    
    
    +---------------------+--------------+
    |name                 |company       |
    +---------------------+--------------+
    |Tanner Brennan       |Adams Group   |
    |Leslie Maxwell       |Santiago Group|
    |Mrs. Jacqueline Brown|Maynard Inc   |
    +---------------------+--------------+
    

To load the data source with a custom number of rows, specify the `numRows`
option. The following example specifies 5 rows:

    
    
    spark.read.format("fake").option("numRows", 5).load().show()
    
    
    
    +--------------+----------+-------+------------+
    |          name|      date|zipcode|       state|
    +--------------+----------+-------+------------+
    |  Pam Mitchell|1988-10-20|  23788|   Tennessee|
    |Melissa Turner|1996-06-14|  30851|      Nevada|
    |  Brian Ramsey|2021-08-21|  55277|  Washington|
    |  Caitlin Reed|1983-06-22|  89813|Pennsylvania|
    | Douglas James|2007-01-18|  46226|     Alabama|
    +--------------+----------+-------+------------+
    

## Example 2: Create PySpark DataSource for streaming read and write

To demonstrate PySpark DataSource stream reader and writer capabilities,
create an example data source that generates two rows in every microbatch
using the `faker` Python package. For more information about `faker`, see the
[Faker documentation](https://faker.readthedocs.io/en/master/).

Install the `faker` package using the following command:

    
    
    %pip install faker
    

### Step 1: Define the example DataSource

First, define your new PySpark DataSource as a subclass of `DataSource` with a
name, schema, and methods `streamReader()` and `streamWriter()`.

    
    
    from pyspark.sql.datasource import DataSource, DataSourceStreamReader, SimpleDataSourceStreamReader, DataSourceStreamWriter
    from pyspark.sql.types import StructType
    
    class FakeStreamDataSource(DataSource):
        """
        An example data source for streaming read and write using the `faker` library.
        """
    
        @classmethod
        def name(cls):
            return "fakestream"
    
        def schema(self):
            return "name string, state string"
    
        def streamReader(self, schema: StructType):
            return FakeStreamReader(schema, self.options)
    
        # If you don't need partitioning, you can implement the simpleStreamReader method instead of streamReader.
        # def simpleStreamReader(self, schema: StructType):
        #    return SimpleStreamReader()
    
        def streamWriter(self, schema: StructType, overwrite: bool):
            return FakeStreamWriter(self.options)
    

### Step 2: Implement the stream reader

Next, implement the example streaming data reader that generates two rows in
every microbatch. You can implement `DataSourceStreamReader`, or if the data
source has low throughput and doesnât require partitioning, you can
implement `SimpleDataSourceStreamReader` instead. Either
`simpleStreamReader()` or `streamReader()` must be implemented, and
`simpleStreamReader()` is only invoked when `streamReader()` is not
implemented.

#### DataSourceStreamReader implementation

The `streamReader` instance has an integer offset that increases by 2 in every
microbatch, implemented with the `DataSourceStreamReader` interface.

    
    
    class RangePartition(InputPartition):
        def __init__(self, start, end):
            self.start = start
            self.end = end
    
    class FakeStreamReader(DataSourceStreamReader):
        def __init__(self, schema, options):
            self.current = 0
    
        def initialOffset(self) -> dict:
            """
            Returns the initial start offset of the reader.
            """
            return {"offset": 0}
    
        def latestOffset(self) -> dict:
            """
            Returns the current latest offset that the next microbatch will read to.
            """
            self.current += 2
            return {"offset": self.current}
    
        def partitions(self, start: dict, end: dict):
            """
            Plans the partitioning of the current microbatch defined by start and end offset. It
            needs to return a sequence of :class:`InputPartition` objects.
            """
            return [RangePartition(start["offset"], end["offset"])]
    
        def commit(self, end: dict):
            """
            This is invoked when the query has finished processing data before end offset. This
            can be used to clean up the resource.
            """
            pass
    
        def read(self, partition) -> Iterator[Tuple]:
            """
            Takes a partition as an input and reads an iterator of tuples from the data source.
            """
            start, end = partition.start, partition.end
            for i in range(start, end):
                yield (i, str(i))
    

#### SimpleDataSourceStreamReader implementation

The `SimpleStreamReader` instance is the same as the `FakeStreamReader`
instance that generates two rows in every batch, but implemented with the
`SimpleDataSourceStreamReader` interface without partitioning.

    
    
    class SimpleStreamReader(SimpleDataSourceStreamReader):
        def initialOffset(self):
            """
            Returns the initial start offset of the reader.
            """
            return {"offset": 0}
    
        def read(self, start: dict) -> (Iterator[Tuple], dict):
            """
            Takes start offset as an input, then returns an iterator of tuples and the start offset of the next read.
            """
            start_idx = start["offset"]
            it = iter([(i,) for i in range(start_idx, start_idx + 2)])
            return (it, {"offset": start_idx + 2})
    
        def readBetweenOffsets(self, start: dict, end: dict) -> Iterator[Tuple]:
            """
            Takes start and end offset as inputs, then reads an iterator of data deterministically.
            This is called when the query replays batches during restart or after a failure.
            """
            start_idx = start["offset"]
            end_idx = end["offset"]
            return iter([(i,) for i in range(start_idx, end_idx)])
    
        def commit(self, end):
            """
            This is invoked when the query has finished processing data before end offset. This can be used to clean up resources.
            """
            pass
    

### Step 3: Implement the stream writer

Now implement the streaming writer. This streaming data writer writes the
metadata information of each microbatch to a local path.

    
    
    class SimpleCommitMessage(WriterCommitMessage):
       partition_id: int
       count: int
    
    class FakeStreamWriter(DataSourceStreamWriter):
       def __init__(self, options):
           self.options = options
           self.path = self.options.get("path")
           assert self.path is not None
    
       def write(self, iterator):
           """
           Writes the data, then returns the commit message of that partition. Library imports must be within the method.
           """
           from pyspark import TaskContext
           context = TaskContext.get()
           partition_id = context.partitionId()
           cnt = 0
           for row in iterator:
               cnt += 1
           return SimpleCommitMessage(partition_id=partition_id, count=cnt)
    
       def commit(self, messages, batchId) -> None:
           """
           Receives a sequence of :class:`WriterCommitMessage` when all write tasks have succeeded, then decides what to do with it.
           In this FakeStreamWriter, the metadata of the microbatch(number of rows and partitions) is written into a JSON file inside commit().
           """
           status = dict(num_partitions=len(messages), rows=sum(m.count for m in messages))
           with open(os.path.join(self.path, f"{batchId}.json"), "a") as file:
               file.write(json.dumps(status) + "\n")
    
       def abort(self, messages, batchId) -> None:
           """
           Receives a sequence of :class:`WriterCommitMessage` from successful tasks when some other tasks have failed, then decides what to do with it.
           In this FakeStreamWriter, a failure message is written into a text file inside abort().
           """
           with open(os.path.join(self.path, f"{batchId}.txt"), "w") as file:
               file.write(f"failed in batch {batchId}")
    

### Step 4: Register and use the example data source

To use the data source, register it. After it is regsitered, you can use it in
streaming queries as a source or sink by passing a short name or full name to
`format()`. The following example registers the data source, then starts a
query that reads from the example data source and outputs to the console:

    
    
    spark.dataSource.register(FakeStreamDataSource)
    query = spark.readStream.format("fakestream").load().writeStream.format("console").start()
    

Alternatively, the following example uses the example stream as a sink and
specifies an output path:

    
    
    query = spark.readStream.format("fakestream").load().writeStream.format("fake").start("/output_path")
    

## Troubleshooting

If the output is the following error, your compute does not support PySpark
custom data sources. You must use Databricks Runtime 15.2 or above.

`Error: [UNSUPPORTED_FEATURE.PYTHON_DATA_SOURCE] The feature is not supported:
Python data sources. SQLSTATE: 0A000`

* * *

(C) Databricks 2024. All rights reserved. Apache, Apache Spark, Spark, and the
Spark logo are trademarks of the [Apache Software
Foundation](http://www.apache.org/).

[Send us feedback](mailto:doc-feedback@databricks.com?subject=Documentation Feedback) | [Privacy Policy](https://databricks.com/privacy-policy) | [Terms of Use](https://databricks.com/terms-of-use)

