[ ![](img/spark-logo-rev.svg)](index.html)3.5.3

  * [Overview](index.html)
  * Programming Guides

[Quick Start](quick-start.html) [RDDs, Accumulators, Broadcasts Vars](rdd-
programming-guide.html) [SQL, DataFrames, and Datasets](sql-programming-
guide.html) [Structured Streaming](structured-streaming-programming-
guide.html) [Spark Streaming (DStreams)](streaming-programming-guide.html)
[MLlib (Machine Learning)](ml-guide.html) [GraphX (Graph Processing)](graphx-
programming-guide.html) [SparkR (R on Spark)](sparkr.html) [PySpark (Python on
Spark)](api/python/getting_started/index.html)

  * API Docs

[Scala](api/scala/org/apache/spark/index.html) [Java](api/java/index.html)
[Python](api/python/index.html) [R](api/R/index.html) [SQL, Built-in
Functions](api/sql/index.html)

  * Deploying

[Overview](cluster-overview.html) [Submitting Applications](submitting-
applications.html)

[Spark Standalone](spark-standalone.html) [Mesos](running-on-mesos.html)
[YARN](running-on-yarn.html) [Kubernetes](running-on-kubernetes.html)

  * More

[Configuration](configuration.html) [Monitoring](monitoring.html) [Tuning
Guide](tuning.html) [Job Scheduling](job-scheduling.html)
[Security](security.html) [Hardware Provisioning](hardware-provisioning.html)
[Migration Guide](migration-guide.html)

[Building Spark](building-spark.html) [Contributing to
Spark](https://spark.apache.org/contributing.html) [Third Party
Projects](https://spark.apache.org/third-party-projects.html)

  * 

### [MLlib: Main Guide](ml-guide.html)

  * [ Basic statistics ](ml-statistics.html)
  * [ Data sources ](ml-datasource.html)
  * [ Pipelines ](ml-pipeline.html)
  * [ Extracting, transforming and selecting features ](ml-features.html)
  * [ Classification and Regression ](ml-classification-regression.html)
  * [ Clustering ](ml-clustering.html)
  * [ Collaborative filtering ](ml-collaborative-filtering.html)
  * [ Frequent Pattern Mining ](ml-frequent-pattern-mining.html)
  * [ Model selection and tuning ](ml-tuning.html)
  * [ Advanced topics ](ml-advanced.html)

### [MLlib: RDD-based API Guide](mllib-guide.html)

  * [ Data types ](mllib-data-types.html)
  * [ Basic statistics ](mllib-statistics.html)
  * [ Classification and regression ](mllib-classification-regression.html)
  * [ Collaborative filtering ](mllib-collaborative-filtering.html)
  * [ Clustering ](mllib-clustering.html)
  * [ Dimensionality reduction ](mllib-dimensionality-reduction.html)
  * [ Feature extraction and transformation ](mllib-feature-extraction.html)
  * [ Frequent pattern mining ](mllib-frequent-pattern-mining.html)
  * [ Evaluation metrics ](mllib-evaluation-metrics.html)
  * [ PMML model export ](mllib-pmml-model-export.html)
  * [ Optimization (developer) ](mllib-optimization.html)

# Extracting, transforming and selecting features

This section covers algorithms for working with features, roughly divided into
these groups:

  * Extraction: Extracting features from "raw" data
  * Transformation: Scaling, converting, or modifying features
  * Selection: Selecting a subset from a larger set of features
  * Locality Sensitive Hashing (LSH): This class of algorithms combines aspects of feature transformation with other algorithms.

**Table of Contents**

  * Feature Extractors
    * TF-IDF
    * Word2Vec
    * CountVectorizer
    * FeatureHasher
  * Feature Transformers
    * Tokenizer
    * StopWordsRemover
    * $n$-gram
    * Binarizer
    * PCA
    * PolynomialExpansion
    * Discrete Cosine Transform (DCT)
    * StringIndexer
    * IndexToString
    * OneHotEncoder
    * VectorIndexer
    * Interaction
    * Normalizer
    * StandardScaler
    * RobustScaler
    * MinMaxScaler
    * MaxAbsScaler
    * Bucketizer
    * ElementwiseProduct
    * SQLTransformer
    * VectorAssembler
    * VectorSizeHint
    * QuantileDiscretizer
    * Imputer
  * Feature Selectors
    * VectorSlicer
    * RFormula
    * ChiSqSelector
    * UnivariateFeatureSelector
    * VarianceThresholdSelector
  * Locality Sensitive Hashing
    * LSH Operations
      * Feature Transformation
      * Approximate Similarity Join
      * Approximate Nearest Neighbor Search
    * LSH Algorithms
      * Bucketed Random Projection for Euclidean Distance
      * MinHash for Jaccard Distance

# Feature Extractors

## TF-IDF

[Term frequency-inverse document frequency (TF-
IDF)](http://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a feature vectorization
method widely used in text mining to reflect the importance of a term to a
document in the corpus. Denote a term by `$t$`, a document by `$d$`, and the
corpus by `$D$`. Term frequency `$TF(t, d)$` is the number of times that term
`$t$` appears in document `$d$`, while document frequency `$DF(t, D)$` is the
number of documents that contains term `$t$`. If we only use term frequency to
measure the importance, it is very easy to over-emphasize terms that appear
very often but carry little information about the document, e.g. "a", "the",
and "of". If a term appears very often across the corpus, it means it doesn't
carry special information about a particular document. Inverse document
frequency is a numerical measure of how much information a term provides: `\[
IDF(t, D) = \log \frac{|D| + 1}{DF(t, D) + 1}, \]` where `$|D|$` is the total
number of documents in the corpus. Since logarithm is used, if a term appears
in all documents, its IDF value becomes 0. Note that a smoothing term is
applied to avoid dividing by zero for terms outside the corpus. The TF-IDF
measure is simply the product of TF and IDF: `\[ TFIDF(t, d, D) = TF(t, d)
\cdot IDF(t, D). \]` There are several variants on the definition of term
frequency and document frequency. In MLlib, we separate TF and IDF to make
them flexible.

**TF** : Both `HashingTF` and `CountVectorizer` can be used to generate the
term frequency vectors.

`HashingTF` is a `Transformer` which takes sets of terms and converts those
sets into fixed-length feature vectors. In text processing, a "set of terms"
might be a bag of words. `HashingTF` utilizes the [hashing
trick](http://en.wikipedia.org/wiki/Feature_hashing). A raw feature is mapped
into an index (term) by applying a hash function. The hash function used here
is [MurmurHash 3](https://en.wikipedia.org/wiki/MurmurHash). Then term
frequencies are calculated based on the mapped indices. This approach avoids
the need to compute a global term-to-index map, which can be expensive for a
large corpus, but it suffers from potential hash collisions, where different
raw features may become the same term after hashing. To reduce the chance of
collision, we can increase the target feature dimension, i.e. the number of
buckets of the hash table. Since a simple modulo on the hashed value is used
to determine the vector index, it is advisable to use a power of two as the
feature dimension, otherwise the features will not be mapped evenly to the
vector indices. The default feature dimension is `$2^{18} = 262,144$`. An
optional binary toggle parameter controls term frequency counts. When set to
true all nonzero frequency counts are set to 1. This is especially useful for
discrete probabilistic models that model binary, rather than integer, counts.

`CountVectorizer` converts text documents to vectors of term counts. Refer to
[CountVectorizer ](ml-features.html#countvectorizer) for more details.

**IDF** : `IDF` is an `Estimator` which is fit on a dataset and produces an
`IDFModel`. The `IDFModel` takes feature vectors (generally created from
`HashingTF` or `CountVectorizer`) and scales each feature. Intuitively, it
down-weights features which appear frequently in a corpus.

**Note:** `spark.ml` doesn't provide tools for text segmentation. We refer
users to the [Stanford NLP Group](http://nlp.stanford.edu/) and
[scalanlp/chalk](https://github.com/scalanlp/chalk).

**Examples**

In the following code segment, we start with a set of sentences. We split each
sentence into words using `Tokenizer`. For each sentence (bag of words), we
use `HashingTF` to hash the sentence into a feature vector. We use `IDF` to
rescale the feature vectors; this generally improves performance when using
text as features. Our feature vectors could then be passed to a learning
algorithm.

Refer to the [HashingTF Python
docs](api/python/reference/api/pyspark.ml.feature.HashingTF.html) and the [IDF
Python docs](api/python/reference/api/pyspark.ml.feature.IDF.html) for more
details on the API.

    
    
    from pyspark.ml.feature import HashingTF, IDF, Tokenizer
    
    sentenceData = spark.createDataFrame([
        (0.0, "Hi I heard about Spark"),
        (0.0, "I wish Java could use case classes"),
        (1.0, "Logistic regression models are neat")
    ], ["label", "sentence"])
    
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(sentenceData)
    
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData)
    # alternatively, CountVectorizer can also be used to get term frequency vectors
    
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    
    rescaledData.select("label", "features").show()

Find full example code at "examples/src/main/python/ml/tf_idf_example.py" in
the Spark repo.

Refer to the [HashingTF Scala
docs](api/scala/org/apache/spark/ml/feature/HashingTF.html) and the [IDF Scala
docs](api/scala/org/apache/spark/ml/feature/IDF.html) for more details on the
API.

    
    
    import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
    
    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat")
    )).toDF("label", "sentence")
    
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    
    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    
    val featurizedData = hashingTF.transform(wordsData)
    // alternatively, CountVectorizer can also be used to get term frequency vectors
    
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/TfIdfExample.scala" in
the Spark repo.

Refer to the [HashingTF Java
docs](api/java/org/apache/spark/ml/feature/HashingTF.html) and the [IDF Java
docs](api/java/org/apache/spark/ml/feature/IDF.html) for more details on the
API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.HashingTF;
    import org.apache.spark.ml.feature.IDF;
    import org.apache.spark.ml.feature.IDFModel;
    import org.apache.spark.ml.feature.Tokenizer;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.SparkSession;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(0.0, "Hi I heard about Spark"),
      RowFactory.create(0.0, "I wish Java could use case classes"),
      RowFactory.create(1.0, "Logistic regression models are neat")
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
      new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
    });
    Dataset<Row> sentenceData = spark.createDataFrame(data, schema);
    
    Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
    Dataset<Row> wordsData = tokenizer.transform(sentenceData);
    
    int numFeatures = 20;
    HashingTF hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(numFeatures);
    
    Dataset<Row> featurizedData = hashingTF.transform(wordsData);
    // alternatively, CountVectorizer can also be used to get term frequency vectors
    
    IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
    IDFModel idfModel = idf.fit(featurizedData);
    
    Dataset<Row> rescaledData = idfModel.transform(featurizedData);
    rescaledData.select("label", "features").show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaTfIdfExample.java" in
the Spark repo.

## Word2Vec

`Word2Vec` is an `Estimator` which takes sequences of words representing
documents and trains a `Word2VecModel`. The model maps each word to a unique
fixed-size vector. The `Word2VecModel` transforms each document into a vector
using the average of all words in the document; this vector can then be used
as features for prediction, document similarity calculations, etc. Please
refer to the [MLlib user guide on Word2Vec](mllib-feature-
extraction.html#word2vec) for more details.

**Examples**

In the following code segment, we start with a set of documents, each of which
is represented as a sequence of words. For each document, we transform it into
a feature vector. This feature vector could then be passed to a learning
algorithm.

Refer to the [Word2Vec Python
docs](api/python/reference/api/pyspark.ml.feature.Word2Vec.html) for more
details on the API.

    
    
    from pyspark.ml.feature import Word2Vec
    
    # Input data: Each row is a bag of words from a sentence or document.
    documentDF = spark.createDataFrame([
        ("Hi I heard about Spark".split(" "), ),
        ("I wish Java could use case classes".split(" "), ),
        ("Logistic regression models are neat".split(" "), )
    ], ["text"])
    
    # Learn a mapping from words to Vectors.
    word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
    model = word2Vec.fit(documentDF)
    
    result = model.transform(documentDF)
    for row in result.collect():
        text, vector = row
        print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

Find full example code at "examples/src/main/python/ml/word2vec_example.py" in
the Spark repo.

Refer to the [Word2Vec Scala
docs](api/scala/org/apache/spark/ml/feature/Word2Vec.html) for more details on
the API.

    
    
    import org.apache.spark.ml.feature.Word2Vec
    import org.apache.spark.ml.linalg.Vector
    import org.apache.spark.sql.Row
    
    // Input data: Each row is a bag of words from a sentence or document.
    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")
    
    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2Vec.fit(documentDF)
    
    val result = model.transform(documentDF)
    result.collect().foreach { case Row(text: Seq[_], features: Vector) =>
      println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/Word2VecExample.scala"
in the Spark repo.

Refer to the [Word2Vec Java
docs](api/java/org/apache/spark/ml/feature/Word2Vec.html) for more details on
the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.Word2Vec;
    import org.apache.spark.ml.feature.Word2VecModel;
    import org.apache.spark.ml.linalg.Vector;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.SparkSession;
    import org.apache.spark.sql.types.*;
    
    // Input data: Each row is a bag of words from a sentence or document.
    List<Row> data = Arrays.asList(
      RowFactory.create(Arrays.asList("Hi I heard about Spark".split(" "))),
      RowFactory.create(Arrays.asList("I wish Java could use case classes".split(" "))),
      RowFactory.create(Arrays.asList("Logistic regression models are neat".split(" ")))
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
    });
    Dataset<Row> documentDF = spark.createDataFrame(data, schema);
    
    // Learn a mapping from words to Vectors.
    Word2Vec word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0);
    
    Word2VecModel model = word2Vec.fit(documentDF);
    Dataset<Row> result = model.transform(documentDF);
    
    for (Row row : result.collectAsList()) {
      List<String> text = row.getList(0);
      Vector vector = (Vector) row.get(1);
      System.out.println("Text: " + text + " => \nVector: " + vector + "\n");
    }

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaWord2VecExample.java"
in the Spark repo.

## CountVectorizer

`CountVectorizer` and `CountVectorizerModel` aim to help convert a collection
of text documents to vectors of token counts. When an a-priori dictionary is
not available, `CountVectorizer` can be used as an `Estimator` to extract the
vocabulary, and generates a `CountVectorizerModel`. The model produces sparse
representations for the documents over the vocabulary, which can then be
passed to other algorithms like LDA.

During the fitting process, `CountVectorizer` will select the top `vocabSize`
words ordered by term frequency across the corpus. An optional parameter
`minDF` also affects the fitting process by specifying the minimum number (or
fraction if < 1.0) of documents a term must appear in to be included in the
vocabulary. Another optional binary toggle parameter controls the output
vector. If set to true all nonzero counts are set to 1. This is especially
useful for discrete probabilistic models that model binary, rather than
integer, counts.

**Examples**

Assume that we have the following DataFrame with columns `id` and `texts`:

    
    
     id | texts
    ----|----------
     0  | Array("a", "b", "c")
     1  | Array("a", "b", "b", "c", "a")
    

each row in `texts` is a document of type Array[String]. Invoking fit of
`CountVectorizer` produces a `CountVectorizerModel` with vocabulary (a, b, c).
Then the output column "vector" after transformation contains:

    
    
     id | texts                           | vector
    ----|---------------------------------|---------------
     0  | Array("a", "b", "c")            | (3,[0,1,2],[1.0,1.0,1.0])
     1  | Array("a", "b", "b", "c", "a")  | (3,[0,1,2],[2.0,2.0,1.0])
    

Each vector represents the token counts of the document over the vocabulary.

Refer to the [CountVectorizer Python
docs](api/python/reference/api/pyspark.ml.feature.CountVectorizer.html) and
the [CountVectorizerModel Python
docs](api/python/reference/api/pyspark.ml.feature.CountVectorizerModel.html)
for more details on the API.

    
    
    from pyspark.ml.feature import CountVectorizer
    
    # Input data: Each row is a bag of words with a ID.
    df = spark.createDataFrame([
        (0, "a b c".split(" ")),
        (1, "a b b c a".split(" "))
    ], ["id", "words"])
    
    # fit a CountVectorizerModel from the corpus.
    cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)
    
    model = cv.fit(df)
    
    result = model.transform(df)
    result.show(truncate=False)

Find full example code at
"examples/src/main/python/ml/count_vectorizer_example.py" in the Spark repo.

Refer to the [CountVectorizer Scala
docs](api/scala/org/apache/spark/ml/feature/CountVectorizer.html) and the
[CountVectorizerModel Scala
docs](api/scala/org/apache/spark/ml/feature/CountVectorizerModel.html) for
more details on the API.

    
    
    import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
    
    val df = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a"))
    )).toDF("id", "words")
    
    // fit a CountVectorizerModel from the corpus
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df)
    
    // alternatively, define CountVectorizerModel with a-priori vocabulary
    val cvm = new CountVectorizerModel(Array("a", "b", "c"))
      .setInputCol("words")
      .setOutputCol("features")
    
    cvModel.transform(df).show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/CountVectorizerExample.scala"
in the Spark repo.

Refer to the [CountVectorizer Java
docs](api/java/org/apache/spark/ml/feature/CountVectorizer.html) and the
[CountVectorizerModel Java
docs](api/java/org/apache/spark/ml/feature/CountVectorizerModel.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.CountVectorizer;
    import org.apache.spark.ml.feature.CountVectorizerModel;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.SparkSession;
    import org.apache.spark.sql.types.*;
    
    // Input data: Each row is a bag of words from a sentence or document.
    List<Row> data = Arrays.asList(
      RowFactory.create(Arrays.asList("a", "b", "c")),
      RowFactory.create(Arrays.asList("a", "b", "b", "c", "a"))
    );
    StructType schema = new StructType(new StructField [] {
      new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
    });
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    // fit a CountVectorizerModel from the corpus
    CountVectorizerModel cvModel = new CountVectorizer()
      .setInputCol("text")
      .setOutputCol("feature")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df);
    
    // alternatively, define CountVectorizerModel with a-priori vocabulary
    CountVectorizerModel cvm = new CountVectorizerModel(new String[]{"a", "b", "c"})
      .setInputCol("text")
      .setOutputCol("feature");
    
    cvModel.transform(df).show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaCountVectorizerExample.java"
in the Spark repo.

## FeatureHasher

Feature hashing projects a set of categorical or numerical features into a
feature vector of specified dimension (typically substantially smaller than
that of the original feature space). This is done using the [hashing
trick](https://en.wikipedia.org/wiki/Feature_hashing) to map features to
indices in the feature vector.

The `FeatureHasher` transformer operates on multiple columns. Each column may
contain either numeric or categorical features. Behavior and handling of
column data types is as follows:

  * Numeric columns: For numeric features, the hash value of the column name is used to map the feature value to its index in the feature vector. By default, numeric features are not treated as categorical (even when they are integers). To treat them as categorical, specify the relevant columns using the `categoricalCols` parameter.
  * String columns: For categorical features, the hash value of the string "column_name=value" is used to map to the vector index, with an indicator value of `1.0`. Thus, categorical features are "one-hot" encoded (similarly to using [OneHotEncoder](ml-features.html#onehotencoder) with `dropLast=false`).
  * Boolean columns: Boolean values are treated in the same way as string columns. That is, boolean features are represented as "column_name=true" or "column_name=false", with an indicator value of `1.0`.

Null (missing) values are ignored (implicitly zero in the resulting feature
vector).

The hash function used here is also the [MurmurHash
3](https://en.wikipedia.org/wiki/MurmurHash) used in [HashingTF](ml-
features.html#tf-idf). Since a simple modulo on the hashed value is used to
determine the vector index, it is advisable to use a power of two as the
numFeatures parameter; otherwise the features will not be mapped evenly to the
vector indices.

**Examples**

Assume that we have a DataFrame with 4 input columns `real`, `bool`,
`stringNum`, and `string`. These different data types as input will illustrate
the behavior of the transform to produce a column of feature vectors.

    
    
    real| bool|stringNum|string
    ----|-----|---------|------
     2.2| true|        1|   foo
     3.3|false|        2|   bar
     4.4|false|        3|   baz
     5.5|false|        4|   foo
    

Then the output of `FeatureHasher.transform` on this DataFrame is:

    
    
    real|bool |stringNum|string|features
    ----|-----|---------|------|-------------------------------------------------------
    2.2 |true |1        |foo   |(262144,[51871, 63643,174475,253195],[1.0,1.0,2.2,1.0])
    3.3 |false|2        |bar   |(262144,[6031,  80619,140467,174475],[1.0,1.0,1.0,3.3])
    4.4 |false|3        |baz   |(262144,[24279,140467,174475,196810],[1.0,1.0,4.4,1.0])
    5.5 |false|4        |foo   |(262144,[63643,140467,168512,174475],[1.0,1.0,1.0,5.5])
    

The resulting feature vectors could then be passed to a learning algorithm.

Refer to the [FeatureHasher Python
docs](api/python/reference/api/pyspark.ml.feature.FeatureHasher.html) for more
details on the API.

    
    
    from pyspark.ml.feature import FeatureHasher
    
    dataset = spark.createDataFrame([
        (2.2, True, "1", "foo"),
        (3.3, False, "2", "bar"),
        (4.4, False, "3", "baz"),
        (5.5, False, "4", "foo")
    ], ["real", "bool", "stringNum", "string"])
    
    hasher = FeatureHasher(inputCols=["real", "bool", "stringNum", "string"],
                           outputCol="features")
    
    featurized = hasher.transform(dataset)
    featurized.show(truncate=False)

Find full example code at
"examples/src/main/python/ml/feature_hasher_example.py" in the Spark repo.

Refer to the [FeatureHasher Scala
docs](api/scala/org/apache/spark/ml/feature/FeatureHasher.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.FeatureHasher
    
    val dataset = spark.createDataFrame(Seq(
      (2.2, true, "1", "foo"),
      (3.3, false, "2", "bar"),
      (4.4, false, "3", "baz"),
      (5.5, false, "4", "foo")
    )).toDF("real", "bool", "stringNum", "string")
    
    val hasher = new FeatureHasher()
      .setInputCols("real", "bool", "stringNum", "string")
      .setOutputCol("features")
    
    val featurized = hasher.transform(dataset)
    featurized.show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/FeatureHasherExample.scala"
in the Spark repo.

Refer to the [FeatureHasher Java
docs](api/java/org/apache/spark/ml/feature/FeatureHasher.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.FeatureHasher;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(2.2, true, "1", "foo"),
      RowFactory.create(3.3, false, "2", "bar"),
      RowFactory.create(4.4, false, "3", "baz"),
      RowFactory.create(5.5, false, "4", "foo")
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("real", DataTypes.DoubleType, false, Metadata.empty()),
      new StructField("bool", DataTypes.BooleanType, false, Metadata.empty()),
      new StructField("stringNum", DataTypes.StringType, false, Metadata.empty()),
      new StructField("string", DataTypes.StringType, false, Metadata.empty())
    });
    Dataset<Row> dataset = spark.createDataFrame(data, schema);
    
    FeatureHasher hasher = new FeatureHasher()
      .setInputCols(new String[]{"real", "bool", "stringNum", "string"})
      .setOutputCol("features");
    
    Dataset<Row> featurized = hasher.transform(dataset);
    
    featurized.show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaFeatureHasherExample.java"
in the Spark repo.

# Feature Transformers

## Tokenizer

[Tokenization](http://en.wikipedia.org/wiki/Lexical_analysis#Tokenization) is
the process of taking text (such as a sentence) and breaking it into
individual terms (usually words). A simple
[Tokenizer](api/scala/org/apache/spark/ml/feature/Tokenizer.html) class
provides this functionality. The example below shows how to split sentences
into sequences of words.

[RegexTokenizer](api/scala/org/apache/spark/ml/feature/RegexTokenizer.html)
allows more advanced tokenization based on regular expression (regex)
matching. By default, the parameter "pattern" (regex, default: `"\\s+"`) is
used as delimiters to split the input text. Alternatively, users can set
parameter "gaps" to false indicating the regex "pattern" denotes "tokens"
rather than splitting gaps, and find all matching occurrences as the
tokenization result.

**Examples**

Refer to the [Tokenizer Python
docs](api/python/reference/api/pyspark.ml.feature.Tokenizer.html) and the
[RegexTokenizer Python
docs](api/python/reference/api/pyspark.ml.feature.RegexTokenizer.html) for
more details on the API.

    
    
    from pyspark.ml.feature import Tokenizer, RegexTokenizer
    from pyspark.sql.functions import col, udf
    from pyspark.sql.types import IntegerType
    
    sentenceDataFrame = spark.createDataFrame([
        (0, "Hi I heard about Spark"),
        (1, "I wish Java could use case classes"),
        (2, "Logistic,regression,models,are,neat")
    ], ["id", "sentence"])
    
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    
    regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")
    # alternatively, pattern="\\w+", gaps(False)
    
    countTokens = udf(lambda words: len(words), IntegerType())
    
    tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select("sentence", "words")\
        .withColumn("tokens", countTokens(col("words"))).show(truncate=False)
    
    regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence", "words") \
        .withColumn("tokens", countTokens(col("words"))).show(truncate=False)

Find full example code at "examples/src/main/python/ml/tokenizer_example.py"
in the Spark repo.

Refer to the [Tokenizer Scala
docs](api/scala/org/apache/spark/ml/feature/Tokenizer.html) and the
[RegexTokenizer Scala
docs](api/scala/org/apache/spark/ml/feature/RegexTokenizer.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.sql.functions._
    
    val sentenceDataFrame = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("id", "sentence")
    
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)
    
    val countTokens = udf { (words: Seq[String]) => words.length }
    
    val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select("sentence", "words")
        .withColumn("tokens", countTokens(col("words"))).show(false)
    
    val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence", "words")
        .withColumn("tokens", countTokens(col("words"))).show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/TokenizerExample.scala"
in the Spark repo.

Refer to the [Tokenizer Java
docs](api/java/org/apache/spark/ml/feature/Tokenizer.html) and the
[RegexTokenizer Java
docs](api/java/org/apache/spark/ml/feature/RegexTokenizer.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import scala.collection.mutable.Seq;
    
    import org.apache.spark.ml.feature.RegexTokenizer;
    import org.apache.spark.ml.feature.Tokenizer;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    // col("...") is preferable to df.col("...")
    import static org.apache.spark.sql.functions.call_udf;
    import static org.apache.spark.sql.functions.col;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(0, "Hi I heard about Spark"),
      RowFactory.create(1, "I wish Java could use case classes"),
      RowFactory.create(2, "Logistic,regression,models,are,neat")
    );
    
    StructType schema = new StructType(new StructField[]{
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
    });
    
    Dataset<Row> sentenceDataFrame = spark.createDataFrame(data, schema);
    
    Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
    
    RegexTokenizer regexTokenizer = new RegexTokenizer()
        .setInputCol("sentence")
        .setOutputCol("words")
        .setPattern("\\W");  // alternatively .setPattern("\\w+").setGaps(false);
    
    spark.udf().register(
      "countTokens", (Seq<?> words) -> words.size(), DataTypes.IntegerType);
    
    Dataset<Row> tokenized = tokenizer.transform(sentenceDataFrame);
    tokenized.select("sentence", "words")
        .withColumn("tokens", call_udf("countTokens", col("words")))
        .show(false);
    
    Dataset<Row> regexTokenized = regexTokenizer.transform(sentenceDataFrame);
    regexTokenized.select("sentence", "words")
        .withColumn("tokens", call_udf("countTokens", col("words")))
        .show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaTokenizerExample.java"
in the Spark repo.

## StopWordsRemover

[Stop words](https://en.wikipedia.org/wiki/Stop_words) are words which should
be excluded from the input, typically because the words appear frequently and
don't carry as much meaning.

`StopWordsRemover` takes as input a sequence of strings (e.g. the output of a
[Tokenizer](ml-features.html#tokenizer)) and drops all the stop words from the
input sequences. The list of stopwords is specified by the `stopWords`
parameter. Default stop words for some languages are accessible by calling
`StopWordsRemover.loadDefaultStopWords(language)`, for which available options
are "danish", "dutch", "english", "finnish", "french", "german", "hungarian",
"italian", "norwegian", "portuguese", "russian", "spanish", "swedish" and
"turkish". A boolean parameter `caseSensitive` indicates if the matches should
be case sensitive (false by default).

**Examples**

Assume that we have the following DataFrame with columns `id` and `raw`:

    
    
     id | raw
    ----|----------
     0  | [I, saw, the, red, balloon]
     1  | [Mary, had, a, little, lamb]
    

Applying `StopWordsRemover` with `raw` as the input column and `filtered` as
the output column, we should get the following:

    
    
     id | raw                         | filtered
    ----|-----------------------------|--------------------
     0  | [I, saw, the, red, balloon]  |  [saw, red, balloon]
     1  | [Mary, had, a, little, lamb]|[Mary, little, lamb]
    

In `filtered`, the stop words "I", "the", "had", and "a" have been filtered
out.

Refer to the [StopWordsRemover Python
docs](api/python/reference/api/pyspark.ml.feature.StopWordsRemover.html) for
more details on the API.

    
    
    from pyspark.ml.feature import StopWordsRemover
    
    sentenceData = spark.createDataFrame([
        (0, ["I", "saw", "the", "red", "balloon"]),
        (1, ["Mary", "had", "a", "little", "lamb"])
    ], ["id", "raw"])
    
    remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
    remover.transform(sentenceData).show(truncate=False)

Find full example code at
"examples/src/main/python/ml/stopwords_remover_example.py" in the Spark repo.

Refer to the [StopWordsRemover Scala
docs](api/scala/org/apache/spark/ml/feature/StopWordsRemover.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.StopWordsRemover
    
    val remover = new StopWordsRemover()
      .setInputCol("raw")
      .setOutputCol("filtered")
    
    val dataSet = spark.createDataFrame(Seq(
      (0, Seq("I", "saw", "the", "red", "balloon")),
      (1, Seq("Mary", "had", "a", "little", "lamb"))
    )).toDF("id", "raw")
    
    remover.transform(dataSet).show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/StopWordsRemoverExample.scala"
in the Spark repo.

Refer to the [StopWordsRemover Java
docs](api/java/org/apache/spark/ml/feature/StopWordsRemover.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.StopWordsRemover;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    StopWordsRemover remover = new StopWordsRemover()
      .setInputCol("raw")
      .setOutputCol("filtered");
    
    List<Row> data = Arrays.asList(
      RowFactory.create(Arrays.asList("I", "saw", "the", "red", "balloon")),
      RowFactory.create(Arrays.asList("Mary", "had", "a", "little", "lamb"))
    );
    
    StructType schema = new StructType(new StructField[]{
      new StructField(
        "raw", DataTypes.createArrayType(DataTypes.StringType), false, Metadata.empty())
    });
    
    Dataset<Row> dataset = spark.createDataFrame(data, schema);
    remover.transform(dataset).show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaStopWordsRemoverExample.java"
in the Spark repo.

## $n$-gram

An [n-gram](https://en.wikipedia.org/wiki/N-gram) is a sequence of $n$ tokens
(typically words) for some integer $n$. The `NGram` class can be used to
transform input features into $n$-grams.

`NGram` takes as input a sequence of strings (e.g. the output of a
[Tokenizer](ml-features.html#tokenizer)). The parameter `n` is used to
determine the number of terms in each $n$-gram. The output will consist of a
sequence of $n$-grams where each $n$-gram is represented by a space-delimited
string of $n$ consecutive words. If the input sequence contains fewer than `n`
strings, no output is produced.

**Examples**

Refer to the [NGram Python
docs](api/python/reference/api/pyspark.ml.feature.NGram.html) for more details
on the API.

    
    
    from pyspark.ml.feature import NGram
    
    wordDataFrame = spark.createDataFrame([
        (0, ["Hi", "I", "heard", "about", "Spark"]),
        (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
        (2, ["Logistic", "regression", "models", "are", "neat"])
    ], ["id", "words"])
    
    ngram = NGram(n=2, inputCol="words", outputCol="ngrams")
    
    ngramDataFrame = ngram.transform(wordDataFrame)
    ngramDataFrame.select("ngrams").show(truncate=False)

Find full example code at "examples/src/main/python/ml/n_gram_example.py" in
the Spark repo.

Refer to the [NGram Scala
docs](api/scala/org/apache/spark/ml/feature/NGram.html) for more details on
the API.

    
    
    import org.apache.spark.ml.feature.NGram
    
    val wordDataFrame = spark.createDataFrame(Seq(
      (0, Array("Hi", "I", "heard", "about", "Spark")),
      (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
      (2, Array("Logistic", "regression", "models", "are", "neat"))
    )).toDF("id", "words")
    
    val ngram = new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")
    
    val ngramDataFrame = ngram.transform(wordDataFrame)
    ngramDataFrame.select("ngrams").show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/NGramExample.scala" in
the Spark repo.

Refer to the [NGram Java
docs](api/java/org/apache/spark/ml/feature/NGram.html) for more details on the
API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.NGram;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(0, Arrays.asList("Hi", "I", "heard", "about", "Spark")),
      RowFactory.create(1, Arrays.asList("I", "wish", "Java", "could", "use", "case", "classes")),
      RowFactory.create(2, Arrays.asList("Logistic", "regression", "models", "are", "neat"))
    );
    
    StructType schema = new StructType(new StructField[]{
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField(
        "words", DataTypes.createArrayType(DataTypes.StringType), false, Metadata.empty())
    });
    
    Dataset<Row> wordDataFrame = spark.createDataFrame(data, schema);
    
    NGram ngramTransformer = new NGram().setN(2).setInputCol("words").setOutputCol("ngrams");
    
    Dataset<Row> ngramDataFrame = ngramTransformer.transform(wordDataFrame);
    ngramDataFrame.select("ngrams").show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaNGramExample.java" in
the Spark repo.

## Binarizer

Binarization is the process of thresholding numerical features to binary (0/1)
features.

`Binarizer` takes the common parameters `inputCol` and `outputCol`, as well as
the `threshold` for binarization. Feature values greater than the threshold
are binarized to 1.0; values equal to or less than the threshold are binarized
to 0.0. Both Vector and Double types are supported for `inputCol`.

**Examples**

Refer to the [Binarizer Python
docs](api/python/reference/api/pyspark.ml.feature.Binarizer.html) for more
details on the API.

    
    
    from pyspark.ml.feature import Binarizer
    
    continuousDataFrame = spark.createDataFrame([
        (0, 0.1),
        (1, 0.8),
        (2, 0.2)
    ], ["id", "feature"])
    
    binarizer = Binarizer(threshold=0.5, inputCol="feature", outputCol="binarized_feature")
    
    binarizedDataFrame = binarizer.transform(continuousDataFrame)
    
    print("Binarizer output with Threshold = %f" % binarizer.getThreshold())
    binarizedDataFrame.show()

Find full example code at "examples/src/main/python/ml/binarizer_example.py"
in the Spark repo.

Refer to the [Binarizer Scala
docs](api/scala/org/apache/spark/ml/feature/Binarizer.html) for more details
on the API.

    
    
    import org.apache.spark.ml.feature.Binarizer
    
    val data = Array((0, 0.1), (1, 0.8), (2, 0.2))
    val dataFrame = spark.createDataFrame(data).toDF("id", "feature")
    
    val binarizer: Binarizer = new Binarizer()
      .setInputCol("feature")
      .setOutputCol("binarized_feature")
      .setThreshold(0.5)
    
    val binarizedDataFrame = binarizer.transform(dataFrame)
    
    println(s"Binarizer output with Threshold = ${binarizer.getThreshold}")
    binarizedDataFrame.show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/BinarizerExample.scala"
in the Spark repo.

Refer to the [Binarizer Java
docs](api/java/org/apache/spark/ml/feature/Binarizer.html) for more details on
the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.Binarizer;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(0, 0.1),
      RowFactory.create(1, 0.8),
      RowFactory.create(2, 0.2)
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("feature", DataTypes.DoubleType, false, Metadata.empty())
    });
    Dataset<Row> continuousDataFrame = spark.createDataFrame(data, schema);
    
    Binarizer binarizer = new Binarizer()
      .setInputCol("feature")
      .setOutputCol("binarized_feature")
      .setThreshold(0.5);
    
    Dataset<Row> binarizedDataFrame = binarizer.transform(continuousDataFrame);
    
    System.out.println("Binarizer output with Threshold = " + binarizer.getThreshold());
    binarizedDataFrame.show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaBinarizerExample.java"
in the Spark repo.

## PCA

[PCA](http://en.wikipedia.org/wiki/Principal_component_analysis) is a
statistical procedure that uses an orthogonal transformation to convert a set
of observations of possibly correlated variables into a set of values of
linearly uncorrelated variables called principal components. A
[PCA](api/scala/org/apache/spark/ml/feature/PCA.html) class trains a model to
project vectors to a low-dimensional space using PCA. The example below shows
how to project 5-dimensional feature vectors into 3-dimensional principal
components.

**Examples**

Refer to the [PCA Python
docs](api/python/reference/api/pyspark.ml.feature.PCA.html) for more details
on the API.

    
    
    from pyspark.ml.feature import PCA
    from pyspark.ml.linalg import Vectors
    
    data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
            (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
            (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
    df = spark.createDataFrame(data, ["features"])
    
    pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(df)
    
    result = model.transform(df).select("pcaFeatures")
    result.show(truncate=False)

Find full example code at "examples/src/main/python/ml/pca_example.py" in the
Spark repo.

Refer to the [PCA Scala docs](api/scala/org/apache/spark/ml/feature/PCA.html)
for more details on the API.

    
    
    import org.apache.spark.ml.feature.PCA
    import org.apache.spark.ml.linalg.Vectors
    
    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    
    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(df)
    
    val result = pca.transform(df).select("pcaFeatures")
    result.show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/PCAExample.scala" in the
Spark repo.

Refer to the [PCA Java docs](api/java/org/apache/spark/ml/feature/PCA.html)
for more details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.PCA;
    import org.apache.spark.ml.feature.PCAModel;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(Vectors.sparse(5, new int[]{1, 3}, new double[]{1.0, 7.0})),
      RowFactory.create(Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0)),
      RowFactory.create(Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0))
    );
    
    StructType schema = new StructType(new StructField[]{
      new StructField("features", new VectorUDT(), false, Metadata.empty()),
    });
    
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    PCAModel pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(df);
    
    Dataset<Row> result = pca.transform(df).select("pcaFeatures");
    result.show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaPCAExample.java" in
the Spark repo.

## PolynomialExpansion

[Polynomial expansion](http://en.wikipedia.org/wiki/Polynomial_expansion) is
the process of expanding your features into a polynomial space, which is
formulated by an n-degree combination of original dimensions. A
[PolynomialExpansion](api/scala/org/apache/spark/ml/feature/PolynomialExpansion.html)
class provides this functionality. The example below shows how to expand your
features into a 3-degree polynomial space.

**Examples**

Refer to the [PolynomialExpansion Python
docs](api/python/reference/api/pyspark.ml.feature.PolynomialExpansion.html)
for more details on the API.

    
    
    from pyspark.ml.feature import PolynomialExpansion
    from pyspark.ml.linalg import Vectors
    
    df = spark.createDataFrame([
        (Vectors.dense([2.0, 1.0]),),
        (Vectors.dense([0.0, 0.0]),),
        (Vectors.dense([3.0, -1.0]),)
    ], ["features"])
    
    polyExpansion = PolynomialExpansion(degree=3, inputCol="features", outputCol="polyFeatures")
    polyDF = polyExpansion.transform(df)
    
    polyDF.show(truncate=False)

Find full example code at
"examples/src/main/python/ml/polynomial_expansion_example.py" in the Spark
repo.

Refer to the [PolynomialExpansion Scala
docs](api/scala/org/apache/spark/ml/feature/PolynomialExpansion.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.PolynomialExpansion
    import org.apache.spark.ml.linalg.Vectors
    
    val data = Array(
      Vectors.dense(2.0, 1.0),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(3.0, -1.0)
    )
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    
    val polyExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
      .setDegree(3)
    
    val polyDF = polyExpansion.transform(df)
    polyDF.show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/PolynomialExpansionExample.scala"
in the Spark repo.

Refer to the [PolynomialExpansion Java
docs](api/java/org/apache/spark/ml/feature/PolynomialExpansion.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.PolynomialExpansion;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    PolynomialExpansion polyExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
      .setDegree(3);
    
    List<Row> data = Arrays.asList(
      RowFactory.create(Vectors.dense(2.0, 1.0)),
      RowFactory.create(Vectors.dense(0.0, 0.0)),
      RowFactory.create(Vectors.dense(3.0, -1.0))
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("features", new VectorUDT(), false, Metadata.empty()),
    });
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    Dataset<Row> polyDF = polyExpansion.transform(df);
    polyDF.show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaPolynomialExpansionExample.java"
in the Spark repo.

## Discrete Cosine Transform (DCT)

The [Discrete Cosine
Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform) transforms
a length $N$ real-valued sequence in the time domain into another length $N$
real-valued sequence in the frequency domain. A
[DCT](api/scala/org/apache/spark/ml/feature/DCT.html) class provides this
functionality, implementing the [DCT-
II](https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II) and
scaling the result by $1/\sqrt{2}$ such that the representing matrix for the
transform is unitary. No shift is applied to the transformed sequence (e.g.
the $0$th element of the transformed sequence is the $0$th DCT coefficient and
_not_ the $N/2$th).

**Examples**

Refer to the [DCT Python
docs](api/python/reference/api/pyspark.ml.feature.DCT.html) for more details
on the API.

    
    
    from pyspark.ml.feature import DCT
    from pyspark.ml.linalg import Vectors
    
    df = spark.createDataFrame([
        (Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
        (Vectors.dense([-1.0, 2.0, 4.0, -7.0]),),
        (Vectors.dense([14.0, -2.0, -5.0, 1.0]),)], ["features"])
    
    dct = DCT(inverse=False, inputCol="features", outputCol="featuresDCT")
    
    dctDf = dct.transform(df)
    
    dctDf.select("featuresDCT").show(truncate=False)

Find full example code at "examples/src/main/python/ml/dct_example.py" in the
Spark repo.

Refer to the [DCT Scala docs](api/scala/org/apache/spark/ml/feature/DCT.html)
for more details on the API.

    
    
    import org.apache.spark.ml.feature.DCT
    import org.apache.spark.ml.linalg.Vectors
    
    val data = Seq(
      Vectors.dense(0.0, 1.0, -2.0, 3.0),
      Vectors.dense(-1.0, 2.0, 4.0, -7.0),
      Vectors.dense(14.0, -2.0, -5.0, 1.0))
    
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    
    val dct = new DCT()
      .setInputCol("features")
      .setOutputCol("featuresDCT")
      .setInverse(false)
    
    val dctDf = dct.transform(df)
    dctDf.select("featuresDCT").show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/DCTExample.scala" in the
Spark repo.

Refer to the [DCT Java docs](api/java/org/apache/spark/ml/feature/DCT.html)
for more details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.DCT;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(Vectors.dense(0.0, 1.0, -2.0, 3.0)),
      RowFactory.create(Vectors.dense(-1.0, 2.0, 4.0, -7.0)),
      RowFactory.create(Vectors.dense(14.0, -2.0, -5.0, 1.0))
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("features", new VectorUDT(), false, Metadata.empty()),
    });
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    DCT dct = new DCT()
      .setInputCol("features")
      .setOutputCol("featuresDCT")
      .setInverse(false);
    
    Dataset<Row> dctDf = dct.transform(df);
    
    dctDf.select("featuresDCT").show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaDCTExample.java" in
the Spark repo.

## StringIndexer

`StringIndexer` encodes a string column of labels to a column of label
indices. `StringIndexer` can encode multiple columns. The indices are in `[0,
numLabels)`, and four ordering options are supported: "frequencyDesc":
descending order by label frequency (most frequent label assigned 0),
"frequencyAsc": ascending order by label frequency (least frequent label
assigned 0), "alphabetDesc": descending alphabetical order, and "alphabetAsc":
ascending alphabetical order (default = "frequencyDesc"). Note that in case of
equal frequency when under "frequencyDesc"/"frequencyAsc", the strings are
further sorted by alphabet.

The unseen labels will be put at index numLabels if user chooses to keep them.
If the input column is numeric, we cast it to string and index the string
values. When downstream pipeline components such as `Estimator` or
`Transformer` make use of this string-indexed label, you must set the input
column of the component to this string-indexed column name. In many cases, you
can set the input column with `setInputCol`.

**Examples**

Assume that we have the following DataFrame with columns `id` and `category`:

    
    
     id | category
    ----|----------
     0  | a
     1  | b
     2  | c
     3  | a
     4  | a
     5  | c
    

`category` is a string column with three labels: "a", "b", and "c". Applying
`StringIndexer` with `category` as the input column and `categoryIndex` as the
output column, we should get the following:

    
    
     id | category | categoryIndex
    ----|----------|---------------
     0  | a        | 0.0
     1  | b        | 2.0
     2  | c        | 1.0
     3  | a        | 0.0
     4  | a        | 0.0
     5  | c        | 1.0
    

"a" gets index `0` because it is the most frequent, followed by "c" with index
`1` and "b" with index `2`.

Additionally, there are three strategies regarding how `StringIndexer` will
handle unseen labels when you have fit a `StringIndexer` on one dataset and
then use it to transform another:

  * throw an exception (which is the default)
  * skip the row containing the unseen label entirely
  * put unseen labels in a special additional bucket, at index numLabels

**Examples**

Let's go back to our previous example but this time reuse our previously
defined `StringIndexer` on the following dataset:

    
    
     id | category
    ----|----------
     0  | a
     1  | b
     2  | c
     3  | d
     4  | e
    

If you've not set how `StringIndexer` handles unseen labels or set it to
"error", an exception will be thrown. However, if you had called
`setHandleInvalid("skip")`, the following dataset will be generated:

    
    
     id | category | categoryIndex
    ----|----------|---------------
     0  | a        | 0.0
     1  | b        | 2.0
     2  | c        | 1.0
    

Notice that the rows containing "d" or "e" do not appear.

If you call `setHandleInvalid("keep")`, the following dataset will be
generated:

    
    
     id | category | categoryIndex
    ----|----------|---------------
     0  | a        | 0.0
     1  | b        | 2.0
     2  | c        | 1.0
     3  | d        | 3.0
     4  | e        | 3.0
    

Notice that the rows containing "d" or "e" are mapped to index "3.0"

Refer to the [StringIndexer Python
docs](api/python/reference/api/pyspark.ml.feature.StringIndexer.html) for more
details on the API.

    
    
    from pyspark.ml.feature import StringIndexer
    
    df = spark.createDataFrame(
        [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
        ["id", "category"])
    
    indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
    indexed = indexer.fit(df).transform(df)
    indexed.show()

Find full example code at
"examples/src/main/python/ml/string_indexer_example.py" in the Spark repo.

Refer to the [StringIndexer Scala
docs](api/scala/org/apache/spark/ml/feature/StringIndexer.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.StringIndexer
    
    val df = spark.createDataFrame(
      Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    ).toDF("id", "category")
    
    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
    
    val indexed = indexer.fit(df).transform(df)
    indexed.show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/StringIndexerExample.scala"
in the Spark repo.

Refer to the [StringIndexer Java
docs](api/java/org/apache/spark/ml/feature/StringIndexer.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.StringIndexer;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    import static org.apache.spark.sql.types.DataTypes.*;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(0, "a"),
      RowFactory.create(1, "b"),
      RowFactory.create(2, "c"),
      RowFactory.create(3, "a"),
      RowFactory.create(4, "a"),
      RowFactory.create(5, "c")
    );
    StructType schema = new StructType(new StructField[]{
      createStructField("id", IntegerType, false),
      createStructField("category", StringType, false)
    });
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    StringIndexer indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex");
    
    Dataset<Row> indexed = indexer.fit(df).transform(df);
    indexed.show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaStringIndexerExample.java"
in the Spark repo.

## IndexToString

Symmetrically to `StringIndexer`, `IndexToString` maps a column of label
indices back to a column containing the original labels as strings. A common
use case is to produce indices from labels with `StringIndexer`, train a model
with those indices and retrieve the original labels from the column of
predicted indices with `IndexToString`. However, you are free to supply your
own labels.

**Examples**

Building on the `StringIndexer` example, let's assume we have the following
DataFrame with columns `id` and `categoryIndex`:

    
    
     id | categoryIndex
    ----|---------------
     0  | 0.0
     1  | 2.0
     2  | 1.0
     3  | 0.0
     4  | 0.0
     5  | 1.0
    

Applying `IndexToString` with `categoryIndex` as the input column,
`originalCategory` as the output column, we are able to retrieve our original
labels (they will be inferred from the columns' metadata):

    
    
     id | categoryIndex | originalCategory
    ----|---------------|-----------------
     0  | 0.0           | a
     1  | 2.0           | b
     2  | 1.0           | c
     3  | 0.0           | a
     4  | 0.0           | a
     5  | 1.0           | c
    

Refer to the [IndexToString Python
docs](api/python/reference/api/pyspark.ml.feature.IndexToString.html) for more
details on the API.

    
    
    from pyspark.ml.feature import IndexToString, StringIndexer
    
    df = spark.createDataFrame(
        [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
        ["id", "category"])
    
    indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
    model = indexer.fit(df)
    indexed = model.transform(df)
    
    print("Transformed string column '%s' to indexed column '%s'"
          % (indexer.getInputCol(), indexer.getOutputCol()))
    indexed.show()
    
    print("StringIndexer will store labels in output column metadata\n")
    
    converter = IndexToString(inputCol="categoryIndex", outputCol="originalCategory")
    converted = converter.transform(indexed)
    
    print("Transformed indexed column '%s' back to original string column '%s' using "
          "labels in metadata" % (converter.getInputCol(), converter.getOutputCol()))
    converted.select("id", "categoryIndex", "originalCategory").show()

Find full example code at
"examples/src/main/python/ml/index_to_string_example.py" in the Spark repo.

Refer to the [IndexToString Scala
docs](api/scala/org/apache/spark/ml/feature/IndexToString.html) for more
details on the API.

    
    
    import org.apache.spark.ml.attribute.Attribute
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
    
    val df = spark.createDataFrame(Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )).toDF("id", "category")
    
    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)
    val indexed = indexer.transform(df)
    
    println(s"Transformed string column '${indexer.getInputCol}' " +
        s"to indexed column '${indexer.getOutputCol}'")
    indexed.show()
    
    val inputColSchema = indexed.schema(indexer.getOutputCol)
    println(s"StringIndexer will store labels in output column metadata: " +
        s"${Attribute.fromStructField(inputColSchema).toString}\n")
    
    val converter = new IndexToString()
      .setInputCol("categoryIndex")
      .setOutputCol("originalCategory")
    
    val converted = converter.transform(indexed)
    
    println(s"Transformed indexed column '${converter.getInputCol}' back to original string " +
        s"column '${converter.getOutputCol}' using labels in metadata")
    converted.select("id", "categoryIndex", "originalCategory").show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/IndexToStringExample.scala"
in the Spark repo.

Refer to the [IndexToString Java
docs](api/java/org/apache/spark/ml/feature/IndexToString.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.attribute.Attribute;
    import org.apache.spark.ml.feature.IndexToString;
    import org.apache.spark.ml.feature.StringIndexer;
    import org.apache.spark.ml.feature.StringIndexerModel;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(0, "a"),
      RowFactory.create(1, "b"),
      RowFactory.create(2, "c"),
      RowFactory.create(3, "a"),
      RowFactory.create(4, "a"),
      RowFactory.create(5, "c")
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("category", DataTypes.StringType, false, Metadata.empty())
    });
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    StringIndexerModel indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df);
    Dataset<Row> indexed = indexer.transform(df);
    
    System.out.println("Transformed string column '" + indexer.getInputCol() + "' " +
        "to indexed column '" + indexer.getOutputCol() + "'");
    indexed.show();
    
    StructField inputColSchema = indexed.schema().apply(indexer.getOutputCol());
    System.out.println("StringIndexer will store labels in output column metadata: " +
        Attribute.fromStructField(inputColSchema).toString() + "\n");
    
    IndexToString converter = new IndexToString()
      .setInputCol("categoryIndex")
      .setOutputCol("originalCategory");
    Dataset<Row> converted = converter.transform(indexed);
    
    System.out.println("Transformed indexed column '" + converter.getInputCol() + "' back to " +
        "original string column '" + converter.getOutputCol() + "' using labels in metadata");
    converted.select("id", "categoryIndex", "originalCategory").show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaIndexToStringExample.java"
in the Spark repo.

## OneHotEncoder

[One-hot encoding](http://en.wikipedia.org/wiki/One-hot) maps a categorical
feature, represented as a label index, to a binary vector with at most a
single one-value indicating the presence of a specific feature value from
among the set of all feature values. This encoding allows algorithms which
expect continuous features, such as Logistic Regression, to use categorical
features. For string type input data, it is common to encode categorical
features using [StringIndexer](ml-features.html#stringindexer) first.

`OneHotEncoder` can transform multiple columns, returning an one-hot-encoded
output vector column for each input column. It is common to merge these
vectors into a single feature vector using [VectorAssembler](ml-
features.html#vectorassembler).

`OneHotEncoder` supports the `handleInvalid` parameter to choose how to handle
invalid input during transforming data. Available options include 'keep' (any
invalid inputs are assigned to an extra categorical index) and 'error' (throw
an error).

**Examples**

Refer to the [OneHotEncoder Python
docs](api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html) for more
details on the API.

    
    
    from pyspark.ml.feature import OneHotEncoder
    
    df = spark.createDataFrame([
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (0.0, 2.0),
        (0.0, 1.0),
        (2.0, 0.0)
    ], ["categoryIndex1", "categoryIndex2"])
    
    encoder = OneHotEncoder(inputCols=["categoryIndex1", "categoryIndex2"],
                            outputCols=["categoryVec1", "categoryVec2"])
    model = encoder.fit(df)
    encoded = model.transform(df)
    encoded.show()

Find full example code at
"examples/src/main/python/ml/onehot_encoder_example.py" in the Spark repo.

Refer to the [OneHotEncoder Scala
docs](api/scala/org/apache/spark/ml/feature/OneHotEncoder.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.OneHotEncoder
    
    val df = spark.createDataFrame(Seq(
      (0.0, 1.0),
      (1.0, 0.0),
      (2.0, 1.0),
      (0.0, 2.0),
      (0.0, 1.0),
      (2.0, 0.0)
    )).toDF("categoryIndex1", "categoryIndex2")
    
    val encoder = new OneHotEncoder()
      .setInputCols(Array("categoryIndex1", "categoryIndex2"))
      .setOutputCols(Array("categoryVec1", "categoryVec2"))
    val model = encoder.fit(df)
    
    val encoded = model.transform(df)
    encoded.show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/OneHotEncoderExample.scala"
in the Spark repo.

Refer to the [OneHotEncoder Java
docs](api/java/org/apache/spark/ml/feature/OneHotEncoder.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.OneHotEncoder;
    import org.apache.spark.ml.feature.OneHotEncoderModel;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(0.0, 1.0),
      RowFactory.create(1.0, 0.0),
      RowFactory.create(2.0, 1.0),
      RowFactory.create(0.0, 2.0),
      RowFactory.create(0.0, 1.0),
      RowFactory.create(2.0, 0.0)
    );
    
    StructType schema = new StructType(new StructField[]{
      new StructField("categoryIndex1", DataTypes.DoubleType, false, Metadata.empty()),
      new StructField("categoryIndex2", DataTypes.DoubleType, false, Metadata.empty())
    });
    
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    OneHotEncoder encoder = new OneHotEncoder()
      .setInputCols(new String[] {"categoryIndex1", "categoryIndex2"})
      .setOutputCols(new String[] {"categoryVec1", "categoryVec2"});
    
    OneHotEncoderModel model = encoder.fit(df);
    Dataset<Row> encoded = model.transform(df);
    encoded.show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaOneHotEncoderExample.java"
in the Spark repo.

## VectorIndexer

`VectorIndexer` helps index categorical features in datasets of `Vector`s. It
can both automatically decide which features are categorical and convert
original values to category indices. Specifically, it does the following:

  1. Take an input column of type [Vector](api/scala/org/apache/spark/ml/linalg/Vector.html) and a parameter `maxCategories`.
  2. Decide which features should be categorical based on the number of distinct values, where features with at most `maxCategories` are declared categorical.
  3. Compute 0-based category indices for each categorical feature.
  4. Index categorical features and transform original feature values to indices.

Indexing categorical features allows algorithms such as Decision Trees and
Tree Ensembles to treat categorical features appropriately, improving
performance.

**Examples**

In the example below, we read in a dataset of labeled points and then use
`VectorIndexer` to decide which features should be treated as categorical. We
transform the categorical feature values to their indices. This transformed
data could then be passed to algorithms such as `DecisionTreeRegressor` that
handle categorical features.

Refer to the [VectorIndexer Python
docs](api/python/reference/api/pyspark.ml.feature.VectorIndexer.html) for more
details on the API.

    
    
    from pyspark.ml.feature import VectorIndexer
    
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    indexer = VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=10)
    indexerModel = indexer.fit(data)
    
    categoricalFeatures = indexerModel.categoryMaps
    print("Chose %d categorical features: %s" %
          (len(categoricalFeatures), ", ".join(str(k) for k in categoricalFeatures.keys())))
    
    # Create new column "indexed" with categorical values transformed to indices
    indexedData = indexerModel.transform(data)
    indexedData.show()

Find full example code at
"examples/src/main/python/ml/vector_indexer_example.py" in the Spark repo.

Refer to the [VectorIndexer Scala
docs](api/scala/org/apache/spark/ml/feature/VectorIndexer.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.VectorIndexer
    
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    val indexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexed")
      .setMaxCategories(10)
    
    val indexerModel = indexer.fit(data)
    
    val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
    println(s"Chose ${categoricalFeatures.size} " +
      s"categorical features: ${categoricalFeatures.mkString(", ")}")
    
    // Create new column "indexed" with categorical values transformed to indices
    val indexedData = indexerModel.transform(data)
    indexedData.show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/VectorIndexerExample.scala"
in the Spark repo.

Refer to the [VectorIndexer Java
docs](api/java/org/apache/spark/ml/feature/VectorIndexer.html) for more
details on the API.

    
    
    import java.util.Map;
    
    import org.apache.spark.ml.feature.VectorIndexer;
    import org.apache.spark.ml.feature.VectorIndexerModel;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    
    Dataset<Row> data = spark.read().format("libsvm").load("data/mllib/sample_libsvm_data.txt");
    
    VectorIndexer indexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexed")
      .setMaxCategories(10);
    VectorIndexerModel indexerModel = indexer.fit(data);
    
    Map<Integer, Map<Double, Integer>> categoryMaps = indexerModel.javaCategoryMaps();
    System.out.print("Chose " + categoryMaps.size() + " categorical features:");
    
    for (Integer feature : categoryMaps.keySet()) {
      System.out.print(" " + feature);
    }
    System.out.println();
    
    // Create new column "indexed" with categorical values transformed to indices
    Dataset<Row> indexedData = indexerModel.transform(data);
    indexedData.show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaVectorIndexerExample.java"
in the Spark repo.

## Interaction

`Interaction` is a `Transformer` which takes vector or double-valued columns,
and generates a single vector column that contains the product of all
combinations of one value from each input column.

For example, if you have 2 vector type columns each of which has 3 dimensions
as input columns, then you'll get a 9-dimensional vector as the output column.

**Examples**

Assume that we have the following DataFrame with the columns "id1", "vec1",
and "vec2":

    
    
      id1|vec1          |vec2          
      ---|--------------|--------------
      1  |[1.0,2.0,3.0] |[8.0,4.0,5.0] 
      2  |[4.0,3.0,8.0] |[7.0,9.0,8.0] 
      3  |[6.0,1.0,9.0] |[2.0,3.0,6.0] 
      4  |[10.0,8.0,6.0]|[9.0,4.0,5.0] 
      5  |[9.0,2.0,7.0] |[10.0,7.0,3.0]
      6  |[1.0,1.0,4.0] |[2.0,8.0,4.0]     
    

Applying `Interaction` with those input columns, then `interactedCol` as the
output column contains:

    
    
      id1|vec1          |vec2          |interactedCol                                         
      ---|--------------|--------------|------------------------------------------------------
      1  |[1.0,2.0,3.0] |[8.0,4.0,5.0] |[8.0,4.0,5.0,16.0,8.0,10.0,24.0,12.0,15.0]            
      2  |[4.0,3.0,8.0] |[7.0,9.0,8.0] |[56.0,72.0,64.0,42.0,54.0,48.0,112.0,144.0,128.0]     
      3  |[6.0,1.0,9.0] |[2.0,3.0,6.0] |[36.0,54.0,108.0,6.0,9.0,18.0,54.0,81.0,162.0]        
      4  |[10.0,8.0,6.0]|[9.0,4.0,5.0] |[360.0,160.0,200.0,288.0,128.0,160.0,216.0,96.0,120.0]
      5  |[9.0,2.0,7.0] |[10.0,7.0,3.0]|[450.0,315.0,135.0,100.0,70.0,30.0,350.0,245.0,105.0] 
      6  |[1.0,1.0,4.0] |[2.0,8.0,4.0] |[12.0,48.0,24.0,12.0,48.0,24.0,48.0,192.0,96.0]       
    

Refer to the [Interaction Python
docs](api/python/reference/api/pyspark.ml.feature.Interaction.html) for more
details on the API.

    
    
    from pyspark.ml.feature import Interaction, VectorAssembler
    
    df = spark.createDataFrame(
        [(1, 1, 2, 3, 8, 4, 5),
         (2, 4, 3, 8, 7, 9, 8),
         (3, 6, 1, 9, 2, 3, 6),
         (4, 10, 8, 6, 9, 4, 5),
         (5, 9, 2, 7, 10, 7, 3),
         (6, 1, 1, 4, 2, 8, 4)],
        ["id1", "id2", "id3", "id4", "id5", "id6", "id7"])
    
    assembler1 = VectorAssembler(inputCols=["id2", "id3", "id4"], outputCol="vec1")
    
    assembled1 = assembler1.transform(df)
    
    assembler2 = VectorAssembler(inputCols=["id5", "id6", "id7"], outputCol="vec2")
    
    assembled2 = assembler2.transform(assembled1).select("id1", "vec1", "vec2")
    
    interaction = Interaction(inputCols=["id1", "vec1", "vec2"], outputCol="interactedCol")
    
    interacted = interaction.transform(assembled2)
    
    interacted.show(truncate=False)

Find full example code at "examples/src/main/python/ml/interaction_example.py"
in the Spark repo.

Refer to the [Interaction Scala
docs](api/scala/org/apache/spark/ml/feature/Interaction.html) for more details
on the API.

    
    
    import org.apache.spark.ml.feature.Interaction
    import org.apache.spark.ml.feature.VectorAssembler
    
    val df = spark.createDataFrame(Seq(
      (1, 1, 2, 3, 8, 4, 5),
      (2, 4, 3, 8, 7, 9, 8),
      (3, 6, 1, 9, 2, 3, 6),
      (4, 10, 8, 6, 9, 4, 5),
      (5, 9, 2, 7, 10, 7, 3),
      (6, 1, 1, 4, 2, 8, 4)
    )).toDF("id1", "id2", "id3", "id4", "id5", "id6", "id7")
    
    val assembler1 = new VectorAssembler().
      setInputCols(Array("id2", "id3", "id4")).
      setOutputCol("vec1")
    
    val assembled1 = assembler1.transform(df)
    
    val assembler2 = new VectorAssembler().
      setInputCols(Array("id5", "id6", "id7")).
      setOutputCol("vec2")
    
    val assembled2 = assembler2.transform(assembled1).select("id1", "vec1", "vec2")
    
    val interaction = new Interaction()
      .setInputCols(Array("id1", "vec1", "vec2"))
      .setOutputCol("interactedCol")
    
    val interacted = interaction.transform(assembled2)
    
    interacted.show(truncate = false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/InteractionExample.scala"
in the Spark repo.

Refer to the [Interaction Java
docs](api/java/org/apache/spark/ml/feature/Interaction.html) for more details
on the API.

    
    
    List<Row> data = Arrays.asList(
      RowFactory.create(1, 1, 2, 3, 8, 4, 5),
      RowFactory.create(2, 4, 3, 8, 7, 9, 8),
      RowFactory.create(3, 6, 1, 9, 2, 3, 6),
      RowFactory.create(4, 10, 8, 6, 9, 4, 5),
      RowFactory.create(5, 9, 2, 7, 10, 7, 3),
      RowFactory.create(6, 1, 1, 4, 2, 8, 4)
    );
    
    StructType schema = new StructType(new StructField[]{
      new StructField("id1", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("id2", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("id3", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("id4", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("id5", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("id6", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("id7", DataTypes.IntegerType, false, Metadata.empty())
    });
    
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    VectorAssembler assembler1 = new VectorAssembler()
            .setInputCols(new String[]{"id2", "id3", "id4"})
            .setOutputCol("vec1");
    
    Dataset<Row> assembled1 = assembler1.transform(df);
    
    VectorAssembler assembler2 = new VectorAssembler()
            .setInputCols(new String[]{"id5", "id6", "id7"})
            .setOutputCol("vec2");
    
    Dataset<Row> assembled2 = assembler2.transform(assembled1).select("id1", "vec1", "vec2");
    
    Interaction interaction = new Interaction()
            .setInputCols(new String[]{"id1","vec1","vec2"})
            .setOutputCol("interactedCol");
    
    Dataset<Row> interacted = interaction.transform(assembled2);
    
    interacted.show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaInteractionExample.java"
in the Spark repo.

## Normalizer

`Normalizer` is a `Transformer` which transforms a dataset of `Vector` rows,
normalizing each `Vector` to have unit norm. It takes parameter `p`, which
specifies the
[p-norm](http://en.wikipedia.org/wiki/Norm_%28mathematics%29#p-norm) used for
normalization. ($p = 2$ by default.) This normalization can help standardize
your input data and improve the behavior of learning algorithms.

**Examples**

The following example demonstrates how to load a dataset in libsvm format and
then normalize each row to have unit $L^1$ norm and unit $L^\infty$ norm.

Refer to the [Normalizer Python
docs](api/python/reference/api/pyspark.ml.feature.Normalizer.html) for more
details on the API.

    
    
    from pyspark.ml.feature import Normalizer
    from pyspark.ml.linalg import Vectors
    
    dataFrame = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.5, -1.0]),),
        (1, Vectors.dense([2.0, 1.0, 1.0]),),
        (2, Vectors.dense([4.0, 10.0, 2.0]),)
    ], ["id", "features"])
    
    # Normalize each Vector using $L^1$ norm.
    normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
    l1NormData = normalizer.transform(dataFrame)
    print("Normalized using L^1 norm")
    l1NormData.show()
    
    # Normalize each Vector using $L^\infty$ norm.
    lInfNormData = normalizer.transform(dataFrame, {normalizer.p: float("inf")})
    print("Normalized using L^inf norm")
    lInfNormData.show()

Find full example code at "examples/src/main/python/ml/normalizer_example.py"
in the Spark repo.

Refer to the [Normalizer Scala
docs](api/scala/org/apache/spark/ml/feature/Normalizer.html) for more details
on the API.

    
    
    import org.apache.spark.ml.feature.Normalizer
    import org.apache.spark.ml.linalg.Vectors
    
    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.5, -1.0)),
      (1, Vectors.dense(2.0, 1.0, 1.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )).toDF("id", "features")
    
    // Normalize each Vector using $L^1$ norm.
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)
    
    val l1NormData = normalizer.transform(dataFrame)
    println("Normalized using L^1 norm")
    l1NormData.show()
    
    // Normalize each Vector using $L^\infty$ norm.
    val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
    println("Normalized using L^inf norm")
    lInfNormData.show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/NormalizerExample.scala"
in the Spark repo.

Refer to the [Normalizer Java
docs](api/java/org/apache/spark/ml/feature/Normalizer.html) for more details
on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.Normalizer;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
        RowFactory.create(0, Vectors.dense(1.0, 0.1, -8.0)),
        RowFactory.create(1, Vectors.dense(2.0, 1.0, -4.0)),
        RowFactory.create(2, Vectors.dense(4.0, 10.0, 8.0))
    );
    StructType schema = new StructType(new StructField[]{
        new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
        new StructField("features", new VectorUDT(), false, Metadata.empty())
    });
    Dataset<Row> dataFrame = spark.createDataFrame(data, schema);
    
    // Normalize each Vector using $L^1$ norm.
    Normalizer normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0);
    
    Dataset<Row> l1NormData = normalizer.transform(dataFrame);
    l1NormData.show();
    
    // Normalize each Vector using $L^\infty$ norm.
    Dataset<Row> lInfNormData =
      normalizer.transform(dataFrame, normalizer.p().w(Double.POSITIVE_INFINITY));
    lInfNormData.show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaNormalizerExample.java"
in the Spark repo.

## StandardScaler

`StandardScaler` transforms a dataset of `Vector` rows, normalizing each
feature to have unit standard deviation and/or zero mean. It takes parameters:

  * `withStd`: True by default. Scales the data to unit standard deviation.
  * `withMean`: False by default. Centers the data with mean before scaling. It will build a dense output, so take care when applying to sparse input.

`StandardScaler` is an `Estimator` which can be `fit` on a dataset to produce
a `StandardScalerModel`; this amounts to computing summary statistics. The
model can then transform a `Vector` column in a dataset to have unit standard
deviation and/or zero mean features.

Note that if the standard deviation of a feature is zero, it will return
default `0.0` value in the `Vector` for that feature.

**Examples**

The following example demonstrates how to load a dataset in libsvm format and
then normalize each feature to have unit standard deviation.

Refer to the [StandardScaler Python
docs](api/python/reference/api/pyspark.ml.feature.StandardScaler.html) for
more details on the API.

    
    
    from pyspark.ml.feature import StandardScaler
    
    dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=False)
    
    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(dataFrame)
    
    # Normalize each feature to have unit standard deviation.
    scaledData = scalerModel.transform(dataFrame)
    scaledData.show()

Find full example code at
"examples/src/main/python/ml/standard_scaler_example.py" in the Spark repo.

Refer to the [StandardScaler Scala
docs](api/scala/org/apache/spark/ml/feature/StandardScaler.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.StandardScaler
    
    val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)
    
    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(dataFrame)
    
    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/StandardScalerExample.scala"
in the Spark repo.

Refer to the [StandardScaler Java
docs](api/java/org/apache/spark/ml/feature/StandardScaler.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.StandardScaler;
    import org.apache.spark.ml.feature.StandardScalerModel;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    
    Dataset<Row> dataFrame =
      spark.read().format("libsvm").load("data/mllib/sample_libsvm_data.txt");
    
    StandardScaler scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false);
    
    // Compute summary statistics by fitting the StandardScaler
    StandardScalerModel scalerModel = scaler.fit(dataFrame);
    
    // Normalize each feature to have unit standard deviation.
    Dataset<Row> scaledData = scalerModel.transform(dataFrame);
    scaledData.show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaStandardScalerExample.java"
in the Spark repo.

## RobustScaler

`RobustScaler` transforms a dataset of `Vector` rows, removing the median and
scaling the data according to a specific quantile range (by default the IQR:
Interquartile Range, quantile range between the 1st quartile and the 3rd
quartile). Its behavior is quite similar to `StandardScaler`, however the
median and the quantile range are used instead of mean and standard deviation,
which make it robust to outliers. It takes parameters:

  * `lower`: 0.25 by default. Lower quantile to calculate quantile range, shared by all features.
  * `upper`: 0.75 by default. Upper quantile to calculate quantile range, shared by all features.
  * `withScaling`: True by default. Scales the data to quantile range.
  * `withCentering`: False by default. Centers the data with median before scaling. It will build a dense output, so take care when applying to sparse input.

`RobustScaler` is an `Estimator` which can be `fit` on a dataset to produce a
`RobustScalerModel`; this amounts to computing quantile statistics. The model
can then transform a `Vector` column in a dataset to have unit quantile range
and/or zero median features.

Note that if the quantile range of a feature is zero, it will return default
`0.0` value in the `Vector` for that feature.

**Examples**

The following example demonstrates how to load a dataset in libsvm format and
then normalize each feature to have unit quantile range.

Refer to the [RobustScaler Python
docs](api/python/reference/api/pyspark.ml.feature.RobustScaler.html) for more
details on the API.

    
    
    from pyspark.ml.feature import RobustScaler
    
    dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    scaler = RobustScaler(inputCol="features", outputCol="scaledFeatures",
                          withScaling=True, withCentering=False,
                          lower=0.25, upper=0.75)
    
    # Compute summary statistics by fitting the RobustScaler
    scalerModel = scaler.fit(dataFrame)
    
    # Transform each feature to have unit quantile range.
    scaledData = scalerModel.transform(dataFrame)
    scaledData.show()

Find full example code at
"examples/src/main/python/ml/robust_scaler_example.py" in the Spark repo.

Refer to the [RobustScaler Scala
docs](api/scala/org/apache/spark/ml/feature/RobustScaler.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.RobustScaler
    
    val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    val scaler = new RobustScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithScaling(true)
      .setWithCentering(false)
      .setLower(0.25)
      .setUpper(0.75)
    
    // Compute summary statistics by fitting the RobustScaler.
    val scalerModel = scaler.fit(dataFrame)
    
    // Transform each feature to have unit quantile range.
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/RobustScalerExample.scala"
in the Spark repo.

Refer to the [RobustScaler Java
docs](api/java/org/apache/spark/ml/feature/RobustScaler.html) for more details
on the API.

    
    
    import org.apache.spark.ml.feature.RobustScaler;
    import org.apache.spark.ml.feature.RobustScalerModel;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    
    Dataset<Row> dataFrame =
      spark.read().format("libsvm").load("data/mllib/sample_libsvm_data.txt");
    
    RobustScaler scaler = new RobustScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithScaling(true)
      .setWithCentering(false)
      .setLower(0.25)
      .setUpper(0.75);
    
    // Compute summary statistics by fitting the RobustScaler
    RobustScalerModel scalerModel = scaler.fit(dataFrame);
    
    // Transform each feature to have unit quantile range.
    Dataset<Row> scaledData = scalerModel.transform(dataFrame);
    scaledData.show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaRobustScalerExample.java"
in the Spark repo.

## MinMaxScaler

`MinMaxScaler` transforms a dataset of `Vector` rows, rescaling each feature
to a specific range (often [0, 1]). It takes parameters:

  * `min`: 0.0 by default. Lower bound after transformation, shared by all features.
  * `max`: 1.0 by default. Upper bound after transformation, shared by all features.

`MinMaxScaler` computes summary statistics on a data set and produces a
`MinMaxScalerModel`. The model can then transform each feature individually
such that it is in the given range.

The rescaled value for a feature E is calculated as, `\begin{equation}
Rescaled(e_i) = \frac{e_i - E_{min}}{E_{max} - E_{min}} * (max - min) + min
\end{equation}` For the case `$E_{max} == E_{min}$`, `$Rescaled(e_i) = 0.5 *
(max + min)$`

Note that since zero values will probably be transformed to non-zero values,
output of the transformer will be `DenseVector` even for sparse input.

**Examples**

The following example demonstrates how to load a dataset in libsvm format and
then rescale each feature to [0, 1].

Refer to the [MinMaxScaler Python
docs](api/python/reference/api/pyspark.ml.feature.MinMaxScaler.html) and the
[MinMaxScalerModel Python
docs](api/python/reference/api/pyspark.ml.feature.MinMaxScalerModel.html) for
more details on the API.

    
    
    from pyspark.ml.feature import MinMaxScaler
    from pyspark.ml.linalg import Vectors
    
    dataFrame = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.1, -1.0]),),
        (1, Vectors.dense([2.0, 1.1, 1.0]),),
        (2, Vectors.dense([3.0, 10.1, 3.0]),)
    ], ["id", "features"])
    
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    
    # Compute summary statistics and generate MinMaxScalerModel
    scalerModel = scaler.fit(dataFrame)
    
    # rescale each feature to range [min, max].
    scaledData = scalerModel.transform(dataFrame)
    print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
    scaledData.select("features", "scaledFeatures").show()

Find full example code at
"examples/src/main/python/ml/min_max_scaler_example.py" in the Spark repo.

Refer to the [MinMaxScaler Scala
docs](api/scala/org/apache/spark/ml/feature/MinMaxScaler.html) and the
[MinMaxScalerModel Scala
docs](api/scala/org/apache/spark/ml/feature/MinMaxScalerModel.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.MinMaxScaler
    import org.apache.spark.ml.linalg.Vectors
    
    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -1.0)),
      (1, Vectors.dense(2.0, 1.1, 1.0)),
      (2, Vectors.dense(3.0, 10.1, 3.0))
    )).toDF("id", "features")
    
    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
    
    // Compute summary statistics and generate MinMaxScalerModel
    val scalerModel = scaler.fit(dataFrame)
    
    // rescale each feature to range [min, max].
    val scaledData = scalerModel.transform(dataFrame)
    println(s"Features scaled to range: [${scaler.getMin}, ${scaler.getMax}]")
    scaledData.select("features", "scaledFeatures").show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/MinMaxScalerExample.scala"
in the Spark repo.

Refer to the [MinMaxScaler Java
docs](api/java/org/apache/spark/ml/feature/MinMaxScaler.html) and the
[MinMaxScalerModel Java
docs](api/java/org/apache/spark/ml/feature/MinMaxScalerModel.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.MinMaxScaler;
    import org.apache.spark.ml.feature.MinMaxScalerModel;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
        RowFactory.create(0, Vectors.dense(1.0, 0.1, -1.0)),
        RowFactory.create(1, Vectors.dense(2.0, 1.1, 1.0)),
        RowFactory.create(2, Vectors.dense(3.0, 10.1, 3.0))
    );
    StructType schema = new StructType(new StructField[]{
        new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
        new StructField("features", new VectorUDT(), false, Metadata.empty())
    });
    Dataset<Row> dataFrame = spark.createDataFrame(data, schema);
    
    MinMaxScaler scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures");
    
    // Compute summary statistics and generate MinMaxScalerModel
    MinMaxScalerModel scalerModel = scaler.fit(dataFrame);
    
    // rescale each feature to range [min, max].
    Dataset<Row> scaledData = scalerModel.transform(dataFrame);
    System.out.println("Features scaled to range: [" + scaler.getMin() + ", "
        + scaler.getMax() + "]");
    scaledData.select("features", "scaledFeatures").show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaMinMaxScalerExample.java"
in the Spark repo.

## MaxAbsScaler

`MaxAbsScaler` transforms a dataset of `Vector` rows, rescaling each feature
to range [-1, 1] by dividing through the maximum absolute value in each
feature. It does not shift/center the data, and thus does not destroy any
sparsity.

`MaxAbsScaler` computes summary statistics on a data set and produces a
`MaxAbsScalerModel`. The model can then transform each feature individually to
range [-1, 1].

**Examples**

The following example demonstrates how to load a dataset in libsvm format and
then rescale each feature to [-1, 1].

Refer to the [MaxAbsScaler Python
docs](api/python/reference/api/pyspark.ml.feature.MaxAbsScaler.html) and the
[MaxAbsScalerModel Python
docs](api/python/reference/api/pyspark.ml.feature.MaxAbsScalerModel.html) for
more details on the API.

    
    
    from pyspark.ml.feature import MaxAbsScaler
    from pyspark.ml.linalg import Vectors
    
    dataFrame = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.1, -8.0]),),
        (1, Vectors.dense([2.0, 1.0, -4.0]),),
        (2, Vectors.dense([4.0, 10.0, 8.0]),)
    ], ["id", "features"])
    
    scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")
    
    # Compute summary statistics and generate MaxAbsScalerModel
    scalerModel = scaler.fit(dataFrame)
    
    # rescale each feature to range [-1, 1].
    scaledData = scalerModel.transform(dataFrame)
    
    scaledData.select("features", "scaledFeatures").show()

Find full example code at
"examples/src/main/python/ml/max_abs_scaler_example.py" in the Spark repo.

Refer to the [MaxAbsScaler Scala
docs](api/scala/org/apache/spark/ml/feature/MaxAbsScaler.html) and the
[MaxAbsScalerModel Scala
docs](api/scala/org/apache/spark/ml/feature/MaxAbsScalerModel.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.MaxAbsScaler
    import org.apache.spark.ml.linalg.Vectors
    
    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -8.0)),
      (1, Vectors.dense(2.0, 1.0, -4.0)),
      (2, Vectors.dense(4.0, 10.0, 8.0))
    )).toDF("id", "features")
    
    val scaler = new MaxAbsScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
    
    // Compute summary statistics and generate MaxAbsScalerModel
    val scalerModel = scaler.fit(dataFrame)
    
    // rescale each feature to range [-1, 1]
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.select("features", "scaledFeatures").show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/MaxAbsScalerExample.scala"
in the Spark repo.

Refer to the [MaxAbsScaler Java
docs](api/java/org/apache/spark/ml/feature/MaxAbsScaler.html) and the
[MaxAbsScalerModel Java
docs](api/java/org/apache/spark/ml/feature/MaxAbsScalerModel.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.MaxAbsScaler;
    import org.apache.spark.ml.feature.MaxAbsScalerModel;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
        RowFactory.create(0, Vectors.dense(1.0, 0.1, -8.0)),
        RowFactory.create(1, Vectors.dense(2.0, 1.0, -4.0)),
        RowFactory.create(2, Vectors.dense(4.0, 10.0, 8.0))
    );
    StructType schema = new StructType(new StructField[]{
        new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
        new StructField("features", new VectorUDT(), false, Metadata.empty())
    });
    Dataset<Row> dataFrame = spark.createDataFrame(data, schema);
    
    MaxAbsScaler scaler = new MaxAbsScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures");
    
    // Compute summary statistics and generate MaxAbsScalerModel
    MaxAbsScalerModel scalerModel = scaler.fit(dataFrame);
    
    // rescale each feature to range [-1, 1].
    Dataset<Row> scaledData = scalerModel.transform(dataFrame);
    scaledData.select("features", "scaledFeatures").show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaMaxAbsScalerExample.java"
in the Spark repo.

## Bucketizer

`Bucketizer` transforms a column of continuous features to a column of feature
buckets, where the buckets are specified by users. It takes a parameter:

  * `splits`: Parameter for mapping continuous features into buckets. With n+1 splits, there are n buckets. A bucket defined by splits x,y holds values in the range [x,y) except the last bucket, which also includes y. Splits should be strictly increasing. Values at -inf, inf must be explicitly provided to cover all Double values; Otherwise, values outside the splits specified will be treated as errors. Two examples of `splits` are `Array(Double.NegativeInfinity, 0.0, 1.0, Double.PositiveInfinity)` and `Array(0.0, 1.0, 2.0)`.

Note that if you have no idea of the upper and lower bounds of the targeted
column, you should add `Double.NegativeInfinity` and `Double.PositiveInfinity`
as the bounds of your splits to prevent a potential out of Bucketizer bounds
exception.

Note also that the splits that you provided have to be in strictly increasing
order, i.e. `s0 < s1 < s2 < ... < sn`.

More details can be found in the API docs for
[Bucketizer](api/scala/org/apache/spark/ml/feature/Bucketizer.html).

**Examples**

The following example demonstrates how to bucketize a column of `Double`s into
another index-wised column.

Refer to the [Bucketizer Python
docs](api/python/reference/api/pyspark.ml.feature.Bucketizer.html) for more
details on the API.

    
    
    from pyspark.ml.feature import Bucketizer
    
    splits = [-float("inf"), -0.5, 0.0, 0.5, float("inf")]
    
    data = [(-999.9,), (-0.5,), (-0.3,), (0.0,), (0.2,), (999.9,)]
    dataFrame = spark.createDataFrame(data, ["features"])
    
    bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bucketedFeatures")
    
    # Transform original data into its bucket index.
    bucketedData = bucketizer.transform(dataFrame)
    
    print("Bucketizer output with %d buckets" % (len(bucketizer.getSplits()) - 1))
    bucketedData.show()

Find full example code at "examples/src/main/python/ml/bucketizer_example.py"
in the Spark repo.

Refer to the [Bucketizer Scala
docs](api/scala/org/apache/spark/ml/feature/Bucketizer.html) for more details
on the API.

    
    
    import org.apache.spark.ml.feature.Bucketizer
    
    val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)
    
    val data = Array(-999.9, -0.5, -0.3, 0.0, 0.2, 999.9)
    val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    
    val bucketizer = new Bucketizer()
      .setInputCol("features")
      .setOutputCol("bucketedFeatures")
      .setSplits(splits)
    
    // Transform original data into its bucket index.
    val bucketedData = bucketizer.transform(dataFrame)
    
    println(s"Bucketizer output with ${bucketizer.getSplits.length-1} buckets")
    bucketedData.show()
    
    val splitsArray = Array(
      Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity),
      Array(Double.NegativeInfinity, -0.3, 0.0, 0.3, Double.PositiveInfinity))
    
    val data2 = Array(
      (-999.9, -999.9),
      (-0.5, -0.2),
      (-0.3, -0.1),
      (0.0, 0.0),
      (0.2, 0.4),
      (999.9, 999.9))
    val dataFrame2 = spark.createDataFrame(data2).toDF("features1", "features2")
    
    val bucketizer2 = new Bucketizer()
      .setInputCols(Array("features1", "features2"))
      .setOutputCols(Array("bucketedFeatures1", "bucketedFeatures2"))
      .setSplitsArray(splitsArray)
    
    // Transform original data into its bucket index.
    val bucketedData2 = bucketizer2.transform(dataFrame2)
    
    println(s"Bucketizer output with [" +
      s"${bucketizer2.getSplitsArray(0).length-1}, " +
      s"${bucketizer2.getSplitsArray(1).length-1}] buckets for each input column")
    bucketedData2.show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/BucketizerExample.scala"
in the Spark repo.

Refer to the [Bucketizer Java
docs](api/java/org/apache/spark/ml/feature/Bucketizer.html) for more details
on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.Bucketizer;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    double[] splits = {Double.NEGATIVE_INFINITY, -0.5, 0.0, 0.5, Double.POSITIVE_INFINITY};
    
    List<Row> data = Arrays.asList(
      RowFactory.create(-999.9),
      RowFactory.create(-0.5),
      RowFactory.create(-0.3),
      RowFactory.create(0.0),
      RowFactory.create(0.2),
      RowFactory.create(999.9)
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("features", DataTypes.DoubleType, false, Metadata.empty())
    });
    Dataset<Row> dataFrame = spark.createDataFrame(data, schema);
    
    Bucketizer bucketizer = new Bucketizer()
      .setInputCol("features")
      .setOutputCol("bucketedFeatures")
      .setSplits(splits);
    
    // Transform original data into its bucket index.
    Dataset<Row> bucketedData = bucketizer.transform(dataFrame);
    
    System.out.println("Bucketizer output with " + (bucketizer.getSplits().length-1) + " buckets");
    bucketedData.show();
    
    // Bucketize multiple columns at one pass.
    double[][] splitsArray = {
      {Double.NEGATIVE_INFINITY, -0.5, 0.0, 0.5, Double.POSITIVE_INFINITY},
      {Double.NEGATIVE_INFINITY, -0.3, 0.0, 0.3, Double.POSITIVE_INFINITY}
    };
    
    List<Row> data2 = Arrays.asList(
      RowFactory.create(-999.9, -999.9),
      RowFactory.create(-0.5, -0.2),
      RowFactory.create(-0.3, -0.1),
      RowFactory.create(0.0, 0.0),
      RowFactory.create(0.2, 0.4),
      RowFactory.create(999.9, 999.9)
    );
    StructType schema2 = new StructType(new StructField[]{
      new StructField("features1", DataTypes.DoubleType, false, Metadata.empty()),
      new StructField("features2", DataTypes.DoubleType, false, Metadata.empty())
    });
    Dataset<Row> dataFrame2 = spark.createDataFrame(data2, schema2);
    
    Bucketizer bucketizer2 = new Bucketizer()
      .setInputCols(new String[] {"features1", "features2"})
      .setOutputCols(new String[] {"bucketedFeatures1", "bucketedFeatures2"})
      .setSplitsArray(splitsArray);
    // Transform original data into its bucket index.
    Dataset<Row> bucketedData2 = bucketizer2.transform(dataFrame2);
    
    System.out.println("Bucketizer output with [" +
      (bucketizer2.getSplitsArray()[0].length-1) + ", " +
      (bucketizer2.getSplitsArray()[1].length-1) + "] buckets for each input column");
    bucketedData2.show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaBucketizerExample.java"
in the Spark repo.

## ElementwiseProduct

ElementwiseProduct multiplies each input vector by a provided "weight" vector,
using element-wise multiplication. In other words, it scales each column of
the dataset by a scalar multiplier. This represents the [Hadamard
product](https://en.wikipedia.org/wiki/Hadamard_product_%28matrices%29)
between the input vector, `v` and transforming vector, `w`, to yield a result
vector.

`\[ \begin{pmatrix} v_1 \\ \vdots \\ v_N \end{pmatrix} \circ \begin{pmatrix}
w_1 \\ \vdots \\ w_N \end{pmatrix} = \begin{pmatrix} v_1 w_1 \\ \vdots \\ v_N
w_N \end{pmatrix} \]`

**Examples**

This example below demonstrates how to transform vectors using a transforming
vector value.

Refer to the [ElementwiseProduct Python
docs](api/python/reference/api/pyspark.ml.feature.ElementwiseProduct.html) for
more details on the API.

    
    
    from pyspark.ml.feature import ElementwiseProduct
    from pyspark.ml.linalg import Vectors
    
    # Create some vector data; also works for sparse vectors
    data = [(Vectors.dense([1.0, 2.0, 3.0]),), (Vectors.dense([4.0, 5.0, 6.0]),)]
    df = spark.createDataFrame(data, ["vector"])
    transformer = ElementwiseProduct(scalingVec=Vectors.dense([0.0, 1.0, 2.0]),
                                     inputCol="vector", outputCol="transformedVector")
    # Batch transform the vectors to create new column:
    transformer.transform(df).show()

Find full example code at
"examples/src/main/python/ml/elementwise_product_example.py" in the Spark
repo.

Refer to the [ElementwiseProduct Scala
docs](api/scala/org/apache/spark/ml/feature/ElementwiseProduct.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.ElementwiseProduct
    import org.apache.spark.ml.linalg.Vectors
    
    // Create some vector data; also works for sparse vectors
    val dataFrame = spark.createDataFrame(Seq(
      ("a", Vectors.dense(1.0, 2.0, 3.0)),
      ("b", Vectors.dense(4.0, 5.0, 6.0)))).toDF("id", "vector")
    
    val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
    val transformer = new ElementwiseProduct()
      .setScalingVec(transformingVector)
      .setInputCol("vector")
      .setOutputCol("transformedVector")
    
    // Batch transform the vectors to create new column:
    transformer.transform(dataFrame).show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/ElementwiseProductExample.scala"
in the Spark repo.

Refer to the [ElementwiseProduct Java
docs](api/java/org/apache/spark/ml/feature/ElementwiseProduct.html) for more
details on the API.

    
    
    import java.util.ArrayList;
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.ElementwiseProduct;
    import org.apache.spark.ml.linalg.Vector;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    // Create some vector data; also works for sparse vectors
    List<Row> data = Arrays.asList(
      RowFactory.create("a", Vectors.dense(1.0, 2.0, 3.0)),
      RowFactory.create("b", Vectors.dense(4.0, 5.0, 6.0))
    );
    
    List<StructField> fields = new ArrayList<>(2);
    fields.add(DataTypes.createStructField("id", DataTypes.StringType, false));
    fields.add(DataTypes.createStructField("vector", new VectorUDT(), false));
    
    StructType schema = DataTypes.createStructType(fields);
    
    Dataset<Row> dataFrame = spark.createDataFrame(data, schema);
    
    Vector transformingVector = Vectors.dense(0.0, 1.0, 2.0);
    
    ElementwiseProduct transformer = new ElementwiseProduct()
      .setScalingVec(transformingVector)
      .setInputCol("vector")
      .setOutputCol("transformedVector");
    
    // Batch transform the vectors to create new column:
    transformer.transform(dataFrame).show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaElementwiseProductExample.java"
in the Spark repo.

## SQLTransformer

`SQLTransformer` implements the transformations which are defined by SQL
statement. Currently, we only support SQL syntax like `"SELECT ... FROM
__THIS__ ..."` where `"__THIS__"` represents the underlying table of the input
dataset. The select clause specifies the fields, constants, and expressions to
display in the output, and can be any select clause that Spark SQL supports.
Users can also use Spark SQL built-in function and UDFs to operate on these
selected columns. For example, `SQLTransformer` supports statements like:

  * `SELECT a, a + b AS a_b FROM __THIS__`
  * `SELECT a, SQRT(b) AS b_sqrt FROM __THIS__ where a > 5`
  * `SELECT a, b, SUM(c) AS c_sum FROM __THIS__ GROUP BY a, b`

**Examples**

Assume that we have the following DataFrame with columns `id`, `v1` and `v2`:

    
    
     id |  v1 |  v2
    ----|-----|-----
     0  | 1.0 | 3.0  
     2  | 2.0 | 5.0
    

This is the output of the `SQLTransformer` with statement `"SELECT *, (v1 +
v2) AS v3, (v1 * v2) AS v4 FROM __THIS__"`:

    
    
     id |  v1 |  v2 |  v3 |  v4
    ----|-----|-----|-----|-----
     0  | 1.0 | 3.0 | 4.0 | 3.0
     2  | 2.0 | 5.0 | 7.0 |10.0
    

Refer to the [SQLTransformer Python
docs](api/python/reference/api/pyspark.ml.feature.SQLTransformer.html) for
more details on the API.

    
    
    from pyspark.ml.feature import SQLTransformer
    
    df = spark.createDataFrame([
        (0, 1.0, 3.0),
        (2, 2.0, 5.0)
    ], ["id", "v1", "v2"])
    sqlTrans = SQLTransformer(
        statement="SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")
    sqlTrans.transform(df).show()

Find full example code at "examples/src/main/python/ml/sql_transformer.py" in
the Spark repo.

Refer to the [SQLTransformer Scala
docs](api/scala/org/apache/spark/ml/feature/SQLTransformer.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.SQLTransformer
    
    val df = spark.createDataFrame(
      Seq((0, 1.0, 3.0), (2, 2.0, 5.0))).toDF("id", "v1", "v2")
    
    val sqlTrans = new SQLTransformer().setStatement(
      "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")
    
    sqlTrans.transform(df).show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/SQLTransformerExample.scala"
in the Spark repo.

Refer to the [SQLTransformer Java
docs](api/java/org/apache/spark/ml/feature/SQLTransformer.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.SQLTransformer;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.SparkSession;
    import org.apache.spark.sql.types.*;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(0, 1.0, 3.0),
      RowFactory.create(2, 2.0, 5.0)
    );
    StructType schema = new StructType(new StructField [] {
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("v1", DataTypes.DoubleType, false, Metadata.empty()),
      new StructField("v2", DataTypes.DoubleType, false, Metadata.empty())
    });
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    SQLTransformer sqlTrans = new SQLTransformer().setStatement(
      "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__");
    
    sqlTrans.transform(df).show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaSQLTransformerExample.java"
in the Spark repo.

## VectorAssembler

`VectorAssembler` is a transformer that combines a given list of columns into
a single vector column. It is useful for combining raw features and features
generated by different feature transformers into a single feature vector, in
order to train ML models like logistic regression and decision trees.
`VectorAssembler` accepts the following input column types: all numeric types,
boolean type, and vector type. In each row, the values of the input columns
will be concatenated into a vector in the specified order.

**Examples**

Assume that we have a DataFrame with the columns `id`, `hour`, `mobile`,
`userFeatures`, and `clicked`:

    
    
     id | hour | mobile | userFeatures     | clicked
    ----|------|--------|------------------|---------
     0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0
    

`userFeatures` is a vector column that contains three user features. We want
to combine `hour`, `mobile`, and `userFeatures` into a single feature vector
called `features` and use it to predict `clicked` or not. If we set
`VectorAssembler`'s input columns to `hour`, `mobile`, and `userFeatures` and
output column to `features`, after transformation we should get the following
DataFrame:

    
    
     id | hour | mobile | userFeatures     | clicked | features
    ----|------|--------|------------------|---------|-----------------------------
     0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0     | [18.0, 1.0, 0.0, 10.0, 0.5]
    

Refer to the [VectorAssembler Python
docs](api/python/reference/api/pyspark.ml.feature.VectorAssembler.html) for
more details on the API.

    
    
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.feature import VectorAssembler
    
    dataset = spark.createDataFrame(
        [(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0)],
        ["id", "hour", "mobile", "userFeatures", "clicked"])
    
    assembler = VectorAssembler(
        inputCols=["hour", "mobile", "userFeatures"],
        outputCol="features")
    
    output = assembler.transform(dataset)
    print("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
    output.select("features", "clicked").show(truncate=False)

Find full example code at
"examples/src/main/python/ml/vector_assembler_example.py" in the Spark repo.

Refer to the [VectorAssembler Scala
docs](api/scala/org/apache/spark/ml/feature/VectorAssembler.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.linalg.Vectors
    
    val dataset = spark.createDataFrame(
      Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
    ).toDF("id", "hour", "mobile", "userFeatures", "clicked")
    
    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "mobile", "userFeatures"))
      .setOutputCol("features")
    
    val output = assembler.transform(dataset)
    println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
    output.select("features", "clicked").show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/VectorAssemblerExample.scala"
in the Spark repo.

Refer to the [VectorAssembler Java
docs](api/java/org/apache/spark/ml/feature/VectorAssembler.html) for more
details on the API.

    
    
    import java.util.Arrays;
    
    import org.apache.spark.ml.feature.VectorAssembler;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.*;
    import static org.apache.spark.sql.types.DataTypes.*;
    
    StructType schema = createStructType(new StructField[]{
      createStructField("id", IntegerType, false),
      createStructField("hour", IntegerType, false),
      createStructField("mobile", DoubleType, false),
      createStructField("userFeatures", new VectorUDT(), false),
      createStructField("clicked", DoubleType, false)
    });
    Row row = RowFactory.create(0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0);
    Dataset<Row> dataset = spark.createDataFrame(Arrays.asList(row), schema);
    
    VectorAssembler assembler = new VectorAssembler()
      .setInputCols(new String[]{"hour", "mobile", "userFeatures"})
      .setOutputCol("features");
    
    Dataset<Row> output = assembler.transform(dataset);
    System.out.println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column " +
        "'features'");
    output.select("features", "clicked").show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaVectorAssemblerExample.java"
in the Spark repo.

## VectorSizeHint

It can sometimes be useful to explicitly specify the size of the vectors for a
column of `VectorType`. For example, `VectorAssembler` uses size information
from its input columns to produce size information and metadata for its output
column. While in some cases this information can be obtained by inspecting the
contents of the column, in a streaming dataframe the contents are not
available until the stream is started. `VectorSizeHint` allows a user to
explicitly specify the vector size for a column so that `VectorAssembler`, or
other transformers that might need to know vector size, can use that column as
an input.

To use `VectorSizeHint` a user must set the `inputCol` and `size` parameters.
Applying this transformer to a dataframe produces a new dataframe with updated
metadata for `inputCol` specifying the vector size. Downstream operations on
the resulting dataframe can get this size using the metadata.

`VectorSizeHint` can also take an optional `handleInvalid` parameter which
controls its behaviour when the vector column contains nulls or vectors of the
wrong size. By default `handleInvalid` is set to "error", indicating an
exception should be thrown. This parameter can also be set to "skip",
indicating that rows containing invalid values should be filtered out from the
resulting dataframe, or "optimistic", indicating that the column should not be
checked for invalid values and all rows should be kept. Note that the use of
"optimistic" can cause the resulting dataframe to be in an inconsistent state,
meaning the metadata for the column `VectorSizeHint` was applied to does not
match the contents of that column. Users should take care to avoid this kind
of inconsistent state.

Refer to the [VectorSizeHint Python
docs](api/python/reference/api/pyspark.ml.feature.VectorSizeHint.html) for
more details on the API.

    
    
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.feature import (VectorSizeHint, VectorAssembler)
    
    dataset = spark.createDataFrame(
        [(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0),
         (0, 18, 1.0, Vectors.dense([0.0, 10.0]), 0.0)],
        ["id", "hour", "mobile", "userFeatures", "clicked"])
    
    sizeHint = VectorSizeHint(
        inputCol="userFeatures",
        handleInvalid="skip",
        size=3)
    
    datasetWithSize = sizeHint.transform(dataset)
    print("Rows where 'userFeatures' is not the right size are filtered out")
    datasetWithSize.show(truncate=False)
    
    assembler = VectorAssembler(
        inputCols=["hour", "mobile", "userFeatures"],
        outputCol="features")
    
    # This dataframe can be used by downstream transformers as before
    output = assembler.transform(datasetWithSize)
    print("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
    output.select("features", "clicked").show(truncate=False)

Find full example code at
"examples/src/main/python/ml/vector_size_hint_example.py" in the Spark repo.

Refer to the [VectorSizeHint Scala
docs](api/scala/org/apache/spark/ml/feature/VectorSizeHint.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.{VectorAssembler, VectorSizeHint}
    import org.apache.spark.ml.linalg.Vectors
    
    val dataset = spark.createDataFrame(
      Seq(
        (0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0),
        (0, 18, 1.0, Vectors.dense(0.0, 10.0), 0.0))
    ).toDF("id", "hour", "mobile", "userFeatures", "clicked")
    
    val sizeHint = new VectorSizeHint()
      .setInputCol("userFeatures")
      .setHandleInvalid("skip")
      .setSize(3)
    
    val datasetWithSize = sizeHint.transform(dataset)
    println("Rows where 'userFeatures' is not the right size are filtered out")
    datasetWithSize.show(false)
    
    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "mobile", "userFeatures"))
      .setOutputCol("features")
    
    // This dataframe can be used by downstream transformers as before
    val output = assembler.transform(datasetWithSize)
    println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
    output.select("features", "clicked").show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/VectorSizeHintExample.scala"
in the Spark repo.

Refer to the [VectorSizeHint Java
docs](api/java/org/apache/spark/ml/feature/VectorSizeHint.html) for more
details on the API.

    
    
    import java.util.Arrays;
    
    import org.apache.spark.ml.feature.VectorAssembler;
    import org.apache.spark.ml.feature.VectorSizeHint;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    import static org.apache.spark.sql.types.DataTypes.*;
    
    StructType schema = createStructType(new StructField[]{
      createStructField("id", IntegerType, false),
      createStructField("hour", IntegerType, false),
      createStructField("mobile", DoubleType, false),
      createStructField("userFeatures", new VectorUDT(), false),
      createStructField("clicked", DoubleType, false)
    });
    Row row0 = RowFactory.create(0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0);
    Row row1 = RowFactory.create(0, 18, 1.0, Vectors.dense(0.0, 10.0), 0.0);
    Dataset<Row> dataset = spark.createDataFrame(Arrays.asList(row0, row1), schema);
    
    VectorSizeHint sizeHint = new VectorSizeHint()
      .setInputCol("userFeatures")
      .setHandleInvalid("skip")
      .setSize(3);
    
    Dataset<Row> datasetWithSize = sizeHint.transform(dataset);
    System.out.println("Rows where 'userFeatures' is not the right size are filtered out");
    datasetWithSize.show(false);
    
    VectorAssembler assembler = new VectorAssembler()
      .setInputCols(new String[]{"hour", "mobile", "userFeatures"})
      .setOutputCol("features");
    
    // This dataframe can be used by downstream transformers as before
    Dataset<Row> output = assembler.transform(datasetWithSize);
    System.out.println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column " +
        "'features'");
    output.select("features", "clicked").show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaVectorSizeHintExample.java"
in the Spark repo.

## QuantileDiscretizer

`QuantileDiscretizer` takes a column with continuous features and outputs a
column with binned categorical features. The number of bins is set by the
`numBuckets` parameter. It is possible that the number of buckets used will be
smaller than this value, for example, if there are too few distinct values of
the input to create enough distinct quantiles.

NaN values: NaN values will be removed from the column during
`QuantileDiscretizer` fitting. This will produce a `Bucketizer` model for
making predictions. During the transformation, `Bucketizer` will raise an
error when it finds NaN values in the dataset, but the user can also choose to
either keep or remove NaN values within the dataset by setting
`handleInvalid`. If the user chooses to keep NaN values, they will be handled
specially and placed into their own bucket, for example, if 4 buckets are
used, then non-NaN data will be put into buckets[0-3], but NaNs will be
counted in a special bucket[4].

Algorithm: The bin ranges are chosen using an approximate algorithm (see the
documentation for
[approxQuantile](api/scala/org/apache/spark/sql/DataFrameStatFunctions.html)
for a detailed description). The precision of the approximation can be
controlled with the `relativeError` parameter. When set to zero, exact
quantiles are calculated (**Note:** Computing exact quantiles is an expensive
operation). The lower and upper bin bounds will be `-Infinity` and `+Infinity`
covering all real values.

**Examples**

Assume that we have a DataFrame with the columns `id`, `hour`:

    
    
     id | hour
    ----|------
     0  | 18.0
    ----|------
     1  | 19.0
    ----|------
     2  | 8.0
    ----|------
     3  | 5.0
    ----|------
     4  | 2.2
    

`hour` is a continuous feature with `Double` type. We want to turn the
continuous feature into a categorical one. Given `numBuckets = 3`, we should
get the following DataFrame:

    
    
     id | hour | result
    ----|------|------
     0  | 18.0 | 2.0
    ----|------|------
     1  | 19.0 | 2.0
    ----|------|------
     2  | 8.0  | 1.0
    ----|------|------
     3  | 5.0  | 1.0
    ----|------|------
     4  | 2.2  | 0.0
    

Refer to the [QuantileDiscretizer Python
docs](api/python/reference/api/pyspark.ml.feature.QuantileDiscretizer.html)
for more details on the API.

    
    
    from pyspark.ml.feature import QuantileDiscretizer
    
    data = [(0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2)]
    df = spark.createDataFrame(data, ["id", "hour"])
    
    discretizer = QuantileDiscretizer(numBuckets=3, inputCol="hour", outputCol="result")
    
    result = discretizer.fit(df).transform(df)
    result.show()

Find full example code at
"examples/src/main/python/ml/quantile_discretizer_example.py" in the Spark
repo.

Refer to the [QuantileDiscretizer Scala
docs](api/scala/org/apache/spark/ml/feature/QuantileDiscretizer.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.QuantileDiscretizer
    
    val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
    val df = spark.createDataFrame(data).toDF("id", "hour")
    
    val discretizer = new QuantileDiscretizer()
      .setInputCol("hour")
      .setOutputCol("result")
      .setNumBuckets(3)
    
    val result = discretizer.fit(df).transform(df)
    result.show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/QuantileDiscretizerExample.scala"
in the Spark repo.

Refer to the [QuantileDiscretizer Java
docs](api/java/org/apache/spark/ml/feature/QuantileDiscretizer.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.QuantileDiscretizer;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(0, 18.0),
      RowFactory.create(1, 19.0),
      RowFactory.create(2, 8.0),
      RowFactory.create(3, 5.0),
      RowFactory.create(4, 2.2)
    );
    
    StructType schema = new StructType(new StructField[]{
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("hour", DataTypes.DoubleType, false, Metadata.empty())
    });
    
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    QuantileDiscretizer discretizer = new QuantileDiscretizer()
      .setInputCol("hour")
      .setOutputCol("result")
      .setNumBuckets(3);
    
    Dataset<Row> result = discretizer.fit(df).transform(df);
    result.show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaQuantileDiscretizerExample.java"
in the Spark repo.

## Imputer

The `Imputer` estimator completes missing values in a dataset, using the mean,
median or mode of the columns in which the missing values are located. The
input columns should be of numeric type. Currently `Imputer` does not support
categorical features and possibly creates incorrect values for columns
containing categorical features. Imputer can impute custom values other than
'NaN' by `.setMissingValue(custom_value)`. For example, `.setMissingValue(0)`
will impute all occurrences of (0).

**Note** all `null` values in the input columns are treated as missing, and so
are also imputed.

**Examples**

Suppose that we have a DataFrame with the columns `a` and `b`:

    
    
          a     |      b      
    ------------|-----------
         1.0    | Double.NaN
         2.0    | Double.NaN
     Double.NaN |     3.0   
         4.0    |     4.0   
         5.0    |     5.0   
    

In this example, Imputer will replace all occurrences of `Double.NaN` (the
default for the missing value) with the mean (the default imputation strategy)
computed from the other values in the corresponding columns. In this example,
the surrogate values for columns `a` and `b` are 3.0 and 4.0 respectively.
After transformation, the missing values in the output columns will be
replaced by the surrogate value for the relevant column.

    
    
          a     |      b     | out_a | out_b   
    ------------|------------|-------|-------
         1.0    | Double.NaN |  1.0  |  4.0 
         2.0    | Double.NaN |  2.0  |  4.0 
     Double.NaN |     3.0    |  3.0  |  3.0 
         4.0    |     4.0    |  4.0  |  4.0
         5.0    |     5.0    |  5.0  |  5.0 
    

Refer to the [Imputer Python
docs](api/python/reference/api/pyspark.ml.feature.Imputer.html) for more
details on the API.

    
    
    from pyspark.ml.feature import Imputer
    
    df = spark.createDataFrame([
        (1.0, float("nan")),
        (2.0, float("nan")),
        (float("nan"), 3.0),
        (4.0, 4.0),
        (5.0, 5.0)
    ], ["a", "b"])
    
    imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
    model = imputer.fit(df)
    
    model.transform(df).show()

Find full example code at "examples/src/main/python/ml/imputer_example.py" in
the Spark repo.

Refer to the [Imputer Scala
docs](api/scala/org/apache/spark/ml/feature/Imputer.html) for more details on
the API.

    
    
    import org.apache.spark.ml.feature.Imputer
    
    val df = spark.createDataFrame(Seq(
      (1.0, Double.NaN),
      (2.0, Double.NaN),
      (Double.NaN, 3.0),
      (4.0, 4.0),
      (5.0, 5.0)
    )).toDF("a", "b")
    
    val imputer = new Imputer()
      .setInputCols(Array("a", "b"))
      .setOutputCols(Array("out_a", "out_b"))
    
    val model = imputer.fit(df)
    model.transform(df).show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/ImputerExample.scala" in
the Spark repo.

Refer to the [Imputer Java
docs](api/java/org/apache/spark/ml/feature/Imputer.html) for more details on
the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.Imputer;
    import org.apache.spark.ml.feature.ImputerModel;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.SparkSession;
    import org.apache.spark.sql.types.*;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(1.0, Double.NaN),
      RowFactory.create(2.0, Double.NaN),
      RowFactory.create(Double.NaN, 3.0),
      RowFactory.create(4.0, 4.0),
      RowFactory.create(5.0, 5.0)
    );
    StructType schema = new StructType(new StructField[]{
      createStructField("a", DoubleType, false),
      createStructField("b", DoubleType, false)
    });
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    Imputer imputer = new Imputer()
      .setInputCols(new String[]{"a", "b"})
      .setOutputCols(new String[]{"out_a", "out_b"});
    
    ImputerModel model = imputer.fit(df);
    model.transform(df).show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaImputerExample.java"
in the Spark repo.

# Feature Selectors

## VectorSlicer

`VectorSlicer` is a transformer that takes a feature vector and outputs a new
feature vector with a sub-array of the original features. It is useful for
extracting features from a vector column.

`VectorSlicer` accepts a vector column with specified indices, then outputs a
new vector column whose values are selected via those indices. There are two
types of indices,

  1. Integer indices that represent the indices into the vector, `setIndices()`.

  2. String indices that represent the names of features into the vector, `setNames()`. _This requires the vector column to have an`AttributeGroup` since the implementation matches on the name field of an `Attribute`._

Specification by integer and string are both acceptable. Moreover, you can use
integer index and string name simultaneously. At least one feature must be
selected. Duplicate features are not allowed, so there can be no overlap
between selected indices and names. Note that if names of features are
selected, an exception will be thrown if empty input attributes are
encountered.

The output vector will order features with the selected indices first (in the
order given), followed by the selected names (in the order given).

**Examples**

Suppose that we have a DataFrame with the column `userFeatures`:

    
    
     userFeatures
    ------------------
     [0.0, 10.0, 0.5]
    

`userFeatures` is a vector column that contains three user features. Assume
that the first column of `userFeatures` are all zeros, so we want to remove it
and select only the last two columns. The `VectorSlicer` selects the last two
elements with `setIndices(1, 2)` then produces a new vector column named
`features`:

    
    
     userFeatures     | features
    ------------------|-----------------------------
     [0.0, 10.0, 0.5] | [10.0, 0.5]
    

Suppose also that we have potential input attributes for the `userFeatures`,
i.e. `["f1", "f2", "f3"]`, then we can use `setNames("f2", "f3")` to select
them.

    
    
     userFeatures     | features
    ------------------|-----------------------------
     [0.0, 10.0, 0.5] | [10.0, 0.5]
     ["f1", "f2", "f3"] | ["f2", "f3"]
    

Refer to the [VectorSlicer Python
docs](api/python/reference/api/pyspark.ml.feature.VectorSlicer.html) for more
details on the API.

    
    
    from pyspark.ml.feature import VectorSlicer
    from pyspark.ml.linalg import Vectors
    from pyspark.sql.types import Row
    
    df = spark.createDataFrame([
        Row(userFeatures=Vectors.sparse(3, {0: -2.0, 1: 2.3})),
        Row(userFeatures=Vectors.dense([-2.0, 2.3, 0.0]))])
    
    slicer = VectorSlicer(inputCol="userFeatures", outputCol="features", indices=[1])
    
    output = slicer.transform(df)
    
    output.select("userFeatures", "features").show()

Find full example code at
"examples/src/main/python/ml/vector_slicer_example.py" in the Spark repo.

Refer to the [VectorSlicer Scala
docs](api/scala/org/apache/spark/ml/feature/VectorSlicer.html) for more
details on the API.

    
    
    import java.util.Arrays
    
    import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
    import org.apache.spark.ml.feature.VectorSlicer
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.sql.{Row, SparkSession}
    import org.apache.spark.sql.types.StructType
    
    val data = Arrays.asList(
      Row(Vectors.sparse(3, Seq((0, -2.0), (1, 2.3)))),
      Row(Vectors.dense(-2.0, 2.3, 0.0))
    )
    
    val defaultAttr = NumericAttribute.defaultAttr
    val attrs = Array("f1", "f2", "f3").map(defaultAttr.withName)
    val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])
    
    val dataset = spark.createDataFrame(data, StructType(Array(attrGroup.toStructField())))
    
    val slicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")
    
    slicer.setIndices(Array(1)).setNames(Array("f3"))
    // or slicer.setIndices(Array(1, 2)), or slicer.setNames(Array("f2", "f3"))
    
    val output = slicer.transform(dataset)
    output.show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/VectorSlicerExample.scala"
in the Spark repo.

Refer to the [VectorSlicer Java
docs](api/java/org/apache/spark/ml/feature/VectorSlicer.html) for more details
on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.attribute.Attribute;
    import org.apache.spark.ml.attribute.AttributeGroup;
    import org.apache.spark.ml.attribute.NumericAttribute;
    import org.apache.spark.ml.feature.VectorSlicer;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.*;
    
    Attribute[] attrs = {
      NumericAttribute.defaultAttr().withName("f1"),
      NumericAttribute.defaultAttr().withName("f2"),
      NumericAttribute.defaultAttr().withName("f3")
    };
    AttributeGroup group = new AttributeGroup("userFeatures", attrs);
    
    List<Row> data = Arrays.asList(
      RowFactory.create(Vectors.sparse(3, new int[]{0, 1}, new double[]{-2.0, 2.3})),
      RowFactory.create(Vectors.dense(-2.0, 2.3, 0.0))
    );
    
    Dataset<Row> dataset =
      spark.createDataFrame(data, (new StructType()).add(group.toStructField()));
    
    VectorSlicer vectorSlicer = new VectorSlicer()
      .setInputCol("userFeatures").setOutputCol("features");
    
    vectorSlicer.setIndices(new int[]{1}).setNames(new String[]{"f3"});
    // or slicer.setIndices(new int[]{1, 2}), or slicer.setNames(new String[]{"f2", "f3"})
    
    Dataset<Row> output = vectorSlicer.transform(dataset);
    output.show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaVectorSlicerExample.java"
in the Spark repo.

## RFormula

`RFormula` selects columns specified by an [R model
formula](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/formula.html).
Currently we support a limited subset of the R operators, including '~', '.',
':', '+', and '-'. The basic operators are:

  * `~` separate target and terms
  * `+` concat terms, "\+ 0" means removing intercept
  * `-` remove a term, "\- 1" means removing intercept
  * `:` interaction (multiplication for numeric values, or binarized categorical values)
  * `.` all columns except target

Suppose `a` and `b` are double columns, we use the following simple examples
to illustrate the effect of `RFormula`:

  * `y ~ a + b` means model `y ~ w0 + w1 * a + w2 * b` where `w0` is the intercept and `w1, w2` are coefficients.
  * `y ~ a + b + a:b - 1` means model `y ~ w1 * a + w2 * b + w3 * a * b` where `w1, w2, w3` are coefficients.

`RFormula` produces a vector column of features and a double or string column
of label. Like when formulas are used in R for linear regression, numeric
columns will be cast to doubles. As to string input columns, they will first
be transformed with [StringIndexer](ml-features.html#stringindexer) using
ordering determined by `stringOrderType`, and the last category after ordering
is dropped, then the doubles will be one-hot encoded.

Suppose a string feature column containing values `{'b', 'a', 'b', 'a', 'c',
'b'}`, we set `stringOrderType` to control the encoding:

    
    
    stringOrderType | Category mapped to 0 by StringIndexer |  Category dropped by RFormula
    ----------------|---------------------------------------|---------------------------------
    'frequencyDesc' | most frequent category ('b')          | least frequent category ('c')
    'frequencyAsc'  | least frequent category ('c')         | most frequent category ('b')
    'alphabetDesc'  | last alphabetical category ('c')      | first alphabetical category ('a')
    'alphabetAsc'   | first alphabetical category ('a')     | last alphabetical category ('c')
    

If the label column is of type string, it will be first transformed to double
with [StringIndexer](ml-features.html#stringindexer) using `frequencyDesc`
ordering. If the label column does not exist in the DataFrame, the output
label column will be created from the specified response variable in the
formula.

**Note:** The ordering option `stringOrderType` is NOT used for the label
column. When the label column is indexed, it uses the default descending
frequency ordering in `StringIndexer`.

**Examples**

Assume that we have a DataFrame with the columns `id`, `country`, `hour`, and
`clicked`:

    
    
    id | country | hour | clicked
    ---|---------|------|---------
     7 | "US"    | 18   | 1.0
     8 | "CA"    | 12   | 0.0
     9 | "NZ"    | 15   | 0.0
    

If we use `RFormula` with a formula string of `clicked ~ country + hour`,
which indicates that we want to predict `clicked` based on `country` and
`hour`, after transformation we should get the following DataFrame:

    
    
    id | country | hour | clicked | features         | label
    ---|---------|------|---------|------------------|-------
     7 | "US"    | 18   | 1.0     | [0.0, 0.0, 18.0] | 1.0
     8 | "CA"    | 12   | 0.0     | [0.0, 1.0, 12.0] | 0.0
     9 | "NZ"    | 15   | 0.0     | [1.0, 0.0, 15.0] | 0.0
    

Refer to the [RFormula Python
docs](api/python/reference/api/pyspark.ml.feature.RFormula.html) for more
details on the API.

    
    
    from pyspark.ml.feature import RFormula
    
    dataset = spark.createDataFrame(
        [(7, "US", 18, 1.0),
         (8, "CA", 12, 0.0),
         (9, "NZ", 15, 0.0)],
        ["id", "country", "hour", "clicked"])
    
    formula = RFormula(
        formula="clicked ~ country + hour",
        featuresCol="features",
        labelCol="label")
    
    output = formula.fit(dataset).transform(dataset)
    output.select("features", "label").show()

Find full example code at "examples/src/main/python/ml/rformula_example.py" in
the Spark repo.

Refer to the [RFormula Scala
docs](api/scala/org/apache/spark/ml/feature/RFormula.html) for more details on
the API.

    
    
    import org.apache.spark.ml.feature.RFormula
    
    val dataset = spark.createDataFrame(Seq(
      (7, "US", 18, 1.0),
      (8, "CA", 12, 0.0),
      (9, "NZ", 15, 0.0)
    )).toDF("id", "country", "hour", "clicked")
    
    val formula = new RFormula()
      .setFormula("clicked ~ country + hour")
      .setFeaturesCol("features")
      .setLabelCol("label")
    
    val output = formula.fit(dataset).transform(dataset)
    output.select("features", "label").show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/RFormulaExample.scala"
in the Spark repo.

Refer to the [RFormula Java
docs](api/java/org/apache/spark/ml/feature/RFormula.html) for more details on
the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.RFormula;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    import static org.apache.spark.sql.types.DataTypes.*;
    
    StructType schema = createStructType(new StructField[]{
      createStructField("id", IntegerType, false),
      createStructField("country", StringType, false),
      createStructField("hour", IntegerType, false),
      createStructField("clicked", DoubleType, false)
    });
    
    List<Row> data = Arrays.asList(
      RowFactory.create(7, "US", 18, 1.0),
      RowFactory.create(8, "CA", 12, 0.0),
      RowFactory.create(9, "NZ", 15, 0.0)
    );
    
    Dataset<Row> dataset = spark.createDataFrame(data, schema);
    RFormula formula = new RFormula()
      .setFormula("clicked ~ country + hour")
      .setFeaturesCol("features")
      .setLabelCol("label");
    Dataset<Row> output = formula.fit(dataset).transform(dataset);
    output.select("features", "label").show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaRFormulaExample.java"
in the Spark repo.

## ChiSqSelector

`ChiSqSelector` stands for Chi-Squared feature selection. It operates on
labeled data with categorical features. ChiSqSelector uses the [Chi-Squared
test of independence](https://en.wikipedia.org/wiki/Chi-squared_test) to
decide which features to choose. It supports five selection methods:
`numTopFeatures`, `percentile`, `fpr`, `fdr`, `fwe`:

  * `numTopFeatures` chooses a fixed number of top features according to a chi-squared test. This is akin to yielding the features with the most predictive power.
  * `percentile` is similar to `numTopFeatures` but chooses a fraction of all features instead of a fixed number.
  * `fpr` chooses all features whose p-values are below a threshold, thus controlling the false positive rate of selection.
  * `fdr` uses the [Benjamini-Hochberg procedure](https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini.E2.80.93Hochberg_procedure) to choose all features whose false discovery rate is below a threshold.
  * `fwe` chooses all features whose p-values are below a threshold. The threshold is scaled by 1/numFeatures, thus controlling the family-wise error rate of selection. By default, the selection method is `numTopFeatures`, with the default number of top features set to 50. The user can choose a selection method using `setSelectorType`.

**Examples**

Assume that we have a DataFrame with the columns `id`, `features`, and
`clicked`, which is used as our target to be predicted:

    
    
    id | features              | clicked
    ---|-----------------------|---------
     7 | [0.0, 0.0, 18.0, 1.0] | 1.0
     8 | [0.0, 1.0, 12.0, 0.0] | 0.0
     9 | [1.0, 0.0, 15.0, 0.1] | 0.0
    

If we use `ChiSqSelector` with `numTopFeatures = 1`, then according to our
label `clicked` the last column in our `features` is chosen as the most useful
feature:

    
    
    id | features              | clicked | selectedFeatures
    ---|-----------------------|---------|------------------
     7 | [0.0, 0.0, 18.0, 1.0] | 1.0     | [1.0]
     8 | [0.0, 1.0, 12.0, 0.0] | 0.0     | [0.0]
     9 | [1.0, 0.0, 15.0, 0.1] | 0.0     | [0.1]
    

Refer to the [ChiSqSelector Python
docs](api/python/reference/api/pyspark.ml.feature.ChiSqSelector.html) for more
details on the API.

    
    
    from pyspark.ml.feature import ChiSqSelector
    from pyspark.ml.linalg import Vectors
    
    df = spark.createDataFrame([
        (7, Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0,),
        (8, Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0,),
        (9, Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0,)], ["id", "features", "clicked"])
    
    selector = ChiSqSelector(numTopFeatures=1, featuresCol="features",
                             outputCol="selectedFeatures", labelCol="clicked")
    
    result = selector.fit(df).transform(df)
    
    print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
    result.show()

Find full example code at
"examples/src/main/python/ml/chisq_selector_example.py" in the Spark repo.

Refer to the [ChiSqSelector Scala
docs](api/scala/org/apache/spark/ml/feature/ChiSqSelector.html) for more
details on the API.

    
    
    import org.apache.spark.ml.feature.ChiSqSelector
    import org.apache.spark.ml.linalg.Vectors
    
    val data = Seq(
      (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
      (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
      (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
    )
    
    val df = spark.createDataset(data).toDF("id", "features", "clicked")
    
    val selector = new ChiSqSelector()
      .setNumTopFeatures(1)
      .setFeaturesCol("features")
      .setLabelCol("clicked")
      .setOutputCol("selectedFeatures")
    
    val result = selector.fit(df).transform(df)
    
    println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
    result.show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/ChiSqSelectorExample.scala"
in the Spark repo.

Refer to the [ChiSqSelector Java
docs](api/java/org/apache/spark/ml/feature/ChiSqSelector.html) for more
details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.ChiSqSelector;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
      RowFactory.create(8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
      RowFactory.create(9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("features", new VectorUDT(), false, Metadata.empty()),
      new StructField("clicked", DataTypes.DoubleType, false, Metadata.empty())
    });
    
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    ChiSqSelector selector = new ChiSqSelector()
      .setNumTopFeatures(1)
      .setFeaturesCol("features")
      .setLabelCol("clicked")
      .setOutputCol("selectedFeatures");
    
    Dataset<Row> result = selector.fit(df).transform(df);
    
    System.out.println("ChiSqSelector output with top " + selector.getNumTopFeatures()
        + " features selected");
    result.show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaChiSqSelectorExample.java"
in the Spark repo.

## UnivariateFeatureSelector

`UnivariateFeatureSelector` operates on categorical/continuous labels with
categorical/continuous features. User can set `featureType` and `labelType`,
and Spark will pick the score function to use based on the specified
`featureType` and `labelType`.

    
    
    featureType |  labelType |score function
    ------------|------------|--------------
    categorical |categorical | chi-squared (chi2)
    continuous  |categorical | ANOVATest (f_classif)
    continuous  |continuous  | F-value (f_regression)
    

It supports five selection modes: `numTopFeatures`, `percentile`, `fpr`,
`fdr`, `fwe`:

  * `numTopFeatures` chooses a fixed number of top features.
  * `percentile` is similar to `numTopFeatures` but chooses a fraction of all features instead of a fixed number.
  * `fpr` chooses all features whose p-values are below a threshold, thus controlling the false positive rate of selection.
  * `fdr` uses the [Benjamini-Hochberg procedure](https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini.E2.80.93Hochberg_procedure) to choose all features whose false discovery rate is below a threshold.
  * `fwe` chooses all features whose p-values are below a threshold. The threshold is scaled by 1/numFeatures, thus controlling the family-wise error rate of selection.

By default, the selection mode is `numTopFeatures`, with the default
selectionThreshold sets to 50.

**Examples**

Assume that we have a DataFrame with the columns `id`, `features`, and
`label`, which is used as our target to be predicted:

    
    
    id | features                       | label
    ---|--------------------------------|---------
     1 | [1.7, 4.4, 7.6, 5.8, 9.6, 2.3] | 3.0
     2 | [8.8, 7.3, 5.7, 7.3, 2.2, 4.1] | 2.0
     3 | [1.2, 9.5, 2.5, 3.1, 8.7, 2.5] | 3.0
     4 | [3.7, 9.2, 6.1, 4.1, 7.5, 3.8] | 2.0
     5 | [8.9, 5.2, 7.8, 8.3, 5.2, 3.0] | 4.0
     6 | [7.9, 8.5, 9.2, 4.0, 9.4, 2.1] | 4.0
    

If we set `featureType` to `continuous` and `labelType` to `categorical` with
`numTopFeatures = 1`, the last column in our `features` is chosen as the most
useful feature:

    
    
    id | features                       | label   | selectedFeatures
    ---|--------------------------------|---------|------------------
     1 | [1.7, 4.4, 7.6, 5.8, 9.6, 2.3] | 3.0     | [2.3]
     2 | [8.8, 7.3, 5.7, 7.3, 2.2, 4.1] | 2.0     | [4.1]
     3 | [1.2, 9.5, 2.5, 3.1, 8.7, 2.5] | 3.0     | [2.5]
     4 | [3.7, 9.2, 6.1, 4.1, 7.5, 3.8] | 2.0     | [3.8]
     5 | [8.9, 5.2, 7.8, 8.3, 5.2, 3.0] | 4.0     | [3.0]
     6 | [7.9, 8.5, 9.2, 4.0, 9.4, 2.1] | 4.0     | [2.1]
    

Refer to the [UnivariateFeatureSelector Python
docs](api/python/reference/api/pyspark.ml.feature.UnivariateFeatureSelector.html)
for more details on the API.

    
    
    from pyspark.ml.feature import UnivariateFeatureSelector
    from pyspark.ml.linalg import Vectors
    
    df = spark.createDataFrame([
        (1, Vectors.dense([1.7, 4.4, 7.6, 5.8, 9.6, 2.3]), 3.0,),
        (2, Vectors.dense([8.8, 7.3, 5.7, 7.3, 2.2, 4.1]), 2.0,),
        (3, Vectors.dense([1.2, 9.5, 2.5, 3.1, 8.7, 2.5]), 3.0,),
        (4, Vectors.dense([3.7, 9.2, 6.1, 4.1, 7.5, 3.8]), 2.0,),
        (5, Vectors.dense([8.9, 5.2, 7.8, 8.3, 5.2, 3.0]), 4.0,),
        (6, Vectors.dense([7.9, 8.5, 9.2, 4.0, 9.4, 2.1]), 4.0,)], ["id", "features", "label"])
    
    selector = UnivariateFeatureSelector(featuresCol="features", outputCol="selectedFeatures",
                                         labelCol="label", selectionMode="numTopFeatures")
    selector.setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(1)
    
    result = selector.fit(df).transform(df)
    
    print("UnivariateFeatureSelector output with top %d features selected using f_classif"
          % selector.getSelectionThreshold())
    result.show()

Find full example code at
"examples/src/main/python/ml/univariate_feature_selector_example.py" in the
Spark repo.

Refer to the [UnivariateFeatureSelector Scala
docs](api/scala/org/apache/spark/ml/feature/UnivariateFeatureSelector.html)
for more details on the API.

    
    
    import org.apache.spark.ml.feature.UnivariateFeatureSelector
    import org.apache.spark.ml.linalg.Vectors
    
    val data = Seq(
      (1, Vectors.dense(1.7, 4.4, 7.6, 5.8, 9.6, 2.3), 3.0),
      (2, Vectors.dense(8.8, 7.3, 5.7, 7.3, 2.2, 4.1), 2.0),
      (3, Vectors.dense(1.2, 9.5, 2.5, 3.1, 8.7, 2.5), 3.0),
      (4, Vectors.dense(3.7, 9.2, 6.1, 4.1, 7.5, 3.8), 2.0),
      (5, Vectors.dense(8.9, 5.2, 7.8, 8.3, 5.2, 3.0), 4.0),
      (6, Vectors.dense(7.9, 8.5, 9.2, 4.0, 9.4, 2.1), 4.0)
    )
    
    val df = spark.createDataset(data).toDF("id", "features", "label")
    
    val selector = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("categorical")
      .setSelectionMode("numTopFeatures")
      .setSelectionThreshold(1)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setOutputCol("selectedFeatures")
    
    val result = selector.fit(df).transform(df)
    
    println(s"UnivariateFeatureSelector output with top ${selector.getSelectionThreshold}" +
      s" features selected using f_classif")
    result.show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/UnivariateFeatureSelectorExample.scala"
in the Spark repo.

Refer to the [UnivariateFeatureSelector Java
docs](api/java/org/apache/spark/ml/feature/UnivariateFeatureSelector.html) for
more details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.UnivariateFeatureSelector;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.*;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(1, Vectors.dense(1.7, 4.4, 7.6, 5.8, 9.6, 2.3), 3.0),
      RowFactory.create(2, Vectors.dense(8.8, 7.3, 5.7, 7.3, 2.2, 4.1), 2.0),
      RowFactory.create(3, Vectors.dense(1.2, 9.5, 2.5, 3.1, 8.7, 2.5), 3.0),
      RowFactory.create(4, Vectors.dense(3.7, 9.2, 6.1, 4.1, 7.5, 3.8), 2.0),
      RowFactory.create(5, Vectors.dense(8.9, 5.2, 7.8, 8.3, 5.2, 3.0), 4.0),
      RowFactory.create(6, Vectors.dense(7.9, 8.5, 9.2, 4.0, 9.4, 2.1), 4.0)
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("features", new VectorUDT(), false, Metadata.empty()),
      new StructField("label", DataTypes.DoubleType, false, Metadata.empty())
    });
    
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    UnivariateFeatureSelector selector = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("categorical")
      .setSelectionMode("numTopFeatures")
      .setSelectionThreshold(1)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setOutputCol("selectedFeatures");
    
    Dataset<Row> result = selector.fit(df).transform(df);
    
    System.out.println("UnivariateFeatureSelector output with top "
        + selector.getSelectionThreshold() + " features selected using f_classif");
    result.show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaUnivariateFeatureSelectorExample.java"
in the Spark repo.

## VarianceThresholdSelector

`VarianceThresholdSelector` is a selector that removes low-variance features.
Features with a (sample) variance not greater than the `varianceThreshold`
will be removed. If not set, `varianceThreshold` defaults to 0, which means
only features with variance 0 (i.e. features that have the same value in all
samples) will be removed.

**Examples**

Assume that we have a DataFrame with the columns `id` and `features`, which is
used as our target to be predicted:

    
    
    id | features
    ---|--------------------------------
     1 | [6.0, 7.0, 0.0, 7.0, 6.0, 0.0]
     2 | [0.0, 9.0, 6.0, 0.0, 5.0, 9.0]
     3 | [0.0, 9.0, 3.0, 0.0, 5.0, 5.0]
     4 | [0.0, 9.0, 8.0, 5.0, 6.0, 4.0]
     5 | [8.0, 9.0, 6.0, 5.0, 4.0, 4.0]
     6 | [8.0, 9.0, 6.0, 0.0, 0.0, 0.0]
    

The sample variances for the 6 features are 16.67, 0.67, 8.17, 10.17, 5.07,
and 11.47 respectively. If we use `VarianceThresholdSelector` with
`varianceThreshold = 8.0`, then the features with variance <= 8.0 are removed:

    
    
    id | features                       | selectedFeatures
    ---|--------------------------------|-------------------
     1 | [6.0, 7.0, 0.0, 7.0, 6.0, 0.0] | [6.0,0.0,7.0,0.0]
     2 | [0.0, 9.0, 6.0, 0.0, 5.0, 9.0] | [0.0,6.0,0.0,9.0]
     3 | [0.0, 9.0, 3.0, 0.0, 5.0, 5.0] | [0.0,3.0,0.0,5.0]
     4 | [0.0, 9.0, 8.0, 5.0, 6.0, 4.0] | [0.0,8.0,5.0,4.0]
     5 | [8.0, 9.0, 6.0, 5.0, 4.0, 4.0] | [8.0,6.0,5.0,4.0]
     6 | [8.0, 9.0, 6.0, 0.0, 0.0, 0.0] | [8.0,6.0,0.0,0.0]
    

Refer to the [VarianceThresholdSelector Python
docs](api/python/reference/api/pyspark.ml.feature.VarianceThresholdSelector.html)
for more details on the API.

    
    
    from pyspark.ml.feature import VarianceThresholdSelector
    from pyspark.ml.linalg import Vectors
    
    df = spark.createDataFrame([
        (1, Vectors.dense([6.0, 7.0, 0.0, 7.0, 6.0, 0.0])),
        (2, Vectors.dense([0.0, 9.0, 6.0, 0.0, 5.0, 9.0])),
        (3, Vectors.dense([0.0, 9.0, 3.0, 0.0, 5.0, 5.0])),
        (4, Vectors.dense([0.0, 9.0, 8.0, 5.0, 6.0, 4.0])),
        (5, Vectors.dense([8.0, 9.0, 6.0, 5.0, 4.0, 4.0])),
        (6, Vectors.dense([8.0, 9.0, 6.0, 0.0, 0.0, 0.0]))], ["id", "features"])
    
    selector = VarianceThresholdSelector(varianceThreshold=8.0, outputCol="selectedFeatures")
    
    result = selector.fit(df).transform(df)
    
    print("Output: Features with variance lower than %f are removed." %
          selector.getVarianceThreshold())
    result.show()

Find full example code at
"examples/src/main/python/ml/variance_threshold_selector_example.py" in the
Spark repo.

Refer to the [VarianceThresholdSelector Scala
docs](api/scala/org/apache/spark/ml/feature/VarianceThresholdSelector.html)
for more details on the API.

    
    
    import org.apache.spark.ml.feature.VarianceThresholdSelector
    import org.apache.spark.ml.linalg.Vectors
    
    val data = Seq(
      (1, Vectors.dense(6.0, 7.0, 0.0, 7.0, 6.0, 0.0)),
      (2, Vectors.dense(0.0, 9.0, 6.0, 0.0, 5.0, 9.0)),
      (3, Vectors.dense(0.0, 9.0, 3.0, 0.0, 5.0, 5.0)),
      (4, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0)),
      (5, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0)),
      (6, Vectors.dense(8.0, 9.0, 6.0, 0.0, 0.0, 0.0))
    )
    
    val df = spark.createDataset(data).toDF("id", "features")
    
    val selector = new VarianceThresholdSelector()
      .setVarianceThreshold(8.0)
      .setFeaturesCol("features")
      .setOutputCol("selectedFeatures")
    
    val result = selector.fit(df).transform(df)
    
    println(s"Output: Features with variance lower than" +
      s" ${selector.getVarianceThreshold} are removed.")
    result.show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/VarianceThresholdSelectorExample.scala"
in the Spark repo.

Refer to the [VarianceThresholdSelector Java
docs](api/java/org/apache/spark/ml/feature/VarianceThresholdSelector.html) for
more details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.VarianceThresholdSelector;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.*;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(1, Vectors.dense(6.0, 7.0, 0.0, 7.0, 6.0, 0.0)),
      RowFactory.create(2, Vectors.dense(0.0, 9.0, 6.0, 0.0, 5.0, 9.0)),
      RowFactory.create(3, Vectors.dense(0.0, 9.0, 3.0, 0.0, 5.0, 5.0)),
      RowFactory.create(4, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0)),
      RowFactory.create(5, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0)),
      RowFactory.create(6, Vectors.dense(8.0, 9.0, 6.0, 0.0, 0.0, 0.0))
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("features", new VectorUDT(), false, Metadata.empty())
    });
    
    Dataset<Row> df = spark.createDataFrame(data, schema);
    
    VarianceThresholdSelector selector = new VarianceThresholdSelector()
      .setVarianceThreshold(8.0)
      .setFeaturesCol("features")
      .setOutputCol("selectedFeatures");
    
    Dataset<Row> result = selector.fit(df).transform(df);
    
    System.out.println("Output: Features with variance lower than "
        + selector.getVarianceThreshold() + " are removed.");
    result.show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaVarianceThresholdSelectorExample.java"
in the Spark repo.

# Locality Sensitive Hashing

[Locality Sensitive Hashing (LSH)](https://en.wikipedia.org/wiki/Locality-
sensitive_hashing) is an important class of hashing techniques, which is
commonly used in clustering, approximate nearest neighbor search and outlier
detection with large datasets.

The general idea of LSH is to use a family of functions ("LSH families") to
hash data points into buckets, so that the data points which are close to each
other are in the same buckets with high probability, while data points that
are far away from each other are very likely in different buckets. An LSH
family is formally defined as follows.

In a metric space `(M, d)`, where `M` is a set and `d` is a distance function
on `M`, an LSH family is a family of functions `h` that satisfy the following
properties: `\[ \forall p, q \in M,\\ d(p,q) \leq r1 \Rightarrow Pr(h(p)=h(q))
\geq p1\\ d(p,q) \geq r2 \Rightarrow Pr(h(p)=h(q)) \leq p2 \]` This LSH family
is called `(r1, r2, p1, p2)`-sensitive.

In Spark, different LSH families are implemented in separate classes (e.g.,
`MinHash`), and APIs for feature transformation, approximate similarity join
and approximate nearest neighbor are provided in each class.

In LSH, we define a false positive as a pair of distant input features (with
`$d(p,q) \geq r2$`) which are hashed into the same bucket, and we define a
false negative as a pair of nearby features (with `$d(p,q) \leq r1$`) which
are hashed into different buckets.

## LSH Operations

We describe the major types of operations which LSH can be used for. A fitted
LSH model has methods for each of these operations.

### Feature Transformation

Feature transformation is the basic functionality to add hashed values as a
new column. This can be useful for dimensionality reduction. Users can specify
input and output column names by setting `inputCol` and `outputCol`.

LSH also supports multiple LSH hash tables. Users can specify the number of
hash tables by setting `numHashTables`. This is also used for [OR-
amplification](https://en.wikipedia.org/wiki/Locality-
sensitive_hashing#Amplification) in approximate similarity join and
approximate nearest neighbor. Increasing the number of hash tables will
increase the accuracy but will also increase communication cost and running
time.

The type of `outputCol` is `Seq[Vector]` where the dimension of the array
equals `numHashTables`, and the dimensions of the vectors are currently set to
1. In future releases, we will implement AND-amplification so that users can
specify the dimensions of these vectors.

### Approximate Similarity Join

Approximate similarity join takes two datasets and approximately returns pairs
of rows in the datasets whose distance is smaller than a user-defined
threshold. Approximate similarity join supports both joining two different
datasets and self-joining. Self-joining will produce some duplicate pairs.

Approximate similarity join accepts both transformed and untransformed
datasets as input. If an untransformed dataset is used, it will be transformed
automatically. In this case, the hash signature will be created as
`outputCol`.

In the joined dataset, the origin datasets can be queried in `datasetA` and
`datasetB`. A distance column will be added to the output dataset to show the
true distance between each pair of rows returned.

### Approximate Nearest Neighbor Search

Approximate nearest neighbor search takes a dataset (of feature vectors) and a
key (a single feature vector), and it approximately returns a specified number
of rows in the dataset that are closest to the vector.

Approximate nearest neighbor search accepts both transformed and untransformed
datasets as input. If an untransformed dataset is used, it will be transformed
automatically. In this case, the hash signature will be created as
`outputCol`.

A distance column will be added to the output dataset to show the true
distance between each output row and the searched key.

**Note:** Approximate nearest neighbor search will return fewer than `k` rows
when there are not enough candidates in the hash bucket.

## LSH Algorithms

### Bucketed Random Projection for Euclidean Distance

[Bucketed Random Projection](https://en.wikipedia.org/wiki/Locality-
sensitive_hashing#Stable_distributions) is an LSH family for Euclidean
distance. The Euclidean distance is defined as follows: `\[ d(\mathbf{x},
\mathbf{y}) = \sqrt{\sum_i (x_i - y_i)^2} \]` Its LSH family projects feature
vectors `$\mathbf{x}$` onto a random unit vector `$\mathbf{v}$` and portions
the projected results into hash buckets: `\[ h(\mathbf{x}) = \Big\lfloor
\frac{\mathbf{x} \cdot \mathbf{v}}{r} \Big\rfloor \]` where `r` is a user-
defined bucket length. The bucket length can be used to control the average
size of hash buckets (and thus the number of buckets). A larger bucket length
(i.e., fewer buckets) increases the probability of features being hashed to
the same bucket (increasing the numbers of true and false positives).

Bucketed Random Projection accepts arbitrary vectors as input features, and
supports both sparse and dense vectors.

Refer to the [BucketedRandomProjectionLSH Python
docs](api/python/reference/api/pyspark.ml.feature.BucketedRandomProjectionLSH.html)
for more details on the API.

    
    
    from pyspark.ml.feature import BucketedRandomProjectionLSH
    from pyspark.ml.linalg import Vectors
    from pyspark.sql.functions import col
    
    dataA = [(0, Vectors.dense([1.0, 1.0]),),
             (1, Vectors.dense([1.0, -1.0]),),
             (2, Vectors.dense([-1.0, -1.0]),),
             (3, Vectors.dense([-1.0, 1.0]),)]
    dfA = spark.createDataFrame(dataA, ["id", "features"])
    
    dataB = [(4, Vectors.dense([1.0, 0.0]),),
             (5, Vectors.dense([-1.0, 0.0]),),
             (6, Vectors.dense([0.0, 1.0]),),
             (7, Vectors.dense([0.0, -1.0]),)]
    dfB = spark.createDataFrame(dataB, ["id", "features"])
    
    key = Vectors.dense([1.0, 0.0])
    
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0,
                                      numHashTables=3)
    model = brp.fit(dfA)
    
    # Feature Transformation
    print("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(dfA).show()
    
    # Compute the locality sensitive hashes for the input rows, then perform approximate
    # similarity join.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxSimilarityJoin(transformedA, transformedB, 1.5)`
    print("Approximately joining dfA and dfB on Euclidean distance smaller than 1.5:")
    model.approxSimilarityJoin(dfA, dfB, 1.5, distCol="EuclideanDistance")\
        .select(col("datasetA.id").alias("idA"),
                col("datasetB.id").alias("idB"),
                col("EuclideanDistance")).show()
    
    # Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    # neighbor search.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxNearestNeighbors(transformedA, key, 2)`
    print("Approximately searching dfA for 2 nearest neighbors of the key:")
    model.approxNearestNeighbors(dfA, key, 2).show()

Find full example code at
"examples/src/main/python/ml/bucketed_random_projection_lsh_example.py" in the
Spark repo.

Refer to the [BucketedRandomProjectionLSH Scala
docs](api/scala/org/apache/spark/ml/feature/BucketedRandomProjectionLSH.html)
for more details on the API.

    
    
    import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.sql.functions.col
    
    val dfA = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (2, Vectors.dense(-1.0, -1.0)),
      (3, Vectors.dense(-1.0, 1.0))
    )).toDF("id", "features")
    
    val dfB = spark.createDataFrame(Seq(
      (4, Vectors.dense(1.0, 0.0)),
      (5, Vectors.dense(-1.0, 0.0)),
      (6, Vectors.dense(0.0, 1.0)),
      (7, Vectors.dense(0.0, -1.0))
    )).toDF("id", "features")
    
    val key = Vectors.dense(1.0, 0.0)
    
    val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(2.0)
      .setNumHashTables(3)
      .setInputCol("features")
      .setOutputCol("hashes")
    
    val model = brp.fit(dfA)
    
    // Feature Transformation
    println("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(dfA).show()
    
    // Compute the locality sensitive hashes for the input rows, then perform approximate
    // similarity join.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxSimilarityJoin(transformedA, transformedB, 1.5)`
    println("Approximately joining dfA and dfB on Euclidean distance smaller than 1.5:")
    model.approxSimilarityJoin(dfA, dfB, 1.5, "EuclideanDistance")
      .select(col("datasetA.id").alias("idA"),
        col("datasetB.id").alias("idB"),
        col("EuclideanDistance")).show()
    
    // Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    // neighbor search.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxNearestNeighbors(transformedA, key, 2)`
    println("Approximately searching dfA for 2 nearest neighbors of the key:")
    model.approxNearestNeighbors(dfA, key, 2).show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/BucketedRandomProjectionLSHExample.scala"
in the Spark repo.

Refer to the [BucketedRandomProjectionLSH Java
docs](api/java/org/apache/spark/ml/feature/BucketedRandomProjectionLSH.html)
for more details on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.BucketedRandomProjectionLSH;
    import org.apache.spark.ml.feature.BucketedRandomProjectionLSHModel;
    import org.apache.spark.ml.linalg.Vector;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    import static org.apache.spark.sql.functions.col;
    
    List<Row> dataA = Arrays.asList(
      RowFactory.create(0, Vectors.dense(1.0, 1.0)),
      RowFactory.create(1, Vectors.dense(1.0, -1.0)),
      RowFactory.create(2, Vectors.dense(-1.0, -1.0)),
      RowFactory.create(3, Vectors.dense(-1.0, 1.0))
    );
    
    List<Row> dataB = Arrays.asList(
        RowFactory.create(4, Vectors.dense(1.0, 0.0)),
        RowFactory.create(5, Vectors.dense(-1.0, 0.0)),
        RowFactory.create(6, Vectors.dense(0.0, 1.0)),
        RowFactory.create(7, Vectors.dense(0.0, -1.0))
    );
    
    StructType schema = new StructType(new StructField[]{
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("features", new VectorUDT(), false, Metadata.empty())
    });
    Dataset<Row> dfA = spark.createDataFrame(dataA, schema);
    Dataset<Row> dfB = spark.createDataFrame(dataB, schema);
    
    Vector key = Vectors.dense(1.0, 0.0);
    
    BucketedRandomProjectionLSH mh = new BucketedRandomProjectionLSH()
      .setBucketLength(2.0)
      .setNumHashTables(3)
      .setInputCol("features")
      .setOutputCol("hashes");
    
    BucketedRandomProjectionLSHModel model = mh.fit(dfA);
    
    // Feature Transformation
    System.out.println("The hashed dataset where hashed values are stored in the column 'hashes':");
    model.transform(dfA).show();
    
    // Compute the locality sensitive hashes for the input rows, then perform approximate
    // similarity join.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxSimilarityJoin(transformedA, transformedB, 1.5)`
    System.out.println("Approximately joining dfA and dfB on distance smaller than 1.5:");
    model.approxSimilarityJoin(dfA, dfB, 1.5, "EuclideanDistance")
      .select(col("datasetA.id").alias("idA"),
        col("datasetB.id").alias("idB"),
        col("EuclideanDistance")).show();
    
    // Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    // neighbor search.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxNearestNeighbors(transformedA, key, 2)`
    System.out.println("Approximately searching dfA for 2 nearest neighbors of the key:");
    model.approxNearestNeighbors(dfA, key, 2).show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaBucketedRandomProjectionLSHExample.java"
in the Spark repo.

### MinHash for Jaccard Distance

[MinHash](https://en.wikipedia.org/wiki/MinHash) is an LSH family for Jaccard
distance where input features are sets of natural numbers. Jaccard distance of
two sets is defined by the cardinality of their intersection and union: `\[
d(\mathbf{A}, \mathbf{B}) = 1 - \frac{|\mathbf{A} \cap
\mathbf{B}|}{|\mathbf{A} \cup \mathbf{B}|} \]` MinHash applies a random hash
function `g` to each element in the set and take the minimum of all hashed
values: `\[ h(\mathbf{A}) = \min_{a \in \mathbf{A}}(g(a)) \]`

The input sets for MinHash are represented as binary vectors, where the vector
indices represent the elements themselves and the non-zero values in the
vector represent the presence of that element in the set. While both dense and
sparse vectors are supported, typically sparse vectors are recommended for
efficiency. For example, `Vectors.sparse(10, Array[(2, 1.0), (3, 1.0), (5,
1.0)])` means there are 10 elements in the space. This set contains elem 2,
elem 3 and elem 5. All non-zero values are treated as binary "1" values.

**Note:** Empty sets cannot be transformed by MinHash, which means any input
vector must have at least 1 non-zero entry.

Refer to the [MinHashLSH Python
docs](api/python/reference/api/pyspark.ml.feature.MinHashLSH.html) for more
details on the API.

    
    
    from pyspark.ml.feature import MinHashLSH
    from pyspark.ml.linalg import Vectors
    from pyspark.sql.functions import col
    
    dataA = [(0, Vectors.sparse(6, [0, 1, 2], [1.0, 1.0, 1.0]),),
             (1, Vectors.sparse(6, [2, 3, 4], [1.0, 1.0, 1.0]),),
             (2, Vectors.sparse(6, [0, 2, 4], [1.0, 1.0, 1.0]),)]
    dfA = spark.createDataFrame(dataA, ["id", "features"])
    
    dataB = [(3, Vectors.sparse(6, [1, 3, 5], [1.0, 1.0, 1.0]),),
             (4, Vectors.sparse(6, [2, 3, 5], [1.0, 1.0, 1.0]),),
             (5, Vectors.sparse(6, [1, 2, 4], [1.0, 1.0, 1.0]),)]
    dfB = spark.createDataFrame(dataB, ["id", "features"])
    
    key = Vectors.sparse(6, [1, 3], [1.0, 1.0])
    
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(dfA)
    
    # Feature Transformation
    print("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(dfA).show()
    
    # Compute the locality sensitive hashes for the input rows, then perform approximate
    # similarity join.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxSimilarityJoin(transformedA, transformedB, 0.6)`
    print("Approximately joining dfA and dfB on distance smaller than 0.6:")
    model.approxSimilarityJoin(dfA, dfB, 0.6, distCol="JaccardDistance")\
        .select(col("datasetA.id").alias("idA"),
                col("datasetB.id").alias("idB"),
                col("JaccardDistance")).show()
    
    # Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    # neighbor search.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxNearestNeighbors(transformedA, key, 2)`
    # It may return less than 2 rows when not enough approximate near-neighbor candidates are
    # found.
    print("Approximately searching dfA for 2 nearest neighbors of the key:")
    model.approxNearestNeighbors(dfA, key, 2).show()

Find full example code at
"examples/src/main/python/ml/min_hash_lsh_example.py" in the Spark repo.

Refer to the [MinHashLSH Scala
docs](api/scala/org/apache/spark/ml/feature/MinHashLSH.html) for more details
on the API.

    
    
    import org.apache.spark.ml.feature.MinHashLSH
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.sql.functions.col
    
    val dfA = spark.createDataFrame(Seq(
      (0, Vectors.sparse(6, Seq((0, 1.0), (1, 1.0), (2, 1.0)))),
      (1, Vectors.sparse(6, Seq((2, 1.0), (3, 1.0), (4, 1.0)))),
      (2, Vectors.sparse(6, Seq((0, 1.0), (2, 1.0), (4, 1.0))))
    )).toDF("id", "features")
    
    val dfB = spark.createDataFrame(Seq(
      (3, Vectors.sparse(6, Seq((1, 1.0), (3, 1.0), (5, 1.0)))),
      (4, Vectors.sparse(6, Seq((2, 1.0), (3, 1.0), (5, 1.0)))),
      (5, Vectors.sparse(6, Seq((1, 1.0), (2, 1.0), (4, 1.0))))
    )).toDF("id", "features")
    
    val key = Vectors.sparse(6, Seq((1, 1.0), (3, 1.0)))
    
    val mh = new MinHashLSH()
      .setNumHashTables(5)
      .setInputCol("features")
      .setOutputCol("hashes")
    
    val model = mh.fit(dfA)
    
    // Feature Transformation
    println("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(dfA).show()
    
    // Compute the locality sensitive hashes for the input rows, then perform approximate
    // similarity join.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxSimilarityJoin(transformedA, transformedB, 0.6)`
    println("Approximately joining dfA and dfB on Jaccard distance smaller than 0.6:")
    model.approxSimilarityJoin(dfA, dfB, 0.6, "JaccardDistance")
      .select(col("datasetA.id").alias("idA"),
        col("datasetB.id").alias("idB"),
        col("JaccardDistance")).show()
    
    // Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    // neighbor search.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxNearestNeighbors(transformedA, key, 2)`
    // It may return less than 2 rows when not enough approximate near-neighbor candidates are
    // found.
    println("Approximately searching dfA for 2 nearest neighbors of the key:")
    model.approxNearestNeighbors(dfA, key, 2).show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/MinHashLSHExample.scala"
in the Spark repo.

Refer to the [MinHashLSH Java
docs](api/java/org/apache/spark/ml/feature/MinHashLSH.html) for more details
on the API.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.feature.MinHashLSH;
    import org.apache.spark.ml.feature.MinHashLSHModel;
    import org.apache.spark.ml.linalg.Vector;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    import static org.apache.spark.sql.functions.col;
    
    List<Row> dataA = Arrays.asList(
      RowFactory.create(0, Vectors.sparse(6, new int[]{0, 1, 2}, new double[]{1.0, 1.0, 1.0})),
      RowFactory.create(1, Vectors.sparse(6, new int[]{2, 3, 4}, new double[]{1.0, 1.0, 1.0})),
      RowFactory.create(2, Vectors.sparse(6, new int[]{0, 2, 4}, new double[]{1.0, 1.0, 1.0}))
    );
    
    List<Row> dataB = Arrays.asList(
      RowFactory.create(0, Vectors.sparse(6, new int[]{1, 3, 5}, new double[]{1.0, 1.0, 1.0})),
      RowFactory.create(1, Vectors.sparse(6, new int[]{2, 3, 5}, new double[]{1.0, 1.0, 1.0})),
      RowFactory.create(2, Vectors.sparse(6, new int[]{1, 2, 4}, new double[]{1.0, 1.0, 1.0}))
    );
    
    StructType schema = new StructType(new StructField[]{
      new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("features", new VectorUDT(), false, Metadata.empty())
    });
    Dataset<Row> dfA = spark.createDataFrame(dataA, schema);
    Dataset<Row> dfB = spark.createDataFrame(dataB, schema);
    
    int[] indices = {1, 3};
    double[] values = {1.0, 1.0};
    Vector key = Vectors.sparse(6, indices, values);
    
    MinHashLSH mh = new MinHashLSH()
      .setNumHashTables(5)
      .setInputCol("features")
      .setOutputCol("hashes");
    
    MinHashLSHModel model = mh.fit(dfA);
    
    // Feature Transformation
    System.out.println("The hashed dataset where hashed values are stored in the column 'hashes':");
    model.transform(dfA).show();
    
    // Compute the locality sensitive hashes for the input rows, then perform approximate
    // similarity join.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxSimilarityJoin(transformedA, transformedB, 0.6)`
    System.out.println("Approximately joining dfA and dfB on Jaccard distance smaller than 0.6:");
    model.approxSimilarityJoin(dfA, dfB, 0.6, "JaccardDistance")
      .select(col("datasetA.id").alias("idA"),
        col("datasetB.id").alias("idB"),
        col("JaccardDistance")).show();
    
    // Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    // neighbor search.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxNearestNeighbors(transformedA, key, 2)`
    // It may return less than 2 rows when not enough approximate near-neighbor candidates are
    // found.
    System.out.println("Approximately searching dfA for 2 nearest neighbors of the key:");
    model.approxNearestNeighbors(dfA, key, 2).show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaMinHashLSHExample.java"
in the Spark repo.

