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

# Classification and regression

`\[ \newcommand{\R}{\mathbb{R}} \newcommand{\E}{\mathbb{E}}
\newcommand{\x}{\mathbf{x}} \newcommand{\y}{\mathbf{y}}
\newcommand{\wv}{\mathbf{w}} \newcommand{\av}{\mathbf{\alpha}}
\newcommand{\bv}{\mathbf{b}} \newcommand{\N}{\mathbb{N}}
\newcommand{\id}{\mathbf{I}} \newcommand{\ind}{\mathbf{1}}
\newcommand{\0}{\mathbf{0}} \newcommand{\unit}{\mathbf{e}}
\newcommand{\one}{\mathbf{1}} \newcommand{\zero}{\mathbf{0}} \]`

This page covers algorithms for Classification and Regression. It also
includes sections discussing specific classes of algorithms, such as linear
methods, trees, and ensembles.

**Table of Contents**

  * Classification
    * Logistic regression
      * Binomial logistic regression
      * Multinomial logistic regression
    * Decision tree classifier
    * Random forest classifier
    * Gradient-boosted tree classifier
    * Multilayer perceptron classifier
    * Linear Support Vector Machine
    * One-vs-Rest classifier (a.k.a. One-vs-All)
    * Naive Bayes
    * Factorization machines classifier
  * Regression
    * Linear regression
    * Generalized linear regression
      * Available families
    * Decision tree regression
    * Random forest regression
    * Gradient-boosted tree regression
    * Survival regression
    * Isotonic regression
    * Factorization machines regressor
  * Linear methods
  * Factorization Machines
  * Decision trees
    * Inputs and Outputs
      * Input Columns
      * Output Columns
  * Tree Ensembles
    * Random Forests
      * Inputs and Outputs
        * Input Columns
        * Output Columns (Predictions)
    * Gradient-Boosted Trees (GBTs)
      * Inputs and Outputs
        * Input Columns
        * Output Columns (Predictions)

# Classification

## Logistic regression

Logistic regression is a popular method to predict a categorical response. It
is a special case of [Generalized Linear
models](https://en.wikipedia.org/wiki/Generalized_linear_model) that predicts
the probability of the outcomes. In `spark.ml` logistic regression can be used
to predict a binary outcome by using binomial logistic regression, or it can
be used to predict a multiclass outcome by using multinomial logistic
regression. Use the `family` parameter to select between these two algorithms,
or leave it unset and Spark will infer the correct variant.

> Multinomial logistic regression can be used for binary classification by
> setting the `family` param to "multinomial". It will produce two sets of
> coefficients and two intercepts.

> When fitting LogisticRegressionModel without intercept on dataset with
> constant nonzero column, Spark MLlib outputs zero coefficients for constant
> nonzero columns. This behavior is the same as R glmnet but different from
> LIBSVM.

### Binomial logistic regression

For more background and more details about the implementation of binomial
logistic regression, refer to the documentation of [logistic regression in
`spark.mllib`](mllib-linear-methods.html#logistic-regression).

**Examples**

The following example shows how to train binomial and multinomial logistic
regression models for binary classification with elastic net regularization.
`elasticNetParam` corresponds to $\alpha$ and `regParam` corresponds to
$\lambda$.

More details on parameters can be found in the [Python API
documentation](api/python/reference/api/pyspark.ml.classification.LogisticRegression.html).

    
    
    from pyspark.ml.classification import LogisticRegression
    
    # Load training data
    training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    
    # Fit the model
    lrModel = lr.fit(training)
    
    # Print the coefficients and intercept for logistic regression
    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))
    
    # We can also use the multinomial family for binary classification
    mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")
    
    # Fit the model
    mlrModel = mlr.fit(training)
    
    # Print the coefficients and intercepts for logistic regression with multinomial family
    print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
    print("Multinomial intercepts: " + str(mlrModel.interceptVector))

Find full example code at
"examples/src/main/python/ml/logistic_regression_with_elastic_net.py" in the
Spark repo.

More details on parameters can be found in the [Scala API
documentation](api/scala/org/apache/spark/ml/classification/LogisticRegression.html).

    
    
    import org.apache.spark.ml.classification.LogisticRegression
    
    // Load training data
    val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
    
    // Fit the model
    val lrModel = lr.fit(training)
    
    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    
    // We can also use the multinomial family for binary classification
    val mlr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFamily("multinomial")
    
    val mlrModel = mlr.fit(training)
    
    // Print the coefficients and intercepts for logistic regression with multinomial family
    println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
    println(s"Multinomial intercepts: ${mlrModel.interceptVector}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/LogisticRegressionWithElasticNetExample.scala"
in the Spark repo.

More details on parameters can be found in the [Java API
documentation](api/java/org/apache/spark/ml/classification/LogisticRegression.html).

    
    
    import org.apache.spark.ml.classification.LogisticRegression;
    import org.apache.spark.ml.classification.LogisticRegressionModel;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load training data
    Dataset<Row> training = spark.read().format("libsvm")
      .load("data/mllib/sample_libsvm_data.txt");
    
    LogisticRegression lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8);
    
    // Fit the model
    LogisticRegressionModel lrModel = lr.fit(training);
    
    // Print the coefficients and intercept for logistic regression
    System.out.println("Coefficients: "
      + lrModel.coefficients() + " Intercept: " + lrModel.intercept());
    
    // We can also use the multinomial family for binary classification
    LogisticRegression mlr = new LogisticRegression()
            .setMaxIter(10)
            .setRegParam(0.3)
            .setElasticNetParam(0.8)
            .setFamily("multinomial");
    
    // Fit the model
    LogisticRegressionModel mlrModel = mlr.fit(training);
    
    // Print the coefficients and intercepts for logistic regression with multinomial family
    System.out.println("Multinomial coefficients: " + lrModel.coefficientMatrix()
      + "\nMultinomial intercepts: " + mlrModel.interceptVector());

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaLogisticRegressionWithElasticNetExample.java"
in the Spark repo.

More details on parameters can be found in the [R API
documentation](api/R/reference/spark.logit.html).

    
    
    # Load training data
    df <- read.df("data/mllib/sample_libsvm_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # Fit an binomial logistic regression model with spark.logit
    model <- spark.logit(training, label ~ features, maxIter = 10, regParam = 0.3, elasticNetParam = 0.8)
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/logit.R" in the Spark repo.

The `spark.ml` implementation of logistic regression also supports extracting
a summary of the model over the training set. Note that the predictions and
metrics which are stored as `DataFrame` in `LogisticRegressionSummary` are
annotated `@transient` and hence only available on the driver.

[`LogisticRegressionTrainingSummary`](api/python/reference/api/pyspark.ml.classification.LogisticRegressionSummary.html)
provides a summary for a
[`LogisticRegressionModel`](api/python/reference/api/pyspark.ml.classification.LogisticRegressionModel.html).
In the case of binary classification, certain additional metrics are
available, e.g. ROC curve. See
[`BinaryLogisticRegressionTrainingSummary`](api/python/reference/api/pyspark.ml.classification.BinaryLogisticRegressionTrainingSummary.html).

Continuing the earlier example:

    
    
    from pyspark.ml.classification import LogisticRegression
    
    # Extract the summary from the returned LogisticRegressionModel instance trained
    # in the earlier example
    trainingSummary = lrModel.summary
    
    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)
    
    # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    trainingSummary.roc.show()
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))
    
    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
        .select('threshold').head()['threshold']
    lr.setThreshold(bestThreshold)

Find full example code at
"examples/src/main/python/ml/logistic_regression_summary_example.py" in the
Spark repo.

[`LogisticRegressionTrainingSummary`](api/scala/org/apache/spark/ml/classification/LogisticRegressionTrainingSummary.html)
provides a summary for a
[`LogisticRegressionModel`](api/scala/org/apache/spark/ml/classification/LogisticRegressionModel.html).
In the case of binary classification, certain additional metrics are
available, e.g. ROC curve. The binary summary can be accessed via the
`binarySummary` method. See
[`BinaryLogisticRegressionTrainingSummary`](api/scala/org/apache/spark/ml/classification/BinaryLogisticRegressionTrainingSummary.html).

Continuing the earlier example:

    
    
    import org.apache.spark.ml.classification.LogisticRegression
    
    // Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
    // example
    val trainingSummary = lrModel.binarySummary
    
    // Obtain the objective per iteration.
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(loss => println(loss))
    
    // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    val roc = trainingSummary.roc
    roc.show()
    println(s"areaUnderROC: ${trainingSummary.areaUnderROC}")
    
    // Set the model threshold to maximize F-Measure
    val fMeasure = trainingSummary.fMeasureByThreshold
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
      .select("threshold").head().getDouble(0)
    lrModel.setThreshold(bestThreshold)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/LogisticRegressionSummaryExample.scala"
in the Spark repo.

[`LogisticRegressionTrainingSummary`](api/java/org/apache/spark/ml/classification/LogisticRegressionTrainingSummary.html)
provides a summary for a
[`LogisticRegressionModel`](api/java/org/apache/spark/ml/classification/LogisticRegressionModel.html).
In the case of binary classification, certain additional metrics are
available, e.g. ROC curve. The binary summary can be accessed via the
`binarySummary` method. See
[`BinaryLogisticRegressionTrainingSummary`](api/java/org/apache/spark/ml/classification/BinaryLogisticRegressionTrainingSummary.html).

Continuing the earlier example:

    
    
    import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
    import org.apache.spark.ml.classification.LogisticRegression;
    import org.apache.spark.ml.classification.LogisticRegressionModel;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    import org.apache.spark.sql.functions;
    
    // Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
    // example
    BinaryLogisticRegressionTrainingSummary trainingSummary = lrModel.binarySummary();
    
    // Obtain the loss per iteration.
    double[] objectiveHistory = trainingSummary.objectiveHistory();
    for (double lossPerIteration : objectiveHistory) {
      System.out.println(lossPerIteration);
    }
    
    // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    Dataset<Row> roc = trainingSummary.roc();
    roc.show();
    roc.select("FPR").show();
    System.out.println(trainingSummary.areaUnderROC());
    
    // Get the threshold corresponding to the maximum F-Measure and rerun LogisticRegression with
    // this selected threshold.
    Dataset<Row> fMeasure = trainingSummary.fMeasureByThreshold();
    double maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0);
    double bestThreshold = fMeasure.where(fMeasure.col("F-Measure").equalTo(maxFMeasure))
      .select("threshold").head().getDouble(0);
    lrModel.setThreshold(bestThreshold);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaLogisticRegressionSummaryExample.java"
in the Spark repo.

### Multinomial logistic regression

Multiclass classification is supported via multinomial logistic (softmax)
regression. In multinomial logistic regression, the algorithm produces $K$
sets of coefficients, or a matrix of dimension $K \times J$ where $K$ is the
number of outcome classes and $J$ is the number of features. If the algorithm
is fit with an intercept term then a length $K$ vector of intercepts is
available.

> Multinomial coefficients are available as `coefficientMatrix` and intercepts
> are available as `interceptVector`.

> `coefficients` and `intercept` methods on a logistic regression model
> trained with multinomial family are not supported. Use `coefficientMatrix`
> and `interceptVector` instead.

The conditional probabilities of the outcome classes $k \in {1, 2, …, K}$ are
modeled using the softmax function.

`\[ P(Y=k|\mathbf{X}, \boldsymbol{\beta}_k, \beta_{0k}) =
\frac{e^{\boldsymbol{\beta}_k \cdot \mathbf{X} +
\beta_{0k}}}{\sum_{k'=0}^{K-1} e^{\boldsymbol{\beta}_{k'} \cdot \mathbf{X} +
\beta_{0k'}}} \]`

We minimize the weighted negative log-likelihood, using a multinomial response
model, with elastic-net penalty to control for overfitting.

`\[ \min_{\beta, \beta_0} -\left[\sum_{i=1}^L w_i \cdot \log P(Y =
y_i|\mathbf{x}_i)\right] + \lambda \left[\frac{1}{2}\left(1 -
\alpha\right)||\boldsymbol{\beta}||_2^2 + \alpha
||\boldsymbol{\beta}||_1\right] \]`

For a detailed derivation please see
[here](https://en.wikipedia.org/wiki/Multinomial_logistic_regression#As_a_log-
linear_model).

**Examples**

The following example shows how to train a multiclass logistic regression
model with elastic net regularization, as well as extract the multiclass
training summary for evaluating the model.

    
    
    from pyspark.ml.classification import LogisticRegression
    
    # Load training data
    training = spark \
        .read \
        .format("libsvm") \
        .load("data/mllib/sample_multiclass_classification_data.txt")
    
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    
    # Fit the model
    lrModel = lr.fit(training)
    
    # Print the coefficients and intercept for multinomial logistic regression
    print("Coefficients: \n" + str(lrModel.coefficientMatrix))
    print("Intercept: " + str(lrModel.interceptVector))
    
    trainingSummary = lrModel.summary
    
    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)
    
    # for multiclass, we can inspect metrics on a per-label basis
    print("False positive rate by label:")
    for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
        print("label %d: %s" % (i, rate))
    
    print("True positive rate by label:")
    for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
        print("label %d: %s" % (i, rate))
    
    print("Precision by label:")
    for i, prec in enumerate(trainingSummary.precisionByLabel):
        print("label %d: %s" % (i, prec))
    
    print("Recall by label:")
    for i, rec in enumerate(trainingSummary.recallByLabel):
        print("label %d: %s" % (i, rec))
    
    print("F-measure by label:")
    for i, f in enumerate(trainingSummary.fMeasureByLabel()):
        print("label %d: %s" % (i, f))
    
    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

Find full example code at
"examples/src/main/python/ml/multiclass_logistic_regression_with_elastic_net.py"
in the Spark repo.

    
    
    import org.apache.spark.ml.classification.LogisticRegression
    
    // Load training data
    val training = spark
      .read
      .format("libsvm")
      .load("data/mllib/sample_multiclass_classification_data.txt")
    
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
    
    // Fit the model
    val lrModel = lr.fit(training)
    
    // Print the coefficients and intercept for multinomial logistic regression
    println(s"Coefficients: \n${lrModel.coefficientMatrix}")
    println(s"Intercepts: \n${lrModel.interceptVector}")
    
    val trainingSummary = lrModel.summary
    
    // Obtain the objective per iteration
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(println)
    
    // for multiclass, we can inspect metrics on a per-label basis
    println("False positive rate by label:")
    trainingSummary.falsePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
      println(s"label $label: $rate")
    }
    
    println("True positive rate by label:")
    trainingSummary.truePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
      println(s"label $label: $rate")
    }
    
    println("Precision by label:")
    trainingSummary.precisionByLabel.zipWithIndex.foreach { case (prec, label) =>
      println(s"label $label: $prec")
    }
    
    println("Recall by label:")
    trainingSummary.recallByLabel.zipWithIndex.foreach { case (rec, label) =>
      println(s"label $label: $rec")
    }
    
    
    println("F-measure by label:")
    trainingSummary.fMeasureByLabel.zipWithIndex.foreach { case (f, label) =>
      println(s"label $label: $f")
    }
    
    val accuracy = trainingSummary.accuracy
    val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    val truePositiveRate = trainingSummary.weightedTruePositiveRate
    val fMeasure = trainingSummary.weightedFMeasure
    val precision = trainingSummary.weightedPrecision
    val recall = trainingSummary.weightedRecall
    println(s"Accuracy: $accuracy\nFPR: $falsePositiveRate\nTPR: $truePositiveRate\n" +
      s"F-measure: $fMeasure\nPrecision: $precision\nRecall: $recall")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/MulticlassLogisticRegressionWithElasticNetExample.scala"
in the Spark repo.

    
    
    import org.apache.spark.ml.classification.LogisticRegression;
    import org.apache.spark.ml.classification.LogisticRegressionModel;
    import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load training data
    Dataset<Row> training = spark.read().format("libsvm")
            .load("data/mllib/sample_multiclass_classification_data.txt");
    
    LogisticRegression lr = new LogisticRegression()
            .setMaxIter(10)
            .setRegParam(0.3)
            .setElasticNetParam(0.8);
    
    // Fit the model
    LogisticRegressionModel lrModel = lr.fit(training);
    
    // Print the coefficients and intercept for multinomial logistic regression
    System.out.println("Coefficients: \n"
            + lrModel.coefficientMatrix() + " \nIntercept: " + lrModel.interceptVector());
    LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();
    
    // Obtain the loss per iteration.
    double[] objectiveHistory = trainingSummary.objectiveHistory();
    for (double lossPerIteration : objectiveHistory) {
        System.out.println(lossPerIteration);
    }
    
    // for multiclass, we can inspect metrics on a per-label basis
    System.out.println("False positive rate by label:");
    int i = 0;
    double[] fprLabel = trainingSummary.falsePositiveRateByLabel();
    for (double fpr : fprLabel) {
        System.out.println("label " + i + ": " + fpr);
        i++;
    }
    
    System.out.println("True positive rate by label:");
    i = 0;
    double[] tprLabel = trainingSummary.truePositiveRateByLabel();
    for (double tpr : tprLabel) {
        System.out.println("label " + i + ": " + tpr);
        i++;
    }
    
    System.out.println("Precision by label:");
    i = 0;
    double[] precLabel = trainingSummary.precisionByLabel();
    for (double prec : precLabel) {
        System.out.println("label " + i + ": " + prec);
        i++;
    }
    
    System.out.println("Recall by label:");
    i = 0;
    double[] recLabel = trainingSummary.recallByLabel();
    for (double rec : recLabel) {
        System.out.println("label " + i + ": " + rec);
        i++;
    }
    
    System.out.println("F-measure by label:");
    i = 0;
    double[] fLabel = trainingSummary.fMeasureByLabel();
    for (double f : fLabel) {
        System.out.println("label " + i + ": " + f);
        i++;
    }
    
    double accuracy = trainingSummary.accuracy();
    double falsePositiveRate = trainingSummary.weightedFalsePositiveRate();
    double truePositiveRate = trainingSummary.weightedTruePositiveRate();
    double fMeasure = trainingSummary.weightedFMeasure();
    double precision = trainingSummary.weightedPrecision();
    double recall = trainingSummary.weightedRecall();
    System.out.println("Accuracy: " + accuracy);
    System.out.println("FPR: " + falsePositiveRate);
    System.out.println("TPR: " + truePositiveRate);
    System.out.println("F-measure: " + fMeasure);
    System.out.println("Precision: " + precision);
    System.out.println("Recall: " + recall);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaMulticlassLogisticRegressionWithElasticNetExample.java"
in the Spark repo.

More details on parameters can be found in the [R API
documentation](api/R/reference/spark.logit.html).

    
    
    # Load training data
    df <- read.df("data/mllib/sample_multiclass_classification_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # Fit a multinomial logistic regression model with spark.logit
    model <- spark.logit(training, label ~ features, maxIter = 10, regParam = 0.3, elasticNetParam = 0.8)
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/logit.R" in the Spark repo.

## Decision tree classifier

Decision trees are a popular family of classification and regression methods.
More information about the `spark.ml` implementation can be found further in
the section on decision trees.

**Examples**

The following examples load a dataset in LibSVM format, split it into training
and test sets, train on the first dataset, and then evaluate on the held-out
test set. We use two feature transformers to prepare the data; these help
index categories for the label and categorical features, adding metadata to
the `DataFrame` which the Decision Tree algorithm can recognize.

More details on parameters can be found in the [Python API
documentation](api/python/reference/api/pyspark.ml.classification.DecisionTreeClassifier.html).

    
    
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import DecisionTreeClassifier
    from pyspark.ml.feature import StringIndexer, VectorIndexer
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
    # Load the data stored in LIBSVM format as a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    
    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    
    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
    
    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)
    
    # Make predictions.
    predictions = model.transform(testData)
    
    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)
    
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    
    treeModel = model.stages[2]
    # summary only
    print(treeModel)

Find full example code at
"examples/src/main/python/ml/decision_tree_classification_example.py" in the
Spark repo.

More details on parameters can be found in the [Scala API
documentation](api/scala/org/apache/spark/ml/classification/DecisionTreeClassifier.html).

    
    
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel
    import org.apache.spark.ml.classification.DecisionTreeClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
    
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
      .fit(data)
    
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    
    // Train a DecisionTree model.
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
    
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray(0))
    
    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
    
    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)
    
    // Make predictions.
    val predictions = model.transform(testData)
    
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)
    
    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")
    
    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/DecisionTreeClassificationExample.scala"
in the Spark repo.

More details on parameters can be found in the [Java API
documentation](api/java/org/apache/spark/ml/classification/DecisionTreeClassifier.html).

    
    
    import org.apache.spark.ml.Pipeline;
    import org.apache.spark.ml.PipelineModel;
    import org.apache.spark.ml.PipelineStage;
    import org.apache.spark.ml.classification.DecisionTreeClassifier;
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
    import org.apache.spark.ml.feature.*;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load the data stored in LIBSVM format as a DataFrame.
    Dataset<Row> data = spark
      .read()
      .format("libsvm")
      .load("data/mllib/sample_libsvm_data.txt");
    
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    StringIndexerModel labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data);
    
    // Automatically identify categorical features, and index them.
    VectorIndexerModel featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
      .fit(data);
    
    // Split the data into training and test sets (30% held out for testing).
    Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
    Dataset<Row> trainingData = splits[0];
    Dataset<Row> testData = splits[1];
    
    // Train a DecisionTree model.
    DecisionTreeClassifier dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures");
    
    // Convert indexed labels back to original labels.
    IndexToString labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray()[0]);
    
    // Chain indexers and tree in a Pipeline.
    Pipeline pipeline = new Pipeline()
      .setStages(new PipelineStage[]{labelIndexer, featureIndexer, dt, labelConverter});
    
    // Train model. This also runs the indexers.
    PipelineModel model = pipeline.fit(trainingData);
    
    // Make predictions.
    Dataset<Row> predictions = model.transform(testData);
    
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5);
    
    // Select (prediction, true label) and compute test error.
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy");
    double accuracy = evaluator.evaluate(predictions);
    System.out.println("Test Error = " + (1.0 - accuracy));
    
    DecisionTreeClassificationModel treeModel =
      (DecisionTreeClassificationModel) (model.stages()[2]);
    System.out.println("Learned classification tree model:\n" + treeModel.toDebugString());

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaDecisionTreeClassificationExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.decisionTree.html) for more
details.

    
    
    # Load training data
    df <- read.df("data/mllib/sample_libsvm_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # Fit a DecisionTree classification model with spark.decisionTree
    model <- spark.decisionTree(training, label ~ features, "classification")
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/decisionTree.R" in the Spark
repo.

## Random forest classifier

Random forests are a popular family of classification and regression methods.
More information about the `spark.ml` implementation can be found further in
the section on random forests.

**Examples**

The following examples load a dataset in LibSVM format, split it into training
and test sets, train on the first dataset, and then evaluate on the held-out
test set. We use two feature transformers to prepare the data; these help
index categories for the label and categorical features, adding metadata to
the `DataFrame` which the tree-based algorithms can recognize.

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html)
for more details.

    
    
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    
    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    
    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
    
    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)
    
    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])
    
    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)
    
    # Make predictions.
    predictions = model.transform(testData)
    
    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)
    
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    
    rfModel = model.stages[2]
    print(rfModel)  # summary only

Find full example code at
"examples/src/main/python/ml/random_forest_classifier_example.py" in the Spark
repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/classification/RandomForestClassifier.html)
for more details.

    
    
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
    
    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)
    
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    
    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
    
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray(0))
    
    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    
    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)
    
    // Make predictions.
    val predictions = model.transform(testData)
    
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)
    
    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")
    
    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/RandomForestClassifierExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/classification/RandomForestClassifier.html)
for more details.

    
    
    import org.apache.spark.ml.Pipeline;
    import org.apache.spark.ml.PipelineModel;
    import org.apache.spark.ml.PipelineStage;
    import org.apache.spark.ml.classification.RandomForestClassificationModel;
    import org.apache.spark.ml.classification.RandomForestClassifier;
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
    import org.apache.spark.ml.feature.*;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load and parse the data file, converting it to a DataFrame.
    Dataset<Row> data = spark.read().format("libsvm").load("data/mllib/sample_libsvm_data.txt");
    
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    StringIndexerModel labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data);
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    VectorIndexerModel featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data);
    
    // Split the data into training and test sets (30% held out for testing)
    Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
    Dataset<Row> trainingData = splits[0];
    Dataset<Row> testData = splits[1];
    
    // Train a RandomForest model.
    RandomForestClassifier rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures");
    
    // Convert indexed labels back to original labels.
    IndexToString labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray()[0]);
    
    // Chain indexers and forest in a Pipeline
    Pipeline pipeline = new Pipeline()
      .setStages(new PipelineStage[] {labelIndexer, featureIndexer, rf, labelConverter});
    
    // Train model. This also runs the indexers.
    PipelineModel model = pipeline.fit(trainingData);
    
    // Make predictions.
    Dataset<Row> predictions = model.transform(testData);
    
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5);
    
    // Select (prediction, true label) and compute test error
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy");
    double accuracy = evaluator.evaluate(predictions);
    System.out.println("Test Error = " + (1.0 - accuracy));
    
    RandomForestClassificationModel rfModel = (RandomForestClassificationModel)(model.stages()[2]);
    System.out.println("Learned classification forest model:\n" + rfModel.toDebugString());

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaRandomForestClassifierExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.randomForest.html) for more
details.

    
    
    # Load training data
    df <- read.df("data/mllib/sample_libsvm_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # Fit a random forest classification model with spark.randomForest
    model <- spark.randomForest(training, label ~ features, "classification", numTrees = 10)
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/randomForest.R" in the Spark
repo.

## Gradient-boosted tree classifier

Gradient-boosted trees (GBTs) are a popular classification and regression
method using ensembles of decision trees. More information about the
`spark.ml` implementation can be found further in the section on GBTs.

**Examples**

The following examples load a dataset in LibSVM format, split it into training
and test sets, train on the first dataset, and then evaluate on the held-out
test set. We use two feature transformers to prepare the data; these help
index categories for the label and categorical features, adding metadata to
the `DataFrame` which the tree-based algorithms can recognize.

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.classification.GBTClassifier.html)
for more details.

    
    
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.feature import StringIndexer, VectorIndexer
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    
    # Train a GBT model.
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)
    
    # Chain indexers and GBT in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])
    
    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)
    
    # Make predictions.
    predictions = model.transform(testData)
    
    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)
    
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    
    gbtModel = model.stages[2]
    print(gbtModel)  # summary only

Find full example code at
"examples/src/main/python/ml/gradient_boosted_tree_classifier_example.py" in
the Spark repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/classification/GBTClassifier.html) for
more details.

    
    
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
    
    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)
    
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    
    // Train a GBT model.
    val gbt = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)
      .setFeatureSubsetStrategy("auto")
    
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray(0))
    
    // Chain indexers and GBT in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
    
    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)
    
    // Make predictions.
    val predictions = model.transform(testData)
    
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)
    
    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1.0 - accuracy}")
    
    val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
    println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/GradientBoostedTreeClassifierExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/classification/GBTClassifier.html) for more
details.

    
    
    import org.apache.spark.ml.Pipeline;
    import org.apache.spark.ml.PipelineModel;
    import org.apache.spark.ml.PipelineStage;
    import org.apache.spark.ml.classification.GBTClassificationModel;
    import org.apache.spark.ml.classification.GBTClassifier;
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
    import org.apache.spark.ml.feature.*;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load and parse the data file, converting it to a DataFrame.
    Dataset<Row> data = spark
      .read()
      .format("libsvm")
      .load("data/mllib/sample_libsvm_data.txt");
    
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    StringIndexerModel labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data);
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    VectorIndexerModel featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data);
    
    // Split the data into training and test sets (30% held out for testing)
    Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
    Dataset<Row> trainingData = splits[0];
    Dataset<Row> testData = splits[1];
    
    // Train a GBT model.
    GBTClassifier gbt = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10);
    
    // Convert indexed labels back to original labels.
    IndexToString labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray()[0]);
    
    // Chain indexers and GBT in a Pipeline.
    Pipeline pipeline = new Pipeline()
      .setStages(new PipelineStage[] {labelIndexer, featureIndexer, gbt, labelConverter});
    
    // Train model. This also runs the indexers.
    PipelineModel model = pipeline.fit(trainingData);
    
    // Make predictions.
    Dataset<Row> predictions = model.transform(testData);
    
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5);
    
    // Select (prediction, true label) and compute test error.
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy");
    double accuracy = evaluator.evaluate(predictions);
    System.out.println("Test Error = " + (1.0 - accuracy));
    
    GBTClassificationModel gbtModel = (GBTClassificationModel)(model.stages()[2]);
    System.out.println("Learned classification GBT model:\n" + gbtModel.toDebugString());

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaGradientBoostedTreeClassifierExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.gbt.html) for more details.

    
    
    # Load training data
    df <- read.df("data/mllib/sample_libsvm_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # Fit a GBT classification model with spark.gbt
    model <- spark.gbt(training, label ~ features, "classification", maxIter = 10)
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/gbt.R" in the Spark repo.

## Multilayer perceptron classifier

Multilayer perceptron classifier (MLPC) is a classifier based on the
[feedforward artificial neural
network](https://en.wikipedia.org/wiki/Feedforward_neural_network). MLPC
consists of multiple layers of nodes. Each layer is fully connected to the
next layer in the network. Nodes in the input layer represent the input data.
All other nodes map inputs to outputs by a linear combination of the inputs
with the node's weights `$\wv$` and bias `$\bv$` and applying an activation
function. This can be written in matrix form for MLPC with `$K+1$` layers as
follows: `\[ \mathrm{y}(\x) =
\mathrm{f_K}(...\mathrm{f_2}(\wv_2^T\mathrm{f_1}(\wv_1^T \x+b_1)+b_2)...+b_K)
\]` Nodes in intermediate layers use sigmoid (logistic) function: `\[
\mathrm{f}(z_i) = \frac{1}{1 + e^{-z_i}} \]` Nodes in the output layer use
softmax function: `\[ \mathrm{f}(z_i) = \frac{e^{z_i}}{\sum_{k=1}^N e^{z_k}}
\]` The number of nodes `$N$` in the output layer corresponds to the number of
classes.

MLPC employs backpropagation for learning the model. We use the logistic loss
function for optimization and L-BFGS as an optimization routine.

**Examples**

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.classification.MultilayerPerceptronClassifier.html)
for more details.

    
    
    from pyspark.ml.classification import MultilayerPerceptronClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
    # Load training data
    data = spark.read.format("libsvm")\
        .load("data/mllib/sample_multiclass_classification_data.txt")
    
    # Split the data into train and test
    splits = data.randomSplit([0.6, 0.4], 1234)
    train = splits[0]
    test = splits[1]
    
    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)
    layers = [4, 5, 4, 3]
    
    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
    
    # train the model
    model = trainer.fit(train)
    
    # compute accuracy on the test set
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

Find full example code at
"examples/src/main/python/ml/multilayer_perceptron_classification.py" in the
Spark repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/classification/MultilayerPerceptronClassifier.html)
for more details.

    
    
    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm")
      .load("data/mllib/sample_multiclass_classification_data.txt")
    
    // Split the data into train and test
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
    
    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](4, 5, 4, 3)
    
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
    
    // train the model
    val model = trainer.fit(train)
    
    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/MultilayerPerceptronClassifierExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/classification/MultilayerPerceptronClassifier.html)
for more details.

    
    
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
    
    // Load training data
    String path = "data/mllib/sample_multiclass_classification_data.txt";
    Dataset<Row> dataFrame = spark.read().format("libsvm").load(path);
    
    // Split the data into train and test
    Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
    Dataset<Row> train = splits[0];
    Dataset<Row> test = splits[1];
    
    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    int[] layers = new int[] {4, 5, 4, 3};
    
    // create the trainer and set its parameters
    MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100);
    
    // train the model
    MultilayerPerceptronClassificationModel model = trainer.fit(train);
    
    // compute accuracy on the test set
    Dataset<Row> result = model.transform(test);
    Dataset<Row> predictionAndLabels = result.select("prediction", "label");
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy");
    
    System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaMultilayerPerceptronClassifierExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.mlp.html) for more details.

    
    
    # Load training data
    df <- read.df("data/mllib/sample_multiclass_classification_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)
    layers = c(4, 5, 4, 3)
    
    # Fit a multi-layer perceptron neural network model with spark.mlp
    model <- spark.mlp(training, label ~ features, maxIter = 100,
                       layers = layers, blockSize = 128, seed = 1234)
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/mlp.R" in the Spark repo.

## Linear Support Vector Machine

A [support vector
machine](https://en.wikipedia.org/wiki/Support_vector_machine) constructs a
hyperplane or set of hyperplanes in a high- or infinite-dimensional space,
which can be used for classification, regression, or other tasks. Intuitively,
a good separation is achieved by the hyperplane that has the largest distance
to the nearest training-data points of any class (so-called functional
margin), since in general the larger the margin the lower the generalization
error of the classifier. LinearSVC in Spark ML supports binary classification
with linear SVM. Internally, it optimizes the [Hinge
Loss](https://en.wikipedia.org/wiki/Hinge_loss) using OWLQN optimizer.

**Examples**

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.classification.LinearSVC.html) for
more details.

    
    
    from pyspark.ml.classification import LinearSVC
    
    # Load training data
    training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    lsvc = LinearSVC(maxIter=10, regParam=0.1)
    
    # Fit the model
    lsvcModel = lsvc.fit(training)
    
    # Print the coefficients and intercept for linear SVC
    print("Coefficients: " + str(lsvcModel.coefficients))
    print("Intercept: " + str(lsvcModel.intercept))

Find full example code at "examples/src/main/python/ml/linearsvc.py" in the
Spark repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/classification/LinearSVC.html) for more
details.

    
    
    import org.apache.spark.ml.classification.LinearSVC
    
    // Load training data
    val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)
    
    // Fit the model
    val lsvcModel = lsvc.fit(training)
    
    // Print the coefficients and intercept for linear svc
    println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/LinearSVCExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/classification/LinearSVC.html) for more
details.

    
    
    import org.apache.spark.ml.classification.LinearSVC;
    import org.apache.spark.ml.classification.LinearSVCModel;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load training data
    Dataset<Row> training = spark.read().format("libsvm")
      .load("data/mllib/sample_libsvm_data.txt");
    
    LinearSVC lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1);
    
    // Fit the model
    LinearSVCModel lsvcModel = lsvc.fit(training);
    
    // Print the coefficients and intercept for LinearSVC
    System.out.println("Coefficients: "
      + lsvcModel.coefficients() + " Intercept: " + lsvcModel.intercept());

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaLinearSVCExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.svmLinear.html) for more
details.

    
    
    # load training data
    t <- as.data.frame(Titanic)
    training <- createDataFrame(t)
    
    # fit Linear SVM model
    model <- spark.svmLinear(training,  Survived ~ ., regParam = 0.01, maxIter = 10)
    
    # Model summary
    summary(model)
    
    # Prediction
    prediction <- predict(model, training)
    showDF(prediction)

Find full example code at "examples/src/main/r/ml/svmLinear.R" in the Spark
repo.

## One-vs-Rest classifier (a.k.a. One-vs-All)

[OneVsRest](http://en.wikipedia.org/wiki/Multiclass_classification#One-
vs.-rest) is an example of a machine learning reduction for performing
multiclass classification given a base classifier that can perform binary
classification efficiently. It is also known as "One-vs-All."

`OneVsRest` is implemented as an `Estimator`. For the base classifier, it
takes instances of `Classifier` and creates a binary classification problem
for each of the k classes. The classifier for class i is trained to predict
whether the label is i or not, distinguishing class i from all other classes.

Predictions are done by evaluating each binary classifier and the index of the
most confident classifier is output as label.

**Examples**

The example below demonstrates how to load the [Iris
dataset](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale),
parse it as a DataFrame and perform multiclass classification using
`OneVsRest`. The test error is calculated to measure the algorithm accuracy.

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.classification.OneVsRest.html) for
more details.

    
    
    from pyspark.ml.classification import LogisticRegression, OneVsRest
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
    # load data file.
    inputData = spark.read.format("libsvm") \
        .load("data/mllib/sample_multiclass_classification_data.txt")
    
    # generate the train/test split.
    (train, test) = inputData.randomSplit([0.8, 0.2])
    
    # instantiate the base classifier.
    lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)
    
    # instantiate the One Vs Rest Classifier.
    ovr = OneVsRest(classifier=lr)
    
    # train the multiclass model.
    ovrModel = ovr.fit(train)
    
    # score the model on test data.
    predictions = ovrModel.transform(test)
    
    # obtain evaluator.
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    
    # compute the classification error on test data.
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

Find full example code at "examples/src/main/python/ml/one_vs_rest_example.py"
in the Spark repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/classification/OneVsRest.html) for more
details.

    
    
    import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    
    // load data file.
    val inputData = spark.read.format("libsvm")
      .load("data/mllib/sample_multiclass_classification_data.txt")
    
    // generate the train/test split.
    val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))
    
    // instantiate the base classifier
    val classifier = new LogisticRegression()
      .setMaxIter(10)
      .setTol(1E-6)
      .setFitIntercept(true)
    
    // instantiate the One Vs Rest Classifier.
    val ovr = new OneVsRest().setClassifier(classifier)
    
    // train the multiclass model.
    val ovrModel = ovr.fit(train)
    
    // score the model on test data.
    val predictions = ovrModel.transform(test)
    
    // obtain evaluator.
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    
    // compute the classification error on test data.
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1 - accuracy}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/OneVsRestExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/classification/OneVsRest.html) for more
details.

    
    
    import org.apache.spark.ml.classification.LogisticRegression;
    import org.apache.spark.ml.classification.OneVsRest;
    import org.apache.spark.ml.classification.OneVsRestModel;
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    
    // load data file.
    Dataset<Row> inputData = spark.read().format("libsvm")
      .load("data/mllib/sample_multiclass_classification_data.txt");
    
    // generate the train/test split.
    Dataset<Row>[] tmp = inputData.randomSplit(new double[]{0.8, 0.2});
    Dataset<Row> train = tmp[0];
    Dataset<Row> test = tmp[1];
    
    // configure the base classifier.
    LogisticRegression classifier = new LogisticRegression()
      .setMaxIter(10)
      .setTol(1E-6)
      .setFitIntercept(true);
    
    // instantiate the One Vs Rest Classifier.
    OneVsRest ovr = new OneVsRest().setClassifier(classifier);
    
    // train the multiclass model.
    OneVsRestModel ovrModel = ovr.fit(train);
    
    // score the model on test data.
    Dataset<Row> predictions = ovrModel.transform(test)
      .select("prediction", "label");
    
    // obtain evaluator.
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setMetricName("accuracy");
    
    // compute the classification error on test data.
    double accuracy = evaluator.evaluate(predictions);
    System.out.println("Test Error = " + (1 - accuracy));

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaOneVsRestExample.java"
in the Spark repo.

## Naive Bayes

[Naive Bayes classifiers](http://en.wikipedia.org/wiki/Naive_Bayes_classifier)
are a family of simple probabilistic, multiclass classifiers based on applying
Bayes' theorem with strong (naive) independence assumptions between every pair
of features.

Naive Bayes can be trained very efficiently. With a single pass over the
training data, it computes the conditional probability distribution of each
feature given each label. For prediction, it applies Bayes' theorem to compute
the conditional probability distribution of each label given an observation.

MLlib supports [Multinomial naive
Bayes](http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes),
[Complement naive
Bayes](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf), [Bernoulli
naive Bayes](http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-
model-1.html) and [Gaussian naive
Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes).

_Input data_ : These Multinomial, Complement and Bernoulli models are
typically used for [document classification](http://nlp.stanford.edu/IR-
book/html/htmledition/naive-bayes-text-classification-1.html). Within that
context, each observation is a document and each feature represents a term. A
feature's value is the frequency of the term (in Multinomial or Complement
Naive Bayes) or a zero or one indicating whether the term was found in the
document (in Bernoulli Naive Bayes). Feature values for Multinomial and
Bernoulli models must be _non-negative_. The model type is selected with an
optional parameter "multinomial", "complement", "bernoulli" or "gaussian",
with "multinomial" as the default. For document classification, the input
feature vectors should usually be sparse vectors. Since the training data is
only used once, it is not necessary to cache it.

[Additive smoothing](http://en.wikipedia.org/wiki/Lidstone_smoothing) can be
used by setting the parameter $\lambda$ (default to $1.0$).

**Examples**

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.classification.NaiveBayes.html) for
more details.

    
    
    from pyspark.ml.classification import NaiveBayes
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
    # Load training data
    data = spark.read.format("libsvm") \
        .load("data/mllib/sample_libsvm_data.txt")
    
    # Split the data into train and test
    splits = data.randomSplit([0.6, 0.4], 1234)
    train = splits[0]
    test = splits[1]
    
    # create the trainer and set its parameters
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    
    # train the model
    model = nb.fit(train)
    
    # select example rows to display.
    predictions = model.transform(test)
    predictions.show()
    
    # compute accuracy on the test set
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test set accuracy = " + str(accuracy))

Find full example code at "examples/src/main/python/ml/naive_bayes_example.py"
in the Spark repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/classification/NaiveBayes.html) for more
details.

    
    
    import org.apache.spark.ml.classification.NaiveBayes
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
    
    // Train a NaiveBayes model.
    val model = new NaiveBayes()
      .fit(trainingData)
    
    // Select example rows to display.
    val predictions = model.transform(testData)
    predictions.show()
    
    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/NaiveBayesExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/classification/NaiveBayes.html) for more
details.

    
    
    import org.apache.spark.ml.classification.NaiveBayes;
    import org.apache.spark.ml.classification.NaiveBayesModel;
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load training data
    Dataset<Row> dataFrame =
      spark.read().format("libsvm").load("data/mllib/sample_libsvm_data.txt");
    // Split the data into train and test
    Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
    Dataset<Row> train = splits[0];
    Dataset<Row> test = splits[1];
    
    // create the trainer and set its parameters
    NaiveBayes nb = new NaiveBayes();
    
    // train the model
    NaiveBayesModel model = nb.fit(train);
    
    // Select example rows to display.
    Dataset<Row> predictions = model.transform(test);
    predictions.show();
    
    // compute accuracy on the test set
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy");
    double accuracy = evaluator.evaluate(predictions);
    System.out.println("Test set accuracy = " + accuracy);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaNaiveBayesExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.naiveBayes.html) for more
details.

    
    
    # Fit a Bernoulli naive Bayes model with spark.naiveBayes
    titanic <- as.data.frame(Titanic)
    titanicDF <- createDataFrame(titanic[titanic$Freq > 0, -5])
    nbDF <- titanicDF
    nbTestDF <- titanicDF
    nbModel <- spark.naiveBayes(nbDF, Survived ~ Class + Sex + Age)
    
    # Model summary
    summary(nbModel)
    
    # Prediction
    nbPredictions <- predict(nbModel, nbTestDF)
    head(nbPredictions)

Find full example code at "examples/src/main/r/ml/naiveBayes.R" in the Spark
repo.

## Factorization machines classifier

For more background and more details about the implementation of factorization
machines, refer to the [Factorization Machines section](ml-classification-
regression.html#factorization-machines).

**Examples**

The following examples load a dataset in LibSVM format, split it into training
and test sets, train on the first dataset, and then evaluate on the held-out
test set. We scale features to be between 0 and 1 to prevent the exploding
gradient problem.

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.classification.FMClassifier.html)
for more details.

    
    
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import FMClassifier
    from pyspark.ml.feature import MinMaxScaler, StringIndexer
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    # Scale features.
    featureScaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures").fit(data)
    
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    
    # Train a FM model.
    fm = FMClassifier(labelCol="indexedLabel", featuresCol="scaledFeatures", stepSize=0.001)
    
    # Create a Pipeline.
    pipeline = Pipeline(stages=[labelIndexer, featureScaler, fm])
    
    # Train model.
    model = pipeline.fit(trainingData)
    
    # Make predictions.
    predictions = model.transform(testData)
    
    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)
    
    # Select (prediction, true label) and compute test accuracy
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test set accuracy = %g" % accuracy)
    
    fmModel = model.stages[2]
    print("Factors: " + str(fmModel.factors))  # type: ignore
    print("Linear: " + str(fmModel.linear))  # type: ignore
    print("Intercept: " + str(fmModel.intercept))  # type: ignore

Find full example code at
"examples/src/main/python/ml/fm_classifier_example.py" in the Spark repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/classification/FMClassifier.html) for more
details.

    
    
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.{FMClassificationModel, FMClassifier}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, MinMaxScaler, StringIndexer}
    
    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Scale features.
    val featureScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .fit(data)
    
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    
    // Train a FM model.
    val fm = new FMClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("scaledFeatures")
      .setStepSize(0.001)
    
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray(0))
    
    // Create a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureScaler, fm, labelConverter))
    
    // Train model.
    val model = pipeline.fit(trainingData)
    
    // Make predictions.
    val predictions = model.transform(testData)
    
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)
    
    // Select (prediction, true label) and compute test accuracy.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")
    
    val fmModel = model.stages(2).asInstanceOf[FMClassificationModel]
    println(s"Factors: ${fmModel.factors} Linear: ${fmModel.linear} " +
      s"Intercept: ${fmModel.intercept}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/FMClassifierExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/classification/FMClassifier.html) for more
details.

    
    
    import org.apache.spark.ml.Pipeline;
    import org.apache.spark.ml.PipelineModel;
    import org.apache.spark.ml.PipelineStage;
    import org.apache.spark.ml.classification.FMClassificationModel;
    import org.apache.spark.ml.classification.FMClassifier;
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
    import org.apache.spark.ml.feature.*;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load and parse the data file, converting it to a DataFrame.
    Dataset<Row> data = spark
        .read()
        .format("libsvm")
        .load("data/mllib/sample_libsvm_data.txt");
    
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    StringIndexerModel labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedLabel")
        .fit(data);
    // Scale features.
    MinMaxScalerModel featureScaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .fit(data);
    
    // Split the data into training and test sets (30% held out for testing)
    Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
    Dataset<Row> trainingData = splits[0];
    Dataset<Row> testData = splits[1];
    
    // Train a FM model.
    FMClassifier fm = new FMClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("scaledFeatures")
        .setStepSize(0.001);
    
    // Convert indexed labels back to original labels.
    IndexToString labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictedLabel")
        .setLabels(labelIndexer.labelsArray()[0]);
    
    // Create a Pipeline.
    Pipeline pipeline = new Pipeline()
        .setStages(new PipelineStage[] {labelIndexer, featureScaler, fm, labelConverter});
    
    // Train model.
    PipelineModel model = pipeline.fit(trainingData);
    
    // Make predictions.
    Dataset<Row> predictions = model.transform(testData);
    
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5);
    
    // Select (prediction, true label) and compute test accuracy.
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("indexedLabel")
        .setPredictionCol("prediction")
        .setMetricName("accuracy");
    double accuracy = evaluator.evaluate(predictions);
    System.out.println("Test Accuracy = " + accuracy);
    
    FMClassificationModel fmModel = (FMClassificationModel)(model.stages()[2]);
    System.out.println("Factors: " + fmModel.factors());
    System.out.println("Linear: " + fmModel.linear());
    System.out.println("Intercept: " + fmModel.intercept());

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaFMClassifierExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.fmClassifier.html) for more
details.

Note: At the moment SparkR doesn't support feature scaling.

    
    
    # Load training data
    df <- read.df("data/mllib/sample_libsvm_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # Fit a FM classification model
    model <- spark.fmClassifier(training, label ~ features)
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/fmClassifier.R" in the Spark
repo.

# Regression

## Linear regression

The interface for working with linear regression models and model summaries is
similar to the logistic regression case.

> When fitting LinearRegressionModel without intercept on dataset with
> constant nonzero column by "l-bfgs" solver, Spark MLlib outputs zero
> coefficients for constant nonzero columns. This behavior is the same as R
> glmnet but different from LIBSVM.

**Examples**

The following example demonstrates training an elastic net regularized linear
regression model and extracting model summary statistics.

More details on parameters can be found in the [Python API
documentation](api/python/reference/api/pyspark.ml.regression.LinearRegression.html#pyspark.ml.regression.LinearRegression).

    
    
    from pyspark.ml.regression import LinearRegression
    
    # Load training data
    training = spark.read.format("libsvm")\
        .load("data/mllib/sample_linear_regression_data.txt")
    
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    
    # Fit the model
    lrModel = lr.fit(training)
    
    # Print the coefficients and intercept for linear regression
    print("Coefficients: %s" % str(lrModel.coefficients))
    print("Intercept: %s" % str(lrModel.intercept))
    
    # Summarize the model over the training set and print out some metrics
    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

Find full example code at
"examples/src/main/python/ml/linear_regression_with_elastic_net.py" in the
Spark repo.

More details on parameters can be found in the [Scala API
documentation](api/scala/org/apache/spark/ml/regression/LinearRegression.html).

    
    
    import org.apache.spark.ml.regression.LinearRegression
    
    // Load training data
    val training = spark.read.format("libsvm")
      .load("data/mllib/sample_linear_regression_data.txt")
    
    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
    
    // Fit the model
    val lrModel = lr.fit(training)
    
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    
    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/LinearRegressionWithElasticNetExample.scala"
in the Spark repo.

More details on parameters can be found in the [Java API
documentation](api/java/org/apache/spark/ml/regression/LinearRegression.html).

    
    
    import org.apache.spark.ml.regression.LinearRegression;
    import org.apache.spark.ml.regression.LinearRegressionModel;
    import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load training data.
    Dataset<Row> training = spark.read().format("libsvm")
      .load("data/mllib/sample_linear_regression_data.txt");
    
    LinearRegression lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8);
    
    // Fit the model.
    LinearRegressionModel lrModel = lr.fit(training);
    
    // Print the coefficients and intercept for linear regression.
    System.out.println("Coefficients: "
      + lrModel.coefficients() + " Intercept: " + lrModel.intercept());
    
    // Summarize the model over the training set and print out some metrics.
    LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
    System.out.println("numIterations: " + trainingSummary.totalIterations());
    System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
    trainingSummary.residuals().show();
    System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
    System.out.println("r2: " + trainingSummary.r2());

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaLinearRegressionWithElasticNetExample.java"
in the Spark repo.

More details on parameters can be found in the [R API
documentation](api/R/reference/spark.lm.html).

    
    
    # Load training data
    df <- read.df("data/mllib/sample_linear_regression_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # Fit a linear regression model
    model <- spark.lm(training, label ~ features, regParam = 0.3, elasticNetParam = 0.8)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)
    
    # Summarize
    summary(model)

Find full example code at "examples/src/main/r/ml/lm_with_elastic_net.R" in
the Spark repo.

## Generalized linear regression

Contrasted with linear regression where the output is assumed to follow a
Gaussian distribution, [generalized linear
models](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLMs) are
specifications of linear models where the response variable $Y_i$ follows some
distribution from the [exponential family of
distributions](https://en.wikipedia.org/wiki/Exponential_family). Spark's
`GeneralizedLinearRegression` interface allows for flexible specification of
GLMs which can be used for various types of prediction problems including
linear regression, Poisson regression, logistic regression, and others.
Currently in `spark.ml`, only a subset of the exponential family distributions
are supported and they are listed below.

**NOTE** : Spark currently only supports up to 4096 features through its
`GeneralizedLinearRegression` interface, and will throw an exception if this
constraint is exceeded. See the [advanced section](ml-advanced) for more
details. Still, for linear and logistic regression, models with an increased
number of features can be trained using the `LinearRegression` and
`LogisticRegression` estimators.

GLMs require exponential family distributions that can be written in their
"canonical" or "natural" form, aka [natural exponential family
distributions](https://en.wikipedia.org/wiki/Natural_exponential_family). The
form of a natural exponential family distribution is given as:

\\[f_Y(y|\theta, \tau) = h(y, \tau)\exp{\left( \frac{\theta \cdot y -
A(\theta)}{d(\tau)} \right)}\\]

where $\theta$ is the parameter of interest and $\tau$ is a dispersion
parameter. In a GLM the response variable $Y_i$ is assumed to be drawn from a
natural exponential family distribution:

\\[Y_i \sim f\left(\cdot|\theta_i, \tau \right)\\]

where the parameter of interest $\theta_i$ is related to the expected value of
the response variable $\mu_i$ by

\\[\mu_i = A'(\theta_i)\\]

Here, $A'(\theta_i)$ is defined by the form of the distribution selected. GLMs
also allow specification of a link function, which defines the relationship
between the expected value of the response variable $\mu_i$ and the so called
_linear predictor_ $\eta_i$:

\\[g(\mu_i) = \eta_i = \vec{x_i}^T \cdot \vec{\beta}\\]

Often, the link function is chosen such that $A' = g^{-1}$, which yields a
simplified relationship between the parameter of interest $\theta$ and the
linear predictor $\eta$. In this case, the link function $g(\mu)$ is said to
be the "canonical" link function.

\\[\theta_i = A'^{-1}(\mu_i) = g(g^{-1}(\eta_i)) = \eta_i\\]

A GLM finds the regression coefficients $\vec{\beta}$ which maximize the
likelihood function.

\\[\max_{\vec{\beta}} \mathcal{L}(\vec{\theta}|\vec{y},X) = \prod_{i=1}^{N}
h(y_i, \tau) \exp{\left(\frac{y_i\theta_i - A(\theta_i)}{d(\tau)}\right)}\\]

where the parameter of interest $\theta_i$ is related to the regression
coefficients $\vec{\beta}$ by

\\[\theta_i = A'^{-1}(g^{-1}(\vec{x_i} \cdot \vec{\beta}))\\]

Spark's generalized linear regression interface also provides summary
statistics for diagnosing the fit of GLM models, including residuals,
p-values, deviances, the Akaike information criterion, and others.

[See here](http://data.princeton.edu/wws509/notes/) for a more comprehensive
review of GLMs and their applications.

### Available families

Family | Response Type | Supported Links  
---|---|---  
Gaussian | Continuous | Identity*, Log, Inverse  
Binomial | Binary | Logit*, Probit, CLogLog  
Poisson | Count | Log*, Identity, Sqrt  
Gamma | Continuous | Inverse*, Identity, Log  
Tweedie | Zero-inflated continuous | Power link function  
* Canonical Link  
  
**Examples**

The following example demonstrates training a GLM with a Gaussian response and
identity link function and extracting model summary statistics.

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.regression.GeneralizedLinearRegression.html#pyspark.ml.regression.GeneralizedLinearRegression)
for more details.

    
    
    from pyspark.ml.regression import GeneralizedLinearRegression
    
    # Load training data
    dataset = spark.read.format("libsvm")\
        .load("data/mllib/sample_linear_regression_data.txt")
    
    glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)
    
    # Fit the model
    model = glr.fit(dataset)
    
    # Print the coefficients and intercept for generalized linear regression model
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))
    
    # Summarize the model over the training set and print out some metrics
    summary = model.summary
    print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
    print("T Values: " + str(summary.tValues))
    print("P Values: " + str(summary.pValues))
    print("Dispersion: " + str(summary.dispersion))
    print("Null Deviance: " + str(summary.nullDeviance))
    print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
    print("Deviance: " + str(summary.deviance))
    print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
    print("AIC: " + str(summary.aic))
    print("Deviance Residuals: ")
    summary.residuals().show()

Find full example code at
"examples/src/main/python/ml/generalized_linear_regression_example.py" in the
Spark repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/regression/GeneralizedLinearRegression.html)
for more details.

    
    
    import org.apache.spark.ml.regression.GeneralizedLinearRegression
    
    // Load training data
    val dataset = spark.read.format("libsvm")
      .load("data/mllib/sample_linear_regression_data.txt")
    
    val glr = new GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setMaxIter(10)
      .setRegParam(0.3)
    
    // Fit the model
    val model = glr.fit(dataset)
    
    // Print the coefficients and intercept for generalized linear regression model
    println(s"Coefficients: ${model.coefficients}")
    println(s"Intercept: ${model.intercept}")
    
    // Summarize the model over the training set and print out some metrics
    val summary = model.summary
    println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
    println(s"T Values: ${summary.tValues.mkString(",")}")
    println(s"P Values: ${summary.pValues.mkString(",")}")
    println(s"Dispersion: ${summary.dispersion}")
    println(s"Null Deviance: ${summary.nullDeviance}")
    println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
    println(s"Deviance: ${summary.deviance}")
    println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
    println(s"AIC: ${summary.aic}")
    println("Deviance Residuals: ")
    summary.residuals().show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/GeneralizedLinearRegressionExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/regression/GeneralizedLinearRegression.html)
for more details.

    
    
    import java.util.Arrays;
    
    import org.apache.spark.ml.regression.GeneralizedLinearRegression;
    import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
    import org.apache.spark.ml.regression.GeneralizedLinearRegressionTrainingSummary;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    
    // Load training data
    Dataset<Row> dataset = spark.read().format("libsvm")
      .load("data/mllib/sample_linear_regression_data.txt");
    
    GeneralizedLinearRegression glr = new GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setMaxIter(10)
      .setRegParam(0.3);
    
    // Fit the model
    GeneralizedLinearRegressionModel model = glr.fit(dataset);
    
    // Print the coefficients and intercept for generalized linear regression model
    System.out.println("Coefficients: " + model.coefficients());
    System.out.println("Intercept: " + model.intercept());
    
    // Summarize the model over the training set and print out some metrics
    GeneralizedLinearRegressionTrainingSummary summary = model.summary();
    System.out.println("Coefficient Standard Errors: "
      + Arrays.toString(summary.coefficientStandardErrors()));
    System.out.println("T Values: " + Arrays.toString(summary.tValues()));
    System.out.println("P Values: " + Arrays.toString(summary.pValues()));
    System.out.println("Dispersion: " + summary.dispersion());
    System.out.println("Null Deviance: " + summary.nullDeviance());
    System.out.println("Residual Degree Of Freedom Null: " + summary.residualDegreeOfFreedomNull());
    System.out.println("Deviance: " + summary.deviance());
    System.out.println("Residual Degree Of Freedom: " + summary.residualDegreeOfFreedom());
    System.out.println("AIC: " + summary.aic());
    System.out.println("Deviance Residuals: ");
    summary.residuals().show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaGeneralizedLinearRegressionExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.glm.html) for more details.

    
    
    training <- read.df("data/mllib/sample_multiclass_classification_data.txt", source = "libsvm")
    # Fit a generalized linear model of family "gaussian" with spark.glm
    df_list <- randomSplit(training, c(7, 3), 2)
    gaussianDF <- df_list[[1]]
    gaussianTestDF <- df_list[[2]]
    gaussianGLM <- spark.glm(gaussianDF, label ~ features, family = "gaussian")
    
    # Model summary
    summary(gaussianGLM)
    
    # Prediction
    gaussianPredictions <- predict(gaussianGLM, gaussianTestDF)
    head(gaussianPredictions)
    
    # Fit a generalized linear model with glm (R-compliant)
    gaussianGLM2 <- glm(label ~ features, gaussianDF, family = "gaussian")
    summary(gaussianGLM2)
    
    # Fit a generalized linear model of family "binomial" with spark.glm
    training2 <- read.df("data/mllib/sample_multiclass_classification_data.txt", source = "libsvm")
    training2 <- transform(training2, label = cast(training2$label > 1, "integer"))
    df_list2 <- randomSplit(training2, c(7, 3), 2)
    binomialDF <- df_list2[[1]]
    binomialTestDF <- df_list2[[2]]
    binomialGLM <- spark.glm(binomialDF, label ~ features, family = "binomial")
    
    # Model summary
    summary(binomialGLM)
    
    # Prediction
    binomialPredictions <- predict(binomialGLM, binomialTestDF)
    head(binomialPredictions)
    
    # Fit a generalized linear model of family "tweedie" with spark.glm
    training3 <- read.df("data/mllib/sample_multiclass_classification_data.txt", source = "libsvm")
    tweedieDF <- transform(training3, label = training3$label * exp(randn(10)))
    tweedieGLM <- spark.glm(tweedieDF, label ~ features, family = "tweedie",
                            var.power = 1.2, link.power = 0)
    
    # Model summary
    summary(tweedieGLM)

Find full example code at "examples/src/main/r/ml/glm.R" in the Spark repo.

## Decision tree regression

Decision trees are a popular family of classification and regression methods.
More information about the `spark.ml` implementation can be found further in
the section on decision trees.

**Examples**

The following examples load a dataset in LibSVM format, split it into training
and test sets, train on the first dataset, and then evaluate on the held-out
test set. We use a feature transformer to index categorical features, adding
metadata to the `DataFrame` which the Decision Tree algorithm can recognize.

More details on parameters can be found in the [Python API
documentation](api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html#pyspark.ml.regression.DecisionTreeRegressor).

    
    
    from pyspark.ml import Pipeline
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.feature import VectorIndexer
    from pyspark.ml.evaluation import RegressionEvaluator
    
    # Load the data stored in LIBSVM format as a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    
    # Train a DecisionTree model.
    dt = DecisionTreeRegressor(featuresCol="indexedFeatures")
    
    # Chain indexer and tree in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, dt])
    
    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)
    
    # Make predictions.
    predictions = model.transform(testData)
    
    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)
    
    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    
    treeModel = model.stages[1]
    # summary only
    print(treeModel)

Find full example code at
"examples/src/main/python/ml/decision_tree_regression_example.py" in the Spark
repo.

More details on parameters can be found in the [Scala API
documentation](api/scala/org/apache/spark/ml/regression/DecisionTreeRegressor.html).

    
    
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    import org.apache.spark.ml.feature.VectorIndexer
    import org.apache.spark.ml.regression.DecisionTreeRegressionModel
    import org.apache.spark.ml.regression.DecisionTreeRegressor
    
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    // Automatically identify categorical features, and index them.
    // Here, we treat features with > 4 distinct values as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)
    
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    
    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
    
    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dt))
    
    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)
    
    // Make predictions.
    val predictions = model.transform(testData)
    
    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)
    
    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    
    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println(s"Learned regression tree model:\n ${treeModel.toDebugString}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/DecisionTreeRegressionExample.scala"
in the Spark repo.

More details on parameters can be found in the [Java API
documentation](api/java/org/apache/spark/ml/regression/DecisionTreeRegressor.html).

    
    
    import org.apache.spark.ml.Pipeline;
    import org.apache.spark.ml.PipelineModel;
    import org.apache.spark.ml.PipelineStage;
    import org.apache.spark.ml.evaluation.RegressionEvaluator;
    import org.apache.spark.ml.feature.VectorIndexer;
    import org.apache.spark.ml.feature.VectorIndexerModel;
    import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
    import org.apache.spark.ml.regression.DecisionTreeRegressor;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load the data stored in LIBSVM format as a DataFrame.
    Dataset<Row> data = spark.read().format("libsvm")
      .load("data/mllib/sample_libsvm_data.txt");
    
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    VectorIndexerModel featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data);
    
    // Split the data into training and test sets (30% held out for testing).
    Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
    Dataset<Row> trainingData = splits[0];
    Dataset<Row> testData = splits[1];
    
    // Train a DecisionTree model.
    DecisionTreeRegressor dt = new DecisionTreeRegressor()
      .setFeaturesCol("indexedFeatures");
    
    // Chain indexer and tree in a Pipeline.
    Pipeline pipeline = new Pipeline()
      .setStages(new PipelineStage[]{featureIndexer, dt});
    
    // Train model. This also runs the indexer.
    PipelineModel model = pipeline.fit(trainingData);
    
    // Make predictions.
    Dataset<Row> predictions = model.transform(testData);
    
    // Select example rows to display.
    predictions.select("label", "features").show(5);
    
    // Select (prediction, true label) and compute test error.
    RegressionEvaluator evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse");
    double rmse = evaluator.evaluate(predictions);
    System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
    
    DecisionTreeRegressionModel treeModel =
      (DecisionTreeRegressionModel) (model.stages()[1]);
    System.out.println("Learned regression tree model:\n" + treeModel.toDebugString());

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaDecisionTreeRegressionExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.decisionTree.html) for more
details.

    
    
    # Load training data
    df <- read.df("data/mllib/sample_linear_regression_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # Fit a DecisionTree regression model with spark.decisionTree
    model <- spark.decisionTree(training, label ~ features, "regression")
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/decisionTree.R" in the Spark
repo.

## Random forest regression

Random forests are a popular family of classification and regression methods.
More information about the `spark.ml` implementation can be found further in
the section on random forests.

**Examples**

The following examples load a dataset in LibSVM format, split it into training
and test sets, train on the first dataset, and then evaluate on the held-out
test set. We use a feature transformer to index categorical features, adding
metadata to the `DataFrame` which the tree-based algorithms can recognize.

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.regression.RandomForestRegressor.html#pyspark.ml.regression.RandomForestRegressor)
for more details.

    
    
    from pyspark.ml import Pipeline
    from pyspark.ml.regression import RandomForestRegressor
    from pyspark.ml.feature import VectorIndexer
    from pyspark.ml.evaluation import RegressionEvaluator
    
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    
    # Train a RandomForest model.
    rf = RandomForestRegressor(featuresCol="indexedFeatures")
    
    # Chain indexer and forest in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, rf])
    
    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)
    
    # Make predictions.
    predictions = model.transform(testData)
    
    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)
    
    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    
    rfModel = model.stages[1]
    print(rfModel)  # summary only

Find full example code at
"examples/src/main/python/ml/random_forest_regressor_example.py" in the Spark
repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/regression/RandomForestRegressor.html) for
more details.

    
    
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    import org.apache.spark.ml.feature.VectorIndexer
    import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
    
    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)
    
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    
    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
    
    // Chain indexer and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, rf))
    
    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)
    
    // Make predictions.
    val predictions = model.transform(testData)
    
    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)
    
    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    
    val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
    println(s"Learned regression forest model:\n ${rfModel.toDebugString}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/RandomForestRegressorExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/regression/RandomForestRegressor.html) for
more details.

    
    
    import org.apache.spark.ml.Pipeline;
    import org.apache.spark.ml.PipelineModel;
    import org.apache.spark.ml.PipelineStage;
    import org.apache.spark.ml.evaluation.RegressionEvaluator;
    import org.apache.spark.ml.feature.VectorIndexer;
    import org.apache.spark.ml.feature.VectorIndexerModel;
    import org.apache.spark.ml.regression.RandomForestRegressionModel;
    import org.apache.spark.ml.regression.RandomForestRegressor;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load and parse the data file, converting it to a DataFrame.
    Dataset<Row> data = spark.read().format("libsvm").load("data/mllib/sample_libsvm_data.txt");
    
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    VectorIndexerModel featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data);
    
    // Split the data into training and test sets (30% held out for testing)
    Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
    Dataset<Row> trainingData = splits[0];
    Dataset<Row> testData = splits[1];
    
    // Train a RandomForest model.
    RandomForestRegressor rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures");
    
    // Chain indexer and forest in a Pipeline
    Pipeline pipeline = new Pipeline()
      .setStages(new PipelineStage[] {featureIndexer, rf});
    
    // Train model. This also runs the indexer.
    PipelineModel model = pipeline.fit(trainingData);
    
    // Make predictions.
    Dataset<Row> predictions = model.transform(testData);
    
    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5);
    
    // Select (prediction, true label) and compute test error
    RegressionEvaluator evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse");
    double rmse = evaluator.evaluate(predictions);
    System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
    
    RandomForestRegressionModel rfModel = (RandomForestRegressionModel)(model.stages()[1]);
    System.out.println("Learned regression forest model:\n" + rfModel.toDebugString());

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaRandomForestRegressorExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.randomForest.html) for more
details.

    
    
    # Load training data
    df <- read.df("data/mllib/sample_linear_regression_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # Fit a random forest regression model with spark.randomForest
    model <- spark.randomForest(training, label ~ features, "regression", numTrees = 10)
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/randomForest.R" in the Spark
repo.

## Gradient-boosted tree regression

Gradient-boosted trees (GBTs) are a popular regression method using ensembles
of decision trees. More information about the `spark.ml` implementation can be
found further in the section on GBTs.

**Examples**

Note: For this example dataset, `GBTRegressor` actually only needs 1
iteration, but that will not be true in general.

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.regression.GBTRegressor.html#pyspark.ml.regression.GBTRegressor)
for more details.

    
    
    from pyspark.ml import Pipeline
    from pyspark.ml.regression import GBTRegressor
    from pyspark.ml.feature import VectorIndexer
    from pyspark.ml.evaluation import RegressionEvaluator
    
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    
    # Train a GBT model.
    gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)
    
    # Chain indexer and GBT in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, gbt])
    
    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)
    
    # Make predictions.
    predictions = model.transform(testData)
    
    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)
    
    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    
    gbtModel = model.stages[1]
    print(gbtModel)  # summary only

Find full example code at
"examples/src/main/python/ml/gradient_boosted_tree_regressor_example.py" in
the Spark repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/regression/GBTRegressor.html) for more
details.

    
    
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    import org.apache.spark.ml.feature.VectorIndexer
    import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
    
    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)
    
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    
    // Train a GBT model.
    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)
    
    // Chain indexer and GBT in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, gbt))
    
    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)
    
    // Make predictions.
    val predictions = model.transform(testData)
    
    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)
    
    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    
    val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
    println(s"Learned regression GBT model:\n ${gbtModel.toDebugString}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/GradientBoostedTreeRegressorExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/regression/GBTRegressor.html) for more
details.

    
    
    import org.apache.spark.ml.Pipeline;
    import org.apache.spark.ml.PipelineModel;
    import org.apache.spark.ml.PipelineStage;
    import org.apache.spark.ml.evaluation.RegressionEvaluator;
    import org.apache.spark.ml.feature.VectorIndexer;
    import org.apache.spark.ml.feature.VectorIndexerModel;
    import org.apache.spark.ml.regression.GBTRegressionModel;
    import org.apache.spark.ml.regression.GBTRegressor;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load and parse the data file, converting it to a DataFrame.
    Dataset<Row> data = spark.read().format("libsvm").load("data/mllib/sample_libsvm_data.txt");
    
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    VectorIndexerModel featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data);
    
    // Split the data into training and test sets (30% held out for testing).
    Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
    Dataset<Row> trainingData = splits[0];
    Dataset<Row> testData = splits[1];
    
    // Train a GBT model.
    GBTRegressor gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10);
    
    // Chain indexer and GBT in a Pipeline.
    Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {featureIndexer, gbt});
    
    // Train model. This also runs the indexer.
    PipelineModel model = pipeline.fit(trainingData);
    
    // Make predictions.
    Dataset<Row> predictions = model.transform(testData);
    
    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5);
    
    // Select (prediction, true label) and compute test error.
    RegressionEvaluator evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse");
    double rmse = evaluator.evaluate(predictions);
    System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
    
    GBTRegressionModel gbtModel = (GBTRegressionModel)(model.stages()[1]);
    System.out.println("Learned regression GBT model:\n" + gbtModel.toDebugString());

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaGradientBoostedTreeRegressorExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.gbt.html) for more details.

    
    
    # Load training data
    df <- read.df("data/mllib/sample_linear_regression_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # Fit a GBT regression model with spark.gbt
    model <- spark.gbt(training, label ~ features, "regression", maxIter = 10)
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/gbt.R" in the Spark repo.

## Survival regression

In `spark.ml`, we implement the [Accelerated failure time
(AFT)](https://en.wikipedia.org/wiki/Accelerated_failure_time_model) model
which is a parametric survival regression model for censored data. It
describes a model for the log of survival time, so it's often called a log-
linear model for survival analysis. Different from a [Proportional
hazards](https://en.wikipedia.org/wiki/Proportional_hazards_model) model
designed for the same purpose, the AFT model is easier to parallelize because
each instance contributes to the objective function independently.

Given the values of the covariates $x^{'}$, for random lifetime $t_{i}$ of
subjects i = 1, …, n, with possible right-censoring, the likelihood function
under the AFT model is given as: `\[
L(\beta,\sigma)=\prod_{i=1}^n[\frac{1}{\sigma}f_{0}(\frac{\log{t_{i}}-x^{'}\beta}{\sigma})]^{\delta_{i}}S_{0}(\frac{\log{t_{i}}-x^{'}\beta}{\sigma})^{1-\delta_{i}}
\]` Where $\delta_{i}$ is the indicator of the event has occurred i.e.
uncensored or not. Using $\epsilon_{i}=\frac{\log{t_{i}}-x^{'}\beta}{\sigma}$,
the log-likelihood function assumes the form: `\[
\iota(\beta,\sigma)=\sum_{i=1}^{n}[-\delta_{i}\log\sigma+\delta_{i}\log{f_{0}}(\epsilon_{i})+(1-\delta_{i})\log{S_{0}(\epsilon_{i})}]
\]` Where $S_{0}(\epsilon_{i})$ is the baseline survivor function, and
$f_{0}(\epsilon_{i})$ is the corresponding density function.

The most commonly used AFT model is based on the Weibull distribution of the
survival time. The Weibull distribution for lifetime corresponds to the
extreme value distribution for the log of the lifetime, and the
$S_{0}(\epsilon)$ function is: `\[ S_{0}(\epsilon_{i})=\exp(-e^{\epsilon_{i}})
\]` the $f_{0}(\epsilon_{i})$ function is: `\[
f_{0}(\epsilon_{i})=e^{\epsilon_{i}}\exp(-e^{\epsilon_{i}}) \]` The log-
likelihood function for AFT model with a Weibull distribution of lifetime is:
`\[ \iota(\beta,\sigma)=
-\sum_{i=1}^n[\delta_{i}\log\sigma-\delta_{i}\epsilon_{i}+e^{\epsilon_{i}}]
\]` Due to minimizing the negative log-likelihood equivalent to maximum a
posteriori probability, the loss function we use to optimize is
$-\iota(\beta,\sigma)$. The gradient functions for $\beta$ and $\log\sigma$
respectively are: `\[ \frac{\partial (-\iota)}{\partial
\beta}=\sum_{1=1}^{n}[\delta_{i}-e^{\epsilon_{i}}]\frac{x_{i}}{\sigma} \]` `\[
\frac{\partial (-\iota)}{\partial
(\log\sigma)}=\sum_{i=1}^{n}[\delta_{i}+(\delta_{i}-e^{\epsilon_{i}})\epsilon_{i}]
\]`

The AFT model can be formulated as a convex optimization problem, i.e. the
task of finding a minimizer of a convex function $-\iota(\beta,\sigma)$ that
depends on the coefficients vector $\beta$ and the log of scale parameter
$\log\sigma$. The optimization algorithm underlying the implementation is
L-BFGS. The implementation matches the result from R's survival function
[survreg](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html)

> When fitting AFTSurvivalRegressionModel without intercept on dataset with
> constant nonzero column, Spark MLlib outputs zero coefficients for constant
> nonzero columns. This behavior is different from R survival::survreg.

**Examples**

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.regression.AFTSurvivalRegression.html#pyspark.ml.regression.AFTSurvivalRegression)
for more details.

    
    
    from pyspark.ml.regression import AFTSurvivalRegression
    from pyspark.ml.linalg import Vectors
    
    training = spark.createDataFrame([
        (1.218, 1.0, Vectors.dense(1.560, -0.605)),
        (2.949, 0.0, Vectors.dense(0.346, 2.158)),
        (3.627, 0.0, Vectors.dense(1.380, 0.231)),
        (0.273, 1.0, Vectors.dense(0.520, 1.151)),
        (4.199, 0.0, Vectors.dense(0.795, -0.226))], ["label", "censor", "features"])
    quantileProbabilities = [0.3, 0.6]
    aft = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities,
                                quantilesCol="quantiles")
    
    model = aft.fit(training)
    
    # Print the coefficients, intercept and scale parameter for AFT survival regression
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))
    print("Scale: " + str(model.scale))
    model.transform(training).show(truncate=False)

Find full example code at
"examples/src/main/python/ml/aft_survival_regression.py" in the Spark repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/regression/AFTSurvivalRegression.html) for
more details.

    
    
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.ml.regression.AFTSurvivalRegression
    
    val training = spark.createDataFrame(Seq(
      (1.218, 1.0, Vectors.dense(1.560, -0.605)),
      (2.949, 0.0, Vectors.dense(0.346, 2.158)),
      (3.627, 0.0, Vectors.dense(1.380, 0.231)),
      (0.273, 1.0, Vectors.dense(0.520, 1.151)),
      (4.199, 0.0, Vectors.dense(0.795, -0.226))
    )).toDF("label", "censor", "features")
    val quantileProbabilities = Array(0.3, 0.6)
    val aft = new AFTSurvivalRegression()
      .setQuantileProbabilities(quantileProbabilities)
      .setQuantilesCol("quantiles")
    
    val model = aft.fit(training)
    
    // Print the coefficients, intercept and scale parameter for AFT survival regression
    println(s"Coefficients: ${model.coefficients}")
    println(s"Intercept: ${model.intercept}")
    println(s"Scale: ${model.scale}")
    model.transform(training).show(false)

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/AFTSurvivalRegressionExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/regression/AFTSurvivalRegression.html) for
more details.

    
    
    import java.util.Arrays;
    import java.util.List;
    
    import org.apache.spark.ml.regression.AFTSurvivalRegression;
    import org.apache.spark.ml.regression.AFTSurvivalRegressionModel;
    import org.apache.spark.ml.linalg.VectorUDT;
    import org.apache.spark.ml.linalg.Vectors;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.RowFactory;
    import org.apache.spark.sql.SparkSession;
    import org.apache.spark.sql.types.DataTypes;
    import org.apache.spark.sql.types.Metadata;
    import org.apache.spark.sql.types.StructField;
    import org.apache.spark.sql.types.StructType;
    
    List<Row> data = Arrays.asList(
      RowFactory.create(1.218, 1.0, Vectors.dense(1.560, -0.605)),
      RowFactory.create(2.949, 0.0, Vectors.dense(0.346, 2.158)),
      RowFactory.create(3.627, 0.0, Vectors.dense(1.380, 0.231)),
      RowFactory.create(0.273, 1.0, Vectors.dense(0.520, 1.151)),
      RowFactory.create(4.199, 0.0, Vectors.dense(0.795, -0.226))
    );
    StructType schema = new StructType(new StructField[]{
      new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
      new StructField("censor", DataTypes.DoubleType, false, Metadata.empty()),
      new StructField("features", new VectorUDT(), false, Metadata.empty())
    });
    Dataset<Row> training = spark.createDataFrame(data, schema);
    double[] quantileProbabilities = new double[]{0.3, 0.6};
    AFTSurvivalRegression aft = new AFTSurvivalRegression()
      .setQuantileProbabilities(quantileProbabilities)
      .setQuantilesCol("quantiles");
    
    AFTSurvivalRegressionModel model = aft.fit(training);
    
    // Print the coefficients, intercept and scale parameter for AFT survival regression
    System.out.println("Coefficients: " + model.coefficients());
    System.out.println("Intercept: " + model.intercept());
    System.out.println("Scale: " + model.scale());
    model.transform(training).show(false);

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaAFTSurvivalRegressionExample.java"
in the Spark repo.

Refer to the [R API docs](api/R/reference/spark.survreg.html) for more
details.

    
    
    # Use the ovarian dataset available in R survival package
    library(survival)
    
    # Fit an accelerated failure time (AFT) survival regression model with spark.survreg
    ovarianDF <- suppressWarnings(createDataFrame(ovarian))
    aftDF <- ovarianDF
    aftTestDF <- ovarianDF
    aftModel <- spark.survreg(aftDF, Surv(futime, fustat) ~ ecog_ps + rx)
    
    # Model summary
    summary(aftModel)
    
    # Prediction
    aftPredictions <- predict(aftModel, aftTestDF)
    head(aftPredictions)

Find full example code at "examples/src/main/r/ml/survreg.R" in the Spark
repo.

## Isotonic regression

[Isotonic regression](http://en.wikipedia.org/wiki/Isotonic_regression)
belongs to the family of regression algorithms. Formally isotonic regression
is a problem where given a finite set of real numbers `$Y = {y_1, y_2, ...,
y_n}$` representing observed responses and `$X = {x_1, x_2, ..., x_n}$` the
unknown response values to be fitted finding a function that minimizes

`\begin{equation} f(x) = \sum_{i=1}^n w_i (y_i - x_i)^2 \end{equation}`

with respect to complete order subject to `$x_1\le x_2\le ...\le x_n$` where
`$w_i$` are positive weights. The resulting function is called isotonic
regression and it is unique. It can be viewed as least squares problem under
order restriction. Essentially isotonic regression is a [monotonic
function](http://en.wikipedia.org/wiki/Monotonic_function) best fitting the
original data points.

We implement a [pool adjacent violators
algorithm](https://doi.org/10.1198/TECH.2010.10111) which uses an approach to
[parallelizing isotonic
regression](https://doi.org/10.1007/978-3-642-99789-1_10). The training input
is a DataFrame which contains three columns label, features and weight.
Additionally, IsotonicRegression algorithm has one optional parameter called
$isotonic$ defaulting to true. This argument specifies if the isotonic
regression is isotonic (monotonically increasing) or antitonic (monotonically
decreasing).

Training returns an IsotonicRegressionModel that can be used to predict labels
for both known and unknown features. The result of isotonic regression is
treated as piecewise linear function. The rules for prediction therefore are:

  * If the prediction input exactly matches a training feature then associated prediction is returned. In case there are multiple predictions with the same feature then one of them is returned. Which one is undefined (same as java.util.Arrays.binarySearch).
  * If the prediction input is lower or higher than all training features then prediction with lowest or highest feature is returned respectively. In case there are multiple predictions with the same feature then the lowest or highest is returned respectively.
  * If the prediction input falls between two training features then prediction is treated as piecewise linear function and interpolated value is calculated from the predictions of the two closest features. In case there are multiple values with the same feature then the same rules as in previous point are used.

**Examples**

Refer to the [`IsotonicRegression` Python
docs](api/python/reference/api/pyspark.ml.regression.IsotonicRegression.html#pyspark.ml.regression.IsotonicRegression)
for more details on the API.

    
    
    from pyspark.ml.regression import IsotonicRegression
    
    # Loads data.
    dataset = spark.read.format("libsvm")\
        .load("data/mllib/sample_isotonic_regression_libsvm_data.txt")
    
    # Trains an isotonic regression model.
    model = IsotonicRegression().fit(dataset)
    print("Boundaries in increasing order: %s\n" % str(model.boundaries))
    print("Predictions associated with the boundaries: %s\n" % str(model.predictions))
    
    # Makes predictions.
    model.transform(dataset).show()

Find full example code at
"examples/src/main/python/ml/isotonic_regression_example.py" in the Spark
repo.

Refer to the [`IsotonicRegression` Scala
docs](api/scala/org/apache/spark/ml/regression/IsotonicRegression.html) for
details on the API.

    
    
    import org.apache.spark.ml.regression.IsotonicRegression
    
    // Loads data.
    val dataset = spark.read.format("libsvm")
      .load("data/mllib/sample_isotonic_regression_libsvm_data.txt")
    
    // Trains an isotonic regression model.
    val ir = new IsotonicRegression()
    val model = ir.fit(dataset)
    
    println(s"Boundaries in increasing order: ${model.boundaries}\n")
    println(s"Predictions associated with the boundaries: ${model.predictions}\n")
    
    // Makes predictions.
    model.transform(dataset).show()

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/IsotonicRegressionExample.scala"
in the Spark repo.

Refer to the [`IsotonicRegression` Java
docs](api/java/org/apache/spark/ml/regression/IsotonicRegression.html) for
details on the API.

    
    
    import org.apache.spark.ml.regression.IsotonicRegression;
    import org.apache.spark.ml.regression.IsotonicRegressionModel;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    
    // Loads data.
    Dataset<Row> dataset = spark.read().format("libsvm")
      .load("data/mllib/sample_isotonic_regression_libsvm_data.txt");
    
    // Trains an isotonic regression model.
    IsotonicRegression ir = new IsotonicRegression();
    IsotonicRegressionModel model = ir.fit(dataset);
    
    System.out.println("Boundaries in increasing order: " + model.boundaries() + "\n");
    System.out.println("Predictions associated with the boundaries: " + model.predictions() + "\n");
    
    // Makes predictions.
    model.transform(dataset).show();

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaIsotonicRegressionExample.java"
in the Spark repo.

Refer to the [`IsotonicRegression` R API
docs](api/R/reference/spark.isoreg.html) for more details on the API.

    
    
    # Load training data
    df <- read.df("data/mllib/sample_isotonic_regression_libsvm_data.txt", source = "libsvm")
    training <- df
    test <- df
    
    # Fit an isotonic regression model with spark.isoreg
    model <- spark.isoreg(training, label ~ features, isotonic = FALSE)
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/isoreg.R" in the Spark repo.

## Factorization machines regressor

For more background and more details about the implementation of factorization
machines, refer to the [Factorization Machines section](ml-classification-
regression.html#factorization-machines).

**Examples**

The following examples load a dataset in LibSVM format, split it into training
and test sets, train on the first dataset, and then evaluate on the held-out
test set. We scale features to be between 0 and 1 to prevent the exploding
gradient problem.

Refer to the [Python API
docs](api/python/reference/api/pyspark.ml.regression.FMRegressor.html#pyspark.ml.regression.FMRegressor)
for more details.

    
    
    from pyspark.ml import Pipeline
    from pyspark.ml.regression import FMRegressor
    from pyspark.ml.feature import MinMaxScaler
    from pyspark.ml.evaluation import RegressionEvaluator
    
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    # Scale features.
    featureScaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures").fit(data)
    
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    
    # Train a FM model.
    fm = FMRegressor(featuresCol="scaledFeatures", stepSize=0.001)
    
    # Create a Pipeline.
    pipeline = Pipeline(stages=[featureScaler, fm])
    
    # Train model.
    model = pipeline.fit(trainingData)
    
    # Make predictions.
    predictions = model.transform(testData)
    
    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)
    
    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    
    fmModel = model.stages[1]
    print("Factors: " + str(fmModel.factors))  # type: ignore
    print("Linear: " + str(fmModel.linear))  # type: ignore
    print("Intercept: " + str(fmModel.intercept))  # type: ignore

Find full example code at
"examples/src/main/python/ml/fm_regressor_example.py" in the Spark repo.

Refer to the [Scala API
docs](api/scala/org/apache/spark/ml/regression/FMRegressor.html) for more
details.

    
    
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    import org.apache.spark.ml.feature.MinMaxScaler
    import org.apache.spark.ml.regression.{FMRegressionModel, FMRegressor}
    
    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    
    // Scale features.
    val featureScaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .fit(data)
    
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    
    // Train a FM model.
    val fm = new FMRegressor()
      .setLabelCol("label")
      .setFeaturesCol("scaledFeatures")
      .setStepSize(0.001)
    
    // Create a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureScaler, fm))
    
    // Train model.
    val model = pipeline.fit(trainingData)
    
    // Make predictions.
    val predictions = model.transform(testData)
    
    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)
    
    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    
    val fmModel = model.stages(1).asInstanceOf[FMRegressionModel]
    println(s"Factors: ${fmModel.factors} Linear: ${fmModel.linear} " +
      s"Intercept: ${fmModel.intercept}")

Find full example code at
"examples/src/main/scala/org/apache/spark/examples/ml/FMRegressorExample.scala"
in the Spark repo.

Refer to the [Java API
docs](api/java/org/apache/spark/ml/regression/FMRegressor.html) for more
details.

    
    
    import org.apache.spark.ml.Pipeline;
    import org.apache.spark.ml.PipelineModel;
    import org.apache.spark.ml.PipelineStage;
    import org.apache.spark.ml.evaluation.RegressionEvaluator;
    import org.apache.spark.ml.feature.MinMaxScaler;
    import org.apache.spark.ml.feature.MinMaxScalerModel;
    import org.apache.spark.ml.regression.FMRegressionModel;
    import org.apache.spark.ml.regression.FMRegressor;
    import org.apache.spark.sql.Dataset;
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.SparkSession;
    
    // Load and parse the data file, converting it to a DataFrame.
    Dataset<Row> data = spark.read().format("libsvm").load("data/mllib/sample_libsvm_data.txt");
    
    // Scale features.
    MinMaxScalerModel featureScaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .fit(data);
    
    // Split the data into training and test sets (30% held out for testing).
    Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
    Dataset<Row> trainingData = splits[0];
    Dataset<Row> testData = splits[1];
    
    // Train a FM model.
    FMRegressor fm = new FMRegressor()
        .setLabelCol("label")
        .setFeaturesCol("scaledFeatures")
        .setStepSize(0.001);
    
    // Create a Pipeline.
    Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {featureScaler, fm});
    
    // Train model.
    PipelineModel model = pipeline.fit(trainingData);
    
    // Make predictions.
    Dataset<Row> predictions = model.transform(testData);
    
    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5);
    
    // Select (prediction, true label) and compute test error.
    RegressionEvaluator evaluator = new RegressionEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("rmse");
    double rmse = evaluator.evaluate(predictions);
    System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
    
    FMRegressionModel fmModel = (FMRegressionModel)(model.stages()[1]);
    System.out.println("Factors: " + fmModel.factors());
    System.out.println("Linear: " + fmModel.linear());
    System.out.println("Intercept: " + fmModel.intercept());

Find full example code at
"examples/src/main/java/org/apache/spark/examples/ml/JavaFMRegressorExample.java"
in the Spark repo.

Refer to the [R API documentation](api/R/reference/spark.fmRegressor.html) for
more details.

Note: At the moment SparkR doesn't support feature scaling.

    
    
    # Load training data
    df <- read.df("data/mllib/sample_linear_regression_data.txt", source = "libsvm")
    training_test <- randomSplit(df, c(0.7, 0.3))
    training <- training_test[[1]]
    test <- training_test[[2]]
    
    # Fit a FM regression model
    model <- spark.fmRegressor(training, label ~ features)
    
    # Model summary
    summary(model)
    
    # Prediction
    predictions <- predict(model, test)
    head(predictions)

Find full example code at "examples/src/main/r/ml/fmRegressor.R" in the Spark
repo.

# Linear methods

We implement popular linear methods such as logistic regression and linear
least squares with $L_1$ or $L_2$ regularization. Refer to [the linear methods
guide for the RDD-based API](mllib-linear-methods.html) for details about
implementation and tuning; this information is still relevant.

We also include a DataFrame API for [Elastic
net](http://en.wikipedia.org/wiki/Elastic_net_regularization), a hybrid of
$L_1$ and $L_2$ regularization proposed in [Zou et al, Regularization and
variable selection via the elastic
net](http://users.stat.umn.edu/~zouxx019/Papers/elasticnet.pdf).
Mathematically, it is defined as a convex combination of the $L_1$ and the
$L_2$ regularization terms: `\[ \alpha \left( \lambda \|\wv\|_1 \right) +
(1-\alpha) \left( \frac{\lambda}{2}\|\wv\|_2^2 \right) , \alpha \in [0, 1],
\lambda \geq 0 \]` By setting $\alpha$ properly, elastic net contains both
$L_1$ and $L_2$ regularization as special cases. For example, if a [linear
regression](https://en.wikipedia.org/wiki/Linear_regression) model is trained
with the elastic net parameter $\alpha$ set to $1$, it is equivalent to a
[Lasso](http://en.wikipedia.org/wiki/Least_squares#Lasso_method) model. On the
other hand, if $\alpha$ is set to $0$, the trained model reduces to a [ridge
regression](http://en.wikipedia.org/wiki/Tikhonov_regularization) model. We
implement Pipelines API for both linear regression and logistic regression
with elastic net regularization.

# Factorization Machines

[Factorization
Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) are able
to estimate interactions between features even in problems with huge sparsity
(like advertising and recommendation system). The `spark.ml` implementation
supports factorization machines for binary classification and for regression.

Factorization machines formula is:

\\[\hat{y} = w_0 + \sum\limits^n_{i-1} w_i x_i + \sum\limits^n_{i=1}
\sum\limits^n_{j=i+1} \langle v_i, v_j \rangle x_i x_j\\]

The first two terms denote intercept and linear term (same as in linear
regression), and the last term denotes pairwise interactions term. \\(v_i\\)
describes the i-th variable with k factors.

FM can be used for regression and optimization criterion is mean square error.
FM also can be used for binary classification through sigmoid function. The
optimization criterion is logistic loss.

The pairwise interactions can be reformulated:

\\[\sum\limits^n_{i=1} \sum\limits^n_{j=i+1} \langle v_i, v_j \rangle x_i x_j
= \frac{1}{2}\sum\limits^k_{f=1} \left(\left( \sum\limits^n_{i=1}v_{i,f}x_i
\right)^2 - \sum\limits^n_{i=1}v_{i,f}^2x_i^2 \right)\\]

This equation has only linear complexity in both k and n - i.e. its
computation is in \\(O(kn)\\).

In general, in order to prevent the exploding gradient problem, it is best to
scale continuous features to be between 0 and 1, or bin the continuous
features and one-hot encode them.

# Decision trees

[Decision trees](http://en.wikipedia.org/wiki/Decision_tree_learning) and
their ensembles are popular methods for the machine learning tasks of
classification and regression. Decision trees are widely used since they are
easy to interpret, handle categorical features, extend to the multiclass
classification setting, do not require feature scaling, and are able to
capture non-linearities and feature interactions. Tree ensemble algorithms
such as random forests and boosting are among the top performers for
classification and regression tasks.

The `spark.ml` implementation supports decision trees for binary and
multiclass classification and for regression, using both continuous and
categorical features. The implementation partitions data by rows, allowing
distributed training with millions or even billions of instances.

Users can find more information about the decision tree algorithm in the
[MLlib Decision Tree guide](mllib-decision-tree.html). The main differences
between this API and the [original MLlib Decision Tree API](mllib-decision-
tree.html) are:

  * support for ML Pipelines
  * separation of Decision Trees for classification vs. regression
  * use of DataFrame metadata to distinguish continuous and categorical features

The Pipelines API for Decision Trees offers a bit more functionality than the
original API. In particular, for classification, users can get the predicted
probability of each class (a.k.a. class conditional probabilities); for
regression, users can get the biased sample variance of prediction.

Ensembles of trees (Random Forests and Gradient-Boosted Trees) are described
below in the Tree ensembles section.

## Inputs and Outputs

We list the input and output (prediction) column types here. All output
columns are optional; to exclude an output column, set its corresponding Param
to an empty string.

### Input Columns

Param name | Type(s) | Default | Description  
---|---|---|---  
labelCol | Double | "label" | Label to predict  
featuresCol | Vector | "features" | Feature vector  
  
### Output Columns

Param name | Type(s) | Default | Description | Notes  
---|---|---|---|---  
predictionCol | Double | "prediction" | Predicted label |   
rawPredictionCol | Vector | "rawPrediction" | Vector of length # classes, with the counts of training instance labels at the tree node which makes the prediction | Classification only  
probabilityCol | Vector | "probability" | Vector of length # classes equal to rawPrediction normalized to a multinomial distribution | Classification only  
varianceCol | Double |  | The biased sample variance of prediction | Regression only  
  
# Tree Ensembles

The DataFrame API supports two major tree ensemble algorithms: [Random
Forests](http://en.wikipedia.org/wiki/Random_forest) and [Gradient-Boosted
Trees (GBTs)](http://en.wikipedia.org/wiki/Gradient_boosting). Both use
[`spark.ml` decision trees](ml-classification-regression.html#decision-trees)
as their base models.

Users can find more information about ensemble algorithms in the [MLlib
Ensemble guide](mllib-ensembles.html). In this section, we demonstrate the
DataFrame API for ensembles.

The main differences between this API and the [original MLlib ensembles
API](mllib-ensembles.html) are:

  * support for DataFrames and ML Pipelines
  * separation of classification vs. regression
  * use of DataFrame metadata to distinguish continuous and categorical features
  * more functionality for random forests: estimates of feature importance, as well as the predicted probability of each class (a.k.a. class conditional probabilities) for classification.

## Random Forests

[Random forests](http://en.wikipedia.org/wiki/Random_forest) are ensembles of
[decision trees](ml-classification-regression.html#decision-trees). Random
forests combine many decision trees in order to reduce the risk of
overfitting. The `spark.ml` implementation supports random forests for binary
and multiclass classification and for regression, using both continuous and
categorical features.

For more information on the algorithm itself, please see the [`spark.mllib`
documentation on random forests](mllib-ensembles.html#random-forests).

### Inputs and Outputs

We list the input and output (prediction) column types here. All output
columns are optional; to exclude an output column, set its corresponding Param
to an empty string.

#### Input Columns

Param name | Type(s) | Default | Description  
---|---|---|---  
labelCol | Double | "label" | Label to predict  
featuresCol | Vector | "features" | Feature vector  
  
#### Output Columns (Predictions)

Param name | Type(s) | Default | Description | Notes  
---|---|---|---|---  
predictionCol | Double | "prediction" | Predicted label |   
rawPredictionCol | Vector | "rawPrediction" | Vector of length # classes, with the counts of training instance labels at the tree node which makes the prediction | Classification only  
probabilityCol | Vector | "probability" | Vector of length # classes equal to rawPrediction normalized to a multinomial distribution | Classification only  
  
## Gradient-Boosted Trees (GBTs)

[Gradient-Boosted Trees
(GBTs)](http://en.wikipedia.org/wiki/Gradient_boosting) are ensembles of
[decision trees](ml-classification-regression.html#decision-trees). GBTs
iteratively train decision trees in order to minimize a loss function. The
`spark.ml` implementation supports GBTs for binary classification and for
regression, using both continuous and categorical features.

For more information on the algorithm itself, please see the [`spark.mllib`
documentation on GBTs](mllib-ensembles.html#gradient-boosted-trees-gbts).

### Inputs and Outputs

We list the input and output (prediction) column types here. All output
columns are optional; to exclude an output column, set its corresponding Param
to an empty string.

#### Input Columns

Param name | Type(s) | Default | Description  
---|---|---|---  
labelCol | Double | "label" | Label to predict  
featuresCol | Vector | "features" | Feature vector  
  
Note that `GBTClassifier` currently only supports binary labels.

#### Output Columns (Predictions)

Param name | Type(s) | Default | Description | Notes  
---|---|---|---|---  
predictionCol | Double | "prediction" | Predicted label |   
  
In the future, `GBTClassifier` will also output columns for `rawPrediction`
and `probability`, just as `RandomForestClassifier` does.

