Home Topics Ensemble learning What is ensemble learning?

Explore watsonx.ai

![Isometric style illustration for Ensemble
Learning](/content/dam/connectedassets-adobe-cms/worldwide-content/creative-
assets/s-migr/ul/g/12/17/ensemble-learning-
ls.component.xl.ts=1723743540470.png/content/adobe-cms/us/en/topics/ensemble-
learning/_jcr_content/root/leadspace)

**Published:** 18 March 2024  
**Contributors:** Jacob Murel Ph.D., Eda Kavlakoglu

Ensemble learning combines multiple learners to improve predictive
performance. It has been adopted in response to issues resulting from limited
datasets.

Ensemble learning is a [machine learning](https://www.ibm.com/topics/machine-
learning) technique that aggregates two or more learners (e.g.
[regression](https://www.ibm.com/topics/linear-regression) models, [neural
networks](https://www.ibm.com/topics/neural-networks)) in order to produce
better predictions. In other words, an ensemble model combines several
individual models to produce more accurate predictions than a single model
alone.1 At times, sources may refer to this technique as committee-based
learning. Ensemble learning rests on the principle that a collectivity of
learners yields greater overall accuracy than an individual learner.2 Indeed,
research supports its efficacy with machine learning models and [convolutional
neural networks](https://www.ibm.com/topics/convolutional-neural-networks)
(CNNs).

A note on terminology: _base learner_ , _base model_ , and, in some cases,
_base estimator_ refers to the individual model or models used to in ensemble
algorithms. Literature further divides base learners into strong learner and
weak learners. Weak models or learners are defined as those that perform
little better than random guessing. For binary classification problems, weak
classifiers are more formally those that achieve approximately fifty percent
accuracy. By contrast, strong models or learners achieve excellent predictive
performance, which in binary classification is formalized as equal to or
greater than eighty percent accuracy.3

Note that some sources conflate _weak learner_ and _base learner_ given that
ensemble methods, particularly sequential ones, effectively boost weak
learners into strong learners.4

Demo  Take a tour of IBM watsonx

Explore IBM watsonx and learn how to create machine learning models using
statistical datasets.

Related content

Subscribe to the IBM newsletter

Why use ensemble learning?

Bias-variance tradeoff

Bias-variance tradeoff is a well-known problem in machine learning and a
motivating principle behind many
[regularization](https://www.ibm.com/topics/regularization) techniques. We can
define them as:

\- **Bias** measures the average difference between predicted values and true
values. As bias increases, a model predicts less accurately on a training
dataset. High bias refers to high error in training. Optimization signifies
attempts to reduce bias.

\- **Variance** measures the difference between predictions across various
realizations of a given model. As variance increases, a model predicts less
accurately on unseen data. High variance refers to high error during testing
and validation. Generalization refers to attempts to reduce variance.

Bias and variance thus inversely represent model accuracy on training and test
data respectively.5 They are two of three terms that comprise a model’s total
error rate, the third being irreducible error. This third term denotes error
resulting from inherent randomness in a dataset. Total model error can be
defined by the formula:6

Many models versus one

Any one model training algorithm consists of numerous variables—e.g. training
data, hyperparameters, and so forth—that affect the consequent model’s total
error. Thus, even a single training algorithm can produce different models,
each with their own bias, variance, and irreducible error rates. By combining
several diverse models, ensemble algorithms can yield a lower overall error
rate while retaining each individual model’s own complexities and advantages,
such as notably low bias for a specific data subset.7

Research suggests that, in general, the greater diversity among combined
models, the more accurate the resulting ensemble model. Ensemble learning can
thus address regression problems such as
[overfitting](https://www.ibm.com/topics/overfitting) without trading away
model bias. Indeed, research suggests that ensembles comprised of diverse
under-regularized models (i.e. models that overfit to their training data)
outperform single regularized models.8 Moreover, ensemble learning techniques
can help resolve issues stemming from high-dimensional data, and so
effectively serve as an alternative to [dimensionality
reduction](https://www.ibm.com/topics/dimensionality-reduction).

Types of ensemble models

Literature widely categorizes ensemble learning methods in machine learning
into two groups: parallel and sequential.

-**Parallel** methods train each base learner apart from the others of the others. Per its name, then, parallel ensembles train base learners in parallel and independent of one another.

\- **Sequential** methods train a new base learner so that it minimizes errors
made by the previous model trained in the preceding step. In other words,
sequential methods construct base models sequentially in stages.9

Parallel methods are further divided into homogenous and heterogenous methods.
Homogenous parallel ensembles use the same base learning algorithm to produce
all of the component base learners. Heterogenous parallel ensembles use
different algorithms to produce base learners.10

Voting

How do ensemble methods combine base learners into a final learner? Some
techniques—e.g. stacking—use separate machine learning algorithms to train an
ensemble learner from the base learners. But one common method for
consolidating base learner predictions is voting—and more precisely, majority
voting.

Majority voting considers each base learner’s prediction for a given data
instance and outputs a final prediction determined by whatever the majority of
learners predict. For instance, in a binary classification problem, majority
voting takes predictions from each base classifier for a given data instance
and uses the majority prediction as the end prediction. Weighted majority
voting is an extension of this technique that gives greater weight to certain
learner’s predictions over others.11

Ensemble learning techniques

Perhaps three of the most popular ensemble learning techniques are bagging,
boosting, and stacking. In fact, these together exemplify distinctions between
sequential, parallel, homogenous, and heterogenous types of ensemble methods.

Note that this overview is not exhaustive; there are several additional
ensemble methods, such as blending and weighted average ensembles. This is
merely meant to survey some of the more prominent methods in literature.

Bagging

[Bagging](https://www.ibm.com/topics/bagging) is a homogenous parallel method
sometimes called _bootstrap aggregating_. It uses modified replicates of a
given training data set to train multiple base learners with the same training
algorithm.12 Scikit-learn’s ensemble module in Python contains functions for
implementing bagging, such as BaggingClassifier.

More specifically, bagging uses a technique called bootstrap resampling to
derive multiple new datasets from one initial training dataset in order to
train multiple base learners. How does this work? Say a training dataset
contains _n_ training examples. Bootstrap resampling copies _n_ data instances
from that set into a new subsample dataset, with some initial instances
appearing more than once and others excluded entirely. These are bootstrap
samples. Repeating this process _x_ times produces _x_ iterations of the
original dataset, each containing _n_ samples from the initial set. Each
iteration of the initial set is then used to train a separate base learner
with the same learning algorithm.13

Random forest is an extension of bagging that specifically denotes the use of
bagging to construct ensembles of randomized [decision
trees](https://www.ibm.com/topics/decision-trees). This differs from standard
decision trees in that the latter samples every feature to identify the best
for splitting. By contrast, random forests iteratively sample random subsets
of features to create a decision node.14

Stacking

Stacking, or stacked generalization,15 is a heterogenous parallel method that
exemplifies what is known as meta-learning. Meta-learning consists of training
a meta-learner from the output of multiple base learners. Stacking
specifically trains several base learners from the same dataset using a
different training algorithm for each learner. Each base learner makes
predictions on an unseen dataset. These first model predictions are then
compiled and used to train a final model, being the meta-model.16

Note the importance of using a different dataset from that used to train the
base learners in order to train the meta-learner. Using the same dataset to
train the base learners and the meta-learner can result in overfitting. This
can require excluding data instances from the base learner training data to
serve as its test set data, which in turn becomes training data for the meta-
learner. Literature often recommends techniques such as cross-validation to
ensure these datasets do not overlap.17

Much as bagging, the sklearn.ensemble module in Python provides various
functions for implementing stacking techniques.

Boosting

[Boosting](https://www.ibm.com/topics/boosting) algorithms are a sequential
ensemble method. Boosting has many variations, but they all follow the same
general procedure. Boosting trains a learner on some initial dataset, _d_. The
resultant learner is typically weak, misclassifying many samples in the
dataset. Much like bagging, boosting then samples instances from the initial
dataset to create a new dataset (_d 2_). Unlike bagging, however, boosting
prioritizes misclassified data instances from the first model or learner. A
new learner is trained on this new dataset _d 2_. Then a third dataset (_d 3_)
is then compiled from _d 1_ and _d 2_, prioritizes the second learner’s
misclassified samples and instances in which _d 1_ and _d 2_ disagree. The
process repeats _n_ times to produce _n_ learners. Boosting then combines and
weights the all the learners together to produce final predictions.18

Boosting algorithms largely differ in how they prioritize erroneously
predicted data instances when creating a new dataset. Two of the most
prominent boosting methods may illustrate this:

\- **Adaptive boosting** (AdaBoost) weights model errors. That is, when
creating a new iteration of a dataset for training the next learner, AdaBoost
adds weights to the previous learner’s misclassified samples, causing the next
learner to prioritize those misclassified samples.

\- **Gradient boosting** uses residual errors when training new learners.
Rather than weight misclassified samples, gradient boosting uses residual
errors from a previous model to set target predictions for the next model. In
this way, it attempts to close the gap of error left by one model.19

Unfortunately, sklearn contains no pre-defined functions for implementing
boosting. The Extreme Gradient Boosting (XGBoost) open-source library,
however, provides code for implementing gradient boosting in Python.

Recent research

Given difficulties in acquiring large, fair-use, labeled datasets for training
learners, ensemble learning has seen many applications in an attempt to
improve learner performance with less data. For instance, several recent
studies show promising results with improving model generalizability using
ensemble methods for computer vision task, such as training several models
with different representations of a dataset20 or combining several biased
models.21

Despite ensemble methods’ ability to improve generalizability, they
nevertheless can suffer unfairness. In machine learning, fairness denotes
attempts to alleviate algorithmic bias (often against minority groups) in
automated systems, usually resulting from learners trained on sensitive data.
A handful of studies propose metrics, preprocessing, and postprocessing
techniques for improving fairness in ensemble models.22 Continued efforts to
improve fairness and [ethical practices in AI](https://www.ibm.com/impact/ai-
ethics) remain a much needed area for future research.

Related products and solutions

AI consulting services

Reimagine how you work with AI: Our diverse, global team of more than 20,000
AI experts can help you quickly and confidently design and scale AI and
automation across your business, working across our own IBM watsonx™
technology and an open ecosystem of partners to deliver any AI model, on any
cloud, guided by ethics and trust.  

Explore IBM AI consulting services

AI solutions

Operationalize AI across your business to deliver benefits quickly and
ethically. Our rich portfolio of business-grade AI products and analytics
solutions are designed to reduce the hurdles of AI adoption and establish the
right data foundation while optimizing for outcomes and responsible use.

Explore IBM AI solutions

* * *

IBM watsonx

Multiply the power of AI with our next-generation AI and data platform. IBM
watsonx is a portfolio of business-ready tools, applications and solutions,
designed to reduce the costs and hurdles of AI adoption while optimizing
outcomes and responsible use of AI.

Explore watsonx  Try watsonx.ai

* * *

Take the next step

Build an AI strategy for your business on one collaborative AI and data
platform—IBM watsonx. Train, validate, tune and deploy AI models to help you
scale and accelerate the impact of AI with trusted data across your business.

Explore watsonx Book a live demo

#####  Footnotes

1 Zhi-Hua Zhou, _Ensemble Methods: Foundations and Algorithms_ , CRC Press,
2012.

2 Gavin Brown, “Ensemble Learning,” _Encyclopedia of Machine Learning and Data
Mining_ , Springer, 2017.

3 Gautam Kunapuli, _Ensemble Methods for Machine Learning_ , Manning
Publications, 2023. Lior Rokach, _Pattern Classification Using Ensemble
Methods_ , World Scientific Publishing Company, 2010.

4 Zhi-Hua Zhou, _Ensemble Methods: Foundations and Algorithms_ , CRC Press,
2012.

5 Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan
Taylor, _An Introduction to Statistical Learning with Applications in Python,_
Springer, 2023.

6 George Kyriakides and Konstantinos G. Margaritis, _Hands-On Ensemble
Learning with Python_ , Packt Publishing, 2019.

7 Zhi-Hua Zhou, _Machine Learning_ , translated by Shaowu Liu,**** Springer,
2021.__ George Kyriakides and Konstantinos G. Margaritis, _Hands-On Ensemble
Learning with Python_ , Packt Publishing, 2019.

8 Peter Sollich and Anders Krogh, "Learning with ensembles: How overfitting
can be useful," Advances in Neural Information Processing Systems, Vol. 8,
1995,
<https://papers.nips.cc/paper_files/paper/1995/hash/1019c8091693ef5c5f55970346633f92-Abstract.html>
(link resides outside of ibm.com).

9 Gautam Kunapuli, _Ensemble Methods for Machine Learning_ , Manning
Publications, 2023.

10 Zhi-Hua Zhou, _Ensemble Methods: Foundations and Algorithms_ , CRC Press,
2012.

11 Ibomoiye Domor Mienye and Yanxia Sun, “A Survey of Ensemble Learning:
Concepts, Algorithms, Applications, and Prospects,” _IEEE Access_ , Vol. 10,
2022, pp. 99129-99149, <https://ieeexplore.ieee.org/document/9893798> (link
resides outside of ibm.com). Lior Rokach, “Ensemble-based Classifiers,”
_Artificial Intelligence Review_ , Vol. 33, 2010, pp. 1-39,
<https://link.springer.com/article/10.1007/s10462-009-9124-7> (link resdies
outside of ibm.com).

12 M. Galar, A. Fernandez, E. Barrenechea, H. Bustince and F. Herrera, "A
Review on Ensembles for the Class Imbalance Problem: Bagging-, Boosting-, and
Hybrid-Based Approaches," _IEEE Transactions on Systems, Man, and Cybernetics_
, Vol. 42, No. 4, 2012, pp. 463-484,
<https://ieeexplore.ieee.org/document/5978225> (link resides outside of
ibm.com).

13 Zhi-Hua Zhou, _Ensemble Methods: Foundations and Algorithms_ , CRC Press,
2012.

14 Gautam Kunapuli, _Ensemble Methods for Machine Learning_ , Manning
Publications, 2023.

15 Robi Palikar, “Ensemble Learning,” _Ensemble Machine Learning: Methods and
Applications_ , Springer, 2012.

16 Ibomoiye Domor Mienye and Yanxia Sun, “A Survey of Ensemble Learning:
Concepts, Algorithms, Applications, and Prospects,” _IEEE Access_ , Vol. 10,
2022, pp. 99129-99149, <https://ieeexplore.ieee.org/document/9893798> (link
resides outside of ibm.com).

17 Zhi-Hua Zhou, _Ensemble Methods: Foundations and Algorithms_ , CRC Press,
2012. Gautam Kunapuli, _Ensemble Methods for Machine Learning_ , Manning
Publications, 2023.

18 Robi Palikar, “Ensemble Learning,” _Ensemble Machine Learning: Methods and
Applications_ , Springer, 2012. Zhi-Hua Zhou, _Ensemble Methods: Foundations
and Algorithms_ , CRC Press, 2012.

19 Gautam Kunapuli, _Ensemble Methods for Machine Learning_ , Manning
Publications, 2023.

20 Devesh Walawalkar, Zhiqiang Shen, and Marios Savvides, "Online Ensemble
Model Compression Using Knowledge Distillation," 2020, pp. 18-35,
<https://link.springer.com/chapter/10.1007/978-3-030-58529-7_2> (link resides
outside of ibm.com).

21 Xinzhe Han, Shuhui Wang, Chi Su, Qingming Huang, and Qi Tian, "Greedy
Gradient Ensemble for Robust Visual Question Answering," Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp.
1584-1593,
<https://openaccess.thecvf.com/content/ICCV2021/html/Han_Greedy_Gradient_Ensemble_for_Robust_Visual_Question_Answering_ICCV_2021_paper.html>
(link resides outside of ibm.com).

22 Usman Gohar, Sumon Biswas, and Hridesh Rajan, "Towards Understanding
Fairness and its Composition in Ensemble Machine Learning," 2023 IEEE/ACM 45th
International Conference on Software Engineering (ICSE), 2023, pp. 1533-1545,
<https://ieeexplore.ieee.org/abstract/document/10172501> (link resides outside
of ibm.com). Khaled Badran, Pierre-Olivier Côté, Amanda Kolopanis, Rached
Bouchoucha, Antonio Collante, Diego Elias Costa, Emad Shihab, and Foutse
Khomh, "Can Ensembling Preprocessing Algorithms Lead to Better Machine
Learning Fairness?" _Computer_ , Vol. 56, No. 4, 2023, pp. 71-79,
<https://ieeexplore.ieee.org/abstract/document/10098174> (link resides outside
of ibm.com). Swanand Kadhe, Anisa Halimi, Ambrish Rawat, and Nathalie
Baracaldo, "FairSISA: Ensemble Post-Processing to Improve Fairness of
Unlearning in LLMs," _Socially Responsible Language Modelling Research
Workshop_ , 2023, <https://neurips.cc/virtual/2023/78908> (link resides
outside of ibm.com).

