# CS5344 Group Project EDA-Clustering
## Step 1: Load the necessary packages
```
from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import StringIndexer, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GBTClassifier
# from util import *
from pyspark.sql.functions import lit, concat, col, rand, when
from pyspark.ml.clustering import KMeans, LDA
from pyspark.ml.evaluation import ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import DataFrame
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sparknlp.base import Finisher, DocumentAssembler
from sparknlp.annotator import (BertSentenceEmbeddings, ClassifierDLApproach, UniversalSentenceEncoder,
                                SentimentDLApproach, SentimentDLModel)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType, StructField, StructType
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sparknlp.annotator import (Tokenizer, Normalizer,
                                LemmatizerModel, StopWordsCleaner)
from pyspark.sql.functions import explode, col, udf
from pyspark.ml.linalg import Vectors, VectorUDT

```
## Step 2: Build the PySpark environment
```
spark = SparkSession.builder \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.4.2") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "16g") \
    .getOrCreate()
```
