# CS5344 Group Project EDA-Clustering

## Step 1: Load the necessary packages

## Step 2: Build the PySpark environment

## Step 3: Load data, combine fake and true together, and shuffle
```
path_fake = 'Fake.csv'
path_true = 'True.csv'
fake_df = spark.read.option("header", "true").csv(path_fake).withColumn("type", lit("F"))
true_df = spark.read.option("header", "true").csv(path_true).withColumn("type", lit("T"))
combined_df = true_df.withColumn('fake', lit(0)).union(fake_df.withColumn('fake', lit(1))).orderBy(rand())
combined_df = combined_df.withColumn("combined_text", concat(combined_df.title, combined_df.text))
```
## Step 4: Define embedding function and preprocessing function
```
def compute_embeddings(df: DataFrame, input_column: str, document_column: str) -> DataFrame:
...
def preprocess_text(df: DataFrame, input_column: str, output_column: str) -> DataFrame:
...
```

## Step 5: Plot silhouette score and cluster data using Word2Vec and TFIDF separately

## Step 6: Plot clusters in 2-dimensional space
Using plotly.express to plot the data and save them to pdf
```
fig = px.scatter(
    df_embeddings, x='x', y='y',
    size_max = 100, template = 'simple_white',
    color='label', 
    labels={'color': 'label'},
    opacity=0.5,
    color_discrete_map={1: 'red', 0: 'blue'},
    title = 'Word2Vec visualization')
plt.rcParams.update({'font.size': 12})
fig.write_image("word2vec.pdf")
```
