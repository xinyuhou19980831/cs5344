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
## Step 4: Define embedding function and preprocessing function and transform input data
Define the functions
```
def compute_embeddings(df: DataFrame, input_column: str, document_column: str) -> DataFrame:
    ...
    return transformed_df.withColumn("exploded_embeddings", explode(col("bert_embeddings.embeddings")))\
        .withColumn("bert", dense_vector_udf("exploded_embeddings"))
def preprocess_text(df: DataFrame, input_column: str, output_column: str) -> DataFrame:
    ...
    return pipeline.fit(df).transform(df)
```
Transform the input data
```
input_data = preprocess_text(trainingData, "combined_text", "preprocessed")
input_data = compute_embeddings(input_data, input_column = "preprocessed", document_column = "document")
```
## Step 5: Plot silhouette score and cluster data using Word2Vec and TFIDF separately
Put the two plots together.
```
fig, ax = plt.subplots(1,2, figsize =(16, 6))
ax[0].plot(range(2,11), silhouette_scores)
ax[0].set_xlabel('Number of Clusters')
ax[0].set_ylabel('Silhouette Score')
ax[1].plot(range(2,11), silhouette_scores2)
ax[1].set_xlabel('Number of Clusters')
ax[1].set_ylabel('Silhouette Score')
plt.tick_params(axis='both', which='minor', labelsize=14)
plt.savefig('silhouette_score.pdf')
```
K-means clustering
```
KMeans_1 = KMeans(featuresCol='word2vec', k=2)
KMeans_fit1 = KMeans_1.fit(input_data)
KMeans_transform1 = KMeans_fit1.transform(input_data)

KMeans_2 = KMeans(featuresCol='tfidf', k=2)
KMeans_fit2 = KMeans_2.fit(input_data)
KMeans_transform2 = KMeans_fit2.transform(input_data)
```
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
