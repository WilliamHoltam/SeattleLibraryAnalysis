# Databricks notebook source
# MAGIC %md
# MAGIC Clustering the Seattle Library Collection into Topics
# MAGIC =======================================================
# MAGIC 
# MAGIC The data used in this demo is available at: [https://catalog.data.gov/dataset?tags=seattle-public-library](https://catalog.data.gov/dataset?tags=seattle-public-library "Seattle Public Library Dataset")
# MAGIC 
# MAGIC ![Seattle Public Library](https://upload.wikimedia.org/wikipedia/commons/4/4d/Seattle_Public_Library.jpg)
# MAGIC 
# MAGIC Use case - to assign items held by the Seattle Library to different topics using the item description.
# MAGIC 
# MAGIC This is an unsupervised learning problem solved using a compination of TFIDF vectorising and the K-means clustering algorithm.
# MAGIC   
# MAGIC Import Python Libraries
# MAGIC -----------------------

# COMMAND ----------

# MAGIC %sh /databricks/python/bin/pip install nltk

# COMMAND ----------

import numpy as np
import pandas as pd
import more_itertools
import re
import mpld3
import matplotlib.pyplot as plt

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as SVD

from pyspark.ml.feature import Word2Vec, StringIndexer, RegexTokenizer, StopWordsRemover
from pyspark.sql.functions import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# from sparknlp.annotator import *
# from sparknlp.common import *
# from sparknlp.base import *

# COMMAND ----------

# MAGIC %md
# MAGIC Login to Snowflake and Import Data
# MAGIC ----------------------------------

# COMMAND ----------

# Use secret manager to get the login name and password for the Snowflake user
username = dbutils.secrets.get(scope="snowflake-credentials", key="username")
password = dbutils.secrets.get(scope="snowflake-credentials", key="password")
# snowflake connection options
options = dict(
  sfUrl="datalytyx.east-us-2.azure.snowflakecomputing.com",
  sfUser=str("WILLHOLTAM"),
  sfPassword=str("04MucSfLV"),
  sfDatabase="DATABRICKS_DEMO",
  sfRole="DATABRICKS",
  sfSchema="SEATTLE_LIBRARY",
  sfWarehouse="DATASCIENCE_WH"
)
df = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("query",
          "select BIBNUM, TITLE, AUTHOR, PUBLICATIONYEAR, SUBJECTS from LIBRARY_COLLECTION_INVENTORY where reportdate in (\
          '2017-09-01T00:00:00',\
          '2017-10-01T00:00:00',\
          '2017-11-01T00:00:00',\
          '2017-12-01T00:00:00',\
          '2018-01-01T00:00:00',\
          '2018-01-01T00:00:00',\
          '2018-02-01T00:00:00',\
          '2018-02-01T00:00:00',\
          '2018-03-01T00:00:00',\
          '2018-04-01T00:00:00',\
          '2018-05-01T00:00:00',\
          '2018-06-01T00:00:00',\
          '2018-07-01T00:00:00'\
          )"
         ) \
  .load() \
  .limit(1000)

df = df.cache()

df_pandas = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Preview the Data
# MAGIC ----------------

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.head(3))

# COMMAND ----------

df.count()

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# MAGIC %md 
# MAGIC Set Initial Parameters
# MAGIC ----------------------

# COMMAND ----------

nltk.download('stopwords')
stop = nltk.corpus.stopwords.words('english')

# COMMAND ----------

# MAGIC %md 
# MAGIC Pipeline
# MAGIC --------

# COMMAND ----------

df_nan_removed = df.fillna(value=pd.np.nan)
df_droped_empty_rows = df_nan_removed.dropna(how='any')
df_droped_null = df_droped_empty_rows.na.drop()

# COMMAND ----------

tokenizer = RegexTokenizer(pattern="[^a-zA-Z-_']", inputCol="SUBJECTS", outputCol="WORDS")
df = tokenizer.transform(df_droped_null)
display(df)

# COMMAND ----------

remover = StopWordsRemover(inputCol="WORDS", outputCol="FILTERED")
df = remover.transform(df)
display(df)

# COMMAND ----------

stemmer = SnowballStemmer("english")
tokens = []
stems = []
for row in df.select("FILTERED").rdd.collect():
  for words in row:
    tokens.extend([t for t in words])
    stems.extend([stemmer.stem(t) for t in words])
vocab_frame = pd.DataFrame({"WORDS": tokens}, index=stems)

# COMMAND ----------

from pyspark.sql.column import Column
from pyspark.sql.types import ArrayType, StringType
rdd = df.select("FILTERED").rdd.collect()
stemmed = []
for row in rdd:
  for list_ in row:
    stemmed.append([stemmer.stem(element) for element in list_])
df_new = pd.DataFrame(data={'STEMMED': stemmed})
df_pandas = df.toPandas()
df = spark.createDataFrame(df_pandas.join(df_new))
display(df)

# COMMAND ----------

word2Vec = Word2Vec(vectorSize=100, seed=42, inputCol="STEMMED", outputCol="VECTORS")
w2v = word2Vec.fit(df)
df_w2v = w2v.transform(df)

# COMMAND ----------

# display(w2v.findSynonyms("baby", 4))

# COMMAND ----------

vectors = w2v.getVectors()
terms = vectors.select("word").rdd.collect()
# vocab_frame.loc[terms[81][0]]
# print(type(terms[81][0]))
# print(type(vocab_frame))
# print(vocab_frame.loc[terms[81][0], :])

# COMMAND ----------

for i in range(8):
  kmeans = KMeans(featuresCol="VECTORS", predictionCol="CLUSTER"+str(i+2), k=i+2, seed=42)
  model = kmeans.fit(df_w2v)
  df = model.transform(df_w2v)
  centers = model.clusterCenters()
  cluEval = ClusteringEvaluator(featuresCol="VECTORS", predictionCol="CLUSTER"+str(i+2), metricName="silhouette")
  score = cluEval.evaluate(df)
  print("K =", i+2, ": ", score)  

# COMMAND ----------

# MAGIC %md
# MAGIC Best clustering occurs when K=5

# COMMAND ----------

num_clusters = 5
kmeans = KMeans(featuresCol="VECTORS", predictionCol="CLUSTER", k=num_clusters, seed=42)
model = kmeans.fit(df)
df = model.transform(df)
centers = model.clusterCenters()
cluEval = ClusteringEvaluator(featuresCol="VECTORS", predictionCol="CLUSTER", metricName="silhouette")
score = cluEval.evaluate(df)
print("K="+str(num_clusters), score)  

# COMMAND ----------

display(df)

# COMMAND ----------

terms = vectors.select("word").rdd.collect()
order_centroids = pd.DataFrame(np.array(centers).argsort()[:, ::-1])
top_terms_list = []
# print(vocab_frame.loc[terms[ind][0]])
for cluster in np.arange(num_clusters):
  print("\ncluster %d words: " % int(str(cluster+1))+"\n", end="")
  top_terms = []
  max_index = 6
  for count, ind in enumerate(order_centroids.iloc[cluster, :max_index]):
    if count == max_index-1:
      print(
        str(count),
        " %s" % vocab_frame.
        loc[terms[ind][0], :].
        values.tolist()[0][0].
        encode("utf-8", "ignore")
      ) 
      top_terms.append(
        (
          " %s" % vocab_frame.
          loc[terms[ind][0], :].
          values.tolist()[0][0]
        ).strip()
      )
    else:
      print(
        str(count),
        " %s" % vocab_frame.
        loc[terms[ind][0], :].
        values.tolist()[0][0].
        encode("utf-8", "ignore")
      ) 
      top_terms.append(
        (
          " %s" % vocab_frame.
          loc[terms[ind][0], :].
          values.tolist()[0][0] + ","
        ).strip()
      )
  top_terms_list.append("".join(top_terms))

# COMMAND ----------

from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vector, Vectors

# display(vectors)
vec = vectors.toPandas()
# print(type(vec.iloc[0, 1][0]))
vec1 = vectors.select("vector").rdd
# print(type(vec1.take(20)[0]))
mat = RowMatrix(vectors.rdd)

mat.numRows()
col = vectors.vector.columns
# vectorAss = VectorAssembler(inputCols=col, outputCol="features")
# vdf = vectorAss.transform(vectors)
# vector = vectors

# # # Get its size.
# mat.numRows()  # 4
# n = mat.numCols()  # 3
# df_temp = spark.createDataFrame(pd.DataFrame(data={"Average Vectors": dist}, index=vector.index))

# display(df_temp)
# mat = IndexedRowMatrix(df_temp)
# sims = mat.columnSimilarities()
# display(sims)
# model.init_sims()
# matrix = numpy.dot(model.syn0norm, model.syn0norm.T)
# dist = np.array(dist)
#   for element in row:
#     print(vectors.loc[row,:][element])
# [print(vectors.loc[i,:]) for i in vectors]

# from pyspark.mllib.linalg.distributed import IndexedRowMatrix
# vectors = df_w2v.select("VECTORS").rdd.map(lambda row: row.VECTORS)
# mat = IndexedRowMatrix(vectors)
# sims = mat.columnSimilarities()
# vectors = df_w2v.select("VECTORS").toPandas()
# dist = 1 - cosine_similarity(vectors)
# # print(dist[100,100])
# # convert two components as we're plotting points in a two-dimensional plane
# # "precomputed" because we provide a distance matrix
# # we will also specify `random_state` so the plot is reproducible.
# MDS()
# mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
# pos = mds.fit_transform(dist.reshape(-1, 1))  # shape (n_components, n_samples)
# xs, ys = pos[:, 0], pos[:, 1]

# COMMAND ----------

#set up colors per clusters using a dict
cluster_colors = {
  0: 'deepskyblue',
  1: 'lightcoral',
  2: 'springgreen',
  3: 'orange',
  4: 'hotpink'
}
cluster_numbers = list(range(num_clusters))
cluster_names = dict(zip(cluster_numbers, top_terms_list))

# COMMAND ----------

# create data frame that has the result of the MDS plus the cluster numbers and titles
# clusters = post_processing_pipeline.named_steps['dfr'].clusters
# titles = pre_processing_pipeline.get_params(True)['anrr'].cleaned_data.TITLE
df = pd.DataFrame(dict(x=xs, y=ys, label=df.cluster, title=df.titles)) 
# print(df)
# group by cluster
groups = df.groupby('label')
# set up plot
fig, ax = plt.subplots(figsize=(13, 7)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
  ax.plot(
    group.x,
    group.y,
    marker='o',
    linestyle='',
    ms=8,
    label=cluster_names[name],
    color=cluster_colors[name],
    mec='none'
  )
  ax.set_aspect('auto')
  ax.tick_params(
    axis= 'x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False
  )
  ax.tick_params(
    axis= 'y',  # changes apply to the y-axis
    which='both',  # both major and minor ticks are affected
    left=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelleft=False
  )

ax.legend(numpoints=1)  #show legend with only 1 point
# #add label in x,y position with the label as the film title
# for i in range(len(df)):
#     ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['title'], size=8)  
# plt.savefig("visualisation.jpg")
# plt.show() #show the plot
# #uncomment the below to save the plot if need be
plt.savefig('clusters_small_noaxes.png', dpi=400)
display(fig)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md The mpld3 project brings together Matplotlib, the popular Python-based graphing library, and D3js, the popular JavaScript library for creating interactive data visualizations for the web. The result is a simple API for exporting your matplotlib graphics to HTML code which can be used within the browser, within standard web pages, blogs, or tools such as the IPython notebook.

# COMMAND ----------

#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
  """Plugin for moving toolbar to top of figure"""

  JAVASCRIPT = """
  mpld3.register_plugin("toptoolbar", TopToolbar);
  TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
  TopToolbar.prototype.constructor = TopToolbar;
  function TopToolbar(fig, props){
      mpld3.Plugin.call(this, fig, props);
  };

  TopToolbar.prototype.draw = function(){
    // the toolbar svg doesn't exist
    // yet, so first draw it
    this.fig.toolbar.draw();

    // then change the y position to be
    // at the top of the figure
    this.fig.toolbar.toolbar.attr("x", 50);
    this.fig.toolbar.toolbar.attr("y", 100);

    // then remove the draw function,
    // so that it is not called again
    this.fig.toolbar.draw = function() {}
  }
  """

  def __init__(self):
    self.dict_ = {"type": "toptoolbar"}

# COMMAND ----------

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 
#group by cluster
groups = df.groupby('label')
#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:"Arial Black", Gadget, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: 0px;}
"""
# Plot 
fig, ax = plt.subplots(figsize=(13,7)) #set plot size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
  points = ax.plot(
    group.x,
    group.y,
    marker='o',
    linestyle='',
    ms=8, 
    label=cluster_names[name],
    mec='none', 
    color=cluster_colors[name]
  )
  ax.set_aspect('auto')
  labels = [i for i in group.title]
  #set tooltip using points, labels and the already defined 'css'
  tooltip = mpld3.plugins.PointHTMLTooltip(
    points[0], 
    labels,               
    voffset=10, 
    hoffset=10, 
    css=css
  )
  #connect tooltip to fig
  mpld3.plugins.connect(fig, tooltip, TopToolbar())    
  #set tick marks as blank
  ax.axes.get_xaxis().set_ticks([])
  ax.axes.get_yaxis().set_ticks([])
  #set axis as blank
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)
  ax.patch.set_facecolor('white')

# plt.subplots_adjust(top = 0.75)      # the top of the subplots of the figure
# plt.xlim(-1,1.4)
plt.ylim(-1,1.1)
plt.tight_layout()
plt.legend(numpoints=1, loc='best', ncol=1, fontsize='small', title='')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.legend(numpoints=1, loc=2) #show legend with only one dot
# plt.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gcf().transFigure)
#uncomment the below to export to html
html = mpld3.fig_to_html(fig)
displayHTML(html)  # show the plot
# # file-output.py
# f = open('graph_html.htm','w')
# f.write(html)
# f.close()

# COMMAND ----------

# stringIndexer = StringIndexer(inputCol="AUTHOR", outputCol="indexed")
# model = stringIndexer.fit(df)
# df = model.transform(df)
# encoder = OneHotEncoder(inputCol="indexed", outputCol="page_ohe")
# df = encoder.transform(df)

# COMMAND ----------

# model = VectorAssembler(inputCols=['page_ohe', 'vectors'], outputCol="merged_vectors")
# df = model.transform(df)

# COMMAND ----------

display(df)

# COMMAND ----------

df.createOrReplaceTempView("my_data")

# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)
# MAGIC library(ggplot2)
# MAGIC library(Rtsne)
# MAGIC df <- sql("select subjects, vectors from my_data")
# MAGIC # display(df)
# MAGIC merged_vectors <- collect(select(df, "vectors"))
# MAGIC print(typeof(merged_vectors))
# MAGIC cluster_coords <- Rtsne(merged_vectors, initial_dims=53, perplexity=50, epoch=50)
# MAGIC # df <- as.data.frame()
# MAGIC # display(df)
# MAGIC # df_1 <- data.frame(matrix(unlist(df), nrow=nrow(df)),stringsAsFactors=FALSE)
# MAGIC # print(df_1)
# MAGIC # localDf <- collect(df)
# MAGIC # print(localDf)
# MAGIC # matrix <-data.matrix(localDf)
# MAGIC # print(typeof(matrix))
# MAGIC # cluster_coords <- Rtsne(df, initial_dims=53, perplexity=50, epoch=50)

# COMMAND ----------

# MAGIC %r
# MAGIC ggplot(df_plot, aes(x=x, y=y, color=page_id)) +
# MAGIC     geom_point(alpha=0.75, stroke=0) + 
# MAGIC     theme_bw()

# COMMAND ----------

# def tokenize_and_stem(text):
#   """
#   Function first Tokenizes and then Stems the words, returning the stems.
#   """
#   tokens = []
#   for sent in nltk.sent_tokenize(text):
#     tokens.extend([word for word in nltk.word_tokenize(sent)])
#   filtered_tokens = []
#   for token in tokens:
#     if re.search('[a-zA-Z]', token):
#       filtered_tokens.append(token)
#     else:
#       token
#   stems = [stemmer.stem(t) for t in filtered_tokens]
#   return stems


# def tokenize_only(text):
#   """
#   Function Tokenizes the words, returning tokens that are filtered to contain upper or
#   lowwer case letters.
#   """ 
#   # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
#   tokens = []
#   for sent in nltk.sent_tokenize(text):
#     tokens.extend([word.lower() for word in nltk.word_tokenize(sent)])
#   # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
#   filtered_tokens = []
#   for token in tokens:
#     if re.search('[a-zA-Z]', token):
#       filtered_tokens.append(token)
#     else:
#       token
#   return filtered_tokens


# totalvocab_stemmed = []
# # for each item in 'synopses', tokenize/stem
# allwords_stemmed = [tokenize_and_stem(i) for i in df_pandas.SUBJECTS.tolist()]
# totalvocab_stemmed.extend(allwords_stemmed)
# totalvocab_stemmed = list(more_itertools.collapse(totalvocab_stemmed))
# totalvocab_tokenized = []
# # for each item in 'synopses', tokenize/stem
# allwords_tokenized = [tokenize_only(i) for i in df_pandas.SUBJECTS.tolist()]
# totalvocab_tokenized.extend(allwords_tokenized)
# totalvocab_tokenized = list(more_itertools.collapse(totalvocab_tokenized))
# vocab_frame = pd.DataFrame({"words": totalvocab_tokenized}, index=totalvocab_stemmed)
# subjects = df_pandas.SUBJECTS.tolist()
# print(subjects)

# COMMAND ----------

print(vocab_frame)

# COMMAND ----------

from pyspark.mllib.feature import Word2Vec
inp = sc.parallelize(subjects)

word2Vec = Word2Vec(vectorSize=50, seed=42, inputCol="words", outputCol="vectors")
model = word2vec.fit(rdd)

# Rdd1 = df_droped_empty_rows.select('SUBJECTS').rdd.map(lambda row: row.split(" "))
# doc = df_droped_empty_rows.select('SUBJECTS').rdd.map(lambda line: line.split(" ")).collect()



# synonyms = model.findSynonyms('1', 5)

# for word, cosine_distance in synonyms:
#     print("{}: {}".format(word, cosine_distance))

# COMMAND ----------

# display(doc)

# COMMAND ----------

class TfidfVectorizerNew(TfidfVectorizer):
  
  """
  Transformer inherits from the Sk-learn TfidfVectorizer class and keeps
  the returned tfidf_matrix as a attribute of the class that can be refered to later.
  """
  
  def transform(self, raw_documents, copy=True):
    X = super(TfidfVectorizer, self).transform(raw_documents)
    self.tfidf_matrix = self._tfidf.transform(X, copy=False)
    return self.tfidf_matrix
    
  def fit_transform(self, raw_documents, copy=True):
    X = super(TfidfVectorizerNew, self).fit_transform(raw_documents)
    self._tfidf.fit(X)
    self.tfidf_matrix = self._tfidf.transform(X, copy=False)
    return self.tfidf_matrix

# COMMAND ----------

kwargs = {
  "max_df": 0.8,
  "max_features": 20000,
  "min_df": 0.001,
  "stop_words": "english",
  "strip_accents": "ascii",
  "use_idf": True,
  "tokenizer": TokenizeAndStemer(text_column="SUBJECTS").tokenize_and_stem,
  "ngram_range": (1,3),
  "token_pattern": r"(?u)\b[\w-]+\b"
}
pre_processing_pipeline = Pipeline(
  [
    ("nr", NoneReplacer()),
    ("anrr", AnyNaNRowRemover()),
    ("tas", TokenizeAndStemer(text_column="SUBJECTS")),
    ("tfidf", TfidfVectorizerNew(**kwargs)),
  ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC Clustering Pipeline
# MAGIC -------------------
# MAGIC 
# MAGIC This pipeline performs the steps necessary to perform K-means clustering on natural language data stored as text including Tokenizing, Stemming, TF-IDF-Vectorizing and K-means Clustering.

# COMMAND ----------

clustering_pipeline = Pipeline(
  [
    ("km",KMeans(
      n_clusters=num_clusters,
      random_state=15)
    )
  ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC Latent Sentiment Analysis Pipeline
# MAGIC ----------------------------------

# COMMAND ----------

# class VectorCapture(TransformerMixin, BaseEstimator):
#   def __init__(self):
#     pass
  
#   def fit(self, X, y = None):
#     return self
  
#   def transform(self, X):
#     self.tfidf_matrix = X
#     return self.tfidf_matrix

# COMMAND ----------

# MAGIC %md 
# MAGIC Post-processing Pipeline
# MAGIC ------------------------
# MAGIC This pipeline appends the clusters to the original dataframe so that further analysis can be done and the attributes of each cluster can be investigated.
# MAGIC 
# MAGIC The top terms in each cluster are returned to the screen as an indicator of the topic/genre of each cluster.
# MAGIC 
# MAGIC In Future: Want to Onehot encode the cluster categories and set the top terms as the column names for each cluster.

# COMMAND ----------

class DataFrameRebuild(TransformerMixin, BaseEstimator):

  """
  Transformer concatinates the cleaned dataframe after the pre-processing and
  the cluster categores as a new column.
  """
    
  def fit(self, X, y=None):
    return self
  
  def transform(self, X):
    """
    Transform concatinates the cleaned dataframe after the pre-processing and
    the cluster categores as a new column and returns the new 
    """    
    self.clusters = clustering_pipeline.named_steps["km"].labels_.tolist()
    clusters_df = pd.DataFrame(self.clusters, columns=["clusters"])
    clusters_df = clusters_df.set_index(pre_processing_pipeline.get_params(True)["anrr"].cleaned_data.index)
    cleaned_data = pre_processing_pipeline.get_params(True)["anrr"].cleaned_data
    result = pd.concat([cleaned_data, clusters_df], axis=1)
    result = result.set_index(
      keys="clusters",
      drop=False,
      append=False,
      inplace=False,
      verify_integrity=False
    )
    # number of items per cluster (clusters from 0 to 4)
    print("The number of items per cluster are:")
    print(
      result.groupby("clusters", as_index=False)
      .count()
      .loc[:, "BIBNUM"]
      .rename(columns={"BIBNUM": "COUNT"}))
    return result

# COMMAND ----------

post_processing_pipeline = Pipeline(
  [
    ('dfr', DataFrameRebuild()),
    ('tt', TopTerms())
  ]
)

# COMMAND ----------

# MAGIC %md ## Model Pipeline

# COMMAND ----------

model_pipeline = Pipeline(
  [
    ('pre_p_pipe', pre_processing_pipeline),
#     ("PCA", SVD(n_components=1000)),
    ('c_pipe', clustering_pipeline),
    ('post_p_pipe', post_processing_pipeline)
  ]
)
result_df = model_pipeline.fit_transform(X = features)

# COMMAND ----------

# MAGIC %md 
# MAGIC Multidimentional Scaling
# MAGIC ------------------------
# MAGIC Note that the purpose of the MDS is to find a low-dimensional representation of the data (here 2D) in which the distances respect well the distances in the original high-dimensional space, unlike other manifold-learning algorithms, it does not seeks an isotropic representation of the data in the low-dimensional space. Here the manifold problem matches fairly that of representing a flat map of the Earth, as with map projection.

# COMMAND ----------

dist = 1 - cosine_similarity(pre_processing_pipeline.named_steps['tfidf'].tfidf_matrix)
# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]

# COMMAND ----------

# np.savetxt("mds.gz", pos)
# pos = np.loadtxt("mds.gz")  # To load data back in

# COMMAND ----------

# MAGIC %md ## PCA

# COMMAND ----------

svd = SVD()
cluster_pipe_reduced = svd.fit_transform(pre_processing_pipeline.named_steps['tfidf'].tfidf_matrix)

# COMMAND ----------

f = plt.figure(figsize=(10,7))
ax = f.add_subplot(111)
ax.scatter(cluster_pipe_reduced[:, 0], cluster_pipe_reduced[:, 1], c="y", s=50, edgecolor='k')
ax.set_title(
  "Truncated SVD reduction (2d) of transformed data (%dd)" %
  cluster_pipe_reduced.shape[1]
)
ax.set_xticks(())
ax.set_yticks(())
display(f)

# COMMAND ----------

# MAGIC %md ## Visualisation