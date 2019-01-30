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

import numpy as np
import pandas as pd
import nltk
import more_itertools
import re
import mpld3
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as SVD

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
  .option("dbtable", "LIBRARY_COLLECTION_INVENTORY") \
  .load() \
  .limit(1000)
features = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Preview the Data
# MAGIC ----------------

# COMMAND ----------

display(df.head(2))

# COMMAND ----------

# MAGIC %md 
# MAGIC Set Initial Parameters
# MAGIC ----------------------

# COMMAND ----------

nltk.download('stopwords')  # Common words to ignore.
nltk.download('punkt')  # Punkt Sentence Tokenizer - more useful in large documents.
stopwords = pd.DataFrame(nltk.corpus.stopwords.words('english'))  # Load nltk's English stopwords.
display(stopwords[:10])
# Snowball is a small string processing language designed for creating stemming algorithms
# for use in Information Retrieval.
stemmer = SnowballStemmer("english")  # Load nltk's SnowballStemmer as variabled 'stemmer'.
num_clusters = 5  # Define the number of clusters to be used later in the model

# COMMAND ----------

# MAGIC %md 
# MAGIC Pre-processing Pipeline
# MAGIC -----------------------
# MAGIC This pipeline processes the original dataframe so that the datatypes in the columns of interest are correct. It removes both empty columns and rows containing NaN values.
# MAGIC 
# MAGIC Note: Be careful to return a dataframe containing the same indexes and columns

# COMMAND ----------

class NoneReplacer(TransformerMixin, BaseEstimator):
  
  """
  Transformer fills Nonetype values with pd.np.nan.
  """

  def fit(self, X, y=None):
    return self
      
  def transform(self, X):
    assert isinstance(X, pd.DataFrame)
    X.fillna(value=pd.np.nan, inplace=True)
    return pd.DataFrame(X, index=X.index, columns=X.columns)

# COMMAND ----------

class AnyNaNRowRemover(TransformerMixin, BaseEstimator):
  
  """
  Transformer removes any rows where here any element in row is NaN.
  Note: This is not an appropriate technique for processing missing values in all use cases.
  """
        
  def fit(self, X, y=None):
    return self
  
  def transform(self, X):
    """
    Transform drops rows where any element in row is NaN 
    """
    assert isinstance(X, pd.DataFrame)
    X = X.dropna(axis=0, how='any')
    self.cleaned_data = X
    return pd.DataFrame(X, index=X.index, columns=X.columns)

# COMMAND ----------

class TokenizeAndStemer(TransformerMixin, BaseEstimator):
  
  """
  This transformer combines two processes:
  * Tokenizing: Split sentences down into individual words.
  * Stemming: The process of breaking a word down into its root.
  
  The transformer tokenizes and stems the words, creating a vocab dataframe 
  containing the words and their stems.
  """
  
  def __init__(self, text_column):
    self.text_column = text_column
    pass
  
  def tokenize_and_stem(self, text):
    """
    Function first Tokenizes and then Stems the words, returning the stems.
    """
    tokens = []
    for sent in nltk.sent_tokenize(text):
      tokens.extend([word for word in nltk.word_tokenize(sent)])
    filtered_tokens = []
    for token in tokens:
      if re.search('[a-zA-Z]', token):
        filtered_tokens.append(token)
      else:
        token
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
  
  def tokenize_only(self, text):
    """
    Function Tokenizes the words, returning tokens that are filtered to contain upper or
    lowwer case letters.
    """ 
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = []
    for sent in nltk.sent_tokenize(text):
      tokens.extend([word.lower() for word in nltk.word_tokenize(sent)])
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    filtered_tokens = []
    for token in tokens:
      if re.search('[a-zA-Z]', token):
        filtered_tokens.append(token)
      else:
        token
    return filtered_tokens
  
  def fit(self, X, y=None):
    return self
 
  def transform(self, X):
    """
    Transform applies both of the custom functions defined above.
    
    The tokenize_and_stem function us used to create a flat list 
    containing the entirity of the vocab in stemmed form. 
    
    The tokenize_only is used to create a flat list containing the
    entirity of the vocab as a flat list.
    
    A dataframe is created from both of these flat lists with the
    "words" column consisting of the individual words and the index
    consisting of the stems.
    
    Returns the feature of interest as a list
    """
    totalvocab_stemmed = []
    # for each item in 'synopses', tokenize/stem
    allwords_stemmed = [self.tokenize_and_stem(str(i)) for i in X[self.text_column].tolist()]
    totalvocab_stemmed.extend(allwords_stemmed)
    totalvocab_stemmed = list(more_itertools.collapse(totalvocab_stemmed))
    totalvocab_tokenized = []
    # for each item in 'synopses', tokenize/stem
    allwords_tokenized = [self.tokenize_only(str(i)) for i in X[self.text_column].tolist()]
    totalvocab_tokenized.extend(allwords_tokenized)
    totalvocab_tokenized = list(more_itertools.collapse(totalvocab_tokenized))        
    self.vocab_frame = pd.DataFrame({"words": totalvocab_tokenized}, index=totalvocab_stemmed)
    print("there are", str(self.vocab_frame.shape[0]), "items in vocab_frame\n")
    X = X[self.text_column].tolist()
    return X

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

class TopTerms(TransformerMixin, BaseEstimator):
  
  """
  Transformer prints out words closest to centre of each cluster.
  """
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, X):
    """
    Transform returns the terms the tfidf matrix represents.
    Orders the cluster centers by proximity to centroid.
    Prints terms by proximity to centroid.
    """
    print("\nTop terms per cluster:")
    # the terms tfidf matrix represents
    terms = pre_processing_pipeline.named_steps["tfidf"].get_feature_names()
    # sort cluster centers by proximity to centroid
    order_centroids = pd.DataFrame(
      clustering_pipeline.
      named_steps["km"].
      cluster_centers_.
      argsort()[:, ::-1]
    )
    self.top_terms_list = []
    for cluster in np.arange(num_clusters):
      print("\ncluster %d words: " % cluster, end="")
      pre_processing_pipeline.get_params(True)["tas"].vocab_frame
      top_terms = []
      max_index = 6
      for count, ind in enumerate(order_centroids.iloc[cluster, :max_index]):
        if count == max_index-1:
          print(
            "\n" + str(count),
            " %s" % pre_processing_pipeline.
            get_params(True)["tas"].
            vocab_frame.
            loc[terms[ind].split(' ')].
            values.tolist()[0][0].
            encode("utf-8", "ignore")
          ) 
          top_terms.append(
            (
              " %s" % pre_processing_pipeline.
              get_params(True)["tas"].
              vocab_frame.
              loc[terms[ind].split(' ')].
              values.tolist()[0][0]
            ).strip()
          )
        else:
          print(
            "\n" + str(count),
            " %s" % pre_processing_pipeline.
            get_params(True)["tas"].
            vocab_frame.
            loc[terms[ind].split(' ')].
            values.tolist()[0][0].
            encode("utf-8", "ignore"),
            end=','
          )
          top_terms.append(
            (
              " %s" % pre_processing_pipeline.
              get_params(True)["tas"].
              vocab_frame.
              loc[terms[ind].split(" ")].
              values.tolist()[0][0] + ","
            ).strip()
          )
      self.top_terms_list.append("".join(top_terms))
    return X

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
cluster_description = post_processing_pipeline.named_steps['tt'].top_terms_list
cluster_names = dict(zip(cluster_numbers, cluster_description))

# COMMAND ----------

# create data frame that has the result of the MDS plus the cluster numbers and titles
clusters = post_processing_pipeline.named_steps['dfr'].clusters
titles = pre_processing_pipeline.get_params(True)['anrr'].cleaned_data.TITLE
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 
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