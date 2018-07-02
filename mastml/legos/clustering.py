

#class GroupByClusters():
#    " Returns a new dataframe with a 'clusters' column appended. "
#
#    def __init__(self, clustering_model=None, n_clusters=5):
#        if clustering_model is None:
#            clustering_model = KMeans(n_clusters=n_clusters)
#        else:
#            raise NotImplemented('Cannot take a clustering_model yet.')
#
#        self.clustering_model = clustering_model
#        self.new_column_name = 'clusters'
#
#    def fit(self, df, y=None):
#        self.clustering_model.fit(df)
##
#    def transform(self, df, y=None):
#        clusters_array = self.clustering_model.fit_predict(df)
#        #clusters = pd.Series(clusters_array, name=self.new_column_name)
#        #df.add(clusters)
#        #df = df.copy()
#        #df.loc[:,self.new_column_name] = clusters_array
#        return pd.DataFrame(clusters_array, columns=[self.new_column_name])


import sklearn.cluster as sc
from .util_legos import DoNothing




name_to_constructor = {
    'DoNothing': DoNothing,
    'AffinityPropagation': sc.AffinityPropagation, # Perform Affinity Propagation Clustering of data.
    'AgglomerativeClustering': sc.AgglomerativeClustering, # Agglomerative Clustering
    'Birch': sc.Birch, # Implements the Birch clustering algorithm.
    'DBSCAN': sc.DBSCAN, # Perform DBSCAN clustering from vector array or distance matrix.
    'FeatureAgglomeration': sc.FeatureAgglomeration, # Agglomerate features.
    'KMeans': sc.KMeans, # K-Means clustering
    'MiniBatchKMeans': sc.MiniBatchKMeans, # Mini-Batch K-Means clustering
    'MeanShift': sc.MeanShift, # Mean shift clustering using a flat kernel.  
    'SpectralClustering': sc.SpectralClustering, # Apply clustering to a projection to the normalized laplacian.
    }

