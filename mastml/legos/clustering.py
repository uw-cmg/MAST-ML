import sklearn.cluster as sc

name_to_constructor = {
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
