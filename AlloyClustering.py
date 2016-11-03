from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
import numpy as np
import data_parser

def AlloyClustering(k):
    alloy_data = data_parser.parse("../../AlloyComps.csv")
    data = np.asarray(alloy_data.get_data(["Cu","Ni","Mn","P","Si","C"]))
    #est = KMeans(n_clusters=k)
    #est = AgglomerativeClustering(n_clusters = k)
    est = AffinityPropagation()
    est.fit(data)

    labels = est.labels_
    '''print(len(labels))
    for i in range(k):
        print("Cluster #{}".format(i))
        print(np.asarray(alloy_data.get_data("Alloy"))[np.where(labels == i)])
        print()'''

    return (labels,alloy_data)





