import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class Domain():

    def distance(self, X_train, X_test, metric, **kwargs):
        if metric == 'mahalanobis':
            m = np.mean(X_train)
            X_train_transposed = np.transpose(X_train)
            covM = np.cov(X_train_transposed)
            invCovM = np.linalg.inv(covM)
            max_distance_train = cdist(XA=[m], XB=X_train, metric=metric, VI=invCovM).max()

            #Do the same for X_test
            centroid_dist = cdist(XA=[m], XB=X_test, metric=metric, VI=invCovM)[0];
            #Check every test datapoint to see if they are in or out of domain
            inDomain = []
            for i in centroid_dist:
                if pd.isna(i):
                    inDomain.append("nan")
                elif i < max_distance_train:
                    inDomain.append(True)
                else:
                    inDomain.append(False)

            return pd.DataFrame(inDomain)

        else:
            m = np.mean(X_train)
            max_distance_train = cdist(XA=[m], XB=X_train, metric=metric, **kwargs).max()
            centroid_dist = cdist(XA=[m], XB=X_test, metric=metric, **kwargs)[0];
            inDomain = []
            for i in centroid_dist:
                if pd.isna(i):
                    inDomain.append("nan")
                elif i < max_distance_train:
                    inDomain.append(True)
                else:
                    inDomain.append(False)

            return pd.DataFrame(inDomain)