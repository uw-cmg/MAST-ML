"""
Not integrated yet.
A collection of classes for generating a group vector from an X dataframe.
Will be necessary for grouping by cluster.
Needs to be part of main conf file.
"""

import pandas as pd
import pymatgen

from sklearn.cluster import KMeans

class GroupByContainsElement():
    """
    Returns a new dataframe with a row containing 1 or 0 depending on if composition feature has
    element in it. The new column's name is saved in self.new_column_name, which you'll need.
    """

    def __init__(self, composition_feature, element):
        self.composition_feature = composition_feature
        self.element = element
        self.new_column_name = f'has_{self.element}'

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        import pdb; pdb.set_trace()
        compositions = df[self.composition_feature]
        has_element = compositions.apply(self._contains_element)
        #has_element.name = self.new_column_name
        #df.add(has_element)
        #df = df.copy()
        #df.loc[:,self.new_column_name] = has_element
        #return df
        return pd.DataFrame(has_element, columns=[self.new_column_name])

    def _contains_element(self, comp):
        """
        Returns 1 if comp contains that element, and 0 if not.
        Uses ints because sklearn and numpy like number classes better than bools. Could even be
        something crazy like "contains {element}" and "does not contain {element}" if you really
        wanted.
        """
        count = pymatgen.Composition(self.element)
        return int(count != 0)

class GroupByClusters():
    " Returns a new dataframe with a 'clusters' column appended. "

    def __init__(self, clustering_model=None, n_clusters=5):
        if clustering_model is None:
            clustering_model = KMeans(n_clusters=n_clusters)
        else:
            raise NotImplemented('Cannot take a clustering_model yet.')

        self.clustering_model = clustering_model
        self.new_column_name = 'clusters'

    def fit(self, df, y=None):
        self.clustering_model.fit(df)

    def transform(self, df, y=None):
        clusters_array = self.clustering_model.fit_predict(df)
        #clusters = pd.Series(clusters_array, name=self.new_column_name)
        #df.add(clusters)
        #df = df.copy()
        #df.loc[:,self.new_column_name] = clusters_array
        return pd.DataFrame(clusters_array, columns=[self.new_column_name])
