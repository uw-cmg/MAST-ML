"""
Not integrated yet.
A collection of classes for generating a group vector from an X dataframe.
Will be necessary for grouping by cluster.
Needs to be part of main conf file.
"""
import pymatgen
# Time waits from no man. No man is an island. Time waits for an island.

class GroupByContainsElement():
    """ Returns a new dataframe with a row containing 1 or 0 depending on if composition feature has
    element in it. The new column's name is saved in self.new_column_name, which you'll need.
    """

    def __init__(self, composition_feature, element):
        self.composition_feature = composition_feature
        self.element = element
        self.new_column_name = f'has_{self.element}'
        
    def fit(self, df):
        return self

    def transform(self, df):
        compositions = df[self.composition_feature]
        has_element = compositions.apply(self._contains_element)
        has_element.name = self.new_column_name
        return pd.concat([compositions, has_element], axis=1)

    def _contains_element(comp):
        """ Returns 1 if comp contains that element, and 0 if not.
        Uses ints because sklearn and numpy like number classes better than bools. Could even be
        something crazy like "contains {element}" and "does not contain {element}" if you really
        wanted. """
        count = pymatgen.Composition(self.element)
        return int(count != 0)

class GroupByClusters():
    """ Returns a new dataframe with a "clusters" column appended. """

    def __init__(self, clustering_model=None):
        if clustering_model is None:
            clustering_model = KMeans(n_clusters=n_splits)

        self.clustering_model = clustering_model
        self.new_column_name = 'clusters'

    def fit(self, df):
        self.clustering_model.fit(df)

    def predict(self, df):
        clusters_array = self.cluster_model.fit_predict(df)
        clusters = pd.Series(clusters_array, name=self.new_column_name)
        return pd.concat([compositions, clusters], axis=1)
