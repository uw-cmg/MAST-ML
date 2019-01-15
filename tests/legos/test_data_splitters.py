from mastml.legos.data_splitters import LeaveCloseCompositionsOut, LeaveOneClusterOut
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from unittest import TestCase


class TestSplitters(TestCase):

    def test_close_comps(self):
        # Make entries at a 10% spacing
        X = ['Al{}Cu{}'.format(i, 10-i) for i in range(11)]

        # Generate test splits with a 5% distance cutoff
        splitter = LeaveCloseCompositionsOut(dist_threshold=0.05)
        train_inds, test_inds = zip(*splitter.split(X))
        self.assertEqual(train_inds[0].tolist(), list(range(1, 11)))  # Everything but 0
        self.assertEqual(list(test_inds), [[i] for i in range(11)])  # Only one point

        # Generate test splits with 25% distance cutoff
        splitter.dist_threshold = 0.25
        splitter.nn_kwargs = {'metric': 'l1'}
        train_inds, test_inds = zip(*splitter.split(X))
        self.assertEqual(train_inds[0].tolist(), list(range(2, 11)))  # 1 is too close
        self.assertEqual(train_inds[1].tolist(), list(range(3, 11)))  # 0 and 2 are too close

    def test_loco_cv(self):
        # Make clusters of entries
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, y = make_blobs(n_samples=30, centers=centers, cluster_std=0.1, random_state=0)

        # Make the LOCO CV tool
        tool = LeaveOneClusterOut(KMeans(n_clusters=3))

        # Make sure the number of clusters is 3
        self.assertEqual(3, tool.get_n_splits(X))

        # Make sure the clusters are correct
        for train_ind, test_ind in tool.split(X):
            self.assertEqual(2, len(set(y[train_ind])))  # two clusters in train
            self.assertEqual(1, len(set(y[test_ind])))  # one in the test set
