from sklearn.ensemble import RandomForestRegressor

__author__ = 'hao yuan'


def get():
    model = RandomForestRegressor(n_estimators=100,
                               max_features='auto',
                               max_depth=5,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               min_weight_fraction_leaf=0,
                               max_leaf_nodes=None,
                               n_jobs=1)
    return model