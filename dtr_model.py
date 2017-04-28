import configuration_parser
import sklearn.tree as tree

def get():
    from warnings import warn
    warn('Usage of dtr_model.py will be deprecated', DeprecationWarning)
    config = configuration_parser.parse()
    exec('max_depth=' + config.get(__name__, 'max_depth'),locals(),globals())
    min_samples_split = config.getint(__name__, 'min_samples_split')
    min_samples_leaf = config.getint(__name__, 'min_samples_leaf')
    criterion = config.get(__name__, 'split criterion')
    return tree.DecisionTreeRegressor(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
	                                  min_samples_split=min_samples_split)