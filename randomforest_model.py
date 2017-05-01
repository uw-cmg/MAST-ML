from sklearn.ensemble import RandomForestRegressor
import configuration_parser
import ast
__author__ = 'hao yuan'


def get():
    from warnings import warn
    warn('The usage of randomforest_model.py will be deprecated', DeprecationWarning)
    config = configuration_parser.parse()
    estimators = config.getint(__name__, 'estimators')
    exec('max_depth = ' + config.get(__name__, 'max_depth'),locals(),globals())
    min_samples_split = config.getint(__name__, 'min_samples_split')
    min_samples_leaf = config.getint(__name__, 'min_samples_leaf')
    exec('max_leaf_nodes=' + config.get(__name__, 'max_leaf_nodes'),locals(),globals())
    jobs = config.getint(__name__, 'jobs')

    model = RandomForestRegressor(n_estimators=estimators,
                               max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               max_leaf_nodes=max_leaf_nodes,
                               n_jobs=jobs)
    return model

def __executeStringNoneOrNumber__(readFromConfig,varAssignment):
    try:
        int(readFromConfig)
        return varAssignment + '=' + readFromConfig  # a number
    except:
        try:
            float(readFromConfig)
            return varAssignment + '=' + readFromConfig  # a number
        except:
            try:
                if exec(readFromConfig) == None: return varAssignment + '=' + readFromConfig  # None
            except:
                return varAssignment + '= \' ' + readFromConfig + '\''  # string

def __executeStringOrInt__(readFromConfig,varAssignment):
    try:
        int(readFromConfig)
        return varAssignment + '=' + readFromConfig  # a number
    except:
        return varAssignment + '= \' ' + readFromConfig + '\''  # string