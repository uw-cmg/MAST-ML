from sklearn.kernel_ridge import KernelRidge
import configuration_parser
__author__ = 'haotian'


def get():
    config = configuration_parser.parse()
    alpha = config.getfloat(__name__, 'alpha')
    coef0 = config.getint(__name__, 'coef0')
    degree = config.getint(__name__, 'degree')
    gamma = config.getfloat(__name__, 'gamma')
    kernel = config.get(__name__, 'kernel')
    model = KernelRidge(alpha=alpha, coef0=coef0, degree=degree, gamma=gamma, kernel=kernel, kernel_params=None)
    return model
