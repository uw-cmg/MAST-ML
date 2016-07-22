from sklearn.kernel_ridge import KernelRidge
import configuration_parser

__author__ = 'haotian'


def get():
    config = configuration_parser.parse()
    alpha = config.getfloat(__name__, 'alpha')
    gamma = config.getfloat(__name__, 'gamma')
    kernel = config.get(__name__, 'kernel')
    return KernelRidge(alpha=alpha,gamma=gamma,kernel=kernel)