# Hide benign warnings
# https://github.com/numpy/numpy/pull/432/commits/170ed4e?diff=split
import warnings
warnings.filterwarnings("ignore", message=r".*numpy\.dtype size changed.*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
