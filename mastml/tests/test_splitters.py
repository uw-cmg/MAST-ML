from mastml.legos.data_splitters import LeaveCloseCompositionsOut
import pytest


def test_close_comps():
    # Make entries at a 10% spacing
    X = ['Al{}Cu{}'.format(i, 10-i) for i in range(11)]

    # Generate test splits with a 5% distance cutoff
    splitter = LeaveCloseCompositionsOut(dist_threshold=0.05)
    train_inds, test_inds = zip(*splitter.split(X))
    assert train_inds[0].tolist() == list(range(1, 11))
    assert list(test_inds)== [[i] for i in range(11)]  # Only one point

    # Generate test splits with 25% distance cutoff
    splitter.dist_threshold = 0.25
    splitter.nn_kwargs = {'metric': 'l1'}
    train_inds, test_inds = zip(*splitter.split(X))
    assert train_inds[0].tolist() == list(range(2, 11))  # 1 is too close
    assert train_inds[1].tolist() == list(range(3, 11))  # 0 and 2 are too close