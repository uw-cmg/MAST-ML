import random
from copy import copy

from sklearn.model_selection import train_test_split

def climb_hill(model_constructor, X, y, param_dict, score_func, num_steps=100, num_restarts=5):
    pairs = []
    for _ in range(num_restarts):
        best_score = -float('inf')
        best_params = {key: random.choice(values) for key,values in param_dict.items()}
        non_singleton_params = [param for param in param_dict if len(param_dict[param]) > 1]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        for step in range(100):
            print(f"Step {step}/100")
            if step % 10 == 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y)
            # get random subset of parameters
            params = copy(best_params)
            subset = random_subset(non_singleton_params)
            params.update((key, random.choice(param_dict[key])) for key in subset)
            model = model_constructor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = score_func(y_test, y_pred)
            if score > best_score:
                best_score = score
                best_params = params
        pairs.append((best_score, best_params))
    return max(pairs, key=lambda pair: pair[0])

def random_subset(list1):
    return random.sample(list1, random_power(len(list1)))

def random_power(n):
    """
    Random number from 1 to n.
    1 is the most likely, 2 is half as likely, 3 is a third as likely, etc
    """
    total = sum(1/i for i in range(1,n+1))
    cumulative_probability = 0
    r = random.random()
    for i in range(1,n+1):
        cumulative_probability += 1 / (i * total)
        if r <= cumulative_probability:
            return i

def test():
    #from hill_climbing import climb_hill
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from numpy import geomspace
    from pandas import read_csv

    df = read_csv("mnist_short.csv")
    df_train = df[df['testonly']==0]
    df_test = df[df['testonly']==1]
    X_train = df_train.drop(['digit', 'testonly'], axis=1)
    y_train = df_train['digit']
    X_test = df_test.drop(['digit', 'testonly'], axis=1)
    y_test = df_test['digit']

    param_dict = dict(
            n_neighbors=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            p=[1,2,3],
            leaf_size=[10,15,20,25,30,35,40],
            weights=['uniform'],)

    score, params = climb_hill(KNeighborsClassifier, X_train, y_train, param_dict, accuracy_score)
    knc = KNeighborsClassifier(**params)
    knc.fit(X_train, y_train)
    y_pred = knc.predict(X_test)
    print("FINAL SCORE: ", accuracy_score(y_test, y_pred))

#test()
