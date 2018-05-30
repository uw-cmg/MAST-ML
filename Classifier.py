
# coding: utf-8



import numpy
import sklearn, sklearn.neighbors, sklearn.svm
import matplotlib.pyplot as plt
import itertools


model = sklearn.neighbors.KNeighborsClassifier()
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    From http://scikit-learn.org/stable/_downloads/plot_confusion_matrix.py
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def make_cluster(x,y,sigma,count):
    return numpy.column_stack((numpy.random.normal(x,sigma,count),
                               numpy.random.normal(y,sigma,count)))

def example_run():
    numpy.random.seed(0)

    # Two clusters of points on the real plain
    # One centered at 3,4 and the other at 5,6
    # 1000 data points in the first group and 500 in the second
    group_0 = make_cluster(3, 4, 1, 1000)
    group_1 = make_cluster(4, 6, 1, 600)
    group_2 = make_cluster(8, 6, 2, 400)

    # plot a subset of the data:
    plt.scatter(group_0[:200,0], group_0[:200,1], c='blue')
    plt.scatter(group_1[:200,0], group_1[:200,1], c='red')
    plt.scatter(group_2[:200,0], group_2[:200,1], c='green')
    plt.show()

    perm = numpy.random.permutation(2000)
    X = numpy.row_stack((group_0, group_1, group_2))[perm]
    y = numpy.array([0]*1000 + [1]*600 + [2]*400)[perm]
    X_train, y_train = X[:1700], y[:1700]
    X_test, y_test = X[1700:], y[1700:]

    #model = sklearn.gaussian_process.GaussianProcessClassifier()

    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(confusion_matrix, [0,1,2])

    # # Precision-Recall Curve

    perm = numpy.random.permutation(1600)
    X = numpy.row_stack((group_0, group_1))[perm]
    y = numpy.array([0]*1000 + [1]*600)[perm]
    X_train, y_train = X[:1100], y[:1100]
    X_test, y_test = X[1100:], y[1100:]

    model = sklearn.svm.LinearSVC()
    model.fit(X_train, y_train)
    y_score = model.decision_function(X_test)

    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_score)

    plt.step(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

