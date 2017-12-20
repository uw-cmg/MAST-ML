import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import xlrd
import math

num_batches = 1
num_points = 1000
fraction_to_leave_out = 0.80
number_of_epochs = 1000
num_iterations = 2
learning_rate = 0.1
output_col = 0
workbook = xlrd.open_workbook('TestSheet.xlsx')
sheet_names = workbook.sheet_names()
sheet = workbook.sheet_by_name(sheet_names[0])
ncols = 5

def initialize_data():

    data = []
    col_name = []
    row_name = []
    for col_idx in range(ncols):
        col = []
        for row_idx in range(num_points+1):
            cell = sheet.cell(row_idx, col_idx)
            if (row_idx == 0):
                col_name.append(cell.value)
            else:
                col.append(cell.value)
        if (col_idx) == 0:
            row_name.extend(col)
        else:
            data.append(col)

    return data

def make_training_data(data):
    random_indices_chosen = []
    count = 0

    xTests = [[] for x in range(ncols - 2)]
    xTrains = [[] for x in range(ncols - 2)]
    yTest = []
    yTrain = []

    while (count < fraction_to_leave_out*num_points):
        random_index = np.random.randint(0, num_points)
        if (random_index not in random_indices_chosen):
            output_found = False
            for col in range (0, len(data)):
                if (col == output_col):
                    yTest.append(data[col][random_index])
                    output_found = True
                else:
                    if (output_found == False):
                        xTests[col].append(data[col][random_index])
                    else:
                        xTests[col-1].append(data[col][random_index])
            count += 1
            random_indices_chosen.append(random_index)
        else:
            random_index = np.random.randint(0, num_points)

    for index in range(num_points):
        if (index not in random_indices_chosen):
            skipped = True
            for col in range (0, len(data)):
                if col == output_col:
                    yTrain.append(data[output_col][index])
                    skipped = True
                else:
                    if (skipped == False):
                        xTrains[col].append(data[col][index])
                    else:
                        xTrains[col-1].append(data[col][index])

    return xTrains, yTrain, xTests, yTest

def make_array_from_list(xTrains, yTrain, xTests, yTest):

    xTrains_array = []
    for train in xTrains:
        xTrains_array.append(np.asarray(train).reshape(1,-1))

    xTests_array =[]
    for test in xTests:
        xTests_array.append(np.asarray(test).reshape(1,-1))

    yTrain_array = np.asarray(yTrain).reshape([1, -1])
    yTest_array = np.asarray(yTest).reshape([1, -1])

    return xTrains_array, yTrain_array, xTests_array, yTest_array

def launch_tensorflow(batched_xTrains, batched_yTrain, xTests, yTest, data, xTrains, yTrain):
    placeholders = []
    for i in range(0, len(batched_xTrains)):
        placeholders.append(tf.placeholder(tf.float32, [1, None]))

    W = tf.Variable(tf.truncated_normal(shape=[10, 1], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[10, 1]))

    W2 = tf.Variable(tf.truncated_normal(shape=[1, 10], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[1]))

    temp = tf.matmul(W,placeholders[0])
    for p in range(1, len(placeholders)):
        temp += tf.matmul(W, placeholders[p])
    hidden_layer = tf.nn.sigmoid(temp)

    y = tf.matmul(W2, hidden_layer) + b2

    cost = tf.reduce_mean(tf.square(y - batched_yTrain))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    feed_dict = {}
    for index in range(0, len(batched_xTrains)):
        feed_dict[placeholders[index]] = batched_xTrains[index]

    for epoch in range(number_of_epochs + 1):
        sess.run(optimizer, feed_dict=feed_dict)
        if epoch % 100 == 0:
            print ("Epoch number", epoch, "Training set RMSE:", cost.eval(feed_dict, sess))

    yFit_feed = {}
    test_yFit_feed = {}
    train_yFit_feed = {}

    for i in range (0, len(placeholders)):
        yFit_feed[placeholders[i]] = np.asarray(data[1:][i]).reshape([1, -1])
        test_yFit_feed[placeholders[i]] = np.asarray(xTests[i]).reshape([1, -1])
        train_yFit_feed[placeholders[i]] = np.asarray(xTrains[i]).reshape([1, -1])
    test_yFit =  y.eval(test_yFit_feed, sess)
    train_yFit = y.eval(train_yFit_feed, sess)
    yFit =      np.append(test_yFit, train_yFit)
    residual_test = test_yFit - yTest
    rmse_test = np.sqrt((np.sum(residual_test**2)) / len(test_yFit))

    print ("Testing set RMSE:", rmse_test)

    return yFit, rmse_test, test_yFit, train_yFit

def plot_data(label, xTrains_list, yTrain_list, xTests_list, yTest_list, yFit, data, test_yFit, train_yFit):

    yData = data[output_col]

    x = tf.placeholder(tf.float32, [1, None])
    yFit_list = yFit.transpose().tolist()
    test_yFit_list =  np.asarray(test_yFit.transpose().tolist()).flatten().tolist()
    train_yFit_list = np.asarray(train_yFit.transpose().tolist()).flatten().tolist()

    residual_test = test_yFit -yTest_list
    rmse_test = np.sqrt((np.sum(residual_test**2)) / len(test_yFit))


    fig1 = plt.figure()

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ax2 = fig1.add_subplot(111)
    plt.plot(train_yFit_list, yTrain_list, 'ro', label='training data')
    plt.plot(test_yFit_list,  yTest_list, 'bo', label='test data')
    ax2.set_xlabel('PredictedY')
    ax2.set_ylabel('ActualY')
    ax2.set_title(label + ' ActualY vs PredictedY')
    textstr1='Testing$r-squared=%.4f$'%(polyfit(yFit_list, yData[:len(yData)], 1)['determination'])
    textstr1 = textstr1, '$RMSE=%.4f$'%(rmse_test)
    ax2.text(right, top, textstr1,
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax2.transAxes)
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    ax2.legend()
    plt.draw()
    plt.savefig(label + '_actual_vs_pred.png')

    fig2 = plt.figure()
    yFit_list = np.asarray(yFit_list).flatten().tolist()
    ax4 = fig2.add_subplot(111)
    n, bins, patches = plt.hist([yData, yFit_list], 10, normed=1, histtype='bar', color=['orange', 'red'],label=['Actual','Predicted'])
    textstr1='Predicted\n$Min=%.10f$\n$Max=%.10f$\n$Mean=%.10f$\n$Std Dev=%.10f$'%(min(yFit_list),max(yFit_list),np.mean(yFit_list),np.std(yFit_list))
    textstr2='Actual\n$Min=%.10f$\n$Max=%.10f$\n$Mean=%.10f$\n$Std Dev=%.10f$'%(min(yData),max(yData),np.mean(yData),np.std(yData))
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    ax4.text(0.05,0.95,textstr1,transform=ax4.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    ax4.text(0.05,0.60,textstr2,transform=ax4.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    ax4.legend()
    ax4.set_title(label + ' Predicted Compared with Actual Histogram')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Frequency')
    plt.legend()
    plt.draw()
    plt.savefig(label + '_histogram.png')

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for row in range (0, num_points - 1):
        if (yData[row] < 40 and yFit_list[row] < 40):
            true_positive += 1
        elif(yFit_list[row] < 40):
            false_positive += 1
        elif(yData[row] >=40 and yFit_list[row] >= 40):
            true_negative += 1
        elif(yFit_list[row] > 40):
            false_negative += 1

    conf_arr = [[true_positive,false_positive],
                [true_negative,false_negative]]
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(1.0)
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = np.array(conf_arr).shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    cb = fig.colorbar(res)
    plt.xticks(range(width), ['Actually Stable', 'Actually Not Stable'])
    plt.yticks(range(height), ['Pred Stable','Pred Not Stable'])
    plt.savefig('confusion_matrix.png', format='png')

def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    results['polynomial'] = coeffs.tolist()

    p = np.poly1d(coeffs)
    yhat = p(x)                        
    ybar = np.sum(y)/len(y)          
    ssreg = np.sum((yhat-ybar)**2)   
    sstot = np.sum((y - ybar)**2)   
    results['determination'] = ssreg / sstot

    return results

def main():

    data = initialize_data()

    xTrains, yTrain, xTests, yTest = make_training_data(data)
    xTrains_array, yTrain_array, xTests_array, yTest_array = make_array_from_list(xTrains, yTrain, xTests, yTest)

    min_xTrains, min_yTrain, min_xTests, min_yTest = xTrains, yTrain, xTests, yTest
    min_xTrains_array, min_yTrain_array, min_xTests_array, min_yTest_array = xTrains_array, yTrain_array, xTests_array, yTest_array
    min_error = float("inf")
    min_yFit = None
    min_test_yFit = None
    min_train_yFit = None
    max_xTrains, max_yTrain, max_xTests, max_yTest = xTrains, yTrain, xTests, yTest
    max_xTrains_array, max_yTrain_array, max_xTests_array, max_yTest_array = xTrains_array, yTrain_array, xTests_array, yTest_array
    max_error = float("-inf")
    max_yFit = None
    max_test_yFit = None
    max_train_yFit = None

    rmses = []

    print()
    print()
    for x in range(0,num_iterations):
        print("Iteration Number: ", x)
        batch_length = math.floor(len(yTrain_array[0])/num_batches)
        batch_start_index = 0
        batch_end_index = 0

        for batch in range(num_batches):
            print("Batch Number: ", batch)
            batch_start_index = batch * batch_length
            if ((batch + 1) * batch_length > len(xTrains[0]) - 1):
                batch_end_index = len(xTrains[0]) - 1
            else:
                batch_end_index = (batch + 1) * batch_length
            batched_xTrains_array = []
            for i in range(len(xTrains)):
                batched_xTrains_array.append([xTrains[i][batch_start_index:batch_end_index]])
            batched_yTrain_array = yTrain_array[0][batch_start_index:batch_end_index]
            yFit, rmse_test, test_yFit, train_yFit = launch_tensorflow(batched_xTrains_array, batched_yTrain_array, xTests_array, yTest_array, data, xTrains_array, yTrain_array)
            rmses.append(rmse_test)
            if rmse_test < min_error:
            	min_error = rmse_test
            	min_yFit = yFit.copy()
            	min_test_yFit = test_yFit.copy()
            	min_train_yFit = train_yFit.copy()
            	min_xTrains, min_yTrain, min_xTests, min_yTest = xTrains.copy(), yTrain.copy(), xTests.copy(), yTest.copy()

            if rmse_test > max_error:
            	max_error = rmse_test
            	max_yFit = yFit.copy()
            	max_test_yFit = test_yFit.copy()
            	max_train_yFit = train_yFit.copy()
            	max_xTrains, max_yTrain, max_xTests, max_yTest = xTrains.copy(), yTrain.copy(), xTests.copy(), yTest.copy()


        xTrains, yTrain, xTests, yTest = make_training_data(data)
        xTrains_array, yTrain_array, xTests_array, yTest_array = make_array_from_list(xTrains, yTrain, xTests, yTest)
        print()
        print()

    for x in range(0,len(rmses)):
    	 'Iteration ', x, ': ', rmses[x]
    print ('Best RMSE: ', min_error)
    print ('Worst RMSE: ', max_error)
    print ("Mean RMSE: ", np.array(rmses).mean())
    print ("Std RMSE: ", np.array(rmses).std())
    print()

    plot_data('Best', min_xTrains, min_yTrain, min_xTests, min_yTest, min_yFit, data, min_test_yFit, min_train_yFit)
    plot_data('Worst', max_xTrains, max_yTrain, max_xTests, max_yTest, max_yFit, data, max_test_yFit, max_train_yFit)

main()
