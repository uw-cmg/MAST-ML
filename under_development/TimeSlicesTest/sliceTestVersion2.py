import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


# imports csv file
f = open("DBTT_Data9.csv")
title = f.readline().rstrip().split(',')
data = np.loadtxt(f, delimiter = ',')
dataTemp = data[:, np.newaxis]
# converts each column in the excell file into an array
XCu = (dataTemp[:, :, title.index("N(Cu)")])
XNi = (dataTemp[:, :, title.index("N(Ni)")])
XMn = (dataTemp[:, :, title.index("N(Mn)")])
XP  = (dataTemp[:, :, title.index("N(P)")])
XSi = (dataTemp[:, :, title.index("N(Si)")])
XC  = (dataTemp[:, :, title.index("N(C)")])
XFl = (dataTemp[:, :, title.index("N(log(fluence)")])
XFx = (dataTemp[:, :, title.index("N(log(flux)")])
XT  = (dataTemp[:, :, title.index("N(Temp)")])
Y = (dataTemp[:, :, title.index("delta sigma")]) #original Y0  #check: len(Y0)
X  = np.concatenate((XCu, XNi, XMn, XP, XSi, XC, XFl, XFx, XT), axis=1)


#Get the index and element of folder 1
f1 = open("f1.csv")
data1 = np.loadtxt(f1, delimiter=',')
dataTemp1 = data1[:, np.newaxis]
index1 = list(np.array(dataTemp1).reshape(-1,))
index1 = [int(i-1) for i in index1]

#Get the index and element of folder 2
f2 = open('f2.csv')
data2 = np.loadtxt(f2, delimiter=',')
dataTemp2 = data2[:, np.newaxis]
index2 = list(np.array(dataTemp2).reshape(-1,))
index2 = [int(i-1) for i in index2]

#Get the index and element of folder 3
f3 = open("f3.csv")
data3 = np.loadtxt(f3, delimiter=',')
dataTemp3 = data3[:, np.newaxis]
index3 = list(np.array(dataTemp3).reshape(-1,))
index3 = [int(i-1) for i in index3]

#Get the index and element of folder 4
f4 = open("f4.csv")
data4 = np.loadtxt(f4, delimiter=',')
dataTemp4 = data4[:, np.newaxis]
index4 = list(np.array(dataTemp4).reshape(-1,))
index4 = [int(i-1) for i in index4]

#Get the index and element of folder 5
f5 = open("f5.csv")
data5 = np.loadtxt(f5, delimiter=',')
dataTemp5 = data5[:, np.newaxis]
index5 = list(np.array(dataTemp5).reshape(-1,))
index5 = [int(i-1) for i in index5]


#Combine the index
index = [index1, index2, index3, index4, index5]
index = np.asarray(index)


#Model:
a = .00139;y = .518
model = KernelRidge(alpha= a, gamma=y, kernel='rbf')



#Loop:
rmse_list = []
num_folds = 5
Overall_Y_Pred = np.zeros(len(X))

for i in [t+1 for t in list(range(4))]:
    to_exclude = list(range(i))
    folder_train = np.asarray(to_exclude).astype(int)
    #train index starts with the first folder
    index_train = index[folder_train];
    index_test = [element for k, element in enumerate(index) if k not in to_exclude]
    #train set starts with the first folder
    X_train = X[np.hstack(index_train)]
    Y_train = Y[np.hstack(index_train)]
    model.fit(X_train, Y_train)
    rmse_folder = []
    for item in index_test:
        folder_X = X[item]
        folder_Y = Y[item]
        # train on training sets
        Y_test_Pred = model.predict(folder_X)
        rmse = np.sqrt(mean_squared_error(folder_Y, Y_test_Pred))
        rmse_folder.append(rmse)
    print(rmse_folder)
    rmse_list.append(rmse_folder)



print(rmse_list)
print("\n")


#Plot
sub = [221, 222, 223, 224]
for i in list(range(4)):
    y = np.asarray(rmse_list[i])
    x = np.asarray([t+1 for t in list(range(len(y)))])
    plt.subplot(sub[i])
    plt.plot(x, y, x, y, 'rs')
    plt.axis([1, 4, 20, 70])
    plt.xlabel('Number of Folders in Training Set')
    plt.ylabel('Overall RMSE of Test Set')
    plt.grid(True)

plt.show()
f.close()






