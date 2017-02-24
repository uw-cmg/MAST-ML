import numpy as np
import data_parser
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.kernel_ridge import KernelRidge
import random


# The two Alloy descriptors are placeholders for the 2 effective fluences that will be added in.
# Add one or remove them to change the number of p-values optimized
X=["N(Cu)", "N(Ni)", "N(Mn)", "N(P)","N(Si)", "N( C )","N(Temp)", "Alloy", "Alloy"] 
nEFl = len(X)-7

Y1="CD delta sigma"
Y2="CD delta sigma"
#Y="delta sigma"
datapath="DBTT_Data21.csv"
lwrdatapath='CD_LWR_clean8.csv'
savepath='{}.png'

data = data_parser.pars e(datapath)
data.set_x_features(X)
data.set_y_feature(Y1)
data.remove_all_filters()
data.add_exclusive_filter("Alloy", '=', 1)
data.add_exclusive_filter("Alloy", '=', 2)
data.add_exclusive_filter("Alloy", '=', 8)
data.add_exclusive_filter("Alloy", '=', 14)
data.add_exclusive_filter("Alloy", '=', 29)
#data.add_exclusive_filter("Temp (C)", '=', 270)
#data.add_exclusive_filter("Temp (C)", '=', 310)
#data.add_exclusive_filter("Temp (C)", '=', 300)

lwrdata = data_parser.parse(lwrdatapath)
lwrdata.set_x_features(X)
lwrdata.set_y_feature(Y2)
lwrdata.remove_all_filters()
lwrdata.add_exclusive_filter("Alloy", '=', 1)
lwrdata.add_exclusive_filter("Alloy", '=', 2)
lwrdata.add_exclusive_filter("Alloy", '=', 8)
lwrdata.add_exclusive_filter("Alloy", '=', 14)
lwrdata.add_exclusive_filter("Alloy", '=', 29)
lwrdata.add_exclusive_filter("Time(Years)", '>', 100)
lwrdata.add_exclusive_filter("Time(Years)", '<', 60)
#lwrdata.add_exclusive_filter("Temp (C)", '=', 270)
#lwrdata.add_exclusive_filter("Temp (C)", '=', 310)
#lwrdata.add_exclusive_filter("Temp (C)", '=', 300)

Ydata = data.get_y_data()
Xdata = data.get_x_data()
Ydata_LWR = lwrdata.get_y_data()
Xdata_LWR = lwrdata.get_x_data()

train_fluence = data.get_data("fluence n/cm2")
train_flux = data.get_data("flux n/cm2/s")
test_fluence = lwrdata.get_data("fluence n/cm2")
test_flux = lwrdata.get_data("flux n/cm2/s")

lwrdata.remove_all_filters()
Norm_fluence = lwrdata.get_data("fluence n/cm2")
Norm_flux = lwrdata.get_data("flux n/cm2/s")

lwrdata.remove_all_filters()
Ndata = lwrdata.get_x_data()


def LO90_cv(model, X, Y, num_folds, num_runs):
    Xdata = np.asarray(X)
    Ydata = np.asarray(Y)

    Krms_list = []
    for n in range(num_runs):
        kf = cross_validation.KFold(len(Xdata), n_folds=num_folds, shuffle=True)
        K_fold_rms_list = []
        # split into testing and training sets
        for train_index, test_index in kf:
            #print(np.shape(train_index), np.shape(test_index))
            X_train, X_test = Xdata[test_index], Xdata[train_index]
            Y_train, Y_test = Ydata[test_index], Ydata[train_index]
            # train on training sets
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            Krms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            K_fold_rms_list.append(Krms)

        Krms_list.append(np.mean(K_fold_rms_list))

    #print(np.mean(Krms_list))
    return (np.mean(Krms_list))

def kfold_cv(model, X, Y, num_folds, num_runs):
    Xdata = np.asarray(X)
    Ydata = np.asarray(Y)
    #print(np.shape(Xdata), np.shape(Ydata))
    
    Krms_list = []
    for n in range(num_runs):
        K_fold_rms_list = []
        divisions = divisionsList[n]
        # split into testing and training sets
        for train_index, test_index in divisions:
            #print((train_index), (test_index))
            X_train, X_test = Xdata[train_index], Xdata[test_index]
            Y_train, Y_test = Ydata[train_index], Ydata[test_index]
            # train on training sets
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            Krms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            K_fold_rms_list.append(Krms)

        Krms_list.append(np.mean(K_fold_rms_list))

    #print(np.mean(Krms_list))
    return (np.mean(Krms_list))

def CD_extrapolation (model, Xtrain, Ytrain, Xtest, Ytest ):
    Xtrain = np.asarray(Xtrain)
    Ytrain = np.asarray(Ytrain)
    Xtest = np.asarray(Xtest)
    Ytest = np.asarray(Ytest)
    #print(Xtrain[0], Xtest[0])
    #print(Xtrain[0])
    model.fit(Xtrain, Ytrain)
    Ypredict = model.predict(Xtest)
    #print(Xtrain[0])
    #print(Xtest[0])
    rms = np.sqrt(mean_squared_error(Ypredict, Ytest))
    #print(rms)
    return (rms)

def calculate_EffectiveFluence(Xtrain, Xtest, p):
    Xtrain = np.asarray(Xtrain)
    Xtest = np.asarray(Xtest)
    TrainEFl = np.zeros(len(Xtrain))
    TestEFl = np.zeros(len(Xtest))
    N_TrainEFl = np.zeros(len(Xtrain))
    N_TestEFl = np.zeros(len(Xtest))
    NEFl = np.zeros(len(Norm_fluence))
        
    for n in range(len(Xtrain)):
        TrainEFl[n] = np.log10( train_fluence[n][0]*( 3E10 / train_flux[n][0] )**p )
    for n in range(len(Xtest)):
        TestEFl[n] = np.log10( test_fluence[n][0]*( 3E10 / test_flux[n][0] )**p )
        #print(Xtest[n][7], Xtest[n][8], TestEFl[n], n)
    for n in range(len(Norm_fluence)):
        NEFl[n] = np.log10( Norm_fluence[n][0]*( 3E10 / Norm_flux[n][0] )**p )    
        
    Nmax = np.max([ np.max(TrainEFl), np.max(NEFl)] )
    Nmin = np.min([ np.min(TrainEFl), np.min(NEFl)] )
    #print(np.max(NEFl), np.min(NEFl))
    #print(Nmax, Nmin)
    for n in range(len(Xtrain)):
        N_TrainEFl[n] = (TrainEFl[n]-Nmin)/(Nmax-Nmin)
        #print(N_TrainEFl[n], p,n)
    for n in range(len(Xtest)):
        N_TestEFl[n] = (TestEFl[n]-Nmin)/(Nmax-Nmin)
    #print(N_TestEFl, p)
    
    return (N_TrainEFl, N_TestEFl)

def generation(pop, num_parameters, num_parents, crossover_prob, mutation_prob, shift_prob):

    rmsList = []
    params = []
    for ind in range(len(pop)):
        
        newX_Train = np.copy(Xdata)
        newX_Test = np.copy(Xdata_LWR)
        params = pop[ind]
        
        for gene in range(nEFl):
            #print((gene), (params[gene]), (newX_Train[:, gene]), (newX_Train[:, gene]))
            #print(newX_Train[0][7], newX_Train[0][8])
            #print(Xdata[0][7], Xdata[0][8])
            #print(newX_Train[0, 7+gene])
            EFls = calculate_EffectiveFluence(Xdata, Xdata_LWR, params[gene])
            #print(EFls[0][0], EFls[0][1], EFls[1][0], EFls[1][1], params[gene])
            newX_Train[:, 7+gene] = EFls[0]
            newX_Test[:, 7+gene] = EFls[1]
        

        model = KernelRidge(alpha = 10**(float(params[-2])*(-6)), gamma = 10**((float(params[-1])*(3))-1.5), kernel = 'rbf')
        # un-comment one of these lines to select which test to use (currently CD extrapolate to LWR 60-100 years)
        #rms = LO90_cv(model, newX_Train, Ydata)
        #rms = CD_extrapolation(model, newX_Train, Ydata, newX_Test, Ydata_LWR)
        rms = kfold_cv(model, newX_Train, Ydata, num_folds = number_of_folds, num_runs = number_of_runs)
        
        rmsList.append(rms)

    #select parents
    parents = np.empty([num_parents, num_parameters])
    parentRMS = []
    for newP in range(num_parents):       
        parents[newP] = pop[np.argmin(rmsList)]
        parentRMS.append(np.min(rmsList))
        rmsList[np.argmin(rmsList)] = np.max(rmsList)

    #progenate new population 
    for ind in range(len(pop)):
        p1 = parents[random.randrange(0, num_parents)]
        p2 = parents[random.randrange(0, num_parents)]
        #print(p1, p2)
        
        for par in range(num_parameters):
            p = random.random()
            m = random.random()
            s = random.random()
            #print(p, m, s)
            if p <= crossover_prob:
                pop[ind][par] = p1[par]
            else:
                pop[ind][par] = p2[par]
            if (s <= shift_prob):
                #pop[ind][par] = np.abs( pop[ind][par] + (random.random() - .5)*1 )      # use this line for random shift when using no increments
                pop[ind][par] = np.abs( pop[ind][par] + random.randrange(-4,4)/100 )     # use this line for random shift when using incerenemts of .1
            if m <= mutation_prob:
                #pop[ind][par] = random.random()*10          # use this line for mutation when using no incerenemts
                pop[ind][par] = random.randrange(1,100)/100   # use this line for mutation when using increments of .1 

    return { 'new_population':pop, 'best_parameters':parents, 'best_rms':parentRMS }

population = np.empty([50, nEFl+2])

#initailize random population
for individual in range(len(population)):
    for gene in range(len(population[individual])):
        #population[individual][gene]  = random.random()*10              # use this line for initializing when using no increments
        population[individual][gene]  = random.randrange(1, 100)/100      # use this line for initializing when using increments of .01 

bestRMSs = []
bestGens = []
bestParamSets = []
GAruns=0
runsSinceBest = 0

number_of_folds = 2
number_of_runs = 20

print('running')
while GAruns < 10: # do 10 runs 
    #print(runs)
    runsSinceBest = 0
    GAruns = GAruns+1
    gens = 1
    bestRMS = 200
    bestParams = []
    divisionsList = []
    
    for kn in range(number_of_runs):
        kf = cross_validation.KFold(len(Xdata), n_folds=number_of_folds, shuffle=True)
        divisionsList.append(kf)

    #initailize random population
    for individual in range(len(population)):
        for gene in range(len(population[individual])):
            #population[individual][gene]  = random.random()*10              # use this line for initializing when using no increments
            population[individual][gene]  = random.randrange(1, 100)/100      # use this line for initializing when using increments of .01 

    while runsSinceBest < 30 and gens < 200: # run until convergence condition is met
        # run a generation imputting population, number of parameters, number of parents, crossover probablility, mutation probability, random shift probability
        Gen = generation(population, (nEFl+2), 10, .5, .1, .5 )
        print(Gen['best_rms'])
        #print(Gen['best_parameters'])
        population = Gen['new_population']

        if bestRMS > np.min(Gen['best_rms']) :
            bestRMS = np.min(Gen['best_rms'])
            bestParams = Gen['best_parameters'][np.argmin(Gen['best_rms'])]
            bestParams[-2] =  10**(float(bestParams[-2])*(-6))               # for variables a, y
            bestParams[-1] = 10**((float(bestParams[-1])*(3))-1.5)           # for variables a, y
            bestRMSs.append(bestRMS)
            bestGens.append(gens)
            runsSinceBest = 0

        bestParamsSTR = '  '
        for i in range(len(bestParams)-2):
            bestParamsSTR = bestParamsSTR+"{0:.2f}".format(bestParams[i])+" , "
        bestParamsSTR = bestParamsSTR+str(bestParams[-2])+' , '+str(bestParams[-1])

        print(bestParamsSTR)
        #print(bestParams)
        print(bestRMS)
        
        print(GAruns, gens, runsSinceBest)
        
        gens = gens+1
        runsSinceBest = runsSinceBest+1
    bestParamSets.append(bestParams)

# prints best parameter set for each run, and the results of some tests using it.
EXTRrmsList=[]
CVrmsList=[]
avgParams = np.zeros(len(bestParamSets[0]))
for n in range(len(bestParamSets)):
    params = bestParamSets[n]

    newX_Train = np.copy(Xdata)
    newX_Test = np.copy(Xdata_LWR)
    
    for gene in range(len(params)-2):
        EFls = calculate_EffectiveFluence(Xdata, Xdata_LWR, params[gene])
        #print(EFls[0][0], EFls[0][1], EFls[1][0], EFls[1][1], params[gene])
        newX_Train[:, 7+gene] = EFls[0]
        newX_Test[:, 7+gene] = EFls[1]

    for kn in range(number_of_runs*50):
        kf = cross_validation.KFold(len(Xdata), n_folds=number_of_folds, shuffle=True)
        divisionsList.append(kf)

    model = KernelRidge(alpha = float(params[-2]), gamma = float(params[-1]), kernel = 'rbf')
    EXTRrms = CD_extrapolation(model, newX_Train, Ydata, newX_Test, Ydata_LWR)
    CVrms = kfold_cv(model, newX_Train, Ydata, num_folds = number_of_folds, num_runs = number_of_runs*50)
    STRrms = "{0:.4f}".format(EXTRrms)+'  '+"{0:.4f}".format(CVrms)
    EXTRrmsList.append(EXTRrms)
    CVrmsList.append(CVrms)
    
    for param in range(len(bestParamSets[0])):
        avgParams[param] = avgParams[param] + params[param]

    bestParamsSTR = '[  '
    for i in range(len(bestParams)-2):
            bestParamsSTR = bestParamsSTR+"{0:.2f}".format(bestParamSets[n][i])+' , '
    bestParamsSTR = bestParamsSTR+str(bestParamSets[n][-2])+' , '+str(bestParamSets[n][-1])+' ]    '+STRrms
    print(bestParamsSTR)

print(np.mean(EXTRrmsList), np.mean(CVrmsList))

# averages all parameters
avgParamsSTR = ''
for n in range(len(avgParams)-2):
    avgParams[n] = avgParams[n]/(len(bestParamSets))
for n in range(len(avgParams)-2):
    avgParamsSTR = avgParamsSTR+"{0:.2f}".format(avgParams[n])+' , '

avgParamsSTR = avgParamsSTR+str(avgParams[-2])+' , '+str(avgParams[-1])
print(avgParamsSTR)


