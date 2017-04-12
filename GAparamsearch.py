import numpy as np
import data_parser
import matplotlib
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
import random

class GAParamSearcher():
    def __init__(self, model="", 
                    savepath="",
                    fit_data_csv="",
                    topredict_data_csv="",
                    x_features="",
                    fit_y_feature="",
                    topredict_y_feature="",
                    *args, **kwargs):
        self.number_of_folds = 2
        self.number_of_runs = 20
        self.model = model
        self.savepath = savepath
        self.x_features = x_features.split(",")
        self.nEFl = len(self.x_features)-7 #TTM hardcode needs change
        self.fit_y_feature = fit_y_feature
        self.topredict_y_feature = topredict_y_feature
        self.fit_data = None #fill in below
        self.topredict_data = None #fill in below
        self.fit_Xdata = None
        self.fit_Ydata = None
        self.topredict_Xdata = None
        self.topredict_Ydata = None
        self.train_fluence = None #TTM hardcode needs change
        self.train_flux = None
        self.test_fluence = None
        self.test_flux = None
        self.Norm_fluence = None
        self.Norm_flux = None
        self.divisionsList=list()

        fit_data = data_parser.parse(fit_data_csv)
        fit_data.set_x_features(self.x_features)
        fit_data.set_y_feature(self.fit_y_feature)
        #TTM remove all filter lines after test phase
        fit_data.remove_all_filters()
        fit_data.add_exclusive_filter("Alloy", '=', 1)
        fit_data.add_exclusive_filter("Alloy", '=', 2)
        fit_data.add_exclusive_filter("Alloy", '=', 8)
        fit_data.add_exclusive_filter("Alloy", '=', 14)
        fit_data.add_exclusive_filter("Alloy", '=', 29)

        topredict_data = data_parser.parse(topredict_data_csv)
        topredict_data.set_x_features(self.x_features)
        topredict_data.set_y_feature(self.topredict_y_feature)
        #TTM remove all filter lines after test phase
        topredict_data.remove_all_filters()
        topredict_data.add_exclusive_filter("Alloy", '=', 1)
        topredict_data.add_exclusive_filter("Alloy", '=', 2)
        topredict_data.add_exclusive_filter("Alloy", '=', 8)
        topredict_data.add_exclusive_filter("Alloy", '=', 14)
        topredict_data.add_exclusive_filter("Alloy", '=', 29)
        topredict_data.add_exclusive_filter("Time(Years)", '>', 100)
        topredict_data.add_exclusive_filter("Time(Years)", '<', 60)

        self.fit_data = fit_data
        self.topredict_data = topredict_data
        
        self.fit_Ydata = fit_data.get_y_data()
        self.fit_Xdata = fit_data.get_x_data()
        self.topredict_Ydata = topredict_data.get_y_data()
        self.topredict_Xdata = topredict_data.get_x_data()

        self.train_fluence = fit_data.get_data("fluence n/cm2")
        self.train_flux = fit_data.get_data("flux n/cm2/s")
        self.test_fluence = topredict_data.get_data("fluence n/cm2")
        self.test_flux = topredict_data.get_data("flux n/cm2/s")

        topredict_data.remove_all_filters()
        self.Norm_fluence = topredict_data.get_data("fluence n/cm2")
        self.Norm_flux = topredict_data.get_data("flux n/cm2/s")

        #TTM not needed?
        topredict_data.remove_all_filters()
        Ndata = topredict_data.get_x_data()
        #start GA

        population = np.empty([50, self.nEFl+2])
        bestRMSs = []
        bestGens = []
        bestParamSets = []
        GAruns=0
        runsSinceBest = 0


        print('running', flush=True)
        while GAruns < 10: # do 10 GA runs 
            runsSinceBest = 0
            GAruns = GAruns+1
            gens = 1
            bestRMS = 200
            bestParams = []
            self.divisionsList = list()
            
            for kn in range(self.number_of_runs):
                kf = cross_validation.KFold(len(self.fit_Xdata), n_folds=self.number_of_folds, shuffle=True)
                self.divisionsList.append(kf)

            #initailize random population
            for individual in range(len(population)):
                for gene in range(len(population[individual])):
                    population[individual][gene]  = random.randrange(0, 101)/100

            while runsSinceBest < 30 and gens < 200: # run until convergence condition is met
                # run a generation imputting population, number of parameters, number of parents, crossover probablility, mutation probability, random shift probability
                Gen = self.generation(population, (self.nEFl+2), 10, .5, .1, .5)
                #print(Gen['best_rms'])
                #print(Gen['best_parameters'])
                population = Gen['new_population']

                # updates best parameter set
                if bestRMS > np.min(Gen['best_rms']) :
                    bestRMS = np.min(Gen['best_rms'])
                    bestParams = Gen['best_parameters'][np.argmin(Gen['best_rms'])]
                    bestParams[-2] =  10**(float(bestParams[-2])*(-6))               # for variable a
                    bestParams[-1] = 10**((float(bestParams[-1])*(3))-1.5)           # for variable y
                    bestRMSs.append(bestRMS)
                    bestGens.append(gens)
                    runsSinceBest = 0

                # prints output for each generation
                bestParamsSTR = '  '
                for i in range(len(bestParams)-2):
                    bestParamsSTR = bestParamsSTR+"{0:.2f}".format(bestParams[i])+" , "
                bestParamsSTR = bestParamsSTR+str(bestParams[-2])+' , '+str(bestParams[-1])
                print(bestParamsSTR, flush=True)
                print(bestRMS, flush=True)
                print(GAruns, gens, runsSinceBest, flush=True)
                
                gens = gens+1
                runsSinceBest = runsSinceBest+1
            bestParamSets.append(bestParams)

        self.divisionsList=list()
        for kn in range(self.number_of_runs*10):
            kf = cross_validation.KFold(len(Xdata), n_folds=self.number_of_folds, shuffle=True)
            self.divisionsList.append(kf)

# prints best parameter set for each run, and the results of some tests using it.
        EXTrmsList=[]
        CVrmsList=[]
        avgParams = np.zeros(len(bestParamSets[0]))
        for n in range(len(bestParamSets)):
            params = bestParamSets[n]

            newX_Train = np.copy(Xdata)
            newX_Test = np.copy(Xdata_LWR)
            
            for gene in range(len(params)-2):
                EFls = calculate_EffectiveFluence(Xdata, Xdata_LWR, params[gene])
                newX_Train[:, 7+gene] = EFls[0]
                newX_Test[:, 7+gene] = EFls[1]

            model = KernelRidge(alpha = float(params[-2]), gamma = float(params[-1]), kernel = 'rbf')
            EXTrms = self.CD_extrapolation(model, newX_Train, Ydata, newX_Test, Ydata_LWR)[0]
            EXTR2 = self.CD_extrapolation(model, newX_Train, Ydata, newX_Test, Ydata_LWR)[1]
            CVrms = self.kfold_cv(model, newX_Train, Ydata, num_folds = self.number_of_folds, num_runs = self.number_of_runs*10)
            STRrms = "{0:.4f}".format(EXTrms)+'  '+"{0:.4f}".format(EXTR2)+'  '+"{0:.4f}".format(CVrms)
            EXTrmsList.append(EXTrms)
            CVrmsList.append(CVrms)
            
            for param in range(len(bestParamSets[0])):
                avgParams[param] = avgParams[param] + params[param]

            bestParamsSTR = '[  '
            for i in range(len(bestParams)-2):
                    bestParamsSTR = bestParamsSTR+"{0:.2f}".format(bestParamSets[n][i])+' , '
            bestParamsSTR = bestParamsSTR+str(bestParamSets[n][-2])+' , '+str(bestParamSets[n][-1])+' ]    '+STRrms
            print(bestParamsSTR, flush=True)
        return

    def kfold_cv(self, model, X, Y, num_folds, num_runs):
        Xdata = np.asarray(X)
        Ydata = np.asarray(Y)
        
        Krms_list = []
        for n in range(num_runs):
            K_fold_rms_list = []
            divisions = self.divisionsList[n]
            # split into testing and training sets
            for train_index, test_index in divisions:
                X_train, X_test = Xdata[train_index], Xdata[test_index]
                Y_train, Y_test = Ydata[train_index], Ydata[test_index]
                # train on training sets
                model.fit(X_train, Y_train)
                Y_test_Pred = model.predict(X_test)
                Krms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
                K_fold_rms_list.append(Krms)

            Krms_list.append(np.mean(K_fold_rms_list))
            
        return (np.mean(Krms_list))

    def CD_extrapolation (self, model, Xtrain, Ytrain, Xtest, Ytest ):
        Xtrain = np.asarray(Xtrain)
        Ytrain = np.asarray(Ytrain)
        Xtest = np.asarray(Xtest)
        Ytest = np.asarray(Ytest)
        model.fit(Xtrain, Ytrain)
        Ypredict = model.predict(Xtest)
        
        rms = np.sqrt(mean_squared_error(Ypredict, Ytest))
        R2 = r2_score(Ytest, Ypredict)
        
        return (rms, R2)

    def calculate_EffectiveFluence(self, Xtrain, Xtest, p):
        Xtrain = np.asarray(Xtrain)
        Xtest = np.asarray(Xtest)
        TrainEFl = np.zeros(len(Xtrain))
        TestEFl = np.zeros(len(Xtest))
        N_TrainEFl = np.zeros(len(Xtrain))
        N_TestEFl = np.zeros(len(Xtest))
        NEFl = np.zeros(len(self.Norm_fluence))
            
        for n in range(len(Xtrain)):
            TrainEFl[n] = np.log10( self.train_fluence[n][0]*( 3E10 / self.train_flux[n][0] )**p )
        for n in range(len(Xtest)):
            TestEFl[n] = np.log10( self.test_fluence[n][0]*( 3E10 / self.test_flux[n][0] )**p )
        for n in range(len(self.Norm_fluence)):
            NEFl[n] = np.log10( self.Norm_fluence[n][0]*( 3E10 / self.Norm_flux[n][0] )**p )    
            
        Nmax = np.max([ np.max(TrainEFl), np.max(NEFl)] )
        Nmin = np.min([ np.min(TrainEFl), np.min(NEFl)] )
        
        for n in range(len(Xtrain)):
            N_TrainEFl[n] = (TrainEFl[n]-Nmin)/(Nmax-Nmin)

        for n in range(len(Xtest)):
            N_TestEFl[n] = (TestEFl[n]-Nmin)/(Nmax-Nmin)

        return (N_TrainEFl, N_TestEFl)

    def generation(self, pop, num_parameters, num_parents, crossover_prob, mutation_prob, shift_prob):

        rmsList = []
        params = []
        for ind in range(len(pop)):
            
            newX_Train = np.copy(self.fit_Xdata)
            newX_Test = np.copy(self.topredict_Xdata)
            params = pop[ind]
            
            for gene in range(self.nEFl):
                EFls = self.calculate_EffectiveFluence(self.fit_Xdata, self.topredict_Xdata, params[gene])
                newX_Train[:, 7+gene] = EFls[0]
                newX_Test[:, 7+gene] = EFls[1]
            
            model = KernelRidge(alpha = 10**(float(params[-2])*(-6)), gamma = 10**((float(params[-1])*(3))-1.5), kernel = 'rbf')
            rms = self.kfold_cv(model, newX_Train, self.fit_Ydata, num_folds = self.number_of_folds, num_runs = self.number_of_runs)       
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
            
            for param in range(num_parameters):
                c = random.random()
                m = random.random()
                s = random.random()

                if c <= crossover_prob:
                    pop[ind][param] = p1[param]
                else:
                    pop[ind][param] = p2[param]
                if (s <= shift_prob):
                    if (pop[ind][param] >= 0) and (pop[ind][param] <= 1):
                        pop[ind][param] = np.abs( pop[ind][param] + random.randrange(-4,5)/100 )   # random shift
                    if (pop[ind][param] < 0):    
                        pop[ind][param] = 0
                    if (pop[ind][param] > 1):    
                        pop[ind][param] = 1                 
                if m <= mutation_prob:
                    pop[ind][param] = random.randrange(0,101)/100   # mutation

        return { 'new_population':pop, 'best_parameters':parents, 'best_rms':parentRMS }


def execute(model, data, savepath, lwr_data,
    fit_data_csv="DBTT_Data21.csv",
    topredict_data_csv="CD_LWR_clean8.csv",
    x_features="N(Cu),N(Ni),N(Mn),N(P),N(Si),N( C ),N(Temp),Alloy",
    fit_y_feature="CD delta sigma",
    topredict_y_feature="CD delta sigma",
    *args,**kwargs):
    """
        Args:
            fit_data_csv <str>: csv name for fitting data
            topredict_data_csv <str>: csv name for data to predict
            x_features <str>: comma-delimited x-feature names
            fit_y_feature <str>: y feature name for fitting data
            topredict_y_feature <str>: y feature name for data to predict
    """
    # The two Alloy descriptors are placeholders for the 2 effective fluences that will be added in.
    # Add or remove them to change the number of p-values optimized
    kwargs=dict()
    kwargs['model'] = model
    kwargs['savepath'] = savepath
    kwargs['fit_data_csv'] = fit_data_csv
    kwargs['topredict_data_csv'] = topredict_data_csv
    kwargs['x_features'] = x_features
    kwargs['fit_y_feature'] = fit_y_feature
    kwargs['topredict_y_feature'] = topredict_y_feature
    myga =  GAParamSearcher(**kwargs)
    return

