import numpy as np
import data_parser
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
#from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from multiprocessing import Process,Pool,TimeoutError,Value,Array,Manager
import random
import time

class GAParamSearcher():
    def __init__(self, model="", 
                    savepath="",
                    fit_data_csv="",
                    topredict_data_csv="",
                    x_features="",
                    fit_y_feature="",
                    topredict_y_feature="",
                    num_folds=2,
                    num_runs=20,
                    population_size=50,
                    convergence_generations=30,
                    max_generations=200,
                    multiprocessing_processors=0,
                    additional_feature_methods="",
                    fit_filter_out="",
                    topredict_filter_out="",
                    testing=0,
                    *args, **kwargs):
        self.num_folds = int(num_folds)
        self.num_runs = int(num_runs)
        self.population_size = int(population_size)
        self.convergence_generations = int(convergence_generations)
        self.max_generations = int(max_generations)
        if int(testing) == 1:
            random.seed(0)
            self.cv_random_state = np.random.RandomState(0)
        else:
            self.cv_random_state = None
        self.model = model
        self.savepath = savepath
        self.procs=int(multiprocessing_processors)
        self.x_features = x_features.split(",")
        self.nEFl = len(self.x_features)-7 #TTM hardcode needs change
        self.fit_y_feature = fit_y_feature
        self.topredict_y_feature = topredict_y_feature
        # Additional features are parameters to be optimized
        self.additional_feature_methods = additional_feature_methods
        self.afm_dict = None
        self.set_afm_dict(self.additional_feature_methods)
        self.num_addl_features = len(list(self.afm_dict.keys()))
        self.hp_dict = None
        self.set_hp_dict()
        self.num_hyperparams = len(list(self.hp_dict.keys()))
        self.num_features = len(self.x_features)
        self.gene_length = self.num_hyperparams + self.num_addl_features
        self.gene_keys = list()
        self.gene_keys.extend(list(self.afm_dict.keys()))
        hp_keys = list(self.hp_dict.keys())
        hp_keys.sort()
        self.gene_keys.extend(hp_keys)
        # 
        self.fit_data = None #fill in below
        self.topredict_data = None #fill in below
        self.topredict_data_unfiltered = None
        self.fit_Xdata = None
        self.fit_Ydata = None
        self.topredict_Xdata = None
        self.topredict_Ydata = None
        self.divisionsList=list()

        fit_data = data_parser.parse(fit_data_csv)
        fit_data.set_x_features(self.x_features)
        fit_data.set_y_feature(self.fit_y_feature)
        
        fit_data.remove_all_filters()
        if len(fit_filter_out) > 0:
            ftriplets = fit_filter_out.split(";")
            for ftriplet in ftriplets:
                fpcs = ftriplet.split(",")
                ffield = fpcs[0].strip()
                foperator = fpcs[1].strip()
                fval = fpcs[2].strip()
                try:
                    fval = float(fval)
                except (ValueError, TypeError):
                    pass
                fit_data.add_exclusive_filter(ffield, foperator, fval)

        topredict_data = data_parser.parse(topredict_data_csv)
        topredict_data.set_x_features(self.x_features)
        topredict_data.set_y_feature(self.topredict_y_feature)
        
        topredict_data.remove_all_filters()
        if len(topredict_filter_out) > 0:
            ftriplets = topredict_filter_out.split(";")
            for ftriplet in ftriplets:
                fpcs = ftriplet.split(",")
                ffield = fpcs[0].strip()
                foperator = fpcs[1].strip()
                fval = fpcs[2].strip()
                try:
                    fval = float(fval)
                except (ValueError, TypeError):
                    pass
                topredict_data.add_exclusive_filter(ffield, foperator, fval)

        self.fit_data = fit_data
        self.topredict_data = topredict_data

        self.topredict_data_unfiltered = data_parser.parse(topredict_data_csv)
        self.topredict_data_unfiltered.set_x_features(self.x_features)
        self.topredict_data_unfiltered.set_y_feature(self.topredict_y_feature)
        
        self.fit_Ydata = fit_data.get_y_data()
        self.fit_Xdata = fit_data.get_x_data()
        self.topredict_Ydata = topredict_data.get_y_data()
        self.topredict_Xdata = topredict_data.get_x_data()


        #TTM not needed?
        topredict_data.remove_all_filters()
        Ndata = topredict_data.get_x_data()
        #start GA

        population=list()
        for pidx in range(self.population_size):
            population.append(dict())
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

            self.set_divisionsList(self.num_runs)

            #initailize random population
            for individual in range(len(population)):
                population[individual] = dict()
                for gene in self.gene_keys:
                    population[individual][gene] = random.randrange(0, 101)/100

            while runsSinceBest < self.convergence_generations and gens < self.max_generations: # run until convergence condition is met
                # run a generation imputting population, number of parameters, number of parents, crossover probablility, mutation probability, random shift probability
                print("Generation %i %s" % (gens, time.asctime()), flush=True)
                Gen = self.generation(population, self.gene_length, 10, .5, .1, .5)
                #print(Gen['best_rms'])
                #print(Gen['best_parameters'])
                population = Gen['new_population']

                # updates best parameter set
                if bestRMS > np.min(Gen['best_rms']) :
                    bestRMS = np.min(Gen['best_rms'])
                    bestParams = Gen['best_parameters'][np.argmin(Gen['best_rms'])]
                    bestRMSs.append(bestRMS)
                    bestGens.append(gens)
                    runsSinceBest = 0

                # prints output for each generation
                print(time.asctime())
                genpref = "Results for generation %i" % gens
                self.print_genome(bestParams,preface=genpref)
                print(bestRMS, flush=True)
                print(GAruns, gens, runsSinceBest, flush=True)
                
                gens = gens+1
                runsSinceBest = runsSinceBest+1
            bestParamSets.append(bestParams)

        self.set_divisionsList(self.num_runs*10)

# prints best parameter set for each run, and the results of some tests using it.
        EXTrmsList=[]
        CVrmsList=[]
        for bestidx in range(len(bestParamSets)):
            i_rdict = self.evaluate_individual(bestidx, bestParamSets, 
                            parallel_result_dict=None, 
                            do_extrapolation=1)
            STRrms = "Extrapolation RMS: {0:.4f}".format(i_rdict['e_rms'])+'  '+"Extrapolation R2: {0:.4f}".format(i_rdict['e_r2'])+'  '+"CV RMS: {0:.4f}".format(i_rdict['cv_rms'])
            EXTrmsList.append(i_rdict['e_rms'])
            CVrmsList.append(i_rdict['cv_rms'])
            bestpref = "Results for GA %i" % bestidx
            self.print_genome(bestParamSets[bestidx],preface=bestpref)
            print(STRrms)
        return
    def print_genome(self, genome, preface=""):
        genomestr = "%s: " % preface
        for gene in self.gene_keys:
            genestr = gene
            geneval = genome[gene]
            if gene == 'alpha':
                geneval = 10**(float(geneval)*(-6))
                genedisp = "%3.12f" % geneval
            elif gene == 'gamma':
                geneval = 10**((float(geneval)*(3))-1.5)
                genedisp = "%3.12f" % geneval
            elif gene == "calculate_EffectiveFluence":
                genestr="Eff. fl. p"
                genedisp = "%0.2f" % geneval
            else:
                genedisp = "%s" % geneval
            genomestr = genomestr + "%s: %s, " % (genestr, genedisp)
        genomestr = genomestr[:-2] #remove last space and comma
        print(genomestr, flush=True)
        return

    def set_hp_dict(self):
        """ #for GKRR, alpha and gamma
        """
        hp_dict=dict()
        hp_dict['alpha']=None
        hp_dict['gamma']=None
        self.hp_dict=dict(hp_dict)
        return hp_dict

    def set_afm_dict(self, afm_str):
        """
            Args:
                afm_str <str>: String of "method,argname:argval,argname:argval;method,argname:argval,argname:argval"
                            methods should be able to accept string arguments
        """
        afm_dict=dict()
        if len(afm_str) == 0:
            self.afm_dict=afm_dict
            return afm_dict
        afm_split = afm_str.split(";")
        for afm_item in afm_split:
            isplit = afm_item.split(",")
            methodname = isplit[0]
            methodargs = dict()
            for argidx in range(1, len(isplit)):
                argitem = isplit[argidx]
                argname=argitem.split(":")[0]
                argval = argitem.split(":")[1]
                methodargs[argname] = argval
            afm_dict[methodname] = methodargs
        self.afm_dict=dict(afm_dict)
        return afm_dict

    def set_divisionsList(self, num_runs):
        self.divisionsList = list()
        for kn in range(num_runs):
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.cv_random_state)
            splits = kf.split(range(len(self.fit_Xdata)))
            sdict=dict()
            sdict['train'] = list()
            sdict['test'] = list()
            for train, test in splits:
                sdict['train'].append(train)
                sdict['test'].append(test)
            self.divisionsList.append(dict(sdict))
        return

    def single_avg_cv(self, model, Xdata, Ydata, division_index, 
                            parallel_return_dict=None):
        rms_list = []
        division_dict = self.divisionsList[division_index]
        # split into testing and training sets
        for didx in range(len(division_dict['train'])):
            train_index = division_dict['train'][didx]
            test_index = division_dict['test'][didx]
            X_train, X_test = Xdata[train_index], Xdata[test_index]
            Y_train, Y_test = Ydata[train_index], Ydata[test_index]
            # train on training sets
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            my_rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            rms_list.append(my_rms)
        mean_rms = np.mean(rms_list)
        if parallel_return_dict is None:
            return mean_rms
        else:
            parallel_return_dict[division_index] = mean_rms
        return None
    
    def single_avg_cv_pooluse(self, init_tuple):
        model, Xdata, Ydata, division_index = init_tuple
        rms_list = []
        division_dict = self.divisionsList[division_index]
        # split into testing and training sets
        for didx in range(len(division_dict['train'])):
            train_index = division_dict['train'][didx]
            test_index = division_dict['test'][didx]
            X_train, X_test = Xdata[train_index], Xdata[test_index]
            Y_train, Y_test = Ydata[train_index], Ydata[test_index]
            # train on training sets
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            my_rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            rms_list.append(my_rms)
        mean_rms = np.mean(rms_list)
        return mean_rms

    def num_runs_cv(self, model, X, Y, num_runs, parallelize=1):
        Xdata = np.asarray(X)
        Ydata = np.asarray(Y)
        n_rms_list = list()
        
        if self.procs > 100000: #save code for future use
            cv_pool = Pool(self.procs)
            n_input_list = list()
            for nidx in range(num_runs):
                n_input_list.append((model, Xdata, Ydata, nidx))
            n_rms_list = cv_pool.map(self.single_avg_cv_pooluse, n_input_list)
        elif self.procs > 0: 
            cv_manager=Manager()
            n_rms_dict = cv_manager.dict()
            cv_procs = list()
            for nidx in range(num_runs):
                cv_proc = Process(target=self.single_avg_cv, 
                            args=(model, Xdata, Ydata, nidx, n_rms_dict))
                cv_procs.append(cv_proc)
                cv_proc.start()
            for cvp in cv_procs:
                cvp.join()
            for nidx in range(num_runs):
                n_rms_list.append(n_rms_dict[nidx])
        else:
            for n in range(num_runs):
                n_rms_list.append(self.single_avg_cv(model, Xdata, Ydata, n))
        print("CV time: %s" % time.asctime(), flush=True)
        print(n_rms_list, flush=True)
        return (np.mean(n_rms_list))

    def extrapolation (self, model, Xtrain, Ytrain, Xtest, Ytest ):
        Xtrain = np.asarray(Xtrain)
        Ytrain = np.asarray(Ytrain)
        Xtest = np.asarray(Xtest)
        Ytest = np.asarray(Ytest)
        model.fit(Xtrain, Ytrain)
        Ypredict = model.predict(Xtest)
        
        rms = np.sqrt(mean_squared_error(Ypredict, Ytest))
        R2 = r2_score(Ytest, Ypredict)
        
        return (rms, R2)

    def calculate_EffectiveFluence(self, p, flux_feature="",fluence_feature=""):
        train_fluence = self.fit_data.get_data(fluence_feature)
        train_flux = self.fit_data.get_data(flux_feature)
        test_fluence = self.topredict_data.get_data(fluence_feature)
        test_flux = self.topredict_data.get_data(flux_feature)
        Norm_fluence = self.topredict_data_unfiltered.get_data(fluence_feature)
        Norm_flux = self.topredict_data_unfiltered.get_data(flux_feature)

        Xtrain = np.asarray(np.copy(self.fit_Xdata))
        Xtest = np.asarray(np.copy(self.topredict_Xdata))
        TrainEFl = np.zeros(len(Xtrain))
        TestEFl = np.zeros(len(Xtest))
        N_TrainEFl = np.zeros(len(Xtrain))
        N_TestEFl = np.zeros(len(Xtest))
        NEFl = np.zeros(len(Norm_fluence))
            
        for n in range(len(Xtrain)):
            TrainEFl[n] = np.log10( train_fluence[n][0]*( 3E10 / train_flux[n][0] )**p )
        for n in range(len(Xtest)):
            TestEFl[n] = np.log10( test_fluence[n][0]*( 3E10 / test_flux[n][0] )**p )
        for n in range(len(Norm_fluence)):
            NEFl[n] = np.log10( Norm_fluence[n][0]*( 3E10 / Norm_flux[n][0] )**p )    
            
        Nmax = np.max([ np.max(TrainEFl), np.max(NEFl)] )
        Nmin = np.min([ np.min(TrainEFl), np.min(NEFl)] )
        
        for n in range(len(Xtrain)):
            N_TrainEFl[n] = (TrainEFl[n]-Nmin)/(Nmax-Nmin)

        for n in range(len(Xtest)):
            N_TestEFl[n] = (TestEFl[n]-Nmin)/(Nmax-Nmin)

        return (N_TrainEFl, N_TestEFl)

    
    def evaluate_individual(self, indidx, pop, parallel_result_dict=None, 
                                    do_extrapolation=0):
        newX_Train = np.copy(self.fit_Xdata)
        newX_Test = np.copy(self.topredict_Xdata)
        params = pop[indidx]
        rdict=dict()
        for gene in self.afm_dict.keys():
            afm_kwargs = dict(self.afm_dict[gene])
            (train_feature, test_feature) = getattr(self, gene)(params[gene],
                            **afm_kwargs)
            train_column = np.reshape(train_feature,(len(train_feature),1))
            test_column = np.reshape(test_feature,(len(test_feature),1))
            newX_Train = np.concatenate((newX_Train, train_column), axis=1)
            newX_Test = np.concatenate((newX_Test, test_column), axis=1)
        model = KernelRidge(alpha = 10**(float(params['alpha'])*(-6)), gamma = 10**((float(params['gamma'])*(3))-1.5), kernel = 'rbf')
        if do_extrapolation == 1:
            (e_rms, e_r2) = self.extrapolation(model, newX_Train, self.fit_Ydata, newX_Test, self.topredict_Ydata)
            rdict['e_rms'] = e_rms
            rdict['e_r2'] = e_r2
            cv_rms = self.num_runs_cv(model, newX_Train, self.fit_Ydata, num_runs = self.num_runs*10)       
            rdict['cv_rms'] = cv_rms
        else:
            cv_rms = self.num_runs_cv(model, newX_Train, self.fit_Ydata, num_runs = self.num_runs)       
            rdict['cv_rms'] = cv_rms
        if parallel_result_dict is None:
            return rdict
        else:
            print("individual %i %s" % (indidx, time.asctime()))
            parallel_result_dict[indidx]=rdict
        return 

    def generation(self, pop, num_parameters, num_parents, crossover_prob, mutation_prob, shift_prob):
        rmsList = list()
        if self.procs > 0:
            gen_manager=Manager()
            gen_rms_dict = gen_manager.dict()
            gen_procs = list()
            for indidx in range(len(pop)):
                gen_proc = Process(target=self.evaluate_individual, 
                            args=(indidx, pop, gen_rms_dict))
                gen_procs.append(gen_proc)
                gen_proc.start()
            for genp in gen_procs:
                genp.join()
            for indidx in range(len(pop)):
                rmsList.append(gen_rms_dict[indidx]['cv_rms'])
        else:
            for indidx in range(len(pop)):
                rmsList.append(self.evaluate_individual(indidx, pop)['cv_rms'])
        #select parents
        parents=list()
        for pidx in range(num_parents):
            parents.append(dict)
        parentRMS = []
        for newP in range(num_parents):       
            parents[newP] = dict(pop[np.argmin(rmsList)])
            parentRMS.append(np.min(rmsList))
            rmsList[np.argmin(rmsList)] = np.max(rmsList)

        #progenate new population 
        for ind in range(len(pop)):
            p1 = parents[random.randrange(0, num_parents)]
            p2 = parents[random.randrange(0, num_parents)]
            
            for gene in self.gene_keys:
                c = random.random()
                m = random.random()
                s = random.random()

                if c <= crossover_prob:
                    pop[ind][gene] = p1[gene]
                else:
                    pop[ind][gene] = p2[gene]
                if (s <= shift_prob):
                    if (pop[ind][gene] >= 0) and (pop[ind][gene] <= 1):
                        pop[ind][gene] = np.abs( pop[ind][gene] + random.randrange(-4,5)/100 )   # random shift
                    if (pop[ind][gene] < 0):    
                        pop[ind][gene] = 0
                    if (pop[ind][gene] > 1):    
                        pop[ind][gene] = 1                 
                if m <= mutation_prob:
                    pop[ind][gene] = random.randrange(0,101)/100   # mutation

        return { 'new_population':pop, 'best_parameters':parents, 'best_rms':parentRMS }


def execute(model, data, savepath, lwr_data,
    fit_data_csv="DBTT_Data21.csv",
    topredict_data_csv="CD_LWR_clean8.csv",
    x_features="N(Cu),N(Ni),N(Mn),N(P),N(Si),N( C ),N(Temp),Alloy",
    fit_y_feature="CD delta sigma",
    topredict_y_feature="CD delta sigma",
    multiprocessing_processors=0,
    num_folds=2,
    num_runs=20,
    population_size=50,
    convergence_generations=30,
    max_generations=200,
    testing=0,
    fit_filter_out="",
    topredict_filter_out="",
    additional_feature_methods="",
    *args,**kwargs):
    """
        Args:
            fit_data_csv <str>: csv name for fitting data
            topredict_data_csv <str>: csv name for data to predict
            x_features <str>: comma-delimited x-feature names
            fit_y_feature <str>: y feature name for fitting data
            topredict_y_feature <str>: y feature name for data to predict
            multiprocessing_processors <int>: number of processors
                        (on a single node only) for parallelization.
                        0 (default) - no parallalization.
            num_folds <int>: number of folds for k-fold reverse split
            num_runs <int>: number of CV runs
            population_size <int>: number of individuals in a generation
            convergence_generations <int>: number of generations for which
                                the best RMS parameter set stays the same,
                                after which the GA is considered converged
            max_generations <int>: maximum number of generations if the
                                    GA does not converge
            testing <int>: 0 (default) - not testing; randomize splits
                            1 - testing; seed all CV splits the same
                                        and also all the populations
            additional_feature_methods <str>: semicolon-delimited list of
                    method names and comma-delimited argument list for
                    additional on-the-fly features that will also
                    be parameterized with the gene.
                    Methods must be able to take string arguments.
                    A gene numeric value should be the first argument.
                    These string arguments will then be the next arguments and
                    must be keyword arguments.
                    For two effective fluences with two p-values:
                    "calculate_EffectiveFluence,flux_feature:flux_n_cm2_sec,fluence_feature:fluence_n_cm2;calculate_EffectiveFluence,flux_feature:flux_n_cm2_sec,fluence_feature:flux_n_cm2" or similar
                    additional_feature_methods=calculate_EffectiveFluence,flux_feature:flux n/cm2/s,fluence_feature:fluence n/cm2
        fit_filter_out <str>: semicolon-delimited list of
                        field name, operator, value triplets for filtering
                        out data for fitting
        topredict_filter_out <str>: semicolon-delimited list of
                        field name, operator, value triplets for filtering
                        out data for prediction (extrapolation)
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
    kwargs['multiprocessing_processors'] = multiprocessing_processors
    kwargs['num_folds'] = num_folds
    kwargs['population_size'] = population_size
    kwargs['convergence_generations'] = convergence_generations
    kwargs['max_generations'] = max_generations
    kwargs['num_runs'] = num_runs
    kwargs['testing'] = testing
    kwargs['additional_feature_methods'] = additional_feature_methods
    kwargs['fit_filter_out'] = fit_filter_out
    kwargs['topredict_filter_out'] = topredict_filter_out
    myga =  GAParamSearcher(**kwargs)
    return

