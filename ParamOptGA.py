import numpy as np
import data_parser
import matplotlib
import matplotlib.pyplot as plt
from SingleFit import SingleFit
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
#from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from multiprocessing import Process,Pool,TimeoutError,Value,Array,Manager
import random
import time

class ParamOptGA(SingleFit):
    """Search for paramters using GA.
        Allows custom features.
    Args:
        training_dataset, (Should be the same as the testing set.)
        testing_dataset, (Should be same as training set.)
        model,
        save_path,
        input_features,
        target_feature, see parent class.
        num_folds <int>: Number of folds for K-fold CV. Keep blank to use LO% CV
        percent_leave_out <int>: Percent to leave out for LO%. Keep blank to use                                 K-fold CV.
        num_cvtests <int>: Number of CV tests for each individual
        population_size <int>: Number of individuals in each generation's
                                population
        convergence_generations <int>: Number of generations where the
                                        genome must stay constant in order to
                                        establish convergence
        max_generations <int>: Maximum number of generations
        num_parents <int>: Number of best individuals to pull out of each
                        generation to use as parents in next generation
        fix_random_for_testing <int>: 1 - Fix random seed for testing
                                      0 - Randomize (default)
        use_multiprocessing <int>: 1 - use multiprocessing
                                   0 - do not use multiprocessing
        additional_feature_methods <str or list>: comma-delimited string, or
                        a list, of semicolon-delimited pieces, formatted like:
                        methodname;parameter1:value1;parameter2:value2;...
    Returns:
        Analysis in save_path folder
    Raises:
    """
    def __init__(self, 
        training_dataset=None,
        testing_dataset=None,
        model=None,
        save_path=None,
        input_features=None,
        target_feature=None,
        num_folds=2,
        percent_leave_out=None,
        num_cvtests=20,
        population_size=50,
        convergence_generations=30,
        max_generations=200,
        num_parents=10,
        fix_random_for_testing=0,
        use_multiprocessing=1,
        additional_feature_methods=None,
        *args, **kwargs):
        """
            Additional class attributes not in parent class:
           
            Set by keyword:
            self.num_folds <int>: Number of folds, if using KFold CV
            self.percent_leave_out <int>: Percent to leave out of training set,
                                            if using KFold CV
            self.num_cvtests <int>: Number of CV tests per individual
            self.population_size <int>: Number of individuals in a generation
            self.convergence_generations <int>: Number of generations where
                                the genome stays the same, to be considered
                                converged
            self.max_generations <int>: Maximum number of generations
            self.num_parents <int>: Number of best individuals to pull out of
                                  each generation as parents
            self.use_multiprocessing <int>: 1 to use multiprocessing;
                                            0 otherwise
            self.additional_feature_methods <list of str>: List of additional
                        feature methods for fitting.
                        Each string takes the format:
                        methodname;parameter1:value1;parameter2:value2;...
            Set by code:
        """
        SingleFit.__init__(self, 
            training_dataset=training_dataset, 
            testing_dataset=testing_dataset,
            model=model, 
            save_path = save_path,
            input_features = input_features, 
            target_feature = target_feature)
        #Sets by keyword
        if num_folds is None:
            self.num_folds = None
        else:
            self.num_folds = int(num_folds)
        if percent_leave_out is None:
            self.percent_leave_out = None  
        else:
            self.percent_leave_out = int(percent_leave_out)
        self.num_cvtests = int(num_cvtests)
        self.population_size = int(population_size)
        self.convergence_generations = int(convergence_generations)
        self.max_generations = int(max_generations)
        self.num_parents = int(num_parents)
        if int(fix_random_for_testing) == 1:
            np.random.seed(0) 
        self.use_multiprocessing = int(use_multiprocessing)
        if type(additional_feature_methods) is list:
            self.additional_feature_methods = list(additional_feature_methods)
        else:
            self.additional_feature_methods = additional_feature_methods.split(",")
        #Sets in code
        self.cv_divisions = None
        self.cv_divisions_final = None
        self.population = list() #list of dict
        self.afm_dict = dict()
        self.hp_dict = dict()
        self.gene_keys = list()
        self.gene_length = None
        #
        self.gen_dict = None 



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
        self.divisionsList=list()
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
        return

    def run_ga(self):
        self.gen_dict=dict()

        bestRMSs = []
        bestGens = []
        bestParamSets = []
        runsSinceBest = 0

        print('running', flush=True)
        runsSinceBest = 0
        gens = 1
        bestRMS = 200
        bestParams = []

        while runsSinceBest < self.convergence_generations and gens < self.max_generations: # run until convergence condition is met
            # run a generation imputting population, number of parameters, number of parents, crossover probablility, mutation probability, random shift probability
            print("Generation %i %s" % (gens, time.asctime()), flush=True)
            gen_dict[gens] = dict()
            Gen = self.generation(list(self.population),
                        crossover_prob = .5, 
                        mutation_prob = .1, 
                        shift_prob = .5)
            #print(Gen['best_rms'])
            #print(Gen['best_parameters'])
            gen_dict[gens].update(Gen)
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
        return

    def old_save_for_evaluating_results_of_multiple_GA(self):
        for GA in GAs:
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

    @timeit
    def run(self):
        self.set_up()
        self.run_ga()
        return

    @timeit
    def set_up(self):
        SingleFit.set_up(self)
        self.cv_divisions = self.get_cv_divisions(self.num_runs)
        self.set_hp_dict()
        self.set_afm_dict()
        self.set_gene_info()
        self.initialize_population()
        return

    def set_gene_info(self):
        self.gene_keys = list()
        self.gene_keys.extend(list(self.afm_dict.keys()))
        self.gene_keys.extend(list(self.hp_dict.keys()))
        self.gene_length = len(self.gene_keys)
        return

    def initialize_population(self):
        self.population=list()
        for pidx in range(self.population_size):
            self.population.append(dict())
        #initailize random population
        for individual in range(len(population)):
            self.population[individual] = dict()
            for gene in self.gene_keys:
                self.population[individual][gene] = random.randrange(0, 101)/100
        return
    
    def get_cv_divisions(self, num_runs=0):
        """Preset all CV divisions so that all individuals in all
            generations have the same CV divisions.

            Presetting division ensures that comparisons are consistent,
            with only the genome changing between individual evaluations.
        """
        cv_divisions = list()
        if not(self.num_folds is None):
            for kn in range(num_runs):
                kf = KFold(n_splits=self.num_folds, shuffle=True)
                splits = kf.split(range(len(self.testing_target_data)))
                sdict=dict()
                sdict['train_index'] = list()
                sdict['test_index'] = list()
                for train, test in splits:
                    sdict['train_index'].append(train)
                    sdict['test_index'].append(test)
                cv_divisions.append(dict(sdict))
        elif not(self.percent_leave_out is None):
            test_fraction = self.percent_leave_out / 100.0
            for kn in range(num_runs):
                plo = ShuffleSplit(n_splits=1, 
                                test_size = test_fraction,
                                random_state = None)
                splits = plo.split(range(len(self.testing_target_data)))
                sdict=dict()
                sdict['train_index'] = list()
                sdict['test_index'] = list()
                for train, test in splits:
                    sdict['train_index'].append(train)
                    sdict['test_index'].append(test)
                cv_divisions.append(dict(sdict))
        else:
            raise ValueError("Neither percent_leave_out nor num_folds appears to be set.")
        return cv_divisions

    

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
        cv_rms = self.num_runs_cv(model, newX_Train, self.fit_Ydata, num_runs = self.num_runs)       
        rdict['cv_rms'] = cv_rms
        if parallel_result_dict is None:
            return rdict
        else:
            print("individual %i %s" % (indidx, time.asctime()))
            parallel_result_dict[indidx]=rdict
        return 

    def generation(self, pop, crossover_prob=0.5, mutation_prob=0.1, shift_prob=0.5):
        rmsList = list()
        if self.use_multiprocessing > 0:
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
        for pidx in range(self.num_parents):
            parents.append(dict)
        parentRMS = []
        for newP in range(self.num_parents):       
            parents[newP] = dict(pop[np.argmin(rmsList)])
            parentRMS.append(np.min(rmsList))
            rmsList[np.argmin(rmsList)] = np.max(rmsList)

        #progenate new population 
        for ind in range(len(pop)):
            p1 = parents[random.randrange(0, self.num_parents)]
            p2 = parents[random.randrange(0, self.num_parents)]
            
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


