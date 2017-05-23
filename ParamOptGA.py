import numpy as np
import copy
import os
from SingleFit import SingleFit
from SingleFit import timeit
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from DataParser import FeatureIO
from sklearn.kernel_ridge import KernelRidge
from multiprocessing import Process,Pool,TimeoutError,Value,Array,Manager
import time
from custom_features import cf_help

def print_genome(genome=None, preface=""):
    genomestr = "%s: " % preface
    genes = list(genome.keys())
    genes.sort()
    for gene in genes:
        geneidxs = list(genome[gene].keys())
        geneidxs.sort()
        for geneidx in geneidxs:
            geneval = genome[gene][geneidx]
            if geneidx == 'alpha':
                geneval = 10**(float(geneval)*(-6))
            elif geneidx == 'gamma':
                geneval = 10**((float(geneval)*(3))-1.5)
            genedisp = "%s %s: %3.6f" % (gene, geneidx, geneval)
            genomestr = genomestr + genedisp + ", "
    print(genomestr[:-2], flush=True) #remove last comma and space
    return genomestr[:-2]

def print_genome_for_code(genome=None):
    printlist=list()
    genes = list(genome.keys())
    genes.sort()
    for gene in genes:
        geneidxs = list(genome[gene].keys())
        geneidxs.sort()
        for geneidx in geneidxs:
            geneval = genome[gene][geneidx]
            if geneidx == 'alpha':
                geneval = 10**(float(geneval)*(-6))
            elif geneidx == 'gamma':
                geneval = 10**((float(geneval)*(3))-1.5)
            genedisp = "%s;%s;%3.6f\n" % (gene, geneidx, geneval)
            printlist.append(genedisp)
    return printlist


class GAIndividual():
    """GA Individual
    """
    def __init__(self,
            genome=None,
            testing_dataset=None,
            cv_divisions=None,
            afm_dict=None,
            use_multiprocessing=1,
            *args,**kwargs):
        """
        Attributes:
            #Set by keyword
            self.genome
            self.testing_dataset
            self.cv_divisions
            self.afm_dict
            self.use_multiprocessing
            #Set in code
            self.model
            self.rmse
        """
        #Set by keyword
        self.genome=dict(genome)
        self.testing_dataset=testing_dataset
        self.cv_divisions=cv_divisions
        self.afm_dict=afm_dict
        self.use_multiprocessing = int(use_multiprocessing)
        #Sets later in code
        self.model = None
        self.rmse=None
        self.input_data_with_afm=None
        #Run setup immediately
        self.set_up()
        return

    def set_up(self):
        self.set_up_model()
        self.set_up_input_data_with_additional_features()
        return
    
    def set_up_model(self):
        self.model = KernelRidge(alpha = 10**(float(self.genome['model']['alpha'])*(-6)), 
                gamma = 10**((float(self.genome['model']['gamma'])*(3))-1.5), 
                kernel = 'rbf')
        return

    def set_up_input_data_with_additional_features(self):
        all_data = copy.deepcopy(self.testing_dataset.data)
        input_data = copy.deepcopy(self.testing_dataset.input_data)
        for gene in self.afm_dict.keys():
            afm_kwargs = dict(self.afm_dict[gene]['args'])
            (feature_name,feature_data)=cf_help.get_custom_feature_data(gene,
                        starting_dataframe = all_data,
                        param_dict=dict(self.genome[gene]),
                        addl_feature_method_kwargs = dict(afm_kwargs))
            fio = FeatureIO(input_data)
            input_data = fio.add_custom_features([gene], feature_data)
        self.input_data_with_afm = input_data
        return

    def single_avg_cv(self, division_index):
        rms_list=list()
        division_dict = self.cv_divisions[division_index]
        # split into testing and training sets
        for didx in range(len(division_dict['train_index'])):
            train_index = division_dict['train_index'][didx]
            test_index = division_dict['test_index'][didx]
            X_train = self.input_data_with_afm.iloc[train_index,:]
            X_test = self.input_data_with_afm.iloc[test_index,:]
            Y_train = self.testing_dataset.target_data.iloc[train_index]
            Y_test = self.testing_dataset.target_data.iloc[test_index]
            # train on training sets
            self.model.fit(X_train, Y_train)
            Y_test_Pred = self.model.predict(X_test)
            my_rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            rms_list.append(my_rms)
        mean_rms = np.mean(rms_list)
        return mean_rms

    def num_runs_cv(self, verbose=0):
        num_runs = len(self.cv_divisions)
        if self.use_multiprocessing > 0:
            cv_pool = Pool(processes = self.use_multiprocessing)
            n_rms_list = cv_pool.map(self.single_avg_cv, range(num_runs))
        else:
            n_rms_list = list()
            for nidx in range(num_runs):
                n_rms_list.append(self.single_avg_cv(nidx))
        print("CV time: %s" % time.asctime(), flush=True)
        if verbose > 0:
            print_copy = list(n_rms_list)
            print_copy.sort()
            print("Sorted RMS list for CV:")
            print(print_copy, flush=True)
        return (np.mean(n_rms_list))

    def evaluate_individual(self):
        cv_rmse = self.num_runs_cv()
        self.rmse = cv_rmse
        return cv_rmse

class GAGeneration():
    """GA Generation
    Attributes:
        self.testing_dataset
        self.cv_divisions <dict>
        self.population <list of param dicts>
        self.num_parents <int>
        self.gene_template <dict>
        self.population_size <int>
        self.use_multiprocessing <int>
        self.best_rmse <float>
        self.best_genome <dict>
        self.best_individual <int>
        self.parents <list of param dicts>
        self.new_population <list of param dicts>
        # Set in code
        self.crossover_prob <float>
        self.mutation_prob <float>
        self.shift_prob <float>
        self.population_rmse_list <list of float>
        self.random_state
    """
    def __init__(self, 
                testing_dataset=None,
                cv_divisions=None,
                population=None,
                num_parents=None,
                gene_template=None,
                population_size=50,
                afm_dict=None,
                use_multiprocessing=1,
                *args,**kwargs):
        self.testing_dataset = testing_dataset
        self.cv_divisions = cv_divisions
        self.population = population
        self.num_parents = int(num_parents)
        self.use_multiprocessing = int(use_multiprocessing)
        self.population_size = int(population_size)
        self.gene_template = gene_template
        self.afm_dict = afm_dict
        self.random_state = np.random.RandomState()
        self.best_rmse = 100000000
        self.best_params = None
        self.best_individual = None
        self.parent_indices = list()
        self.new_population = dict()
        self.crossover_prob = 0.5
        self.mutation_prob = 0.1
        self.shift_prob = 0.5
        self.population_rmses = list()
        return
    
    def run(self):
        self.set_up()
        self.evaluate_population()
        self.select_parents()
        self.create_new_population()
        return

    def set_up(self):
        if self.population is None:
            self.initialize_population()
        return
    
    def initialize_population(self, verbose=1):
        self.population=dict()
        for pidx in range(self.population_size):
            genome = dict()
            genes = list(self.gene_template.keys())
            genes.sort()
            for gene in genes:
                genome[gene] = dict()
                geneidxs = list(self.gene_template[gene].keys())
                geneidxs.sort()
                for geneidx in geneidxs:
                    genome[gene][geneidx] =self.random_state.randint(0, 101)/100
            self.population[pidx] = GAIndividual(genome=genome,
                                testing_dataset = self.testing_dataset,
                                cv_divisions = self.cv_divisions,
                                afm_dict = self.afm_dict,
                                use_multiprocessing = self.use_multiprocessing)
        if verbose > 0:
            for pidx in self.population.keys():
                print_genome(self.population[pidx].genome)
        return

    def evaluate_population_multiprocessing(self, indidx):
        return self.population[indidx].evaluate_individual()

    def evaluate_population(self, verbose=1):
        if self.use_multiprocessing > 0:
            pool = Pool(processes = self.use_multiprocessing)
            rmses = pool.map(self.evaluate_population_multiprocessing, range(self.population_size))
            for indidx in range(self.population_size):
                self.population_rmses[indidx] = rmses[indidx]
        else:
            for indidx in range(self.population_size):
                self.population_rmses.append(self.population[indidx].evaluate_individual())
        if (verbose > 0):
            print(self.population_rmses)
        return
    
    def select_parents(self):
        """Select parents.
        """
        self.parent_indices = list()
        temp_rmse_list = list(self.population_rmses)
        lowest_idx=None
        for pridx in range(self.num_parents):       
            min_rmse_index = np.argmin(temp_rmse_list)
            self.parent_indices.append(pridx)
            min_rmse = temp_rmse_list[min_rmse_index]
            if min_rmse < self.best_rmse:
                self.best_rmse = min_rmse
                self.best_genome = dict(self.population[pridx].genome)
                self.best_individual = pridx
            temp_rmse_list[min_rmse_index] = np.max(temp_rmse_list) #so don't find same rmse twice)
        return

    def create_new_population(self):
        """Progenate new population 
        """
        self.new_population=dict()
        for indidx in range(self.population_size):
            p1idx = self.random_state.randint(0, self.num_parents)
            p2idx = self.random_state.randint(0, self.num_parents) #can have two of same parent?
            new_genome = dict() 
            genes = list(self.population[indidx].genome.keys())
            genes.sort()
            for gene in genes:
                new_genome[gene] = dict()
                geneidxs = list(self.population[indidx].genome[gene].keys())
                geneidxs.sort()
                for geneidx in geneidxs:
                    c = self.random_state.rand()
                    m = self.random_state.rand()
                    s = self.random_state.rand()
                    new_val = None
                    
                    if c <= self.crossover_prob:
                        new_val = self.population[p1idx].genome[gene][geneidx]
                    else:
                        new_val = self.population[p2idx].genome[gene][geneidx]
                    if (s <= self.shift_prob):
                        if (new_val >= 0) and (new_val <= 1):
                            new_val = np.abs( new_val + self.random_state.randint(-4,5)/100 )   # random shift
                        if (new_val < 0):    
                            new_val = 0
                        if (new_val > 1):    
                            new_val = 1                 
                    if m <= self.mutation_prob:
                        new_val = self.random_state.randint(0,101)/100   # mutation
                    new_genome[gene][geneidx] = new_val
            self.new_population[indidx] = GAIndividual(genome = new_genome,
                                testing_dataset = self.testing_dataset,
                                cv_divisions = self.cv_divisions,
                                afm_dict = self.afm_dict,
                                use_multiprocessing = self.use_multiprocessing)
        return

class ParamOptGA(SingleFit):
    """Search for paramters using GA.
        Allows custom features.
    Args:
        training_dataset, (Should be the same as first testing set.)
        testing_dataset, (First should be same as training set. GA is run using
                        only the first testing dataset. 
                        Final evaluations are carried out on all sets.
                        All sets must have target data for evaluation.
        model,
        save_path,
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
                        methodname;number_of_genes;parameter1:value1;parameter2:value2;...
    Returns:
        Analysis in save_path folder
    Raises:
    """
    def __init__(self, 
        training_dataset=None,
        testing_dataset=None,
        model=None,
        save_path=None,
        num_folds=2,
        percent_leave_out=None,
        num_cvtests=20,
        num_gas=1,
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
            self.final_testing_datasets <list>: post-GA testing datasets
            Set by code:
        """
        SingleFit.__init__(self, 
            training_dataset=training_dataset, 
            testing_dataset=testing_dataset,
            model=model, 
            save_path = save_path)
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
        self.num_gas = int(num_gas)
        self.population_size = int(population_size)
        self.convergence_generations = int(convergence_generations)
        self.max_generations = int(max_generations)
        self.num_parents = int(num_parents)
        self.fix_random_for_testing = int(fix_random_for_testing)
        if self.fix_random_for_testing == 1:
            self.random_state = np.random.RandomState(0)
        else:
            self.random_state = np.random.RandomState()
        self.use_multiprocessing = int(use_multiprocessing)
        if type(additional_feature_methods) is list:
            self.additional_feature_methods = list(additional_feature_methods)
        else:
            self.additional_feature_methods = additional_feature_methods.split(",")
        self.final_testing_datasets = list(testing_dataset)
        #Sets in code
        self.cv_divisions = None
        self.cv_divisions_final = None
        self.population = list() #list of dict
        self.afm_dict = dict()
        self.hp_dict = dict()
        self.gene_dict = dict()
        #
        self.ga_dict = dict()
        self.gact = 0
        self.final_best_rmse = 100000000
        self.final_best_genome = None
        return

    def run_ga(self):
        self.ga_dict[self.gact] = dict()
        self.ga_dict[self.gact]['generations'] = dict()
        
        print('running', flush=True)
        ga_genct = 0
        ga_best_rmse = 10000000
        ga_best_genome = None
        ga_converged = 0
        ga_runs_since_best = 0

        new_population=None

        while ga_runs_since_best < self.convergence_generations and ga_genct < self.max_generations: 
            print("Generation %i %s" % (ga_genct, time.asctime()), flush=True)
            Gen = GAGeneration(testing_dataset = self.testing_dataset,
                cv_divisions=self.cv_divisions,
                population=new_population,
                num_parents=self.num_parents,
                gene_template=self.gene_template,
                afm_dict=self.afm_dict,
                population_size=self.population_size,
                use_multiprocessing=self.use_multiprocessing)
            if self.fix_random_for_testing == 1:
                Gen.random_state.seed(self.gact)
            Gen.run()
            new_population = dict(Gen.new_population)
            self.ga_dict[self.gact]['generations'][ga_genct] = Gen

            # updates best parameter set
            if Gen.best_rmse < ga_best_rmse:
                ga_best_rmse = Gen.best_rmse
                ga_best_genome = Gen.best_genome
                ga_runs_since_best = 0
            
            # prints output for each generation
            print(time.asctime())
            genpref = "Results for generation %i, rmse %3.3f" % (ga_genct, Gen.best_rmse)
            print_genome(Gen.best_genome, preface=genpref)
            print(ga_genct, ga_runs_since_best, flush=True)
            
            ga_genct = ga_genct + 1

            if Gen.best_genome == ga_best_genome:
                ga_runs_since_best = ga_runs_since_best + 1
        
        self.ga_dict[self.gact]['best_rmse'] = ga_best_rmse
        self.ga_dict[self.gact]['best_genome'] = ga_best_genome
        if ga_genct < self.max_generations:
            self.ga_dict[self.gact]['converged']=True
        else:
            self.ga_dict[self.gact]['converged']=False
        self.gact = self.gact + 1
        return

    def print_ga(self, ga=""):
        self.readme_list.append("----- GA %i -----\n" % ga)
        self.readme_list.append("Converged?: %s\n" % self.ga_dict[ga]['converged'])
        prefacestr= "Best %i-CV avg RMSE: %3.3f" % (self.num_cvtests, self.ga_dict[ga]['best_rmse'])
        genomestr = print_genome(self.ga_dict[ga]['best_genome'], preface=prefacestr)
        self.readme_list.append("%s\n" % genomestr)
        gens = list(self.ga_dict[ga]['generations'].keys())
        gens.sort()
        self.readme_list.append("..... Generations .....\n")
        for gen in gens:
            prefacestr = "Generation %i best: avg rmse %3.3f" % (gen, self.ga_dict[ga]['generations'][gen].best_rmse)
            genomestr = print_genome(self.ga_dict[ga]['generations'][gen].best_genome, preface = prefacestr)
            self.readme_list.append("%s\n" % genomestr)
        return

    def print_final_eval(self, ga=""):
        printstr = "GA %i %i-CV avg RMSEs: " % (ga, self.num_cvtests * 10)
        for fidx in range(0, len(self.final_testing_datasets)):
            printstr = printstr + "%3.3f " % self.ga_dict[ga]['final_eval_rmses'][fidx]
        self.readme_list.append("%s\n" % printstr)
        return

    @timeit
    def run(self):
        self.set_up()
        self.readme_list.append("===== GA info =====\n")
        for ga in range(0, self.num_gas):
            self.run_ga()
            self.print_ga(ga)
        self.do_final_evaluations()
        self.select_final_best()
        self.print_readme()
        self.print_final_best_for_code()
        self.print_afm_dict_for_code()
        return

    @timeit
    def set_up(self):
        SingleFit.set_up(self)
        self.cv_divisions = self.get_cv_divisions(self.num_cvtests)
        self.set_hp_dict()
        self.set_afm_dict()
        self.set_gene_info()
        return

    def set_gene_info(self):
        self.gene_template=dict()
        self.gene_template.update(self.hp_dict) #model, with subkeys of hyperparams
        for afm_key in self.afm_dict.keys():
            self.gene_template[afm_key] = dict()
            self.gene_template[afm_key].update(self.afm_dict[afm_key]['params'])
        print(self.gene_template)
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
                kf = KFold(n_splits=self.num_folds, shuffle=True, random_state = self.random_state)
                splits = kf.split(range(len(self.testing_dataset.target_data)))
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
                                random_state = self.random_state)
                splits = plo.split(range(len(self.testing_dataset.target_data)))
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

    def set_hp_dict(self):
        """ #for GKRR, alpha and gamma
        """
        hp_dict=dict()
        hp_dict['model'] = dict()
        if self.model.__class__.__name__ == "KernelRidge":
            hp_dict['model']['alpha']=None
            hp_dict['model']['gamma']=None
        else:
            raise ValueError("Only Kernel Ridge with alpha and gamma optimization supported so far.")
        self.hp_dict=dict(hp_dict)
        return hp_dict

    def set_afm_dict(self):
        """
        """
        if self.additional_feature_methods is None:
            self.afm_dict=dict()
            return
        afm_dict=dict()
        for afm_item in self.additional_feature_methods:
            isplit = afm_item.split(";")
            methodname = isplit[0]
            num_params = int(isplit[1])
            methodargs = dict()
            for argidx in range(2, len(isplit)):
                argitem = isplit[argidx]
                argname=argitem.split(":")[0]
                argval = argitem.split(":")[1]
                methodargs[argname] = argval
            afm_dict[methodname] = dict()
            afm_dict[methodname]['args'] = dict(methodargs)
            afm_dict[methodname]['params'] = dict()
            for pidx in range(0, num_params):
                afm_dict[methodname]['params'][pidx] = None
        self.afm_dict=dict(afm_dict)
        return afm_dict
    def do_final_evaluations(self):
        self.readme_list.append("===== Final evaluations =====\n")
        self.cv_divisions_final = self.get_cv_divisions(self.num_cvtests * 10)
        gas = list(self.ga_dict.keys())
        gas.sort()
        for ga in gas:
            best_genome = self.ga_dict[ga]['best_genome']
            self.ga_dict[ga]['final_eval_rmses'] = dict()
            for fidx in range(0, len(self.final_testing_datasets)):
                final_testing_dataset = self.final_testing_datasets[fidx]
                eval_indiv = GAIndividual(genome = best_genome,
                                testing_dataset = final_testing_dataset,
                                cv_divisions = self.cv_divisions_final,
                                afm_dict = self.afm_dict,
                                use_multiprocessing = self.use_multiprocessing)
                self.ga_dict[ga]['final_eval_rmses'][fidx] = eval_indiv.evaluate_individual()
            self.print_final_eval(ga)
        return

    def select_final_best(self):
        for ga in self.ga_dict.keys():
            ga_final_rmse_list = list()
            for fidx in range(0, len(self.final_testing_datasets)):
                ga_final_rmse_list.append(self.ga_dict[ga]['final_eval_rmses'][fidx])
            ga_final_rmse = np.mean(ga_final_rmse_list)
            if ga_final_rmse < self.final_best_rmse:
                self.final_best_rmse = ga_final_rmse
                self.final_best_genome = self.ga_dict[ga]['best_genome']
        self.readme_list.append("===== Overall info =====\n")
        self.readme_list.append("%s\n" % time.asctime())
        printstr = print_genome(self.final_best_genome, preface="Overall best genome")
        self.readme_list.append("%s\n" % printstr)
        return

    def print_final_best_for_code(self):
        printlist = print_genome_for_code(self.final_best_genome)
        with open(os.path.join(self.save_path,"OPTIMIZED_PARAMS"),'w') as pfile:
            pfile.writelines(printlist)
        return

    def print_afm_dict_for_code(self):
        afmlist=list()
        for afm_method in self.afm_dict.keys():
            for afm_arg in self.afm_dict[afm_method]['args'].keys():
                afm_str = "%s;%s;%s" % (afm_method, afm_arg, self.afm_dict[afm_method]['args'][afm_arg])
                afmlist.append("%s\n" % afm_str)
        afmlist.sort()
        with open(os.path.join(self.save_path,"ADDITIONAL_FEATURES"),'w') as afile:
            afile.writelines(afmlist)
        return
