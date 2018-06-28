__author__ = 'Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import numpy as np
import os
from ParamGridSearch import ParamGridSearch
import time

class ParamOptGA(ParamGridSearch):
    """Class to perform parameter optimization by genetic algorithm. Allows custom features.

    Args:
        training_dataset (DataHandler object): Training dataset handler
        testing_dataset (DataHandler object): Testing dataset handler
        model (sklearn model object): sklearn model
        save_path (str): Save path

        param_strings (list of str): list of [parameter string made up of semicolon-delimited pieces]

            Piece 1: The word 'model' or a custom feature class.method string, e.g. DBTT.calculate_EffectiveFluence,
            where the custom module has the same name as the custom class, and resides inside the custom_features folder

            Piece 2: The parameter name

            Piece 3: The parameter type. Use only: 'int', 'float', 'bool', 'str' (Some sklearn model hyperparameters are type-sensitive)

            Piece 4: The series type. Use:
            'discrete': List will be given in piece 5
            'bool' and 'str' MUST use 'discrete'
            'continuous': Range and step will be given in piece 5
            'continuous-log: Range and step will be given in piece 5

            Piece 5: A colon-delimited list of
            (a) a discrete list of values to grid over, OR
            (b) start, end, number of points: numpy's np.linspace or np.logspace function will be used to generate this list,
            using an inclusive start and inclusive end. A parameter with only one discrete value will not be considered as an 'optimized' parameter.

        fix_random_for_testing (bool?): 0 - use random numbers
                                      1 - fix randomizer for testing
        num_cvtests (int): Number of CV tests for each validation step
        num_folds (int): Number of folds for K-fold cross validation; leave blank to use LO% CV
        percent_leave_out (int): Percentage to leave out for LO% CV; leave blank to use K-fold CV
        mark_outlying_points (list of int): Number of outlying points to mark in best and worst tests, e.g. [0,3]
        num_bests (int): Number of best individuals to track
        processors (int): Number of processors to use 1 - single processor (serial); 2 - use multiprocessing with this many processors, all on a SINGLE node
        population_size (int): Number of individuals in each generation's population
        convergence_generations (int): Number of generations where the genome must stay constant in order to establish convergence
        max_generations (int): Maximum number of generations
        num_gas (int): Number of GAs to run
        crossover_prob (float): Crossover probability (float < 1.00)
        mutation_prob (float): Mutation probability (float < 1.00)
        shift_prob (float): Shift probability (float < 1.00)
        gen_tol (float): Generation-to-generation RMSE tolerance for considering RMSEs to be equal (absolute float tolerance)

    Returns:
        Analysis in save_path folder

    Raises:

    """
    def __init__(self, training_dataset=None, testing_dataset=None, model=None, save_path=None, num_folds=None,
        percent_leave_out=None, num_cvtests=20, mark_outlying_points='0,3', num_bests=10, fix_random_for_testing=0,
        processors=1, pop_upper_limit=1000000, num_gas=1, ga_pop_size=50, convergence_generations=30,
        max_generations=200, crossover_prob = 0.5, mutation_prob = 0.1, shift_prob = 0.5, gen_tol = 0.00000001):
        """
            Additional class attributes not in parent class:
           
            Set by keyword:
            self.ga_pop_size <int>: GA population size
            self.num_gas <int>
            self.convergence_generations <int>: Number of generations where
                                the genome stays the same, to be considered
                                converged
            self.max_generations <int>: Maximum number of generations
            self.crossover_prob
            self.mutation_prob
            self.shift_prob
            self.gen_tol
            Set by code:
            self.random_state <numpy RandomState>: random state
            self.ga_dict 
            self.gact 
        """
        ParamGridSearch.__init__(self, 
            training_dataset=training_dataset, 
            testing_dataset=training_dataset, 
            model=model, 
            save_path = save_path,
            fix_random_for_testing = fix_random_for_testing,
            num_cvtests = num_cvtests,
            mark_outlying_points = mark_outlying_points,
            num_folds = num_folds,
            percent_leave_out = percent_leave_out,
            processors = processors,
            pop_upper_limit = pop_upper_limit,
            num_bests = num_bests, **kwargs)
        #Sets by keyword
        self.num_gas = int(num_gas)
        self.ga_pop_size = int(ga_pop_size)
        self.convergence_generations = int(convergence_generations)
        self.max_generations = int(max_generations)
        self.processors = int(processors)
        self.crossover_prob = float(crossover_prob)
        self.mutation_prob = float(mutation_prob)
        self.shift_prob = float(shift_prob)
        self.gen_tol = float(gen_tol)
        #Sets in code
        if self.fix_random_for_testing == 1:
            self.random_state = np.random.RandomState(0)
        else:
            self.random_state = np.random.RandomState()
        #
        self.ga_dict = dict()
        self.gact = 0
        return

    def set_up_generation(self, genct=0):
        ga_dir = os.path.join(self.save_path, "GA_%i" % self.gact)
        if not os.path.isdir(ga_dir):
            os.mkdir(ga_dir)
        gen_save_path = os.path.join(ga_dir, "Gen_%i" % genct)
        gen_kwargs = dict()
        for param_ct in self.param_strings.keys():
            gen_kwargs["param_%i" % param_ct] = self.param_strings[param_ct]
        mygen =ParamGridSearch(training_dataset=self.training_dataset,
            testing_dataset=self.training_dataset, #ONLY TRAIN is used
            model=self.model,
            save_path = gen_save_path,
            fix_random_for_testing = self.fix_random_for_testing,
            num_cvtests = self.num_cvtests,
            mark_outlying_points = self.mark_outlying_points,
            num_folds = self.num_folds,
            percent_leave_out = self.percent_leave_out,
            processors = self.processors,
            pop_upper_limit = self.pop_upper_limit,
            num_bests = self.num_bests, **gen_kwargs)
        return mygen

    def get_parent_params(self, prev_gen):
        upper_lim=0
        if prev_gen is None:
            upper_lim = self.pop_size #refers to GA's total combinatorial space
        else:
            upper_lim = self.num_bests #number of potential parents
        p1idx = self.random_state.randint(0, upper_lim)
        p2idx = p1idx
        safety_ct = 0
        while (p2idx == p1idx) and (safety_ct < 1000):
            p2idx = self.random_state.randint(0, upper_lim)
            safety_ct = safety_ct + 1
        if safety_ct == 1000:
            raise ValueError("Error generating parents. Reached 1000 random integers, all identical in second parent.")
        if prev_gen is None:
            pop_keys = list(self.pop_params.keys()) #these keys are combinatorial, not by count
            p1_params = self.pop_params[pop_keys[p1idx]]
            p2_params = self.pop_params[pop_keys[p2idx]]
        else:
            p1_params = prev_gen.best_indivs[p1idx][2] #params in third column
            p2_params = prev_gen.best_indivs[p2idx][2]
        return (p1_params, p2_params)

    def get_new_pop_params(self, prev_gen):
        """
            Args:
                prev_gen <ParamGridSearch object>
            Returns:
                parameter dictionary for new population, to be fed
                directly into ParamGridSearch.pop_params dictionary
        """
        pop_params = dict()
        for ict in range(0, self.ga_pop_size):
            pop_params[ict] = dict()
            (p1_params, p2_params) = self.get_parent_params(prev_gen)
            # add nonoptimized parameters
            for nonopt_param in self.nonopt_param_list:
                (location, param_name) = self.get_split_name(nonopt_param)
                if not (location in pop_params[ict].keys()):
                    pop_params[ict][location] = dict()
                pop_params[ict][location][param_name] = p1_params[location][param_name] #p1_param and p2_param values should be equivalent
            # add optimized parameters
            for opt_param in self.opt_param_list:
                (location, param_name) = self.get_split_name(opt_param)
                if not (location in pop_params[ict].keys()):
                    pop_params[ict][location] = dict()
                crossover_val = self.random_state.rand()
                mutation_val = self.random_state.rand()
                shift_val = self.random_state.rand()
                if crossover_val <= self.crossover_prob:
                    new_val = p1_params[location][param_name]
                else:
                    new_val = p2_params[location][param_name]
                opvalues =list(self.opt_dict[opt_param]) #all choices
                if shift_val <= self.shift_prob: #shift along the value list
                    validx = opvalues.index(new_val)
                    shift_len = self.random_state.randint(-2,3) #-2,-1,0,1,2
                    newidx = validx + shift_len
                    if newidx < 0:
                        newidx == 0
                    elif newidx >= len(opvalues):
                        newidx = len(opvalues) - 1 #maximum index
                    new_val = opvalues[newidx]
                if mutation_val <= self.mutation_prob:
                    newidx = self.random_state.randint(0,len(opvalues))
                    new_val = opvalues[newidx]
                pop_params[ict][location][param_name] = new_val
        return pop_params

    def check_convergence(self, results, best_rmse, best_params):
        """Check convergence.
            Args:
                results <list of (RMSE, parameter dictionary) entries>
                best_rmse <float>: best RMSE to this point
                best_params <dict>: best parameters to this point
            Returns:
                (boolean for convergence, best rmse, best rmse params)
        """
        rmses_same = False
        params_same = False
        converged = False
        rmses=list()
        param_list=list()
        unmatching_params = 0
        overtol_rmses = 0
        for ridx in range(0, len(results)):
            rmse = results[ridx][0]
            params = results[ridx][1]
            rmses.append(rmse) #make rmse list
            param_list.append(params)
        min_idx = np.argmin(rmses)
        min_rmse = rmses[min_idx]
        if min_rmse < best_rmse:
            best_rmse = min_rmse
            best_params = param_list[min_idx]
        for params in param_list:
            if not (params == best_params):
                unmatching_params +=1
        if unmatching_params == 0:
            params_same = True
        deviations = np.abs(np.array(rmses,'float') - best_rmse)
        for deviation in deviations:
            if deviation > self.gen_tol:
                overtol_rmses +=1
        if overtol_rmses == 0:
            rmses_same = True
        #If RMSEs are same/similar but params flip-flop, still converged
        if rmses_same is True:
            converged = True
            if unmatching_params is True:
                logger.info("RMSE convergence is true, but generation parameters are not all the same.")
        return (converged, best_rmse, best_params)

    def run_ga(self):
        self.ga_dict[self.gact] = dict()
        self.ga_dict[self.gact]['generations'] = dict()
        gen_bests = list()
        
        highval = 10000000
        for gbidx in range(0, self.convergence_generations):
            gen_bests.append((highval, dict()))
        
        print('running', flush=True)
        ga_genct = 0
        ga_best_rmse = highval
        ga_best_genome = None
        ga_converged = False
        previous_generation = None

        while (ga_converged is False) and (ga_genct < self.max_generations): 
            print("Generation %i %s" % (ga_genct, time.asctime()), flush=True)
            mygen = self.set_up_generation(ga_genct)
            mygen.set_up_prior_to_population()
            mygen.pop_size = self.ga_pop_size
            new_pop_params = self.get_new_pop_params(previous_generation)
            mygen.pop_params = new_pop_params
            mygen.evaluate_pop()
            mygen.get_best_indivs()
            mygen.print_results()
            #mygen.plot() #do not plot heatmaps etc. for each generation
            mygen.print_readme()
            gen_best_genome = mygen.best_params
            gen_best_rmse = mygen.best_indivs[0][1]
            previous_generation = mygen
            self.ga_dict[self.gact]['generations'][ga_genct] = dict()
            self.ga_dict[self.gact]['generations'][ga_genct]['best_rmse'] = gen_best_rmse
            self.ga_dict[self.gact]['generations'][ga_genct]['best_genome'] = gen_best_genome
            gen_bests.pop(0)
            gen_bests.append((gen_best_rmse, gen_best_genome))
            # prints output for each generation
            print(time.asctime())
            genpref = "Results gen %i, rmse %3.3f" % (ga_genct, gen_best_rmse)
            printlist = self.print_params(gen_best_genome)
            print("%s: %s" % (genpref, printlist))
            ga_genct = ga_genct + 1
            (ga_converged, ga_best_rmse, ga_best_genome) = self.check_convergence(gen_bests, ga_best_rmse, ga_best_genome)
        self.ga_dict[self.gact]['best_rmse'] = ga_best_rmse 
        self.ga_dict[self.gact]['best_genome'] = ga_best_genome
        self.ga_dict[self.gact]['converged'] = ga_converged
        self.ga_dict[self.gact]['gen_bests'] = gen_bests
        self.gact = self.gact + 1
        return

    def print_ga(self, ga=""):
        self.readme_list.append("----- GA %i -----\n" % ga)
        self.readme_list.append("Converged?: %s\n" % self.ga_dict[ga]['converged'])
        prefacestr= "Best %i-CV avg RMSE: %3.3f\n" % (self.num_cvtests, self.ga_dict[ga]['best_rmse'])
        self.readme_list.append(prefacestr)
        printlist = self.print_params(self.ga_dict[ga]['best_genome'])
        for printitem in printlist:
            self.readme_list.append("%s" % printitem)
        return

    def run(self):
        self.set_up()
        self.readme_list.append("===== GA info =====\n")
        for ga in range(0, self.num_gas):
            self.run_ga()
            self.print_ga(ga)
        self.select_final_best()
        self.print_final_results()
        return
    
    def print_final_results(self):
        self.print_best_params()
        self.save_best_model()
        self.print_best_dataframe()
        self.print_readme()
        return

    def set_up(self):
        ParamGridSearch.set_up(self)
        return

    def select_final_best(self):
        ga_final_rmse_list = list()
        for gact in range(0, self.gact): 
            ga_final_rmse_list.append(self.ga_dict[gact]['best_rmse'])
        ga_min_idx = np.argmin(ga_final_rmse_list)
        self.best_params = self.ga_dict[ga_min_idx]['best_genome']
        self.readme_list.append("===== Overall info =====\n")
        self.readme_list.append("%s\n" % time.asctime())
        self.readme_list.append("Overall best genome:\n")
        printlist = self.print_params(self.best_params)
        for printitem in printlist:
            self.readme_list.append(printitem)
        return
