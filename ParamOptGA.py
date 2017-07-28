import numpy as np
import copy
import os
from SingleFit import SingleFit
from SingleFit import timeit
from ParamGridSearch import ParamGridSearch
import time
from custom_features import cf_help

class ParamOptGA(ParamGridSearch):
    """Search for paramters using GA.
        Allows custom features.
    Args:
        training_dataset, (Should be the same as the testing set.)
        testing_dataset, ONLY THE TRAINING DATASET will be used.
        model,
        save_path,
        xlabel,
        ylabel,
        param_1 and other param_xxx,
        fix_random_for_testing,
        num_cvtests,
        num_folds,
        percent_leave_out,
        mark_outlying_points,
        processors,
        num_bests, see parent class.
        population_size <int>: Number of individuals in each generation's
                                population
        convergence_generations <int>: Number of generations where the
                                        genome must stay constant in order to
                                        establish convergence
        max_generations <int>: Maximum number of generations
        num_gas <int>: Number of GAs to run
        crossover_prob <float>: Crossover probability (float < 1.00)
        mutation_prob <float>: Mutation probability (float < 1.00)
        shift_prob <float>: Shift probability (float < 1.00)
        gen_tol <float>: Generation-to-generation RMSE tolerance for considering
                            RMSEs to be equal (absolute float tolerance)

    Returns:
        Analysis in save_path folder
    Raises:
    """
    def __init__(self, 
        training_dataset=None,
        testing_dataset=None,
        model=None,
        save_path=None,
        num_folds=None,
        percent_leave_out=None,
        num_cvtests=20,
        mark_outlying_points='0,3',
        num_bests=10,
        fix_random_for_testing=0,
        processors=1,
        pop_upper_limit=1000000,
        num_gas=1,
        ga_pop_size=50,
        convergence_generations=30,
        max_generations=200,
        crossover_prob = 0.5,
        mutation_prob = 0.1,
        shift_prob = 0.5,
        gen_tol = 0.00000001,
        *args, **kwargs):
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


    def run_ga(self):
        self.ga_dict[self.gact] = dict()
        self.ga_dict[self.gact]['generations'] = dict()
        
        print('running', flush=True)
        ga_genct = 0
        ga_best_rmse = 10000000
        ga_best_genome = None
        ga_converged = 0
        ga_repetitions_of_best = 0

        previous_generation = None

        while ga_repetitions_of_best < self.convergence_generations and ga_genct < self.max_generations: 
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
            

            # updates best parameter set
            if (gen_best_rmse < ga_best_rmse):
                ga_best_rmse = gen_best_rmse
                if ga_best_genome == gen_best_genome:
                    # slightly different CV may produce slightly diff results
                    # for same genome; account for this case
                    ga_repetitions_of_best += 1
                else:
                    ga_best_genome = gen_best_genome
                    ga_repetitions_of_best = 0
            elif (np.isclose(gen_best_rmse, ga_best_rmse, atol=self.gen_tol)):
                # slightly higher but similar RMSE: avoid non-convergence by
                # flip-flopping between two genomes with similar RMSE
                # keep lower RMSE but take this genome
                ga_best_genome = gen_best_genome
                # convergence adds because of plateau
                ga_repetitions_of_best += 1
            else:
                ga_repetitions_of_best = 0
            
            # prints output for each generation
            print(time.asctime())
            genpref = "Results gen %i (%i/%i convergence), rmse %3.3f" % (ga_genct, ga_repetitions_of_best, self.convergence_generations, gen_best_rmse)
            printlist = self.print_params(gen_best_genome)
            print("%s: %s" % (genpref, printlist))
            ga_genct = ga_genct + 1
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
        prefacestr= "Best %i-CV avg RMSE: %3.3f\n" % (self.num_cvtests, self.ga_dict[ga]['best_rmse'])
        self.readme_list.append(prefacestr)
        printlist = self.print_params(self.ga_dict[ga]['best_genome'])
        for printitem in printlist:
            self.readme_list.append("%s" % printitem)
        return

    @timeit
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

    @timeit
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
