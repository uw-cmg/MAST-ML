===================
ParamOptGA
===================

Optimize hyperparameter values using a genetic algorithm (GA).

-----------------
Input keywords
-----------------

See :doc:`4_p1_paramgridsearch` for the following keywords:

* training_dataset
* testing_dataset
* param_1, param_2, etc.
* fix_random_for_testing
* num_cvtests
* num_folds
* percent_leave_out
* num_bests
* processors
* pop_upper_limit

Additional keywords:

* num_gas: Number of independent GAs to run (default 1)
* ga_pop_size: Population size for each GA (default 50)
    * For each GA, the initial population of ga_pop_size will be drawn randomly from a total population spanning all hyperparameter values given in param_1, param_2, etc.
* convergence_generations: Number of convergence generations (default 30)
    * This is the number of successive generations where the RMSE must be similar (see keyword ``gen_tol``) for the GA to be considered converged.
* max_generations: Maximum number of generations for each GA (default 200)
* crossover_prob: Probability of inheriting from parent 1 as opposed to parent 2 (default 0.5)
    * For each individual, two distinct parents are drawn randomly from the ``num_bests`` best individuals of the previous generation.
    * Crossover is determined separately for each hyperparameter of each individual.
* mutation_prob: Probability that the hyperparameter ('gene') for an individual will randomly mutate (default 0.1)
    * The mutation is the adoption of a random value from the given range for that hyperparameter, regardless of any value that might have been assigned through crossover (supersedes any parental value).
    * Mutation is determined separately for each hyperparameter of each individual.
* shift_prob: Probability that the hyperparameter value (evaluated first for crossover and then for mutation) will shift (default 0.5)
    * The shift is a random amount of -2, -1, 0, 1, or 2 spaces along the given resolution for the given range in the hyperparameter.
    * Shifts stop at the lowest and highest value for that hyperparameter (no wraparound).
* gen_tol: Tolerance for considering RMSEs between generations to be equal (default 0.00000001)
    * This value should be chosen based on the units of the target feature and any knowledge about the expected accuracy in measurement or prediction of the target feature.

----------------
Code
----------------

.. autoclass:: ParamOptGA.ParamOptGA

