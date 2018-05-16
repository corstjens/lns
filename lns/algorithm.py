# -*- coding: utf-8 -*-

import random
import copy
import math
from lns.factories import repair_factory
from lns.factories import destroy_factory
from lns.solution import Solution
import time

class AlnsBase(object):

    """
    A default (A)LNS implementation based on Pisinger and Ropke (2007)

    Parameters
    ----------
    seed : it
        Seed value used for stochastic elements in the algorithm. The default
        value is 12345

    Attributes
    ----------
    number_of_requests: int
        Number of requests in the problem
    factor_lower_bound: float
        Fraction to determine the minimum numnber of requests to remove
    factor_upper_bound: float
        Fraction to determine the maximum number of requests to remove
    max_runtime: int
        The maximum CPU time the algorithm is allowed to run
    score_increaser1 : int
        Score when the last remove-insert operation resulted in a new global
        best solution. (default = 33)
    score_increaser2 : int
        Score when the last remove-insert operation resulted in a solution that
        has not been accepted before, and the cost of the new solution is
        better than the cost of the current solution. (default = 9)
    score_increaser3 : int
        Score when the last remove-insert operation resulted in a solution that
        has not been accepted before, and the cost of the new solution is worse
        than the cost of the current solution, but the solution was accepted.
        (default = 13)
    timesegment_length : int
        Length of time segment for which scores are collected, expressed in
        terms of the number of iterations (default = 100)
    repair_heuristics : ['insert methods']
        A list of insert heuristics
    destroy_heuristics : ['removal methods']
        A list of removal heuristics
    cooling_rate : float
        Defines the rate at which the simulated annealing temperature reduces
        (default = 0.99975)
    start_temp_control_param : float
        Parameter controlling the start temperature of simulated annealing
        (default = 0.05)
    reaction_factor : float
        The reaction factor controls to what extent weights are influenced by
        the weight of the previous timesegment. (default = 0.1)
    """

    def __init__(self, seed=12345, number_of_requests=25, factor_lower_bound=0.1, 
                 factor_upper_bound=0.5, random_parameter = 20):
        random.seed(seed)
        self.number_of_requests = number_of_requests
        self.factor_lower_bound = factor_lower_bound
        self.factor_upper_bound = factor_upper_bound
        self.max_runtime = 20
        self.number_of_iterations = 500
        self.score_increaser1 = 33
        self.score_increaser2 = 9
        self.score_increaser3 = 13
        self.timesegment_length = 100
        self.B_parameter = 5
        self.random_parameter = random_parameter
        self.noise_parameter = 0.025
        self.list_of_all_destroy_heuristics = [
            destroy_factory.produce_remove_random(self.number_of_requests,
                                                  self.factor_lower_bound, 
                                                  self.factor_upper_bound),
            destroy_factory.produce_remove_worst(self.number_of_requests,
                                                 self.factor_lower_bound, 
                                                 self.factor_upper_bound,
                                                 self.random_parameter), 
            destroy_factory.produce_remove_related(self.number_of_requests,
                                                   self.factor_lower_bound, 
                                                   self.factor_upper_bound,
                                                   self.random_parameter),
            destroy_factory.produce_remove_timeoriented(self.number_of_requests, 
                                                        self.factor_lower_bound,
                                                        self.factor_upper_bound,
                                                        self.random_parameter,
                                                        self.B_parameter),
           destroy_factory.produce_remove_neighbor_graph(self.number_of_requests,
                                                         self.factor_lower_bound,
                                                         self.factor_upper_bound,
                                                         self.random_parameter)]
        self.list_of_all_repair_heuristics = [
            repair_factory.produce_insert_greedy_parallel(),
            repair_factory.produce_insert_greedy_sequential(),
            repair_factory.produce_insert_regret_2(),
            repair_factory.produce_insert_regret_3(),
            repair_factory.produce_insert_regret_4()]
        self.destroy_heuristics = []
        self.repair_heuristics = []
        self.cooling_rate = 0.99975
        self.start_temp_control_param = 0.05
        self.reaction_factor = 0.1

    def solve_problem(self, start_solution, start_time):
        """
        The basic ALNS framework
        """
        self._reset_algorithm()
        self._set_start_temperature(start_solution)
        self._solution = start_solution
        self._previous_solution = copy.deepcopy(self._solution)
        self._best_solution = copy.deepcopy(self._solution)
        self._currently_accepted_solution_cost = (
            self._solution.calculate_solution_cost())
        self._best_solution_cost = (
            self._best_solution.calculate_solution_cost())
        runtime = time.clock() - start_time
        while self._next_iteration(start_time):
            self._previous_solution = copy.deepcopy(self._solution)
            self._select_destroy_heuristic()
            self._select_repair_heuristic()
            self._select_noise()
            new_solution = self._destroy_and_repair()
            self.accept = False
            self.accept_best = False
            if self._accept_solution(new_solution):
                self.accept = True
                #self._update_scores(new_solution)
                self._update_solution(new_solution)
                if (self._currently_accepted_solution_cost <
                   self._best_solution_cost):
                       self.accept_best = True
                       self._update_best_solution()
            else:
                self._solution = copy.deepcopy(self._previous_solution)
                self._solution._rebuild_insert_matrices()  
            runtime = time.clock() - start_time
        return self._best_solution

    def _reset_algorithm(self):
        """
        Resets the internal variables of the algorithm
        """
        #Currently accepted solution
        self._solution = None
        self._currently_accepted_solution_cost = 0
        self._previous_solution = copy.deepcopy(self._solution)
        self._previous_accepted_solution_cost = 0
        #Best solution found so far
        self._best_solution = None
        self._best_solution_cost = 0
        self._previous_best_solution = copy.deepcopy(self._best_solution)
        self._previous_best_solution_cost = 0
        #Set of hashes of already visited solutions
        #self._visited_solutions = set()
        # Keeps track of the score of a heuristic in the current segment
        # Convert scores to floats for calculating purposes
        self._destroy_score = [0.0]*len(self.destroy_heuristics)
        map(float, self._destroy_score)
        self._repair_score = [0.0]*len(self.repair_heuristics)
        map(float, self._repair_score)
        self._noise_score = [0.0]*2  # 0: no noise, 1: noise
        map(float, self._noise_score)
        # Holds the current weight of a heuristic
        # Initially all destroy and repair heuristics have equal weights
        destroy_initial_weights = 1.0 / len(self.destroy_heuristics)
        repair_initial_weights = 1.0 / len(self.repair_heuristics)
        self._destroy_weights = ([destroy_initial_weights]*
                                 len(self.destroy_heuristics))
        self._repair_weights = ([repair_initial_weights]*
                                len(self.repair_heuristics))
        self._noise_weights = [0.5]*2
        # Keeps track how often a specific heuristic has been applied in the
        # current segment
        self._destroy_counter = [0.0]*len(self.destroy_heuristics)
        self._repair_counter = [0.0]*len(self.repair_heuristics)
        self._noise_counter = [0.0]*2
        # Currently selected destroy/repair heuristic and whether noising
        # should be applied
        self._destroy_heuristic = None
        self._repair_heuristic = None
        # Keeps track of the number of alns iterations
        self._alns_counter = 0
        # Determines whether or not noise should be applied to repair
        # heuristics
        self._noise = None
        self.accept = False
        self.accept_best = False

    def _set_start_temperature(self, start_solution):
        """
        Determine and set the start temperature for simulated annealing.
        """
        initial_solution = start_solution
        modified_solution_cost = initial_solution.calculate_solution_cost(0)
        self._temperature = (-
                            (self.start_temp_control_param *
                             modified_solution_cost) /
                             math.log(0.5))

    def _next_iteration(self, start_time):
        """
        Evaluate if next interation is necessary and prepare for it
        """
        #if stop criterium is met alns should stop
        if self._stop_criterium(start_time):
            self._solution = copy.deepcopy(self._previous_solution)
            if self.accept is True:
                self._currently_accepted_solution_cost = copy.deepcopy(self._previous_accepted_solution_cost)
                if self.accept_best is True:
                    self._best_solution = copy.deepcopy(self._previous_best_solution)
                    self._best_solution_cost = copy.deepcopy(self._previous_best_solution_cost)
            else: pass
            return False
        #if stop criterium is not met:
        elif self._stop_criterium(start_time) is False:
            #update alns_counter
            self._alns_counter += 1
            #update temperature
            if self._alns_counter == 1:
                self._temperature = self._temperature
            else: self._temperature = self.cooling_rate * self._temperature
            #after every segment, weights need to be updated (only for ALNS, not LNS)
            """if (self._alns_counter % self.timesegment_length == 0):
                self._update_weights()
                #heuristic counters need to be set to zero at the start of
                #the next segment
                self._destroy_counter = [0.0]*(len(self.destroy_heuristics))
                self._repair_counter = [0.0]*(len(self.repair_heuristics))
                self._noise_counter = [0.0]*2"""
            return True

    def _select_destroy_heuristic(self):
        """
        Sets a specific destroy heuristics in a non-deterministic manner.
        """
        heuristic_id = self._roulette_wheel_selection(self._destroy_weights)
        self._destroy_heuristic_id = heuristic_id
        self._destroy_heuristic = self.destroy_heuristics[heuristic_id]
        #self._destroy_counter[heuristic_id] += 1

    def _select_repair_heuristic(self):
        """
        Sets a specific repair heuristics in a non-deterministic manner.
        """
        heuristic_id = self._roulette_wheel_selection(self._repair_weights)
        self._repair_heuristic_id = heuristic_id
        self._repair_heuristic = self.repair_heuristics[heuristic_id]
        #self._repair_counter[heuristic_id] += 1

    def _select_noise(self):
        """
        Sets whether or not noising is used in the repair heuristic
        """

        noise = (self._roulette_wheel_selection(self._noise_weights) == 1)
        self._noise = noise
        """if self._noise is False:
            self._noise_counter[0] += 1
        elif self._noise is True:
            self._noise_counter[1] += 1"""

    def _destroy_and_repair(self):
        """
        Create a new solution by destroying and repairing the current solution.
        """
        requests_in_request_bank = len(self._solution.request_bank)
        self._destroy_heuristic(self._solution)
        self._repair_heuristic(self._solution, self._noise, self.noise_parameter)
        return self._solution

    def _accept_solution(self, new_solution):
        """
        Test if the new solution is acceptable using simulated annealing.

        Returns
        -------
        boolean
        """

        #1. Calculate acceptance probability
        current_cost = new_solution.calculate_solution_cost()
        try:
            acceptance_probability = math.exp(
            -(current_cost-self._currently_accepted_solution_cost)
            / self._temperature)
        except OverflowError:
            acceptance_probability = float('inf')
        except ZeroDivisionError:
            if (current_cost-self._currently_accepted_solution_cost)>0:
                acceptance_probability = 0.0
            elif (current_cost-self._currently_accepted_solution_cost)<=0:
                acceptance_probability = 1.0
        acceptance_probability = min(1, acceptance_probability)
        #2. Use roulette wheel selection to determine on acceptance
        accept = (self._roulette_wheel_selection([1-acceptance_probability,
                                                 acceptance_probability]) == 1)
        return accept

    def _update_scores(self, new_solution):
        """
        Update the scores that keep track of the performance of the various
        heuristics
        """

        #1. Determine score increment
        current_cost = new_solution.calculate_solution_cost()
        score_increment = 0
        if (current_cost < self._best_solution_cost):
            score_increment = self.score_increaser1
        elif new_solution:
            if (current_cost < self._currently_accepted_solution_cost and
               self._add_solution_to_hashset(new_solution) is True):
                score_increment = self.score_increaser2
            elif (self._accept_solution(new_solution) and
                  self._add_solution_to_hashset(new_solution) is True):
                    score_increment = self.score_increaser3
        #2. Add the score increment to the current score of the applied
        #heuristic
        self._destroy_score[self._destroy_heuristic_id] += score_increment
        self._repair_score[self._repair_heuristic_id] += score_increment
        if self._noise is False:
            self._noise_score[0] += score_increment
        elif self._noise is True:
            self._noise_score[1] += score_increment

    def _update_weights(self):
        """
        Update the weights of the heuristics that are used in the roulette
        wheel selection.
        """
        for w in range(0, len(self._destroy_weights)):
            current_weight = self._destroy_weights[w]
            if self._destroy_counter[w] == 0:
                new_weight = current_weight
            else:
                new_weight = (current_weight
                              * (1 - self.reaction_factor)
                              + self.reaction_factor
                              * (self._destroy_score[w]
                                 / self._destroy_counter[w]))
            self._destroy_weights[w] = new_weight
        for w in range(0, len(self._repair_weights)):
            current_weight = self._repair_weights[w]
            if self._repair_counter[w] == 0:
                new_weight = current_weight
            else:
                new_weight = (current_weight
                              * (1 - self.reaction_factor)
                              + self.reaction_factor
                              * (self._repair_score[w]
                                 / self._repair_counter[w]))
            self._repair_weights[w] = new_weight
        for w in range(0, len(self._noise_weights)):
            current_weight = self._noise_weights[w]
            if self._noise_counter[w] == 0:
                new_weight = current_weight
            else:
                new_weight = (current_weight
                              * (1 - self.reaction_factor)
                              + self.reaction_factor
                              * (self._noise_score[w]
                                 / self._noise_counter[w]))
            self._noise_weights[w] = new_weight

    def _update_solution(self, new_solution):
        """
        Update the currently accepted solution to the provided new solution.
        """
        self._previous_solution = copy.deepcopy(self._solution)
        self._previous_accepted_solution_cost = copy.deepcopy(self._currently_accepted_solution_cost)
        self._solution = copy.deepcopy(new_solution)
        self._currently_accepted_solution_cost = (
            self._solution.calculate_solution_cost())
        #self._add_solution_to_hashset(new_solution)

    def _update_best_solution(self):
        """
        Update the best solution to the currently accepted solution.
        """
        self._previous_best_solution = copy.deepcopy(self._best_solution)
        self._previous_best_solution_cost = copy.deepcopy(self._best_solution_cost)
        self._best_solution = copy.deepcopy(self._solution)
        self._best_solution_cost = self._solution.calculate_solution_cost()
        #self._solution.update_neighbor_graph(self._best_solution_cost)

    def _stop_criterium(self, start_time):
        """
        Stop criterium for alns algorithm:
        The total number of ALNS iterations has reached its maximum.
        """
        #Return value is default False
        returnvalue = False
        self.runtime = time.clock() - start_time
        #After 25000 iterations the ALNS is stopped (Ropke & Pisinger, 2006)
        if (self.runtime >= self.max_runtime):
            returnvalue = True
        return returnvalue

    def _roulette_wheel_selection(self, weights):
        """
        Based on a list of weights, apply a roulette wheel selection.

        Returns
        -------
        i: int
            Index used for selection is returned

        Notes
        ------
        The weights are normalized into probabilities such that they sum to 1
        """
        #In order to prevent ZeroDivisionError when all weights are 0, a small
        #amount is added to weights. This will result in all destroy/repair 
        #heuristics getting the same normalized weight.
        if sum(weights) == 0:
            for index in range(len(weights)):
                weights[index] += 0.000000000001
        normalized_weights = [x / sum(weights) for x in weights]
        upperlimit = [sum(normalized_weights[0:1+i])
                      for i in range(len(normalized_weights))]
        random_number = random.random()
        # Verify random number against the upperlimits.
        for i in range(len(upperlimit)):
            if random_number < upperlimit[i]:
                return i

    def _add_solution_to_hashset(self, new_solution):
        """
        Add the current solution to the hashset of visited solutions.

        Returns
        --------
        returnvalue: boolean
            Returns True if solution was not visited yet, False if solution
            has already been visited.
        """
        #Return value is default False
        returnvalue = False
        #Calculate hash
        #Get list of route objects from solution
        routes = new_solution.routes
        #Retrieve the actual routes
        routes = [r.route for r in routes]
        #Sort routes
        routes.sort()
        #Transform routes to tuples and the list of routes to a tuple of routes
        routes = tuple([tuple(route) for route in routes])
        #Hash the solution
        hashtag = routes.__hash__()
        #Add to hashset
        if hashtag not in self._visited_solutions:
            self._visited_solutions.add(hashtag)
            returnvalue = True
        return returnvalue


class Alns_Solver(object):
    """
    Solves an optimization problem using the ALNS framework.
    
    Parameters
    ----------
    alns_vehicle_minimization : AlnsBase
        The ALNS algorithm used to find the minimum number of vehicles
    alns_optimization : AlnsBase
        The ALNS algorithm used to optimize the problem
    
    Attributes
    ----------
    phi : int
        Maximal number of iterations allowed during the vehicle minimization
        phase (for a single run of the AlnsBase algorithm)
    """
    def __init__(self, alns_vehicle_minimization, alns_optimization, id=1):
        self.id = id
        #self.phi = 50
        self.alns_vm = alns_vehicle_minimization
        self.alns_main = alns_optimization

    def solve(self, problem, start_time):
        """
        Optimize a problem with the Pisinger/Ropke ALNS implementation.

        Parameters
        ----------
        problem : Problem
            The problem object which needs to be optimized

        Returns
        -------
        Optimized solution object for the given problem
        """
        self.start_time = start_time
        #print "get start solution"
        start_solution = self._get_start_solution(problem)
        #print "minimize vehicles"
        start_solution = self._minimize_vehicles(start_solution, self.start_time)
        #print "optimize"
        solution = self._optimize(start_solution, self.start_time)
        return solution

    def _get_start_solution(self, problem):
        """
        Create an initial solution for the given problem.

        The approach followed is described in Pisinger and Ropke (2007). The
        regret-2 repair heuristic is used to add all requests to the available
        routes. Note that it is assumed enough vehicles are available to plan
        each request.

        Parameters
        ----------
        problem : Problem
            The problem object for which an initial solution is created

        Returns
        -------
        Initial solution object for the given problem

        Raises
        ------
        IndexError
            If there are not enough vehicles to assign all requests
        """
        solution = Solution(problem)
        repair_heuristic = repair_factory.produce_insert_regret_k(k=2)
        repair_heuristic(solution, False, 0)
        if len(solution.request_bank) > 0:
            raise IndexError("""There are not enough vehicles to assign all
                             requests""")
        solution.routes = [route for route in solution.routes[:solution._number_of_used_vehicles()]]
        solution._rebuild_insert_matrices()
        return solution

    def _minimize_vehicles(self, solution, start_time):
        """
        Minimize the number required vehicles for a specific solution

        The number of vehicles used in a specific solution is minimized until
        there are not enough vehicles anymore to assign all requests. The
        approach used is described in Pisinger & Ropke (2007) and applies ALNS
        with specific stop criteria.

        Parameters
        ----------
        solution : Solution
            The solution object for which the number of vehicles have to be
            minimized.

        Returns
        -------
        A solution object with a minimized number of vehicles
        """
        self.alns_vm.max_runtime = self.alns_vm.max_runtime*0.2
        solution.available_vehicles = [vehicle_id for vehicle_id in range(0,solution._number_of_used_vehicles())]
        solution_minimized_vehicles = copy.deepcopy(solution)
        while len(solution.request_bank) == 0:
            solution_minimized_vehicles = copy.deepcopy(solution)
            solution.remove_last_vehicle()
            solution = copy.deepcopy(self.alns_vm.solve_problem(solution, self.start_time))
            empty_routes_to_remove = []
            for route in solution.routes:
                if len(route) == 2:
                    empty_routes_to_remove.append(route)                   
            for empty_route in empty_routes_to_remove:
                solution.routes.remove(empty_route)
            solution.available_vehicles = [vehicle_id for vehicle_id in range(0,solution._number_of_used_vehicles())]
        solution_minimized_vehicles._rebuild_insert_matrices()
        return solution_minimized_vehicles

    def _optimize(self, solution, start_time):
        """
        Apply ALNS to further optimize a solution object.

        Parameters
        ----------
        solution : Solution
            The solution object which needs to further optimized
        
        Returns
        -------
        The optimized solution object
        """
        return self.alns_main.solve_problem(solution, self.start_time)


class Alns_PR_Minimization(AlnsBase):
    """
    Extends ALNS_Base used for vehicle minimization.

    The ALNS implementation used by Pisinger and Ropke (2007) during the
    vehicle minization stage. This implementation differs from the base
    implementation in terms of stopping criteria

    [Additional] Attributes
    -----------------------
    tau : int
        The maximum number of iterations allowed without going under the
        tau_treshold.
    tau_treshold : int
        A number of unplanned requests. If the algorithm has tau iterations
        without having less unplanned requests than tau_treshold, the
        algorithm is stopped.
    """

    def __init__(self, seed=12345, number_of_requests=25, factor_lower_bound=0.1, 
                 factor_upper_bound=0.5, random_parameter = 20):
        AlnsBase.__init__(self, seed, number_of_requests, factor_lower_bound, 
                          factor_upper_bound, random_parameter)
        #self.tau = 2000
        #self.tau_treshold = 5

    def _reset_algorithm(self):
        """
        Extends the _reset_algorithm method of ALNS_Base.

        This method adds the reset of the tau_counter.
        """
        AlnsBase._reset_algorithm(self)
        #self.tau_counter = 0

    def _next_iteration(self, start_time):
        """
        Extends the _next_iteration method of ALNS_Base.

        In addition to the base implementation, this method increments the
        tau_counter when appropriate.
        """
        #n_unplanned = len(self._solution.request_bank)
        #if n_unplanned >= self.tau_treshold:
        #    self.tau_counter += 1
        #else:
        #    self.tau_counter = 0
        return AlnsBase._next_iteration(self, start_time)

    def _stop_criterium(self, start_time):
        """
        Extends the _stop_criterium.

        In addition to the base implementation, this method now also checks
        the tau_counter and the number of requests on the request bank to
        determine whether one must stop.
        """
        if len(self._solution.request_bank) == 0:
            self._previous_solution = copy.deepcopy(self._solution)
            self._previous_best_solution = copy.deepcopy(self._solution)
            return True
        #'elif self.tau_counter >= self.tau:
        #    return True
        else:
            return AlnsBase._stop_criterium(self, start_time)
