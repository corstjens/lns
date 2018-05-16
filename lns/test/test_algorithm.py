# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 18:06:06 2014

@author: Benoit
"""
import numpy
from alns.factories import destroy_factory
import random
import pytest



def test_reset_algorithm(alns):
    alns._reset_algorithm()
    assert numpy.alltrue(alns._destroy_score == ([0.0, 0.0]))
    assert numpy.alltrue(alns._repair_score == ([0.0, 0.0, 0.0]))
    assert numpy.alltrue(alns._noise_score == ([0.0, 0.0]))
    assert numpy.alltrue(alns._destroy_weights == ([0.5, 0.5]))
    assert numpy.alltrue(alns._repair_weights == ([1.0/3.0, 1.0/3.0, 1.0/3.0]))
    assert numpy.alltrue(alns._noise_weights == ([0.5, 0.5]))
    assert numpy.alltrue(alns._destroy_counter == ([0, 0]))
    assert numpy.alltrue(alns._repair_counter == ([0, 0, 0]))
    assert numpy.alltrue(alns._noise_counter == ([0, 0]))


#score_increaser1 voldaan
def test_update_scores1(alns, solution):
    alns._reset_algorithm()
    alns._noise = False
    alns._best_solution_cost = 305000
    alns._currently_accepted_solution_cost = 305000
    alns._temperature = 10
    alns._destroy_heuristic_id = 1
    alns._repair_heuristic_id = 1
    alns._update_scores(solution)
    assert numpy.alltrue(alns._destroy_score == ([0, 33]))
    assert numpy.alltrue(alns._repair_score == ([0, 33, 0]))
    assert numpy.alltrue(alns._noise_score == ([33, 0]))


def test_update_scores1_noise(alns, solution):
    alns._reset_algorithm()
    alns._noise = True
    alns._best_solution_cost = 305000
    alns._currently_accepted_solution_cost = 305000
    alns._temperature = 10
    alns._destroy_heuristic_id = 1
    alns._repair_heuristic_id = 1
    alns._update_scores(solution)
    assert numpy.alltrue(alns._destroy_score == ([0, 33]))
    assert numpy.alltrue(alns._repair_score == ([0, 33, 0]))
    assert numpy.alltrue(alns._noise_score == ([0, 33]))


def test_update_scores1_no_noise(alns, solution):
    alns._reset_algorithm()
    alns._noise = False
    alns._best_solution_cost = 305000
    alns._currently_accepted_solution_cost = 305000
    alns._temperature = 10
    alns._destroy_heuristic_id = 1
    alns._repair_heuristic_id = 1
    alns._update_scores(solution)
    assert numpy.alltrue(alns._destroy_score == ([0, 33]))
    assert numpy.alltrue(alns._repair_score == ([0, 33, 0]))
    assert numpy.alltrue(alns._noise_score == ([33, 0]))


#score_increaser2 voldaan
def test_update_scores2(alns, solution):
    alns._reset_algorithm()
    alns._best_solution_cost = 10
    alns._currently_accepted_solution_cost = 305000
    alns._temperature = 10
    alns._destroy_heuristic_id = 1
    alns._repair_heuristic_id = 1

    def temp(x):
        return True
    alns._add_solution_to_hashset = temp
    alns._update_scores(solution)
    assert numpy.alltrue(alns._destroy_score == ([0, 9]))
    assert numpy.alltrue(alns._repair_score == ([0, 9, 0]))


#score_increaser3 voldaan
def test_update_scores3(alns, solution):
    alns._reset_algorithm()
    alns._best_solution_cost = 10
    alns._currently_accepted_solution_cost = 10
    alns._temperature = 10
    alns._destroy_heuristic_id = 1
    alns._repair_heuristic_id = 1

    def temp(x):
        return True
    alns._accept_solution = temp

    def temp1(x):
        return True
    alns._add_solution_to_hashset = temp1
    alns._update_scores(solution)
    assert numpy.alltrue(alns._destroy_score == ([0, 13]))
    assert numpy.alltrue(alns._repair_score == ([0, 13, 0]))


#score_increaser3 niet voldaan (accept = False)
def test_update_scores4(alns, solution):
    alns._reset_algorithm()
    alns._best_solution_cost = 10
    alns._currently_accepted_solution_cost = 10
    alns._temperature = 10
    alns._destroy_heuristic_id = 1
    alns._repair_heuristic_id = 1

    def temp(x):
        return False
    alns._accept_solution = temp

    def temp1(x):
        return True
    alns._add_solution_to_hashset = temp1
    alns._update_scores(solution)
    assert numpy.alltrue(alns._destroy_score == ([0, 0]))
    assert numpy.alltrue(alns._repair_score == ([0, 0, 0]))


#score_increaser3 niet voldaan (no new solution)
def test_update_scores5(alns, solution):
    alns._reset_algorithm()
    alns._best_solution_cost = 10
    alns._currently_accepted_solution_cost = 10
    alns._temperature = 10
    alns._destroy_heuristic_id = 1
    alns._repair_heuristic_id = 1

    def temp(x):
        return True
    alns._accept_solution = temp

    def temp1(x):
        return False
    alns._add_solution_to_hashset = temp1
    alns._update_scores(solution)
    assert numpy.alltrue(alns._destroy_score == ([0, 0]))
    assert numpy.alltrue(alns._repair_score == ([0, 0, 0]))


#score_increaser2 niet voldaan (no new solution)
def test_update_scores6(alns, solution):
    alns._reset_algorithm()
    alns._best_solution_cost = 10
    alns._currently_accepted_solution_cost = 305000
    alns._temperature = 10
    alns._destroy_heuristic_id = 1
    alns._repair_heuristic_id = 1

    def temp1(x):
        return False
    alns._add_solution_to_hashset = temp1
    alns._update_scores(solution)
    assert numpy.alltrue(alns._destroy_score == ([0, 0]))
    assert numpy.alltrue(alns._repair_score == ([0, 0, 0]))


#score_increaser2 niet voldaan (cost > currently accepted cost)
def test_update_scores7(alns, solution):
    alns._reset_algorithm()
    alns._best_solution_cost = 10
    alns._currently_accepted_solution_cost = 10
    alns._temperature = 10
    alns._destroy_heuristic_id = 1
    alns._repair_heuristic_id = 1

    def temp1(x):
        return True
    alns._add_solution_to_hashset = temp1
    alns._update_scores(solution)
    assert numpy.alltrue(alns._destroy_score == ([0, 0]))
    assert numpy.alltrue(alns._repair_score == ([0, 0, 0]))


#score_increaser1 niet voldaan
def test_update_scores8(alns, solution):
    alns._reset_algorithm()
    alns._best_solution_cost = 10
    alns._currently_accepted_solution_cost = 10
    alns._temperature = 10
    alns._destroy_heuristic_id = 1
    alns._repair_heuristic_id = 1

    def temp(x):
        return False
    alns._accept_solution = temp

    def temp1(x):
        return False
    alns._add_solution_to_hashset = temp1
    alns._update_scores(solution)
    assert numpy.alltrue(alns._destroy_score == ([0, 0]))
    assert numpy.alltrue(alns._repair_score == ([0, 0, 0]))


def test_select_destroy_heuristic(alns):
    alns._reset_algorithm()

    def temp(x):
        return 0
    alns._roulette_wheel_selection = temp
    alns._select_destroy_heuristic()
    assert (alns._destroy_heuristic ==
            alns.destroy_heuristics[0])


def test_select_destroy_heuristic2(alns):
    alns._reset_algorithm()

    def temp(x):
        return 1
    alns._roulette_wheel_selection = temp
    alns._select_destroy_heuristic()
    assert (alns._destroy_heuristic ==
            alns.destroy_heuristics[1])


def test_select_repair_heuristic(alns):
    alns._reset_algorithm()

    def temp(x):
        return 0
    alns._roulette_wheel_selection = temp
    alns._select_repair_heuristic()
    assert (alns._repair_heuristic ==
            alns.repair_heuristics[0])


def test_select_repair_heuristic2(alns):
    alns._reset_algorithm()

    def temp(x):
        return 1
    alns._roulette_wheel_selection = temp
    alns._select_repair_heuristic()
    assert (alns._repair_heuristic ==
            alns.repair_heuristics[1])


def test_select_repair_heuristic3(alns):
    alns._reset_algorithm()

    def temp(x):
        return 2
    alns._roulette_wheel_selection = temp
    alns._select_repair_heuristic()
    assert (alns._repair_heuristic ==
            alns.repair_heuristics[2])


def test_select_noise1(alns):
    alns._reset_algorithm()

    def temp(x):
        return 0
    alns._roulette_wheel_selection = temp
    alns._select_noise()
    assert (alns._noise is False)
    assert numpy.alltrue(alns._noise_counter == [1, 0])


def test_select_noise2(alns):
    alns._reset_algorithm()

    def temp(x):
        return 1
    alns._roulette_wheel_selection = temp
    alns._select_noise()
    assert (alns._noise is True)
    assert numpy.alltrue(alns._noise_counter == [0, 1])


def test_update_weights1(alns):
    alns._reset_algorithm()
    alns._destroy_counter = ([1, 1])
    alns._repair_counter = ([1, 1, 1])
    alns._noise_counter = ([1, 1])
    alns._update_weights()
    assert numpy.alltrue(alns._destroy_weights == ([0.45, 0.45]))
    assert numpy.alltrue(alns._repair_weights == ([0.3, 0.3, 0.3]))
    assert numpy.alltrue(alns._noise_weights == ([0.45, 0.45]))


def test_update_weights2(alns):
    alns._reset_algorithm()
    alns._destroy_counter = ([1, 1])
    alns._repair_counter = ([1, 1, 1])
    alns._noise_counter = ([1, 1])
    alns._destroy_score = ([1.0, 1.0])
    alns._noise_score = ([1.0, 1.0])
    alns._repair_score = ([1.0, 1.0, 1.0])
    alns._update_weights()
    assert numpy.alltrue(alns._destroy_weights == ([0.55, 0.55]))
    assert numpy.alltrue(alns._repair_weights == ([0.4, 0.4, 0.4]))
    assert numpy.alltrue(alns._noise_weights == ([0.55, 0.55]))


def test_update_weights3(alns):
    alns._reset_algorithm()
    alns._destroy_weights = ([1.0/3.0, 2.0/3.0])
    alns._repair_weights = ([1.0/6.0, 3.0/6.0, 2.0/6.0])
    alns._noise_weights = ([1.0/3.0, 2.0/3.0])
    alns._destroy_counter = ([1, 1])
    alns._repair_counter = ([1, 1, 1])
    alns._noise_counter = ([1, 1])
    alns._update_weights()
    assert numpy.alltrue(alns._destroy_weights == ([0.3, 0.6]))
    assert numpy.alltrue(alns._repair_weights == ([0.15, 0.45, 0.3]))
    assert numpy.alltrue(alns._noise_weights == ([0.3, 0.6]))


def test_update_weights4(alns):
    alns._reset_algorithm()
    alns._destroy_weights = ([1.0/3.0, 2.0/3.0])
    alns._repair_weights = ([1.0/6.0, 3.0/6.0, 2.0/6.0])
    alns._noise_weights = ([1.0/3.0, 2.0/3.0])
    alns._destroy_counter = ([1, 1])
    alns._repair_counter = ([1, 1, 1])
    alns._noise_counter = ([1, 1])
    alns._destroy_score = ([1.0, 1.0])
    alns._repair_score = ([1.0, 1.0, 1.0])
    alns._noise_score = ([1.0, 1.0])
    alns._update_weights()
    assert numpy.alltrue(alns._destroy_weights == ([0.4, 0.7]))
    assert numpy.alltrue(alns._repair_weights == ([0.25, 0.55, 0.4]))
    assert numpy.alltrue(alns._noise_weights == ([0.4, 0.7]))


def test_update_weights5(alns):
    alns._reset_algorithm()
    alns._destroy_counter = ([2, 2])
    alns._repair_counter = ([2, 2, 2])
    alns._noise_counter = ([2, 2])
    alns._update_weights()
    assert numpy.alltrue(alns._destroy_weights == ([0.45, 0.45]))
    assert numpy.alltrue(alns._repair_weights == ([0.3, 0.3, 0.3]))
    assert numpy.alltrue(alns._noise_weights == ([0.45, 0.45]))


def test_update_weights6(alns):
    alns._reset_algorithm()
    alns._destroy_counter = ([2, 2])
    alns._repair_counter = ([2, 2, 2])
    alns._noise_counter = ([2, 2])
    alns._destroy_score = ([1.0, 1.0])
    alns._repair_score = ([1.0, 1.0, 1.0])
    alns._noise_score = ([1.0, 1.0])
    alns._update_weights()
    assert numpy.alltrue(alns._destroy_weights == ([0.5, 0.5]))
    assert numpy.alltrue(alns._repair_weights == ([0.35, 0.35, 0.35]))
    assert numpy.alltrue(alns._noise_weights == ([0.5, 0.5]))


def test_update_weights7(alns):
    alns._reset_algorithm()
    alns._destroy_weights = ([1.0/3.0, 2.0/3.0])
    alns._repair_weights = ([1.0/6.0, 3.0/6.0, 2.0/6.0])
    alns._noise_weights = ([1.0/3.0, 2.0/3.0])
    alns._destroy_counter = ([2, 2])
    alns._repair_counter = ([2, 2, 2])
    alns._noise_counter = ([2, 2])
    alns._update_weights()
    assert numpy.alltrue(alns._destroy_weights == ([0.3, 0.6]))
    assert numpy.alltrue(alns._repair_weights == ([0.15, 0.45, 0.3]))
    assert numpy.alltrue(alns._noise_weights == ([0.3, 0.6]))


def test_update_weights8(alns):
    alns._reset_algorithm()
    alns._destroy_weights = ([1.0/3.0, 2.0/3.0])
    alns._repair_weights = ([1.0/6.0, 3.0/6.0, 2.0/6.0])
    alns._noise_weights = ([1.0/3.0, 2.0/3.0])
    alns._destroy_counter = ([2, 2])
    alns._repair_counter = ([2, 2, 2])
    alns._noise_counter = ([2, 2])
    alns._destroy_score = ([1.0, 1.0])
    alns._repair_score = ([1.0, 1.0, 1.0])
    alns._noise_score = ([1.0, 1.0])
    alns._update_weights()
    assert numpy.alltrue(alns._destroy_weights == ([0.35, 0.65]))
    assert numpy.alltrue(alns._repair_weights == ([0.20, 0.5, 0.35]))
    assert numpy.alltrue(alns._noise_weights == ([0.35, 0.65]))


def test_update_solution(alns, solution):
    alns._reset_algorithm()
    alns._solution = solution
    noise = False
    solution_manual = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    heuristic = alns.repair_heuristics[1]
    heuristic(solution, noise)
    found_solution = {0: alns._solution.routes[0].route,
                      1: alns._solution.routes[1].route}
    alns._update_solution(solution)
    assert (found_solution == solution_manual)
    assert (round(alns._currently_accepted_solution_cost, 2) == 82.20)
    assert (alns._add_solution_to_hashset(alns._solution) is False)


def test_update_best_solution(alns, solution):
    alns._reset_algorithm()
    alns._solution = solution
    alns._best_solution = solution
    noise = False
    solution_manual = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    heuristic = alns.repair_heuristics[1]
    heuristic(solution, noise)
    found_solution = {0: alns._best_solution.routes[0].route,
                      1: alns._best_solution.routes[1].route}
    alns._update_solution(solution)
    alns._update_best_solution()
    assert (found_solution == solution_manual)
    assert (round(alns._best_solution_cost, 2) == 82.20)


def test_accept_solution1(alns, solution):
    alns._reset_algorithm()
    alns._currently_accepted_solution_cost = 310000
    alns._temperature = 10
    assert (alns._accept_solution(solution) is True)


def test_accept_solution2(alns, solution):
    alns._reset_algorithm()
    alns._currently_accepted_solution_cost = 280000
    alns._temperature = 10
    assert (alns._accept_solution(solution) is False)


def test_stop_criterium1(alns):
    alns._reset_algorithm()
    assert (alns._stop_criterium() is False)


def test_stop_criterium2(alns):
    alns._reset_algorithm()
    alns._alns_counter = 25000
    assert (alns._stop_criterium() is True)


def test_next_iteration1(alns):
    alns._reset_algorithm()
    alns._alns_counter = 25000
    assert (alns._next_iteration() is False)


def test_next_iteration2(alns):
    alns._reset_algorithm()
    alns._temperature = 10
    alns._destroy_counter = ([1, 1])
    alns._repair_counter = ([1, 1, 1])
    alns._noise_counter = ([1, 1])
    alns._next_iteration()
    assert (alns._alns_counter == 1)
    assert (alns._temperature == 9.9975)
    assert numpy.alltrue(alns._destroy_weights == ([0.5, 0.5]))
    assert numpy.alltrue(alns._repair_weights == ([1.0/3.0, 1.0/3.0, 1.0/3.0]))
    assert numpy.alltrue(alns._destroy_counter == ([1, 1]))
    assert numpy.alltrue(alns._repair_counter == ([1, 1, 1]))
    assert numpy.alltrue(alns._noise_counter == ([1, 1]))
    assert (alns._next_iteration() is True)


def test_next_iteration3(alns):
    alns._reset_algorithm()
    alns.number_of_iterations = 25000
    alns._temperature = 10
    alns._destroy_counter = ([1, 1])
    alns._repair_counter = ([1, 1, 1])
    alns._noise_counter = ([1, 1])
    alns._alns_counter = 99
    alns._next_iteration()
    assert (alns._alns_counter == 100)
    assert (alns._temperature == 9.9975)
    assert numpy.alltrue(alns._destroy_weights == ([0.45, 0.45]))
    assert numpy.alltrue(alns._repair_weights == ([0.3, 0.3, 0.3]))
    assert numpy.alltrue(alns._destroy_counter == ([0, 0]))
    assert numpy.alltrue(alns._repair_counter == ([0, 0, 0]))
    assert numpy.alltrue(alns._noise_counter == ([0, 0]))
    assert (alns._next_iteration() is True)


#parallel insertion & random removal(3)
def test_destroy_and_repair1_1(alns, solution):
    alns._reset_algorithm()
    alns._noise = False
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, alns._noise)
    alns._solution = solution
    alns._destroy_heuristic = destroy_factory.produce_remove_random(3)
    alns._repair_heuristic = alns.repair_heuristics[0]
    possible_solution1 = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    possible_solution2 = {0: [6, 8], 1: [7, 1, 2, 0, 4, 5, 3, 9]}
    found_solution = {0: alns._solution.routes[0].route,
                      1: alns._solution.routes[1].route}
    alns._destroy_and_repair()
    assert (found_solution == possible_solution1
            or found_solution == possible_solution2)


#parallel insertion & random removal(1)
def test_destroy_and_repair1_2(alns, solution):
    alns._reset_algorithm()
    alns._noise = False
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, alns._noise)
    alns._solution = solution
    alns._destroy_heuristic = destroy_factory.produce_remove_random(1)
    alns._repair_heuristic = alns.repair_heuristics[0]
    possible_solution1 = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    possible_solution2 = {0: [6, 8], 1: [7, 1, 2, 0, 4, 5, 3, 9]}
    possible_solution3 = {0: [6, 1, 0, 2, 4, 5, 3, 8], 1: [7, 9]}
    possible_solution4 = {0: [6, 8], 1: [7, 1, 0, 2, 4, 5, 3, 9]}
    possible_solution5 = {0: [6, 2, 0, 1, 4, 5, 3, 8], 1: [7, 9]}
    possible_solution6 = {0: [6, 8], 1: [7, 2, 0, 1, 4, 5, 3, 9]}
    found_solution = {0: alns._solution.routes[0].route,
                      1: alns._solution.routes[1].route}
    alns._destroy_and_repair()
    assert (found_solution == possible_solution1
            or found_solution == possible_solution2
            or found_solution == possible_solution3
            or found_solution == possible_solution4
            or found_solution == possible_solution5
            or found_solution == possible_solution6)


#parallel insertion & worst removal
def test_destroy_and_repair2(alns, solution):
    alns._reset_algorithm()
    alns._noise = False
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, alns._noise)
    alns._solution = solution
    alns._destroy_heuristic = destroy_factory.produce_remove_worst(3, 2)
    alns._repair_heuristic = alns.repair_heuristics[0]
    possible_solution1 = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    possible_solution2 = {0: [6, 8], 1: [7, 1, 2, 0, 4, 5, 3, 9]}
    found_solution = {0: alns._solution.routes[0].route,
                      1: alns._solution.routes[1].route}
    alns._destroy_and_repair()
    assert (found_solution == possible_solution1
            or found_solution == possible_solution2)


#sequential insertion & random removal(3)
def test_destroy_and_repair3_1(alns, solution):
    alns._reset_algorithm()
    alns._noise = False
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, alns._noise)
    alns._solution = solution
    alns._destroy_heuristic = destroy_factory.produce_remove_random(3)
    alns._repair_heuristic = alns.repair_heuristics[1]
    solution_manual = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    found_solution = {0: alns._solution.routes[0].route,
                      1: alns._solution.routes[1].route}
    alns._destroy_and_repair()
    assert (found_solution == solution_manual)


#sequential insertion & random removal(1)
def test_destroy_and_repair3_2(alns, solution):
    alns._reset_algorithm()
    alns._noise = False
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, alns._noise)
    alns._solution = solution
    alns._destroy_heuristic = destroy_factory.produce_remove_random(1)
    alns._repair_heuristic = alns.repair_heuristics[1]
    possible_solution1 = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    possible_solution2 = {0: [6, 1, 0, 2, 4, 5, 3, 8], 1: [7, 9]}
    possible_solution3 = {0: [6, 2, 0, 1, 4, 5, 3, 8], 1: [7, 9]}
    found_solution = {0: alns._solution.routes[0].route,
                      1: alns._solution.routes[1].route}
    alns._destroy_and_repair()
    assert (found_solution == possible_solution1
            or found_solution == possible_solution2
            or found_solution == possible_solution3)


#sequential insertion & worst removal
def test_destroy_and_repair4(alns, solution):
    alns._reset_algorithm()
    alns._noise = False
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, alns._noise)
    alns._solution = solution
    alns._destroy_heuristic = destroy_factory.produce_remove_worst(3, 2)
    alns._repair_heuristic = alns.repair_heuristics[1]
    solution_manual = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    found_solution = {0: alns._solution.routes[0].route,
                      1: alns._solution.routes[1].route}
    alns._destroy_and_repair()
    assert (found_solution == solution_manual)


#regret_2 insertion & random removal(3)
def test_destroy_and_repair5_1(alns, solution):
    alns._reset_algorithm()
    alns._noise = False
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, alns._noise)
    alns._solution = solution
    alns._destroy_heuristic = destroy_factory.produce_remove_random(3)
    alns._repair_heuristic = alns.repair_heuristics[2]
    solution_manual = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    found_solution = {0: alns._solution.routes[0].route,
                      1: alns._solution.routes[1].route}
    alns._destroy_and_repair()
    assert (found_solution == solution_manual)


#regret_2 insertion & random removal(1)
def test_destroy_and_repair5_2(alns, solution):
    alns._reset_algorithm()
    alns._noise = False
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, alns._noise)
    alns._solution = solution
    alns._destroy_heuristic = destroy_factory.produce_remove_random(1)
    alns._repair_heuristic = alns.repair_heuristics[2]
    possible_solution1 = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    possible_solution2 = {0: [6, 1, 0, 2, 4, 5, 3, 8], 1: [7, 9]}
    possible_solution3 = {0: [6, 2, 0, 1, 4, 5, 3, 8], 1: [7, 9]}
    found_solution = {0: alns._solution.routes[0].route,
                      1: alns._solution.routes[1].route}
    alns._destroy_and_repair()
    assert (found_solution == possible_solution1
            or found_solution == possible_solution2
            or found_solution == possible_solution3)


#regret_2 insertion & worst removal
def test_destroy_and_repair6(alns, solution):
    alns._reset_algorithm()
    alns._noise = False
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, alns._noise)
    alns._solution = solution
    alns._destroy_heuristic = destroy_factory.produce_remove_worst(3, 2)
    alns._repair_heuristic = alns.repair_heuristics[2]
    solution_manual = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    found_solution = {0: alns._solution.routes[0].route,
                      1: alns._solution.routes[1].route}
    alns._destroy_and_repair()
    assert (found_solution == solution_manual)


def test_roulette_wheel_selection1(alns, monkeypatch):
    alns._reset_algorithm()

    def temp():
        return 0.7
    monkeypatch.setattr(random,'random',temp)
    #random.random = temp
    #DANGER. This modifies random.random also for all future calls!
    assert(alns._roulette_wheel_selection(alns._destroy_weights) == 1)


def test_roulette_wheel_selection2(alns, monkeypatch):
    alns._reset_algorithm()

    def temp():
        return 0.4
    monkeypatch.setattr(random,'random',temp)
    #random.random = temp
    #DANGER. This modifies random.random also for all future calls!
    assert(alns._roulette_wheel_selection(alns._destroy_weights) == 0)


def test_roulette_wheel_selection3(alns, monkeypatch):
    alns._reset_algorithm()
    weights = [3.0/15, 5.0/15, 7.0/15]

    def temp():
        return 2.0/15
    monkeypatch.setattr(random,'random',temp)
    #random.random = temp
    #DANGER. This modifies random.random also for all future calls!
    assert(alns._roulette_wheel_selection(weights) == 0)


def test_roulette_wheel_selection4(alns, monkeypatch):
    alns._reset_algorithm()
    weights = [3.0/15, 5.0/15, 7.0/15]

    def temp():
        return 3.0/15
    monkeypatch.setattr(random,'random',temp)
    #random.random = temp
    #DANGER. This modifies random.random also for all future calls!
    assert(alns._roulette_wheel_selection(weights) == 1)


def test_roulette_wheel_selection5(alns, monkeypatch):
    alns._reset_algorithm()
    weights = [3.0/15, 5.0/15, 7.0/15]

    def temp():
        return 9.0/15
    monkeypatch.setattr(random,'random',temp)
    #random.random = temp
    #DANGER. This modifies random.random also for all future calls!
    assert(alns._roulette_wheel_selection(weights) == 2)


def test_add_solution_to_hashset(alns, solution):
    alns._reset_algorithm()
    assert (alns._add_solution_to_hashset(solution) is True)


def test_add_solution_to_hashset2(alns, solution):
    alns._reset_algorithm()
    alns._update_solution(solution)
    assert (alns._add_solution_to_hashset(solution) is False)


def test_add_solution_to_hashset3(alns, solution):
    alns._reset_algorithm()
    noise = False
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, noise)
    assert (alns._add_solution_to_hashset(solution) is True)


def test_add_solution_to_hashset4(alns, solution):
    alns._reset_algorithm()
    noise = False
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, noise)
    alns._update_solution(solution)
    assert (alns._add_solution_to_hashset(solution) is False)


def test_get_start_solution(alnspr2007, vrptw_problem):
    start_solution = alnspr2007._get_start_solution(vrptw_problem)
    solution_manual = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    found_solution = {0: start_solution.routes[0].route,
                      1: start_solution.routes[1].route}
    assert (found_solution == solution_manual)


def test_reset_algorithm_minimization(alnsprminimization):
    alnsprminimization._reset_algorithm()
    assert numpy.alltrue(alnsprminimization._destroy_score == ([0.0, 0.0]))
    assert numpy.alltrue(alnsprminimization._repair_score == ([0.0, 0.0, 0.0]))
    assert numpy.alltrue(alnsprminimization._noise_score == ([0.0, 0.0]))
    assert numpy.alltrue(alnsprminimization._destroy_weights == ([0.5, 0.5]))
    assert numpy.alltrue(alnsprminimization._repair_weights ==
                        ([1.0/3.0, 1.0/3.0, 1.0/3.0]))
    assert numpy.alltrue(alnsprminimization._noise_weights == ([0.5, 0.5]))
    assert numpy.alltrue(alnsprminimization._destroy_counter == ([0, 0]))
    assert numpy.alltrue(alnsprminimization._repair_counter == ([0, 0, 0]))
    assert numpy.alltrue(alnsprminimization._noise_counter == ([0, 0]))
    assert (alnsprminimization.tau_counter == 0)


def test_next_iteration_minimization1(alnsprminimization, solution):
    alnsprminimization._reset_algorithm()
    alnsprminimization._solution = solution
    alnsprminimization._alns_counter = 25000
    assert (alnsprminimization._next_iteration() is False)
    assert (alnsprminimization.tau_counter == 0)


def test_next_iteration_minimization2(alnsprminimization, solution):
    alnsprminimization._reset_algorithm()
    alnsprminimization._solution = solution
    alnsprminimization._alns_counter = 25000
    alnsprminimization.tau_treshold = 2
    assert (alnsprminimization._next_iteration() is False)
    assert (alnsprminimization.tau_counter == 1)


def test_next_iteration_minimization3(alnsprminimization, solution):
    alnsprminimization._reset_algorithm()
    alnsprminimization._solution = solution
    alnsprminimization._temperature = 10
    alnsprminimization._destroy_counter = ([1, 1])
    alnsprminimization._repair_counter = ([1, 1, 1])
    alnsprminimization._noise_counter = ([1, 1])
    assert (alnsprminimization._next_iteration() is True)
    assert (alnsprminimization.tau_counter == 0)
    assert (alnsprminimization._alns_counter == 1)
    assert (alnsprminimization._temperature == 9.9975)
    assert numpy.alltrue(alnsprminimization._destroy_weights == ([0.5, 0.5]))
    assert numpy.alltrue(alnsprminimization._repair_weights ==
                        ([1.0/3.0, 1.0/3.0, 1.0/3.0]))
    assert numpy.alltrue(alnsprminimization._destroy_counter == ([1, 1]))
    assert numpy.alltrue(alnsprminimization._repair_counter == ([1, 1, 1]))
    assert numpy.alltrue(alnsprminimization._noise_counter == ([1, 1]))


def test_next_iteration_minimization4(alnsprminimization, solution):
    alnsprminimization._reset_algorithm()
    alnsprminimization.number_of_iterations = 25000
    alnsprminimization._solution = solution
    alnsprminimization._temperature = 10
    alnsprminimization._destroy_counter = ([1, 1])
    alnsprminimization._repair_counter = ([1, 1, 1])
    alnsprminimization._noise_counter = ([1, 1])
    alnsprminimization._alns_counter = 99
    alnsprminimization.tau_treshold = 2
    assert (alnsprminimization._next_iteration() is True)
    assert (alnsprminimization.tau_counter == 1)
    assert (alnsprminimization._alns_counter == 100)
    assert (alnsprminimization._temperature == 9.9975)
    assert numpy.alltrue(alnsprminimization._destroy_weights == ([0.45, 0.45]))
    assert numpy.alltrue(alnsprminimization._repair_weights ==
                        ([0.3, 0.3, 0.3]))
    assert numpy.alltrue(alnsprminimization._destroy_counter == ([0, 0]))
    assert numpy.alltrue(alnsprminimization._repair_counter == ([0, 0, 0]))
    assert numpy.alltrue(alnsprminimization._noise_counter == ([0, 0]))


def test_stop_criterium_minimization1(alnsprminimization, solution):
    alnsprminimization._reset_algorithm()
    alnsprminimization._solution = solution
    alnsprminimization._solution.request_bank = []
    assert (alnsprminimization._stop_criterium() is True)


def test_stop_criterium_minimization2(alnsprminimization, solution):
    alnsprminimization._reset_algorithm()
    alnsprminimization._solution = solution
    alnsprminimization.tau_counter = 6
    assert (alnsprminimization._stop_criterium() is True)


def test_stop_criterium_minimization3(alnsprminimization, solution):
    alnsprminimization._reset_algorithm()
    alnsprminimization._solution = solution
    assert (alnsprminimization._stop_criterium() is False)


def test_stop_criterium_minimization4(alnsprminimization, solution):
    alnsprminimization._reset_algorithm()
    alnsprminimization._solution = solution
    alnsprminimization._alns_counter = 25000
    assert (alnsprminimization._stop_criterium() is True)

if __name__ == "__main__":
    pytest.main()