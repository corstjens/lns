# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:57:30 2014

@author: lucp2487
"""

import numpy
import pytest
from alns.factories import repair_factory
import copy
import random


def test_noisematrix(solution):

    def temp(x, y):
        return 0.8
    random.uniform = temp
    noisematrix = solution._calculate_noisematrix()
    delta_f_manual = numpy.array([[31.26, 31.26], [36.8, 36.8],
                                 [45.52, 45.52]])
    solution._delta_f = numpy.add(solution._delta_f, noisematrix)
    solution._delta_f = numpy.around(solution._delta_f, 2)
    assert numpy.alltrue(solution._delta_f == delta_f_manual)


def test_insert_greedy_sequential(alns, solution):
    noise = False
    solution_manual = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    heuristic = alns.repair_heuristics[1]
    heuristic(solution, noise)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == solution_manual)


def test_insert_greedy_sequential_noise(alns, solution):
    noise = True

    def temp(x, y):
        return 0.8
    random.uniform = temp
    solution_manual = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    heuristic = alns.repair_heuristics[1]
    heuristic(solution, noise)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == solution_manual)


def test_insert_greedy_sequential2(alns, solution2):
    noise = False
    solution_manual = {0: [6, 2, 0, 5, 3, 8], 1: [7, 1, 4, 9]}
#    pytest.set_trace()
    heuristic = alns.repair_heuristics[1]
    heuristic(solution2, noise)
    found_solution = {0: solution2.routes[0].route,
                      1: solution2.routes[1].route}
    assert (found_solution == solution_manual)


def test_insert_greedy_parallel(alns, solution):
    noise = False
    possible_solution1 = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    possible_solution2 = {0: [6, 8], 1: [7, 1, 2, 0, 4, 5, 3, 9]}
    heuristic = alns.repair_heuristics[0]
    heuristic(solution, noise)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == possible_solution1
            or found_solution == possible_solution2)


def test_insert_greedy_parallel_noise(alns, solution):
    noise = True

    def temp(x, y):
        return 0.8
    random.uniform = temp
    possible_solution1 = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    possible_solution2 = {0: [6, 8], 1: [7, 1, 2, 0, 4, 5, 3, 9]}
    heuristic = alns.repair_heuristics[0]
    heuristic(solution, noise)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == possible_solution1
            or found_solution == possible_solution2)


def test_insert_greedy_parallel2(alns, solution2):
    noise = False
    possible_solution1 = {0: [6, 2, 0, 5, 3, 8], 1: [7, 1, 4, 9]}
    possible_solution2 = {0: [6, 1, 4, 8], 1: [7, 2, 0, 5, 3, 9]}
    #pytest.set_trace()
    heuristic = alns.repair_heuristics[0]
    heuristic(solution2, noise)
    found_solution = {0: solution2.routes[0].route,
                      1: solution2.routes[1].route}
    assert (found_solution == possible_solution1
            or found_solution == possible_solution2)


def test_insert_regret_2(alns, solution):
    noise = False
    solution_manual = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    heuristic = alns.repair_heuristics[2]
    heuristic(solution, noise)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == solution_manual)


def test_insert_regret_2_noise(alns, solution):
    noise = True

    def temp(x, y):
        return 0.8
    random.uniform = temp
    solution_manual = {0: [6, 1, 2, 0, 4, 5, 3, 8], 1: [7, 9]}
    heuristic = alns.repair_heuristics[2]
    heuristic(solution, noise)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == solution_manual)


def test_insert_regret_2_solution2(alns, solution2):
    noise = False
    solution_manual = {0: [6, 2, 0, 5, 3, 8], 1: [7, 1, 4, 9]}
    heuristic = alns.repair_heuristics[2]
    heuristic(solution2, noise)
    found_solution = {0: solution2.routes[0].route,
                      1: solution2.routes[1].route}
    assert (found_solution == solution_manual)


def test_insert_regret_k_basic(solution3):
    noise = False
    #pytest.set_trace()
    solution3._delta_f = numpy.array([[15, 9, 12], [8, 6, 4], [3, 2, 1]],
                                     dtype=float)
    heuristic = repair_factory.produce_insert_regret_k(single_run=True)
    heuristic(solution3, noise)
    #expected = numpy.array([3, 2, 1], dtype = float)
    #expected = numpy.array([0])
    #expected = 1
    expected = [7, 0, 3, 10]
    outcome = solution3.routes[1].route
    assert(expected == outcome)


def test_insert_regret_k_basic_noise(solution3):
    noise = True

    def temp(x, y):
        return 0.8
    random.uniform = temp
    #pytest.set_trace()
    solution3._delta_f = numpy.array([[15, 9, 12], [8, 6, 4], [3, 2, 1]],
                                     dtype=float)
    heuristic = repair_factory.produce_insert_regret_k(single_run=True)
    heuristic(solution3, noise)
    #expected = numpy.array([3, 2, 1], dtype = float)
    #expected = numpy.array([0])
    #expected = 1
    expected = [7, 0, 3, 10]
    outcome = solution3.routes[1].route
    assert(expected == outcome)


def test_insert_regret_k3_basic(solution3):
    noise = False
    #pytest.set_trace()
    solution3._delta_f = numpy.array([[15, 9, 12], [18, 6, 4], [3, 2, 1]],
                                     dtype=float)
    heuristic = repair_factory.produce_insert_regret_k(3, single_run=True)
    heuristic(solution3, noise)
    #expected = numpy.array([9, 16, 3], dtype = float)
    #expected = numpy.array([1])
    #expected = 2
    expected = [8, 1, 4, 11]
    outcome = solution3.routes[2].route
    assert(expected == outcome)


def test_insert_regret_k3_basic_with_ties(solution3):
    noise = False
#    pytest.set_trace()
    solution3._delta_f = numpy.array([[22, 9, 12], [18, 6, 4], [3, 2, 1]],
                                     dtype=float)
    heuristic = repair_factory.produce_insert_regret_k(3, single_run=True)
    heuristic(solution3, noise)
    #expected = numpy.array([16, 16, 3], dtype = float)
    #expected = numpy.array([0, 1])
    #expected = 2
    expected = [8, 1, 4, 11]
    outcome = solution3.routes[2].route
    assert(expected == outcome)


def test_insert_regret_k4_basic(solution3):
    noise = False
    solution3._delta_f = numpy.array([[15, 9, 12], [8, 6, 4], [3, 2, 1]],
                                     dtype=float)
    heuristic = repair_factory.produce_insert_regret_k(4, single_run=True)
    with pytest.raises(IndexError):
        heuristic(solution3, noise)


def test_insert_regret_k_single_route(solution3):
    noise = False
    solution3._delta_f = numpy.array([[7], [8], [9]], dtype=float)
    heuristic = repair_factory.produce_insert_regret_k(single_run=True)
    with pytest.raises(IndexError):
        heuristic(solution3, noise)


def test_insert_regret_k_with_nan(solution3):
    noise = False
    #pytest.set_trace()
    solution3._delta_f = numpy.array([[15, 9, 12], [None, None, None],
                                     [3, 2, 1]], dtype=float)
    heuristic = repair_factory.produce_insert_regret_k(single_run=True)
    heuristic(solution3, noise)
    #expected = numpy.array([3, None, 1], dtype = float)
    #expected = numpy.array([0])
    #expected = 1
    expected = [7, 0, 3, 10]
    outcome = solution3.routes[1].route
    assert(expected == outcome)


def test_insert_regret_k3_with_nan(solution3):
    noise = False
    #pytest.set_trace()
    solution3._delta_f = numpy.array([[15, 9, 12], [None, None, None],
                                     [3, 2, 1]], dtype=float)
    heuristic = repair_factory.produce_insert_regret_k(3, single_run=True)
    outcome = heuristic(solution3, noise)
    #expected = numpy.array([9, None, 3], dtype = float)
    #expected = numpy.array([0])
    #expected = 1
    expected = [7, 0, 3, 10]
    outcome = solution3.routes[1].route
    assert(expected == outcome)


def test_insert_regret_k_with_inf(solution3):
    noise = False
    #pytest.set_trace()
    solution3._delta_f = numpy.array([[15, 9,  numpy.Inf],
                                     [numpy.inf, numpy.inf, numpy.inf],
                                     [1, 2, 3]],
                                     dtype=float)
    heuristic = repair_factory.produce_insert_regret_k(single_run=True)
    outcome = heuristic(solution3, noise)
    #expected = numpy.array([6, 0, 1], dtype=float)
    #expected = numpy.array([0])
    #expected = 1
    expected = [7, 0, 3, 10]
    outcome = solution3.routes[1].route
    assert(expected == outcome)


def test_insert_regret_k3_with_inf(solution3):
    noise = False
    #pytest.set_trace()
    solution3._delta_f = numpy.array([[15, 9,  numpy.Inf],
                                     [numpy.inf, numpy.inf, numpy.inf],
                                     [1, 2, 3]],
                                     dtype=float)
    heuristic = repair_factory.produce_insert_regret_k(3, single_run=True)
    outcome = heuristic(solution3, noise)
    #expected = numpy.array([6, 0, 3], dtype=float)
    #expected = numpy.array([0])
    #expected = 1
    expected = [7, 0, 3, 10]
    outcome = solution3.routes[1].route
    assert(expected == outcome)


def test_insert_regret_k3_with_inf_nan_ties(solution4):
    noise = False
    #pytest.set_trace()
    solution4._delta_f = numpy.array([[15, 9,  numpy.Inf],
                                     [numpy.Inf, numpy.Inf, numpy.Inf],
                                     [None, None, None],
                                     [1, 2, 6]],
                                     dtype=float)
    heuristic = repair_factory.produce_insert_regret_k(3, single_run=True)
    outcome = heuristic(solution4, noise)
    #expected = numpy.array([6, 0, None, 6], dtype=float)
    #expected = numpy.array([3])
    #expected = 0
    expected = [8, 3, 7, 11]
    outcome = solution4.routes[0].route
    assert(expected == outcome)

#For debugging purposes, uncomment below
pytest.main(['test_repair_factory.py'])
