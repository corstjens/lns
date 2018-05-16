# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:03:57 2014

@author: Benoit & Lotte
"""
import numpy
import random
import pytest


def test_calculate_solution_cost(solution):
    assert solution.calculate_solution_cost() == 300000


def test_number_of_used_vehicles(solution):
    assert solution._number_of_used_vehicles() == 0


def test_rebuild_insert_matrices(solution):
    best_insert = numpy.array([[2, 2], [2, 2], [2, 2]])
    delta_f_manual = numpy.array([[30.46, 30.46], [36, 36],
                                 [44.72, 44.72]])

    def temp(x, y):
        return 0.8
    random.uniform = temp
    solution._rebuild_insert_matrices()
    delta_f_rounded = numpy.around(solution._delta_f, decimals=2)
    assert numpy.alltrue(solution._best_insert_position == best_insert)
    assert numpy.alltrue(delta_f_rounded == delta_f_manual)


def test_assign_request(solution):
    def temp(x, y):
        return 0.8
    random.uniform = temp
    solution._rebuild_insert_matrices()
    solution._assign_request(0, 0, 2)
    delta_f = numpy.array([[numpy.nan, numpy.nan], [35.33, 36],
                          [21.69, 44.72]])
    best_insert = numpy.array([[-1, -1], [3, 2], [3, 2]])
    delta_f_produced = solution._delta_f
    best_insert_produced = solution._best_insert_position
    assert solution.routes[0].route == [6, 0, 3, 8]
    assert solution.request_bank == [1, 2]
    assert numpy.isnan(delta_f_produced[0][0] and delta_f_produced[0][1])
    for i in range(1, 3):
        for j in range(0, 2):
            assert delta_f[i][j] == numpy.around(delta_f_produced[i][j],
                                                 decimals=2)
    assert numpy.alltrue(best_insert == best_insert_produced)
    assert solution._number_of_used_vehicles() == 1


def test_remove_request(solution):
    def temp(x, y):
        return 0.8
    random.uniform = temp
    solution._rebuild_insert_matrices()
    solution._assign_request(0, 0, 2)
    solution._remove_request(0)
    delta_f_manual = numpy.array([[30.46, 30.46], [36, 36],
                                 [44.72, 44.72]])
    best_insert = numpy.array([[2, 2], [2, 2], [2, 2]])
    delta_f_rounded = numpy.around(solution._delta_f, decimals=2)
    best_insert_produced = solution._best_insert_position
    assert solution.routes[0].route == [6, 8]
    assert solution.request_bank == [1, 2, 0]
    assert numpy.alltrue(delta_f_manual == delta_f_rounded)
    assert numpy.alltrue(best_insert == best_insert_produced)


def test_find_route_containing_request(solution):
    solution._assign_request(0, 0, 2)
    result = solution._find_route_containing_request(0)
    assert result == 0
    solution._remove_request(0)


def test_find_route_containing_request2(solution):
    solution._assign_request(1, 1, 2)
    result = solution._find_route_containing_request(1)
    assert result == 1
    solution._remove_request(1)


def test_find_route_containing_request3(solution):
    solution._assign_request(1, 1, 2)
    result = solution._find_route_containing_request(1)
    assert result != 2
    solution._remove_request(1)


def test_is_next_insert_possible(solution):
    #pytest.set_trace()
    assert solution._is_next_insert_possible() == True


def test_is_next_insert_possible2(solution):
    assert (solution._is_next_insert_possible(solution.routes[0].vehicle_id)
            == True)


def test_is_next_insert_possible3(solution):
    assert (solution._is_next_insert_possible(solution.routes[1].vehicle_id)
            == True)


def test_is_next_insert_possible4(solution):
    solution._delta_f = numpy.arange(12, dtype=float).reshape(4, 3)
    solution._delta_f[0, 0] = numpy.Inf
    #pytest.set_trace()
    assert(solution._is_next_insert_possible())


def test_calculate_noisematrix(solution):
    noisematrix = numpy.array([[2, 2], [2, 2], [2, 2]])

    def temp(x, y):
        return 2
    random.uniform = temp
    assert numpy.alltrue(solution._calculate_noisematrix() == noisematrix)


def test_remove_last_vehicle(solution):
    solution._assign_request(2, 1, 2)
    solution._assign_request(1, 1, 2)
    solution._assign_request(0, 1, 2)
    solution_manual1 = {0: [6, 8], 1: [7, 0, 3, 1, 4, 2, 5, 9]}
    solution_manual2 = {0: [6, 8]}
    best_insert = numpy.array([[2], [2], [2]])
    delta_f_manual = numpy.array([[30.46], [36],
                                 [44.72]])
    assert solution.request_bank == []
    assert solution.available_vehicles == [0, 1]
    assert solution_manual1 == {0: solution.routes[0].route,
                                1: solution.routes[1].route}
    solution.remove_last_vehicle()
    delta_f_rounded = numpy.around(solution._delta_f, decimals=2)
    assert solution.request_bank == [0, 1, 2]
    assert solution.available_vehicles == [0]
    assert solution_manual2 == {0: solution.routes[0].route}
    assert numpy.alltrue(solution._best_insert_position == best_insert)
    assert numpy.alltrue(delta_f_rounded == delta_f_manual)

#For debugging purposes, uncomment below
#pytest.main(['test_solution.py'])
