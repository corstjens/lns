# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:53:46 2014

@author: Lotte
"""

from alns.problem import Problem


def test_create_pickup_dict(requests):
    p = Problem([], [])
    assert p.create_pickup_dict(requests) == {0: requests[0].pickup_node,
                                              1: requests[1].pickup_node,
                                              2: requests[2].pickup_node}


def test_create_delivery_dict(requests):
    p = Problem([], [])
    assert p.create_delivery_dict(requests) == {3: requests[0].delivery_node,
                                                4: requests[1].delivery_node,
                                                5: requests[2].delivery_node}


def test_create_pk(vrptw_problem):
    assert vrptw_problem.create_pk() == {0: [0, 1, 2], 1: [0, 1, 2]}


def test_create_dk(vrptw_problem):
    assert vrptw_problem.create_dk() == {0: [3, 4, 5], 1: [3, 4, 5]}


def test_create_nk(vrptw_problem):
    assert vrptw_problem.create_nk() == {0: [0, 1, 2, 3, 4, 5],
                                         1: [0, 1, 2, 3, 4, 5]}


def test_create_tauk(vrptw_problem, vehicles):
    assert vrptw_problem.create_tau_k() == {6: vehicles[0].start_terminal,
                                            7: vehicles[1].start_terminal}


def test_create_tauk_bis(vrptw_problem, vehicles):
    assert vrptw_problem.create_tau_k_bis() == {8: vehicles[0].end_terminal,
                                                9: vehicles[1].end_terminal}


def test_create_v(vrptw_problem, requests, vehicles):
    assert vrptw_problem.V == {0: requests[0].pickup_node,
                               1: requests[1].pickup_node,
                               2: requests[2].pickup_node,
                               3: requests[0].delivery_node,
                               4: requests[1].delivery_node,
                               5: requests[2].delivery_node,
                               6: vehicles[0].start_terminal,
                               7: vehicles[1].start_terminal,
                               8: vehicles[0].end_terminal,
                               9: vehicles[1].end_terminal}


def test_create_vk(vrptw_problem):
    assert vrptw_problem.create_Vk() == {0: [0, 1, 2, 3, 4, 5, 6, 8],
                                         1: [0, 1, 2, 3, 4, 5, 7, 9]}


def test_create_ak(vrptw_problem):
    assert vrptw_problem.create_Ak() == {0: [(0, 0), (0, 1), (0, 2), (0, 3),
                                             (0, 4), (0, 5), (0, 6), (0, 8),
                                             (1, 0), (1, 1), (1, 2), (1, 3),
                                             (1, 4), (1, 5), (1, 6), (1, 8),
                                             (2, 0), (2, 1), (2, 2), (2, 3),
                                             (2, 4), (2, 5), (2, 6), (2, 8),
                                             (3, 0), (3, 1), (3, 2), (3, 3),
                                             (3, 4), (3, 5), (3, 6), (3, 8),
                                             (4, 0), (4, 1), (4, 2), (4, 3),
                                             (4, 4), (4, 5), (4, 6), (4, 8),
                                             (5, 0), (5, 1), (5, 2), (5, 3),
                                             (5, 4), (5, 5), (5, 6), (5, 8),
                                             (6, 0), (6, 1), (6, 2), (6, 3),
                                             (6, 4), (6, 5), (6, 6), (6, 8),
                                             (8, 0), (8, 1), (8, 2), (8, 3),
                                             (8, 4), (8, 5), (8, 6), (8, 8)],
                                         1: [(0, 0), (0, 1), (0, 2), (0, 3),
                                             (0, 4), (0, 5), (0, 7), (0, 9),
                                             (1, 0), (1, 1), (1, 2), (1, 3),
                                             (1, 4), (1, 5), (1, 7), (1, 9),
                                             (2, 0), (2, 1), (2, 2), (2, 3),
                                             (2, 4), (2, 5), (2, 7), (2, 9),
                                             (3, 0), (3, 1), (3, 2), (3, 3),
                                             (3, 4), (3, 5), (3, 7), (3, 9),
                                             (4, 0), (4, 1), (4, 2), (4, 3),
                                             (4, 4), (4, 5), (4, 7), (4, 9),
                                             (5, 0), (5, 1), (5, 2), (5, 3),
                                             (5, 4), (5, 5), (5, 7), (5, 9),
                                             (7, 0), (7, 1), (7, 2), (7, 3),
                                             (7, 4), (7, 5), (7, 7), (7, 9),
                                             (9, 0), (9, 1), (9, 2), (9, 3),
                                             (9, 4), (9, 5), (9, 7), (9, 9)]}


def test_calculate_distancematrix(vrptw_problem):
    matrix = {}
    for arc in vrptw_problem.A:
            node1 = vrptw_problem.V[arc[0]]
            node2 = vrptw_problem.V[arc[1]]
            matrix[arc] = (
                round(vrptw_problem.calculate_distance(node1, node2), 2))
    assert matrix == {(0, 0): 0.0, (0, 1): 0.0, (0, 2): 0.0, (0, 3): 15.23,
                      (0, 4): 18.0, (0, 5): 22.36, (0, 6): 0.0, (0, 7): 0.0,
                      (0, 8): 0.0, (0, 9): 0.0, (1, 0): 0.0, (1, 1): 0.0,
                      (1, 2): 0.0, (1, 3): 15.23, (1, 4): 18.0, (1, 5): 22.36,
                      (1, 6): 0.0, (1, 7): 0.0, (1, 8): 0.0, (1, 9): 0.0,
                      (2, 0): 0.0, (2, 1): 0.0, (2, 2): 0.0, (2, 3): 15.23,
                      (2, 4): 18.0, (2, 5): 22.36, (2, 6): 0.0, (2, 7): 0.0,
                      (2, 8): 0.0, (2, 9): 0.0, (3, 0): 15.23, (3, 1): 15.23,
                      (3, 2): 15.23, (3, 3): 0.0, (3, 4): 32.56, (3, 5): 14.56,
                      (3, 6): 15.23, (3, 7): 15.23, (3, 8): 15.23,
                      (3, 9): 15.23, (4, 0): 18.0, (4, 1): 18.0, (4, 2): 18.0,
                      (4, 3): 32.56, (4, 4): 0.0, (4, 5): 34.41, (4, 6): 18.0,
                      (4, 7): 18.0, (4, 8): 18.0, (4, 9): 18.0, (5, 0): 22.36,
                      (5, 1): 22.36, (5, 2): 22.36, (5, 3): 14.56,
                      (5, 4): 34.41, (5, 5): 0.0, (5, 6): 22.36, (5, 7): 22.36,
                      (5, 8): 22.36, (5, 9): 22.36, (6, 0): 0.0, (6, 1): 0.0,
                      (6, 2): 0.0, (6, 3): 15.23, (6, 4): 18.0, (6, 5): 22.36,
                      (6, 6): 0.0, (6, 7): 0.0, (6, 8): 0.0, (6, 9): 0.0,
                      (7, 0): 0.0, (7, 1): 0.0, (7, 2): 0.0, (7, 3): 15.23,
                      (7, 4): 18.0, (7, 5): 22.36, (7, 6): 0.0, (7, 7): 0.0,
                      (7, 8): 0.0, (7, 9): 0.0, (8, 0): 0.0, (8, 1): 0.0,
                      (8, 2): 0.0, (8, 3): 15.23, (8, 4): 18.0, (8, 5): 22.36,
                      (8, 6): 0.0, (8, 7): 0.0, (8, 8): 0.0, (8, 9): 0.0,
                      (9, 0): 0.0, (9, 1): 0.0, (9, 2): 0.0, (9, 3): 15.23,
                      (9, 4): 18.0, (9, 5): 22.36, (9, 6): 0.0, (9, 7): 0.0,
                      (9, 8): 0.0, (9, 9): 0.0}
    assert max(matrix.values()) == 34.41


def test_calculate_timematrix_with_identity_function(vrptw_problem):
    vrptw_problem.distancematrix = {(0, 1): 4, (0, 2): 3, (1, 2): 5}
    expected = {(0, 1): 4, (0, 2): 3, (1, 2): 5}
    assert vrptw_problem.calculate_timematrix() == expected


def test_calculate_timematrix_with_linear_function(vrptw_problem):
    vrptw_problem.distancematrix = {(0, 1): 4, (0, 2): 3, (1, 2): 5}
    expected = {(0, 1): 8, (0, 2): 6, (1, 2): 10}

    def timefunction(distance):
        return distance * 2
    assert vrptw_problem.calculate_timematrix(timefunction) == expected

#For debugging purposes, uncomment below
#pytest.main(['test_problem.py'])
