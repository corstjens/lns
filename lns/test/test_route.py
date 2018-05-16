# -*- coding: utf-8 -*-
"""
Created on Fri Feb 07 10:38:05 2014

@author: Benoit & Lotte
"""
import numpy
import pytest

def test_initial_route(routes):
    earliest_manual = numpy.array([0, 0])
    latest_manual = numpy.array([230, 230])
    earliest_rounded = numpy.around(routes[0].earliest, decimals=2)
    latest_rounded = numpy.around(routes[0].latest, decimals=2)
    assert numpy.alltrue(earliest_manual == earliest_rounded)
    assert numpy.alltrue(latest_manual == latest_rounded)
    assert routes[0].route == [6, 8]


def test_initial_route_pd(routes_pd):
    earliest_manual = numpy.array([0, 0])
    latest_manual = numpy.array([3324, 3324])
    earliest_rounded = numpy.around(routes_pd[0].earliest, decimals=2)
    latest_rounded = numpy.around(routes_pd[0].latest, decimals=2)
    assert numpy.alltrue(earliest_manual == earliest_rounded)
    assert numpy.alltrue(latest_manual == latest_rounded)
    assert routes_pd[0].route == [4, 6]
    assert routes_pd[0].cost == 0


def test_earliest_latest(routes):
    routes[0].insert_node(1, 0)
    earliest_manual = numpy.array([0, 0, 0])
    latest_manual = numpy.array([0, 0, 230])
    earliest_rounded = numpy.around(routes[0].earliest, decimals=2)
    latest_rounded = numpy.around(routes[0].latest, decimals=2)
    assert numpy.alltrue(earliest_manual == earliest_rounded)
    assert numpy.alltrue(latest_manual == latest_rounded)


def test_earliest_latest2(routes):
    routes[0].insert_node(1, 0)
    routes[0].insert_node(2, 3)
    earliest_manual = numpy.array([0, 0, 161, 186.23])
    latest_manual = numpy.array([0, 0, 171, 230])
    earliest_rounded = numpy.around(routes[0].earliest, decimals=2)
    latest_rounded = numpy.around(routes[0].latest, decimals=2)
    assert numpy.alltrue(earliest_manual == earliest_rounded)
    assert numpy.alltrue(latest_manual == latest_rounded)


def test_earliest_latest3(routes):
    routes[0].insert_node(1, 0)
    routes[0].insert_node(2, 3)
    routes[0].insert_node(1, 1)
    earliest_manual = numpy.array([0, 0, 0, 161, 186.23])
    latest_manual = numpy.array([0, 0, 0, 171, 230])
    earliest_rounded = numpy.around(routes[0].earliest, decimals=2)
    latest_rounded = numpy.around(routes[0].latest, decimals=2)
    assert numpy.alltrue(earliest_manual == earliest_rounded)
    assert numpy.alltrue(latest_manual == latest_rounded)


def test_route_long(routes):
    assert routes[1].route == [6, 0, 1, 4, 3, 8]


def test_route_short(routes):
    assert routes[2].route == [6, 2, 5, 8]


def test_calculate_cost(routes):
    assert routes[0].calculate_cost() == 0.0


def test_calculate_cost2(routes):
    assert round(routes[1].calculate_cost(), 3) == 65.789


def test_calculate_cost3(routes):
    assert round(routes[2].calculate_cost(), 3) == 44.721


def test_recalculate_cost_insert(routes):
    routes[0].insert_node(1, 5)
    assert round(routes[0].cost, 3) == 44.721


def test_recalculate_cost_insert2(routes):
    routes[0].insert_node(1, 4)
    assert routes[0].cost == 36


def test_recalculate_cost_delete(routes):
    routes[0].insert_node(1, 5)
    earliest_manual = numpy.array([0, 116, 148.36])
    latest_manual = numpy.array([103.64, 126, 230])
    earliest_rounded = numpy.around(routes[0].earliest, decimals=2)
    latest_rounded = numpy.around(routes[0].latest, decimals=2)
    assert numpy.alltrue(earliest_manual == earliest_rounded)
    assert numpy.alltrue(latest_manual == latest_rounded)


def test_complete_request_initial_route(routes):
    assert routes[0].complete_requests()


def test_complete_request(routes):
    routes[0].insert_node(1, 1)
    routes[0].insert_node(2, 4)
    assert routes[0].complete_requests()


def test_complete_request_incorrect_route(routes):
    routes[0].insert_node(1, 1)
    routes[0].insert_node(2, 4)
    routes[0].insert_node(3, 0)
    assert routes[0].complete_requests() is not True


def test_priority_check_initial_route(routes):
    assert routes[0].priority_check()


def test_priority_incorrect_route(routes):
    routes[0].insert_node(1, 4)
    routes[0].insert_node(2, 1)
    assert routes[0].priority_check() is not True


def test_priority_incorrect_route2(routes):
    routes[0].insert_node(1, 1)
    routes[0].insert_node(2, 4)
    routes[0].insert_node(1, 2)
    routes[0].insert_node(2, 5)
    assert routes[0].priority_check() is not True


def test_priority_correct_route(routes):
    routes[0].insert_node(1, 1)
    routes[0].insert_node(2, 4)
    assert routes[0].priority_check()


def test_precedence_incorrect_route(routes):
    routes[0].insert_node(1, 4)
    routes[0].insert_node(2, 1)
    assert routes[0].precedence_constraints() is not True


def test_precedence_correct_route(routes):
    routes[0].insert_node(1, 1)
    routes[0].insert_node(2, 4)
    assert routes[0].precedence_constraints()


def test_precedence_correct_route2(routes):
    routes[0].insert_node(1, 1)
    routes[0].insert_node(2, 4)
    routes[0].insert_node(1, 2)
    routes[0].insert_node(2, 5)
    assert routes[0].precedence_constraints()


def test_calculate_earliest(routes):
    result = numpy.array([0, 0, 0, 50, 161, 186.23])
    produced = routes[1].calculate_earliest(0)
    assert numpy.allclose(result, produced)


def test_calculate_latest(routes):
    result = numpy.array([0, 0, 0, 60, 171, 230])
    produced = routes[1].calculate_latest(0)
    assert numpy.allclose(result, produced)


def test_calculate_earliest2(routes):
    result = numpy.array([0, 0, 116, 148.36])
    produced = routes[2].calculate_earliest(0)
    assert numpy.allclose(result, produced)


def test_calculate_loads(routes):
    result = numpy.array([0, 10, 17, 10, 0, 0])
    produced = routes[1].calculate_loads()
    assert numpy.alltrue(result == produced)


def test_calculate_loads2(routes):
    result = numpy.array([0, 13, 0, 0])
    produced = routes[2].calculate_loads()
    assert numpy.alltrue(result == produced)


def test_verify_tw_constraints(routes):
    assert routes[1].verify_tw_constraints()


def test_verify_tw_constraints_with_violation_of_start_timewindow(routes):
    routes[1].allnodes[4].tw_start = 172
    assert routes[1].verify_tw_constraints() is not True


def test_verify_tw_constraints_with_violation_of_end_timewindows(routes):
    routes[1].allnodes[4].tw_start = 45
    routes[1].allnodes[4].tw_end = 48
    assert routes[1].verify_tw_constraints() is not True


def test_verify_tw_constraints_with_impossible_early_servicetime(routes):
    routes[1].allnodes[4].tw_start = 10
    routes[1].earliest[3] = 15
    assert routes[1].verify_tw_constraints() is not True


def test_verify_tw_constraints_with_impossible_early_servicetime2(routes):
    routes[1].earliest[5] = 170
    assert routes[1].verify_tw_constraints() is not True


def test_verify_tw_constraints_start_route_before_open_terminal(routes):
    routes[1].allnodes[6].tw_start = 10
    assert routes[1].verify_tw_constraints() is not True


def test_verify_tw_constraints_end_route_after_close_terminal(routes):
    routes[1].allnodes[8].tw_end = 150
    assert routes[1].verify_tw_constraints() is not True


def test_verify_load_constraints(routes):
    assert routes[1].verify_load_constraints()


def test_verify_load_constraints_load_exceeds_capacity(routes):
    routes[1].capacity = 15
    assert routes[1].verify_load_constraints() is not True


def test_verify_load_constraints_empty_route(routes):
    assert routes[0].verify_load_constraints


def test_verify_load_constraints_start_nonempty(routes):
    routes[1].loads[0] = 10
    assert routes[1].verify_load_constraints is not True


def test_verify_load_constraints_arrive_nonempty(routes):
    routes[1].loads[-1] = 10
    assert routes[1].verify_load_constraints is not True

if __name__ == "__main__":
    pytest.main("test_route.py")
