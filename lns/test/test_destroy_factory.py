# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:38:37 2014

@author: lucp2487
"""

import pytest
import numpy
from alns.factories import destroy_factory
from copy import deepcopy

def test_remove_random(alns, solution):
    #pytest.set_trace()
    noise = False
    solution_manual = {0: [6, 8], 1: [7, 9]}
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, noise)
    destroy_heuristic = destroy_factory.produce_remove_random(3)
    destroy_heuristic(solution)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == solution_manual)


def test_remove_random2(alns, solution):
    #pytest.set_trace()
    noise = False
    solution_manual1 = {0: [6, 0, 3, 8], 1: [7, 9]}
    solution_manual2 = {0: [6, 1, 4, 8], 1: [7, 9]}
    solution_manual3 = {0: [6, 2, 5, 8], 1: [7, 9]}
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, noise)
    destroy_heuristic = destroy_factory.produce_remove_random(2)
    destroy_heuristic(solution)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == solution_manual1
            or found_solution == solution_manual2
            or found_solution == solution_manual3)


def test_remove_random3(alns, solution):
    #pytest.set_trace()
    noise = False
    solution_manual1 = {0: [6, 2, 0, 5, 3, 8], 1: [7, 9]}
    solution_manual2 = {0: [6, 1, 0, 4, 3, 8], 1: [7, 9]}
    solution_manual3 = {0: [6, 1, 2, 4, 5, 8], 1: [7, 9]}
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, noise)
    destroy_heuristic = destroy_factory.produce_remove_random(1)
    destroy_heuristic(solution)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == solution_manual1
            or found_solution == solution_manual2
            or found_solution == solution_manual3)


def test_remove_worst1(alns, solution):
    #pytest.set_trace()
    noise = False
    solution_manual = {0: [6, 8], 1: [7, 9]}
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, noise)
    destroy_heuristic = destroy_factory.produce_remove_worst(3, 2)
    #pytest.set_trace()
    destroy_heuristic(solution)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == solution_manual)


def test_remove_worst2(alns, solution):
    #pytest.set_trace()
    noise = False
    solution_manual1 = {0: [6, 0, 3, 8], 1: [7, 9]}
    solution_manual2 = {0: [6, 1, 4, 8], 1: [7, 9]}
    solution_manual3 = {0: [6, 2, 5, 8], 1: [7, 9]}
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, noise)
    destroy_heuristic = destroy_factory.produce_remove_worst(2, 2)
    destroy_heuristic(solution)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == solution_manual1
            or found_solution == solution_manual2
            or found_solution == solution_manual3)


def test_remove_worst3(alns, solution):
    #pytest.set_trace()
    noise = False
    solution_manual1 = {0: [6, 2, 0, 5, 3, 8], 1: [7, 9]}
    solution_manual2 = {0: [6, 1, 0, 4, 3, 8], 1: [7, 9]}
    solution_manual3 = {0: [6, 1, 2, 4, 5, 8], 1: [7, 9]}
    repair_heuristic = alns.repair_heuristics[1]
    repair_heuristic(solution, noise)
    destroy_heuristic = destroy_factory.produce_remove_worst(1, 2)
    destroy_heuristic(solution)
    found_solution = {0: solution.routes[0].route,
                      1: solution.routes[1].route}
    assert (found_solution == solution_manual1
            or found_solution == solution_manual2
            or found_solution == solution_manual3)

def test_remove_related_remove_random_request(solution):
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    test_copy = deepcopy(solution)
    destroy_related = destroy_factory.produce_remove_related(1, 1)
    destroy_related(solution)
    assert len(solution.routes[0].route) != len(test_copy.routes[0].route)
    assert len(solution.routes[0].route) == 6
    assert len(solution.routes[1].route) == 2

def test_remove_related_remove_request_2(solution, monkeypatch):
    def mock_randint(start,end):
        return 2
    monkeypatch.setattr("random.randint", mock_randint)
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    destroy_related = destroy_factory.produce_remove_related(1, 1)
    destroy_related(solution)
    route = solution.routes[0].route
    assert 2 not in route
    assert 5 not in route
    assert len(solution.request_bank) == 1
    #Route 0: [6,0,1,4,3,8]
    
def test_remove_related_remove_least_related_request(solution, monkeypatch):
    def mock_randint(start, end):
        return 2
    def mock_random_number():
        return 0.8
    monkeypatch.setattr("random.randint", mock_randint)
    monkeypatch.setattr("random.random", mock_random_number)
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    #p = 1 means much randomness
    destroy_related = destroy_factory.produce_remove_related(2, 1)
    destroy_related(solution)
    route = solution.routes[0].route
    assert 2 not in route
    assert 5 not in route
    assert 1 not in route
    assert 4 not in route
    assert len(solution.routes[0].route) == 4
    assert len(solution.routes[1].route) == 2
    assert len(solution.request_bank) == 2
    #Route 0: [6,0,3,8], route 1: [7,9]
  
def test_remove_related_remove_most_related_request(solution, monkeypatch):
    def mock_randint(start, end):
        return 2
    def mock_random_number():
        return 0.8
    monkeypatch.setattr("random.randint", mock_randint)
    monkeypatch.setattr("random.random", mock_random_number)
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    #Choose the most/more related request to remove (p >>>1)
    destroy_related = destroy_factory.produce_remove_related(2, 10)
    destroy_related(solution)
    route = solution.routes[0].route
    assert 2 not in route
    assert 5 not in route
    assert 0 not in route
    assert 3 not in route
    assert len(solution.request_bank) == 2
    #Route 0: [6,1,4,8], route 1: [7,9]

def test_remove_related_pickups_1_request_not_on_depot(solution_for_related_removal1, monkeypatch):
    def mock_randint(start, end):
        return 2
    def mock_random_number():
        return 0.8
    monkeypatch.setattr("random.randint", mock_randint)
    monkeypatch.setattr("random.random", mock_random_number)   
    
    solution = solution_for_related_removal1
    #solution.problem.P[1].x_coord = 40
    #solution.problem.P[1].y_coord = 40
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    assert round(destroy_factory.calculate_relatedness(solution, 1, 0, [0, 2], []), 4) == 20.8065
    assert round(destroy_factory.calculate_relatedness(solution, 2, 0, [0, 2], []), 4) == 14.5602
    assert round(destroy_factory.calculate_relatedness(solution, 2, 1, [0, 2], []), 4) == 25.1103
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    #Choose the most/more related request to remove (p >>>1)
    destroy_related = destroy_factory.produce_remove_related(2, 10)
    destroy_related(solution)
    route = solution.routes[0].route
    assert 2 not in route
    assert 5 not in route
    assert 0 not in route
    assert 3 not in route
    assert len(solution.request_bank) == 2
    #Route 0: [6,0,3,8], route 1: [7,9]
    
def test_remove_related_pickups_2_requests_not_on_depot(solution_for_related_removal2, monkeypatch):
    def mock_randint(start, end):
        return 2
    def mock_random_number():
        return 0.8
    monkeypatch.setattr("random.randint", mock_randint)
    monkeypatch.setattr("random.random", mock_random_number)   

    solution = solution_for_related_removal2
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    assert round(destroy_factory.calculate_relatedness(solution, 1, 0, [2], []), 4) == 21.2986
    assert round(destroy_factory.calculate_relatedness(solution, 2, 0, [2], []), 4) == 28.7859
    assert round(destroy_factory.calculate_relatedness(solution, 2, 1, [2], []), 4) == 25.1103
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    #Choose the most/more related request to remove (p >>>1)
    destroy_related = destroy_factory.produce_remove_related(2, 10)
    destroy_related(solution)
    route = solution.routes[0].route
    assert 2 not in route
    assert 5 not in route
    assert 1 not in route
    assert 4 not in route
    assert len(solution.request_bank) == 2
    #Route 0: [6,0,3,8], route 1: [7,9]

def test_remove_related_all_pickups_not_on_depot(solution_for_related_removal3, monkeypatch):
    def mock_randint(start, end):
        return 2
    def mock_random_number():
        return 0.8
    monkeypatch.setattr("random.randint", mock_randint)
    monkeypatch.setattr("random.random", mock_random_number)   
    
    solution = solution_for_related_removal3 
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    assert round(destroy_factory.calculate_relatedness(solution, 1, 0, [], []), 4) == 21.2986    
    assert round(destroy_factory.calculate_relatedness(solution, 2, 0, [], []), 4) == 34.0237 
    assert round(destroy_factory.calculate_relatedness(solution, 2, 1, [], []), 4) == 32.0611
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    #Choose the most/more related request to remove (p >>>1)
    destroy_related = destroy_factory.produce_remove_related(2, 10)
    destroy_related(solution)
    route = solution.routes[0].route
    assert 2 not in route
    assert 5 not in route
    assert 1 not in route
    assert 4 not in route
    assert len(solution.request_bank) == 2
    #Route 0: [6,0,3,8], route 1: [7,9]    

def test_remove_related_pickup0_not_on_depot_delivery1_on_depot(solution_for_related_removal_delivery_on_depot, monkeypatch):
    def mock_randint(start, end):
        return 2
    def mock_random_number():
        return 0.8
    monkeypatch.setattr("random.randint", mock_randint)
    monkeypatch.setattr("random.random", mock_random_number)   
    
    solution = solution_for_related_removal_delivery_on_depot 
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    assert destroy_factory.calculate_relatedness(solution, 2, 1, [1, 2], [4]) is None   
    assert round(destroy_factory.calculate_relatedness(solution, 2, 0, [1, 2], [4]), 4) == 28.7859 
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    #Choose the most/more related request to remove (p >>>1)
    destroy_related = destroy_factory.produce_remove_related(2, 10)
    destroy_related(solution)
    route = solution.routes[0].route
    assert 2 not in route
    assert 5 not in route
    assert 0 not in route
    assert 3 not in route
    assert len(solution.request_bank) == 2
    #Route 0: [6,1,4,8], route 1: [7,9]    

def test_remove_related_pickup2_on_depot_delivery0_and_1_on_depot(solution_for_related_removal_delivery_on_depot2, monkeypatch):
    def mock_randint(start, end):
        return 2
    def mock_random_number():
        return 0.8
    monkeypatch.setattr("random.randint", mock_randint)
    monkeypatch.setattr("random.random", mock_random_number)   
    
    solution = solution_for_related_removal_delivery_on_depot2 
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    assert round(destroy_factory.calculate_relatedness(solution, 2, 0, [2], [3, 4]), 4) == 43.0116  
    assert round(destroy_factory.calculate_relatedness(solution, 2, 1, [2], [3, 4]), 4) == 15.8114
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    #Choose the most/more related request to remove (p >>>1)
    destroy_related = destroy_factory.produce_remove_related(2, 10)
    destroy_related(solution)
    route = solution.routes[0].route
    assert 2 not in route
    assert 5 not in route
    assert 1 not in route
    assert 4 not in route
    assert len(solution.request_bank) == 2
   

def test_remove_related_no_pickup_on_depot_all_deliveries_on_depot(solution_for_related_removal_all_deliveries_on_depot, monkeypatch):
    def mock_randint(start, end):
        return 2
    def mock_random_number():
        return 0.8
    monkeypatch.setattr("random.randint", mock_randint)
    monkeypatch.setattr("random.random", mock_random_number)   
    
    solution = solution_for_related_removal_all_deliveries_on_depot 
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    assert round(destroy_factory.calculate_relatedness(solution, 0, 1, [], [3, 4, 5]), 4) == 28.2843 
    assert round(destroy_factory.calculate_relatedness(solution, 2, 0, [], [3, 4, 5]), 4) == 63.6396  
    assert round(destroy_factory.calculate_relatedness(solution, 2, 1, [], [3, 4, 5]), 4) == 35.3553
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    #Choose the most/more related request to remove (p >>>1)
    destroy_related = destroy_factory.produce_remove_related(2, 10)
    destroy_related(solution)
    route = solution.routes[0].route
    assert 2 not in route
    assert 5 not in route
    assert 1 not in route
    assert 4 not in route
    assert len(solution.request_bank) == 2
    

def test_remove_related_pickup0_and_1_on_depot_all_deliveries_on_depot(solution_for_related_removal_all_deliveries_on_depot2, monkeypatch):
    def mock_randint(start, end):
        return 2
    def mock_random_number():
        return 0.8
    monkeypatch.setattr("random.randint", mock_randint)
    monkeypatch.setattr("random.random", mock_random_number)   
    
    solution = solution_for_related_removal_all_deliveries_on_depot2 
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    assert destroy_factory.calculate_relatedness(solution, 0, 1, [0, 1], [3, 4, 5]) is None
    assert destroy_factory.calculate_relatedness(solution, 2, 0, [0, 1], [3, 4, 5]) is None 
    assert destroy_factory.calculate_relatedness(solution, 2, 1, [0, 1], [3, 4, 5]) is None
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    #Choose the most/more related request to remove (p >>>1)
    destroy_related = destroy_factory.produce_remove_related(2, 10)
    destroy_related(solution)
    route = solution.routes[0].route
    assert 2 not in route
    assert 5 not in route
    assert len(solution.request_bank) == 1

 
def test_remove_related_2nd_vehicle_different_terminal(solution_for_related_removal_different_terminals_2ndvehicle, monkeypatch):
    def mock_randint(start, end):
        return 2
    def mock_random_number():
        return 0.8
    monkeypatch.setattr("random.randint", mock_randint)
    monkeypatch.setattr("random.random", mock_random_number)   
    
    solution = solution_for_related_removal_different_terminals_2ndvehicle 
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    assert destroy_factory.calculate_relatedness(solution, 0, 1, [0, 1, 2], [3]) is None 
    assert destroy_factory.calculate_relatedness(solution, 2, 0, [0, 1, 2], [3]) is None
    assert round(destroy_factory.calculate_relatedness(solution, 2, 1, [0, 1, 2], [3]), 4) == 34.4093
    #route 0: [6, 0, 2, 1, 4, 5, 3, 8], route 1: [7, 9]
    #Choose the most/more related request to remove (p >>>1)
    destroy_related = destroy_factory.produce_remove_related(2, 10)
    destroy_related(solution)
    route = solution.routes[0].route
    assert 2 not in route
    assert 5 not in route
    assert 1 not in route
    assert 4 not in route
    assert len(solution.request_bank) == 2

def test_remove_related_2nd_vehicle_different_terminal_customers4(solution_for_related_removal_different_terminals_2ndvehicle2, monkeypatch):
    #Test with 4 customers, customer 3 located at terminal vehicle 2
    def mock_randint(start, end):
        return 3
    def mock_random_number():
        return 0.8
    monkeypatch.setattr("random.randint", mock_randint)
    monkeypatch.setattr("random.random", mock_random_number)   
    
    solution = solution_for_related_removal_different_terminals_2ndvehicle2 
    solution._assign_request(1, 0, 2)
    solution._assign_request(2, 0, 4)
    solution._assign_request(0, 0, 6)
    solution._assign_request(3, 1, 2)
    assert round(destroy_factory.calculate_relatedness(solution, 1, 0, [0, 1, 2, 3], [7]), 4) == 32.5576 
    assert round(destroy_factory.calculate_relatedness(solution, 2, 0, [0, 1, 2, 3], [7]), 4) == 14.5602  
    assert round(destroy_factory.calculate_relatedness(solution, 2, 1, [0, 1, 2, 3], [7]), 4) == 34.4093
    assert destroy_factory.calculate_relatedness(solution, 3, 0, [0, 1, 2, 3], [7]) is None
    assert destroy_factory.calculate_relatedness(solution, 3, 1, [0, 1, 2, 3], [7]) is None
    assert destroy_factory.calculate_relatedness(solution, 3, 2, [0, 1, 2, 3], [7]) is None
    #route 0: [8, 0, 2, 1, 5, 6, 4, 10], route 1: [9, 3, 7, 11]
    #Choose the most/more related request to remove (p >>>1)
    destroy_related = destroy_factory.produce_remove_related(2, 10)
    destroy_related(solution)
    route = solution.routes[1].route
    assert 3 not in route
    assert 7 not in route
    assert len(solution.request_bank) == 1
    #route 0: [8, 0, 2, 1, 5, 6, 4, 10], route 1: [9, 11]
   
if __name__ == "__main__":
    pytest.main("test_destroy_factory.py")