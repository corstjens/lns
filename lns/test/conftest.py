# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:06:22 2014

@author: Benoit
"""

import pytest
from alns.problem import Problem, Vehicle, Node, Request
from alns.solution import Route, Solution
from alns.solution_pdp import Route_pdp, Solution_pdp
from alns.algorithm import AlnsBase, Alns_Solver, Alns_PR_Minimization
from alns.algorithm_pdp import AlnsBase_pdp
import copy


@pytest.fixture
def depot():
    depot = Node(0, 35, 35, 0, 0, 230, 0, 0)
    return depot


@pytest.fixture
def depotpd():
    depotpd = Node(0, 250, 250, 0, 0, 3324, 0, 0)
    return depotpd


@pytest.fixture
def customers():
    customer1 = Node(1, 41, 49, 10, 161, 171, 10, 0)
    customer2 = Node(2, 35, 17, 7, 50, 60, 10, 0)
    customer3 = Node(3, 55, 45, 13, 116, 126, 10, 0)
    return [customer1, customer2, customer3]


@pytest.fixture
def customers4():
    customer1 = Node(1, 41, 49, 10, 161, 171, 10, 0)
    customer2 = Node(2, 35, 17, 7, 50, 60, 10, 0)
    customer3 = Node(3, 55, 45, 13, 116, 126, 10, 0)
    customer4 = Node(4, 10, 10, 2, 70, 90, 15, 0)
    return [customer1, customer2, customer3, customer4]


@pytest.fixture
def customersp():
    customer1p = Node(0, 243, 248, 30, 993, 1064, 90, 0)
    customer2p = Node(0, 318, 280, 20, 1069, 1534, 90, 0)
    return [customer1p, customer2p]


@pytest.fixture
def customersd():
    customer1d = Node(0, 338, 260, -30, 750, 2396, 90, 0)
    customer2d = Node(0, 315, 287, -20, 1443, 1504, 90, 0)
    return [customer1d, customer2d]


@pytest.fixture
def pickups(depot, customers):
    pickups = []
    for c in customers:
        p = copy.copy(c)
        p.x_coord = depot.x_coord
        p.y_coord = depot.y_coord
        p.tw_start = depot.tw_start
        p.tw_end = p.tw_start
        p.servicetime = 0
        p.priority = 0
        pickups.append(p)
    return pickups


@pytest.fixture
def pickups4(depot, customers4):
    pickups4 = []
    for c in customers4:
        p = copy.copy(c)
        p.x_coord = depot.x_coord
        p.y_coord = depot.y_coord
        p.tw_start = depot.tw_start
        p.tw_end = p.tw_start
        p.servicetime = 0
        p.priority = 0
        pickups4.append(p)
    return pickups4


@pytest.fixture
def pickupsp(customersp):
    pickupsp = []
    for c in customersp:
        p = copy.deepcopy(c)
        pickupsp.append(p)
    return pickupsp


@pytest.fixture
def deliveries(customers):
    deliveries = []
    for c in customers:
        d = copy.copy(c)
        d.load = -d.load
        d.priority = 1
        deliveries.append(d)
    return deliveries


@pytest.fixture
def deliveries2(customers):
    deliveries2 = []
    for c in customers:
        d = copy.copy(c)
        d.load = -d.load
        d.priority = 1
        deliveries2.append(d)
    deliveries2[1].tw_start = 130
    deliveries2[1].tw_end = 140
    return deliveries2


@pytest.fixture
def deliveries4(customers4):
    deliveries4 = []
    for c in customers4:
        d = copy.copy(c)
        d.load = -d.load
        d.priority = 1
        deliveries4.append(d)
    deliveries4[1].tw_start = 130
    deliveries4[1].tw_end = 140
    return deliveries4


@pytest.fixture
def deliveriesd(customersd):
    deliveriesd = []
    for c in customersd:
        d = copy.deepcopy(c)
        deliveriesd.append(d)
    return deliveriesd


@pytest.fixture
def requests(pickups, deliveries):
    requests = []
    for i in range(len(pickups)):
        r = Request(pickups[i], deliveries[i])
        requests.append(r)
    return requests


@pytest.fixture
def requests2(pickups, deliveries2):
    requests2 = []
    for i in range(len(pickups)):
        r = Request(pickups[i], deliveries2[i])
        requests2.append(r)
    return requests2


@pytest.fixture
def requests4(pickups4, deliveries4):
    requests4 = []
    for i in range(len(pickups4)):
        r = Request(pickups4[i], deliveries4[i])
        requests4.append(r)
    return requests4


@pytest.fixture
def requestspd(pickupsp, deliveriesd):
    requestspd = []
    for i in range(len(pickupsp)):
        r = Request(pickupsp[i], deliveriesd[i])
        requestspd.append(r)
    return requestspd


@pytest.fixture
def vehicles(depot):
    start_terminal1 = copy.copy(depot)
    end_terminal1 = copy.copy(depot)
    end_terminal1.priority = 1
    start_terminal2 = copy.copy(depot)
    end_terminal2 = copy.copy(depot)
    end_terminal2.priority = 1
    vehicle1 = Vehicle(200, start_terminal1, end_terminal1)
    vehicle2 = Vehicle(200, start_terminal2, end_terminal2)
    return [vehicle1, vehicle2]


@pytest.fixture
def vehicles3(vehicles):
    vehicles3 = copy.deepcopy(vehicles)
    vehicles3.append(copy.deepcopy(vehicles3[-1]))
    return vehicles3


@pytest.fixture
def vehiclespd(depotpd):
    start_terminal1 = copy.copy(depotpd)
    end_terminal1 = copy.copy(depotpd)
    end_terminal1.priority = 1
    start_terminal2 = copy.copy(depotpd)
    end_terminal2 = copy.copy(depotpd)
    end_terminal2.priority = 1
    vehicle1 = Vehicle(200, start_terminal1, end_terminal1)
    vehicle2 = Vehicle(200, start_terminal2, end_terminal2)
    return [vehicle1, vehicle2]


@pytest.fixture
def vrptw_problem(requests, vehicles):
    problem = Problem(requests, vehicles)
    return problem


@pytest.fixture
def vrptw_problem_for_related_removal_different_pickups1(requests, vehicles):
    requests[1].pickup_node.x_coord = 40
    requests[1].pickup_node.y_coord = 40
    problem = Problem(requests, vehicles)
    return problem


@pytest.fixture
def vrptw_problem_for_related_removal_different_pickups2(requests, vehicles):
    requests[0].pickup_node.x_coord = 20
    requests[0].pickup_node.y_coord = 20
    requests[1].pickup_node.x_coord = 40
    requests[1].pickup_node.y_coord = 40
    problem = Problem(requests, vehicles)
    return problem


@pytest.fixture
def vrptw_problem_for_related_removal_different_pickups3(requests, vehicles):
    requests[0].pickup_node.x_coord = 20
    requests[0].pickup_node.y_coord = 20
    requests[1].pickup_node.x_coord = 40
    requests[1].pickup_node.y_coord = 40
    requests[2].pickup_node.x_coord = 60
    requests[2].pickup_node.y_coord = 60
    problem = Problem(requests, vehicles)
    return problem


@pytest.fixture
def vrptw_problem_for_related_removal_delivery_on_depot(requests, vehicles):
    requests[0].pickup_node.x_coord = 20
    requests[0].pickup_node.y_coord = 20
    requests[1].delivery_node.y_coord = 35
    problem = Problem(requests, vehicles)
    return problem


@pytest.fixture
def vrptw_problem_for_related_removal_delivery_on_depot2(requests, vehicles):
    requests[0].pickup_node.x_coord = 20
    requests[0].pickup_node.y_coord = 20
    requests[1].pickup_node.x_coord = 40
    requests[1].pickup_node.y_coord = 40
    requests[0].delivery_node.x_coord = 35
    requests[0].delivery_node.y_coord = 35
    requests[1].delivery_node.y_coord = 35
    problem = Problem(requests, vehicles)
    return problem


@pytest.fixture
def vrptw_problem_for_related_removal_all_deliveries_on_depot(requests, vehicles):
    requests[0].pickup_node.x_coord = 20
    requests[0].pickup_node.y_coord = 20
    requests[1].pickup_node.x_coord = 40
    requests[1].pickup_node.y_coord = 40
    requests[2].pickup_node.x_coord = 65
    requests[2].pickup_node.y_coord = 65
    requests[0].delivery_node.x_coord = 35
    requests[0].delivery_node.y_coord = 35
    requests[1].delivery_node.y_coord = 35
    requests[2].delivery_node.x_coord = 35
    requests[2].delivery_node.y_coord = 35
    problem = Problem(requests, vehicles)
    return problem

@pytest.fixture
def vrptw_problem_for_related_removal_all_deliveries_on_depot2(requests, vehicles):
    requests[2].pickup_node.x_coord = 65
    requests[2].pickup_node.y_coord = 65
    requests[0].delivery_node.x_coord = 35
    requests[0].delivery_node.y_coord = 35
    requests[1].delivery_node.y_coord = 35
    requests[2].delivery_node.x_coord = 35
    requests[2].delivery_node.y_coord = 35
    problem = Problem(requests, vehicles)
    return problem


@pytest.fixture
def vrptw_problem_for_related_removal_different_terminals_2ndvehicle(requests, vehicles):
    vehicles[1].start_terminal.x_coord = 41
    vehicles[1].start_terminal.y_coord = 49
    vehicles[1].end_terminal.x_coord = 41
    vehicles[1].end_terminal.y_coord = 49
    problem = Problem(requests, vehicles)
    return problem


@pytest.fixture
def vrptw_problem_for_related_removal_different_terminals_2ndvehicle2(requests4, vehicles):
    vehicles[1].start_terminal.x_coord = 10
    vehicles[1].start_terminal.y_coord = 10
    vehicles[1].end_terminal.x_coord = 10
    vehicles[1].end_terminal.y_coord = 10
    problem = Problem(requests4, vehicles)
    return problem


@pytest.fixture
def vrptw_problem2(requests2, vehicles):
    problem2 = Problem(requests2, vehicles)
    return problem2


@pytest.fixture
def vrptw_problem3(requests2, vehicles3):
    return Problem(requests2, vehicles3)


@pytest.fixture
def vrptw_problem4(requests4, vehicles3):
    return Problem(requests4, vehicles3)


@pytest.fixture
def pdp_problem(requestspd, vehiclespd):
    pdp_problem = Problem(requestspd, vehiclespd)
    return pdp_problem


@pytest.fixture
def solution(vrptw_problem):
    solution = Solution(vrptw_problem)
    return solution


@pytest.fixture
def solution_for_related_removal1(vrptw_problem_for_related_removal_different_pickups1):
    solution = Solution(vrptw_problem_for_related_removal_different_pickups1)
    return solution


@pytest.fixture
def solution_for_related_removal2(vrptw_problem_for_related_removal_different_pickups2):
    solution = Solution(vrptw_problem_for_related_removal_different_pickups2)
    return solution


@pytest.fixture
def solution_for_related_removal3(vrptw_problem_for_related_removal_different_pickups3):
    solution = Solution(vrptw_problem_for_related_removal_different_pickups3)
    return solution


@pytest.fixture
def solution_for_related_removal_delivery_on_depot(vrptw_problem_for_related_removal_delivery_on_depot):
    solution = Solution(vrptw_problem_for_related_removal_delivery_on_depot)
    return solution


@pytest.fixture
def solution_for_related_removal_delivery_on_depot2(vrptw_problem_for_related_removal_delivery_on_depot2):
    solution = Solution(vrptw_problem_for_related_removal_delivery_on_depot2)
    return solution


@pytest.fixture
def solution_for_related_removal_all_deliveries_on_depot(vrptw_problem_for_related_removal_all_deliveries_on_depot):
    solution = Solution(vrptw_problem_for_related_removal_all_deliveries_on_depot)
    return solution
    

@pytest.fixture
def solution_for_related_removal_all_deliveries_on_depot2(vrptw_problem_for_related_removal_all_deliveries_on_depot2):
    solution = Solution(vrptw_problem_for_related_removal_all_deliveries_on_depot2)
    return solution


@pytest.fixture
def solution_for_related_removal_different_terminals_2ndvehicle(vrptw_problem_for_related_removal_different_terminals_2ndvehicle):
    solution = Solution(vrptw_problem_for_related_removal_different_terminals_2ndvehicle)
    return solution


@pytest.fixture
def solution_for_related_removal_different_terminals_2ndvehicle2(vrptw_problem_for_related_removal_different_terminals_2ndvehicle2):
    solution = Solution(vrptw_problem_for_related_removal_different_terminals_2ndvehicle2)
    return solution
 

@pytest.fixture
def solution2(vrptw_problem2):
    solution2 = Solution(vrptw_problem2)
    return solution2


@pytest.fixture
def solution3(vrptw_problem3):
    return Solution(vrptw_problem3)


@pytest.fixture
def solution4(vrptw_problem4):
    return Solution(vrptw_problem4)


@pytest.fixture
def solutionpd(pdp_problem):
    solutionpd = Solution_pdp(pdp_problem)
    return solutionpd


@pytest.fixture
def routes(vrptw_problem, vehicles):
    routes = []
    route = Route(vrptw_problem, 0)
    route_long = copy.deepcopy(route)
    route_long.insert_node(1, 3)
    route_long.insert_node(1, 4)
    route_long.insert_node(1, 1)
    route_long.insert_node(1, 0)
    route_short = copy.deepcopy(route)
    route_short.insert_node(1, 5)
    route_short.insert_node(1, 2)
    routes = [route, route_long, route_short]
    return routes


@pytest.fixture
def routes_pd(pdp_problem, vehiclespd):
    routes_pd = []
    route = Route(pdp_problem, 0)
    route_long = copy.deepcopy(route)
    route_long.insert_node(1, 3)
    route_long.insert_node(1, 1)
    route_long.insert_node(1, 2)
    route_long.insert_node(1, 0)
    routes_pd = [route, route_long]
    return routes_pd


@pytest.fixture
def alns():
    alns = AlnsBase()
    return alns


@pytest.fixture
def alns_pdp():
    alns_pdp = AlnsBase_pdp()
    return alns_pdp


@pytest.fixture
def alnspr2007():
    alnspr2007 = Alns_Solver(Alns_PR_Minimization(), AlnsBase())
    return alnspr2007


@pytest.fixture
def alnsprminimization():
    alnsprminimization = Alns_PR_Minimization()
    return alnsprminimization
