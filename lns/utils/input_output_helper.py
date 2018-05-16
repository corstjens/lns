# -*- coding: utf-8 -*-

from lns.problem import Node, Request, Vehicle, Problem
from itertools import islice
import copy


def input_nodes(filename):
    def read_nodes():
        with open(filename) as node_file:
            #Create list of all nodes
            nodes = []
            for line in islice(node_file, 9, None):
                if line[0:7] == "average":
                    break
                else: nodes.append(Node(*line.split()))
            return nodes
    return read_nodes


#Assumption of only one depot and VRPTW
def input_data(filename):
    with open(filename) as node_file:
        nodes_list = input_nodes(filename)
        nodes = nodes_list()
        #Create depot
        depot = nodes[0]
        depot.x_coord = int(depot.x_coord)
        depot.y_coord = int(depot.y_coord)
        depot.tw_start = int(depot.tw_start)
        depot.tw_end = int(depot.tw_end)
        depot.servicetime = int(depot.servicetime)
        depot.load = int(depot.load)
        depot.priority = 0
        start_terminal = copy.copy(depot)
        end_terminal = copy.copy(depot)
        end_terminal.priority = 1
        #Create vehicles
        vehicles = []
        line_vehicles = node_file.readlines()
        vehicle_info = line_vehicles[4].split()
        for i in range(0, int(vehicle_info[0])):
            vehicles.append(Vehicle(int(vehicle_info[1]), start_terminal,
                                    end_terminal, [], i))
            i += 1
        #Identify runtime stopcriterium
        runtime_info = line_vehicles[1].split()
        runtime = runtime_info[1]
        #Create customers
        customers = []
        for i in range(1, len(nodes)):
            customers.append(nodes[i])
        #Create pickups
        pickups = []
        for c in customers:
            p = copy.copy(c)
            p.x_coord = int(depot.x_coord)
            p.y_coord = int(depot.y_coord)
            p.tw_start = int(depot.tw_start)
            p.tw_end = int(p.tw_start)
            p.servicetime = 0
            p.load = int(c.load)
            p.priority = 0
            pickups.append(p)
        #Create deliveries
        deliveries = []
        for c in customers:
            d = copy.copy(c)
            d.x_coord = int(d.x_coord)
            d.y_coord = int(d.y_coord)
            d.tw_start = int(d.tw_start)
            d.tw_end = int(d.tw_end)
            d.servicetime = int(d.servicetime)
            d.load = -int(d.load)
            d.priority = 1
            deliveries.append(d)
        #Create requests
        requests = []
        for i in range(len(pickups)):
            r = Request(pickups[i], deliveries[i])
            requests.append(r)
        #Create problem
        problem = Problem(requests, vehicles, runtime)
    return problem
