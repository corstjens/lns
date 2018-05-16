# -*- coding: utf-8 -*-

import random
import copy
import types
from lns.problem import Problem, Vehicle, Node, Request


class Route(object):

    # All vehicles depart at opening depot, even if this implies
    # a waiting time at the first customer.

    """
    The route of a vehicle.
    Each route consists of a list of ids associated with the nodes visited
    during the route.

    Parameters
    ----------
    problem: 'Problem'
        Problem object for which routes are created.
    vehicle_id: int
        External ID to identify the vehicle running the route.

    Attributes
    ----------
    capacity : int
        The capacity of a vehicle
    distancematrix: {(int,int) : int}
        A dictionary containing the corresponding distance for each arc
        (stored as a tuple).
    timematrix : {(int,int) : int}
        A dictionary containing the corresponding travel time for each arc
        (stored as a tuple)
    n_vehicles : int
        The number of vehicles in the problem
    n_requests : int
        The number of requests
    allnodes : {'Node'}
        A dictionary mapping all nodes (pickup, delivery, start terminal,
        end teminal)
    route : [int]
        A list presenting the sequence in which nodes are visited by a vehicle.
        List contains internal id's of visited nodes and starts with start
        terminal and ends with end terminal.
    cost : int
        Total distance cost of current route
    loads : [int]
        A list enclosing the load of the vehicle after servicing each node
    earliest : [int]
        A list enclosing the et value (= earliest time a node can be
        served according to its time windows and those of predecessors and
        successors) for each node
    latest : [int]
        A list enclosing the lt value (= latest time a node can be
        served according to its time windows and those of predecessors and
        successors) for each node
    """

    def __init__(self, problem, vehicle_id):

        """
        Initialize a new route
        """
        self.problem = problem
        # Vehicle ID of vehicle serving this route
        self.vehicle_id = vehicle_id
        # Max capacity of vehicle running this route
        self.capacity = problem.K[vehicle_id].capacity
        # Distance matrix between nodes {(nodeid1,nodeid2)--> distance}
        self.distancematrix = problem.distancematrix
        # Time matrix between nodes {(nodeid1,nodeid2)--> traveltime}
        self.timematrix = problem.timematrix
        # Number of vehicles in problem
        self.n_vehicles = problem.m
        # Number of requests
        self.n_requests = problem.n
        # dictionary of all nodes (V). {nodeid --> node}
        self.allnodes = problem.V
        # Initialize route with the vehicles start and endterminal
        # This is a list with the node ids, not the nodes!
        self.route = [2*problem.n+vehicle_id, 2*problem.n+problem.m+vehicle_id]
        # Total distance cost of the current route
        self.cost = self.calculate_cost()
        # Initialize array with loads (after servicing)
        self.loads = self.calculate_loads()
        # Initialize array with et values
        self.earliest = self.calculate_earliest(0)
        # Initialize array with lt values
        self.latest = self.calculate_latest(len(self.route)-1)

    def __deepcopy__(self, memo):
        cls = self.__class__
        dup = cls.__new__(cls)
        dup.__dict__.update(self.__dict__)
        setattr(dup, "route", copy.copy(self.route))
        setattr(dup, "loads", copy.copy(self.loads))
        setattr(dup, "earliest", copy.copy(self.earliest))
        setattr(dup, "latest", copy.copy(self.latest))
        return dup

    def __contains__(self, route_id):
        return route_id in self.route

    def calculate_cost(self):
        """
        Calculate the total distance cost of the route.

        Returns
        ----------
        result : int
            Distance cost of the route
        """
        result = 0.0
        for i in range(len(self.route)-1):
                result += self.distancematrix[(self.route[i], self.route[i+1])]
        return result

    def recalculate_cost_insert(self, position):
        """
        Calculate the total distance cost of the route after a node has been
        inserted.

        Returns
        ----------
        result : int
            Change in distance cost of the route
        """
        result = 0.0
        result -= self.distancematrix[(self.route[position-1],
                                       self.route[position+1])]
        result += self.distancematrix[(self.route[position-1],
                                       self.route[position])]
        result += self.distancematrix[(self.route[position],
                                       self.route[position+1])]
        return result

    def recalculate_cost_delete(self, position):
        """
        Calculate the total distance cost of the route after a node has been
        inserted or removed.

        Returns
        ----------
        result : int
            Change in distance cost of the route
        """
        result = 0.0
        result += self.distancematrix[(self.route[position-1],
                                       self.route[position+1])]
        result -= self.distancematrix[(self.route[position-1],
                                       self.route[position])]
        result -= self.distancematrix[(self.route[position],
                                       self.route[position+1])]
        return result

    def complete_requests(self):
        """
        Verify if the route contains complete requests, i.e. the pickup and
        the delivery node of a request must belong to the same route.

        Returns
        ----------
        result : boolean
            True if route contains complete requests, False if route contains
            at least one pickup (delivery) node without its associated delivery
            (pickup) node
        """
        if self.route[-1] - self.route[0] != self.n_vehicles:
            return False
        for i in range(1, len(self.route)-1):
            if(self.route[i] < self.n_requests):
                if (not (self.route[i]+self.n_requests in self.route)):
                    return False
            if(self.route[i] >= self.n_requests):
                if (not (self.route[i]-self.n_requests in self.route)):
                    return False
        return True

    def priority_check(self):
        """
        Verify if priority of nodes is respected in the route.

        Returns
        ----------
        result : boolean
            True if priority of nodes is respected, False if not
        """
        for i in range(len(self.route)-1):
            if (self.allnodes[self.route[i]].priority
                    > self.allnodes[self.route[i+1]].priority):
                return False
        return True

    def precedence_constraints(self):
        """
        Verify if the delivery node is visited after the paired pickup node.

        Returns
        ----------
        result : boolean
            True if precendence of nodes is respected, False if delivery node
            is visited before paired pickup node.
        """
        for i in range(1, len(self.route)-1):
            if(self.route[i] < self.n_requests):
                if (self.route.index(self.route[i] + self.n_requests) < i):
                    return False
            if(self.route[i] >= self.n_requests):
                if (self.route.index(self.route[i] - self.n_requests) > i):
                    return False
        return True

    def calculate_earliest(self, start_position):
        """
        Calculate the earliest time each node can be visited.

        Parameters
        ----------
        start_position : int
            Position from where the array should be (re)calculated.

        Returns
        ----------
        result : [int]
            Array enclosing the et values of all nodes
        """
        result = [None] * len(self.route)
        if start_position > 0:
            result[:start_position] = self.earliest[:start_position]
        else:
            result[0] = self.allnodes[self.route[0]].tw_start
            start_position += 1
            
        for i in xrange(start_position, len(self.route)):
            node_i = self.allnodes[self.route[i]]
            node_previous = self.allnodes[self.route[i-1]]
            a = node_i.tw_start
            b = (result[i-1] + node_previous.servicetime
                 + self.timematrix[(self.route[i-1], self.route[i])])
            result[i] = max(a, b)
        return result

    def calculate_latest(self, start_position):
        """
        Recalculate the latest time each node can be visited. 

        Parameters
        ----------
        start_position : int
            Position from where the array should be (re)calculated.

        Returns
        ----------
        result : [int]
            Array enclosing the lt values of all nodes
        """
        result = [None] * len(self.route)
        n_keep = len(self.route) - start_position -1
        if n_keep > 0:
            result[-n_keep:] = self.latest[-n_keep:]
        else:
            result[-1] = self.allnodes[self.route[-1]].tw_end
            n_keep = 1
        

        for i in xrange(len(self.route) - n_keep - 1, -1, -1):
            node_i = self.allnodes[self.route[i]]
            a = node_i.tw_end
            b = (result[i+1] - node_i.servicetime
                 - self.timematrix[self.route[i], self.route[i+1]])
            result[i] = min(a,b)

        return result

    def calculate_loads(self):
        """
        Calculate the load of the vehicle after service for each node.

        Returns
        ----------
        result : [int]
            Array enclosing the load of the vehicle after servicing each node
        """
        pre_service_load = 0
        result = []
        for node_id in self.route:
            post_service_load = pre_service_load + self.allnodes[node_id].load
            result.append(post_service_load)
            pre_service_load = post_service_load
        return result

    def verify_tw_constraints(self):
        """
        Verify if the time window constraints are respected.

        Returns
        ----------
        result : boolean
            True if time windows are respected, False if not
        """
        for i in range(len(self.route)):
            if self.latest[i] < self.allnodes[self.route[i]].tw_start:
                return False
            if self.earliest[i] > self.allnodes[self.route[i]].tw_end:
                return False
            if i < len(self.route)-1:
                if (self.earliest[i+1]
                        < (self.earliest[i]
                           + self.allnodes[self.route[i]].servicetime
                           + self.timematrix[self.route[i], self.route[i+1]])):
                            return False
        return True

    def verify_load_constraints(self):
        """
        Verify if the load constraints are respected.

        Returns
        ----------
        result : boolean
            True if vehicle capacity is respected, False if not
        """
        if self.loads[0] != 0 or self.loads[-1] != 0:
            return False
        for i in range(1, len(self.route)-1):
            if self.loads[i] > self.capacity:
                return False
        return True

    def insert_node(self, position, node_id):
        """
        Insert a node_id at a given position in the route.

        Parameters
        ----------
        position : int
            Position in route (index of the element before which to insert)
        node_id: int
            External ID to identify node

        Notes
        -----
        After inserting, total distance cost of route, eraliest time of
        nodes, latest time of nodes and load after servicing nodes are updated.

        """
        self.route.insert(position, node_id)
        self.cost += self.recalculate_cost_insert(position)
        self.earliest = self.calculate_earliest(position)
        self.latest = self.calculate_latest(position)
        self.loads = self.calculate_loads()

    def delete_node(self, node_id):
        """
        Remove a node_id from the route.

        Parameters
        ----------
        node_id: int
            External ID to identify node

        Notes
        -----
        After removing, total distance cost of route, eraliest time of
        nodes, latest time of nodes and load after servicing nodes are updated.

        """
        position = self.route.index(node_id)
        self.cost += self.recalculate_cost_delete(position)
        self.route.remove(node_id)
        self.earliest = self.calculate_earliest(position)
        self.latest = self.calculate_latest(position-1)
        self.loads = self.calculate_loads()

    def __len__(self):
        """Retrieve the length of a route"""
        return len(self.route)

    def __repr__(self):
        result = "* Total current cost: %s \n" % self.cost
        result += "* (node: [start, departure_load]):\n"
        for i in range(len(self.route)):
            result += "(%s: [%s, %s]) --> " % (self.route[i],
                                               self.earliest[i],
                                               self.loads[i])
        return result

    def isValid(self):
        """
        Verify if the route passes all constraints.

        Returns
        ----------
        result : boolean
            True if all constraints are respected, False if not
        """
        return (self.verify_load_constraints())


class Solution(object):
    """
    The solution to a RPDPTW problem

    Parameters
    ----------
    problem : `Problem`
        The RPDPTW problem which contains information about nodes, requests
        and vehicles.
    available_vehicles: [int], optional
        The list with available vehicles. If not set, all vehicles in the
        given RPDPTW problem are considered available


    Attributes
    ---------
    problem: `Problem`
        The RPDPTW problem which contains information about nodes, requests and
        vehicles.
    available_vehicles: [int]
        The list with available vehicles. Initially set to all vehicles in the
        RPDPTW problem.
    routes : [`Route`]
        A list of `Route` object. The n-th route of the list corresponds to the
        n-th vehicle in the list of available_vehicles.
    request_bank : [int]
        The id's of requests which still need to be assigned to routes. Note
        that the id of a request corresponds to the id of the related pickup
        node. Initially, all requests of the problem are in the request bank.
    cost_unserved_requests : int
        Cost associated with not servicing a request
    _delta_f : `ndarray`
        A two-dimensional numpy array which is used for the insert heuristics.
        The dimensions are (number_request, number_routes) and the array holds
        the change in the overal solution cost (insertion cost) when a specific
        request would be added to a specific route.
        Once a request has been assigned, its insertion cost will be NaN for
        all routes. If a request cannot validly be added to a route, the value
        is Inf.
    _best_insert_position : `ndarray`
        A two-dimensional numpy array which is used for the insert heuristics.
        The dimensions are (number_requests, number_routes) and the array
        is the counterpart of _delta_f. Whereas _delta_f holds the insertion
        cost, this array holds the best position for a specific request to be
        added to a specific route.
        If the request is already assigned, the value for the request row is
        NaN. To know if the request can be validly added to the route, one must
        inspect _delta_f. This can not be learned from _best_insert_position
    noise_parameter : float
        Parameter controlling the amount of noising added to the repair
        heuristics. (default = 0.025)
    neighbor_graph : {(int,int) : int}
        A dictionary containing the weight f*(u,v) for each arc (u,v). f*(u,v)
        indicates the best solution found so far, in a solution which used
        edge (u,v). Initially, f*(u,v) is set to infinity and each time a new
        solution is found, the weights f*(u,v) of all edges used in the given
        solution are updated. These weights are used in the "historical node-
        pair removal" heuristic (destroy_factory.py).
        (stored as a tuple)
    """
    def __init__(self, problem, available_vehicles=None):
        self.problem = problem
        self.allnodes = problem.V
        self.timematrix = problem.timematrix
        self.distancematrix = problem.distancematrix
        self.neighbor_graph = problem.neighbor_graph
        self.maximum = max(self.distancematrix.values())
        if available_vehicles is None:
            self.available_vehicles = self.problem.K.keys()
        else:
            self.available_vehicles = available_vehicles
        self.cost_unserved_requests = 100000
        self.routes = []
        for k in self.available_vehicles:
            self.routes.append(Route(self.problem, k))
        self.request_bank = self.problem.P.keys()
        #self.noise_parameter = 0.025
        self._rebuild_insert_matrices()

    def __deepcopy__(self, memo):
#        dup = Solution(self.problem)
#        dup.available_vehicles = copy.copy(self.available_vehicles)
#        dup.request_bank = copy.copy(self.request_bank)
#        dup.routes = copy.deepcopy(self.routes)
        cls = self.__class__
        dup = cls.__new__(cls)
        dup.__dict__.update(self.__dict__)
        setattr(dup, "available_vehicles", copy.copy(self.available_vehicles))
        setattr(dup, "request_bank", copy.copy(self.request_bank))
        setattr(dup, "routes", copy.deepcopy(self.routes, memo))
        return dup

    def calculate_solution_cost(self, unserved_cost=None):
        """
        Calculate the cost of the current solution.

        Parameters
        ----------
        unserved_cost: int, optional
            Cost associated with requests that remain unserved

        Returns
        ----------
        total_cost: int
            Total cost of current solution
        """
        if unserved_cost is None:
            unserved_cost = self.cost_unserved_requests
        total_cost = unserved_cost * len(self.request_bank)
        for route in self.routes:
            total_cost += route.cost
        return total_cost

    def remove_last_vehicle(self):
        """
        Removes the last vehicle from the list of available vehicles.

        This method removes the last vehicle from the available vehicle list,
        together with the corresponding route in the routes list. Ultimately,
        the insert matrices are rebuilt.
        """
        # Remove last vehicle from available vehicle list
        self.available_vehicles.pop()
        #Add all the assigned requests to the corresponding route to the
        #request bank
        requests_ids = [i for i in self.routes[-1].route if i < self.problem.n]
        self.request_bank.extend(requests_ids)
        # Remove the last route from the route list
        self.routes.pop()
        #Recalculate the delta_f and best_insert_position matrices
        self._rebuild_insert_matrices()

    def update_neighbor_graph(self, best_solution_cost):
        """
        This method updates the neighbor graph defined in problem.py.
        For each arc (u,v) the weight f*(u,v) is calculated which is equal to
        the best solution value found so far if the arc (u,v) is used in the
        solution.
        
        Parameters
        ----------
        best_solution_cost: int
            Cost associated with the best solution found so far

        """
        for route_id in self.available_vehicles:
            for i in range(len(self.routes[route_id].route)-1):
                self.neighbor_graph[(self.routes[route_id].route[i], self.routes[route_id].route[i+1])] = best_solution_cost

    def _rebuild_insert_matrices(self):
        """
        (Re)build the insert matrices.

        This method (re)builds _delta_f and _best_insert_position.
        Matrices needed by the insert heuristics.
        """
        # Creating and initializing matrices
        self._delta_f = [[float('inf')]*len(self.routes) for x in xrange(self.problem.n)]
        self._best_insert_position = [[-1]*len(self.routes) for x in xrange(self.problem.n)]

        # For each route
        for route_id in self.available_vehicles:
            self._update_best_insert_route(route_id)

    def _calculate_noisematrix(self, noise_parameter):
        """
        Calculates a noise matrix containing the noise values that need to
        be added to the _delta_f and _best_insert_position matrices used in
        the noise-repair heuristics.
        """
        Nmax = noise_parameter * self.maximum
        noisematrix = [[random.uniform(-Nmax, Nmax) for i in
                       range(len(self.routes))] for j in
                       range(self.problem.n)]
        return noisematrix

    def _update_best_insert_route(self, route_id):
        """
        Update cost and position of cheapest insert in specific route
        for all requests at request bank.

        Parameters
        ----------
        route_id : int
            External ID to identify the route
        """
        for request_id in self.request_bank:
            self._update_best_insert(route_id, request_id)

    def _update_best_insert_request(self, request_id):
        """
        Update cost and position of cheapest insert of specific request
        accross all AVAILABLE routes.

        Parameters
        ----------
        request_id : int
            External ID to identify the request
        """
        for route_id in self.available_vehicles:
            self._update_best_insert(route_id, request_id)

    def _update_best_insert(self, route_id, request_id):
        """
        Update cost and position of cheapest request insert.

        Parameters
        ----------
        route_id : int
            External ID of route where request must be inserted
        request_id: int
            External ID of request that must be inserted
        """
        # This function currently assumes a VRPTW problem where the pickup
        # node does not increase the cost (same location as starting terminal)
        # Therefore, the pickup node is added at position 1 (after start
        # terminal and the problem is reduced to inserting the delivery node

        #TOO SLOW!!
        #route = copy.deepcopy(self.routes[route_id])
        #print "request %s route %s" % (request_id, route_id)
        #Hold cost before insertion to compare with
        current_cost = self.routes[route_id].cost
        #Insert the pickup node (the request id = pickup node id)
        self.routes[route_id].insert_node(1, request_id)
        best_insert_position = -1
        lowest_delta_f = float('inf')
        delivery_node_id = request_id + self.problem.n
        for position in range((len(self.routes[route_id]) + 1)/2,
                              len(self.routes[route_id])):
            e_i = max(self.allnodes[delivery_node_id].tw_start,
                      (self.routes[route_id].earliest[position-1] +
                       self.allnodes[
                           self.routes[route_id].route[position-1]].servicetime
                       +
                       self.timematrix[self.routes[route_id].route[position-1],
                                       delivery_node_id]))
            l_i = min(self.allnodes[delivery_node_id].tw_end,
                      (self.routes[route_id].latest[position] -
                       self.allnodes[delivery_node_id].servicetime
                       -
                       self.timematrix[delivery_node_id,
                                       self.routes[route_id].route[position]]))
            new_cost = current_cost + (
                - self.distancematrix[self.routes[route_id].route[position-1],
                                    self.routes[route_id].route[position]]
                + self.distancematrix[self.routes[route_id].route[position-1],
                                      delivery_node_id]
                + self.distancematrix[delivery_node_id,
                                      self.routes[route_id].route[position]])
            delta_f = new_cost - current_cost
            if ((delta_f < lowest_delta_f) and (e_i <= l_i)):
                #Insert delivery node
                self.routes[route_id].insert_node(position, delivery_node_id)
                if (self.routes[route_id].isValid()):
                    new_cost = self.routes[route_id].cost
                delta_f = new_cost - current_cost
                if delta_f < lowest_delta_f:
                    lowest_delta_f = delta_f
                    best_insert_position = position
                self.routes[route_id].delete_node(delivery_node_id)
        # self._delta_f[request_id, route_id] = lowest_delta_f
        self._delta_f[request_id][route_id] = lowest_delta_f
        self._best_insert_position[request_id][route_id] = (
            best_insert_position)
        self.routes[route_id].delete_node(request_id)

    def _assign_request(self, request_id, route_id, position):
        """
        Assign request to specific route at specific position.

        Parameters
        ----------
        route_id : int
            External ID of route where request must be inserted
        request_id: int
            External ID of request that must be inserted
        position: int
            Position in route (index of the element before which to insert)
        """
        position = int(position)
        # Remove the request from the request bank

        self.request_bank.remove(request_id)
        # Add the pickupnode at position 1
        self.routes[route_id].insert_node(1, request_id)
        # Add the delivery node at position "position"
        self.routes[route_id].insert_node(position,
                                          request_id + self.problem.n)
        # Set the delta_f to None for the inserted request for ALL routes
        # self._delta_f[request_id, :] = numpy.NaN
        for id in range(len(self.routes)):
            self._delta_f[request_id][id] = float('Nan')
        # Set the _best_insert_position to -1 for the inserted request
        # for ALL routes
        # self._best_insert_position[request_id, :] = -1
        for id in range(len(self.routes)):
            self._best_insert_position[request_id][id] = -1
        # Update the insert matrix for the route to which the request was
        # assigned
        self._update_best_insert_route(route_id)

    def _remove_request(self, request_id):
        """
        Remove a request from the solution to the request bank.

        Parameters
        ----------
        request_id: int
            External ID of request that must be removed
        """
        route_id = self._find_route_containing_request(request_id)
        # 1. Remove request from route
        route = self.routes[route_id]
        # Remove pickup node
        route.delete_node(request_id)
        # Remove delivery node
        route.delete_node(request_id + self.problem.n)
        # 2. Update the insert matrices (By updating the matrices before adding
        # the request back to the request bank, we prevent the method
        # _update_best_insert_route to recalculate the insert cost for the
        # removed request in its previously assigned route. This is not
        # necessary anymore, since this value is already calculated by
        # method _update_best_insert_request)
        self._update_best_insert_request(request_id)
        self._update_best_insert_route(route_id)
        # 3. Add request to request bank
        self.request_bank.append(request_id)

    def _remove_request_no_update(self, request_id):
        """
        Remove a request from the solution without updating the insert
        matrices or moving them to the request bank.

        Parameters
        ----------
        request_id: int
            External ID of request that must be removed
        """
        route_id = self._find_route_containing_request(request_id)
        # 1. Remove request from route
        route = self.routes[route_id]
        # Remove pickup node
        route.delete_node(request_id)
        # Remove delivery node
        route.delete_node(request_id + self.problem.n)

    def _remove_request_completely(self, request_id):
        """
        Remove a request from the solution completely (request is not moved to
        the request bank).

        Parameters
        ----------
        request_id: int
            External ID of request that must be removed
        """
        route_id = self._find_route_containing_request(request_id)
        # 1. Remove request from route
        route = self.routes[route_id]
        # Remove pickup node
        route.delete_node(request_id)
        # Remove delivery node
        route.delete_node(request_id + self.problem.n)
        # 2. Update the insert matrices
        self._update_best_insert_request(request_id)
        self._update_best_insert_route(route_id)

    def _find_route_containing_request(self, request_id):
        """
        Find the route_id of the route containing the given request.

        Parameters
        ----------
        request_id: int
            External ID of request

        Returns
        ----------
        result: int
            route_id of the route containing the given request
        """
        result = None
        for route_id in self.available_vehicles:
            if request_id in self.routes[route_id]:
                result = route_id
                break
        return result

    def _is_next_insert_possible(self, routes=None):
        """
        Test if it is possible to assign another request to a list of routes.
        If no list of routes are defined, all available routes are considered.

        Parameters
        ----------
        routes : int, optional
            route_id of route that is investigated for possible insertions

        Returns
        ----------
        numpy.any(...): boolean
            True if at least one of the requests has a route for which the
            insert cost does not equal Inf, False if not
        """
        # Return True if at least one of the unserved requests (those in
        # the request bank) has a route for which the insert cost does not
        # equal Inf
        selected_routes = routes
        if routes is None:
            selected_routes = self.available_vehicles
            for request_id in self.request_bank:
                for route_id in selected_routes:
                    if self._delta_f[request_id][route_id] != float('inf'):
                        return True
                    else: continue
        else:
            for request_id in self.request_bank:
                if self._delta_f[request_id][selected_routes] != float('inf'):
                    return True
                else: continue
        return False
                
    def _number_of_used_vehicles(self):
        """
        Calculates the the total number of used vehicles in the
        current solution.

        Returns
        ----------
        used_vehicles: int
            Total number of used vehicles of current solution
        """
        used_vehicles = 0
        for route in self.routes:
            if len(route) > 2:
                used_vehicles += 1
        return used_vehicles

    def __repr__(self):
        result = ""
        for i in range(len(self.routes)):
            result += "Route %s:\n----------\n%r\n\n" % (i, self.routes[i])
        result += "Total cost:\n----------\n%s\n\n" % (
                  (self.calculate_solution_cost()))
        result += "Number of used vehicles:\n----------\n%s\n\n" % (
                  (self._number_of_used_vehicles()))
        result += "Request bank:\n"
        result += "-------------\n"
        for i in range(len(self.request_bank)):
            result += "%s, " % self.request_bank[i]
        return result
