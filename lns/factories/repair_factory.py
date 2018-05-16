# -*- coding: utf-8 -*-

import math
import copy

def produce_insert_greedy_sequential():
    def insert_greedy_sequential(solution, noise, noise_parameter):
        """
        Insert unserved requests based on sequential greedy heuristic.

        Notes
        -------
        This heuristic creates a valid solution. It adds requests until there
        are no unserved requests left OR none of the unserved requests can be
        entered validly.
        """
        #Determine whether or not noise needs to be applied to the
        #repair heuristic
        delta_f_number_of_rows = sum(1 for x in solution._delta_f if 
                                     isinstance(x, list))
        delta_f_number_of_columns = sum(1 for x in solution._delta_f if 
                                        isinstance(x, list))
        noisematrix = [[0.0]*delta_f_number_of_columns]*delta_f_number_of_rows
        number_inserted = 0
        if noise is True:
            noisematrix = solution._calculate_noisematrix(noise_parameter)
            # solution._delta_f = numpy.add(solution._delta_f, noisematrix)
            delta_f_with_noise= []
            delta_f_with_noise = [[solution._delta_f[x][y] + noisematrix[x][y] 
                                   for y in range(len(solution._delta_f[0]))] 
                                   for x in range(len(solution._delta_f))]
            solution._delta_f = delta_f_with_noise
        i = 0
        #Go through all the available routes, starting from the first one
        while(i < len(solution.available_vehicles)):
            #Select the appropriate route
            route_id = solution.available_vehicles[i]
            #For this route, try to add as much requests as possible
            while(solution._is_next_insert_possible(route_id)):
                # Find request id with cheapest insert cost
                # request_id = numpy.nanargmin(solution._delta_f[:, route_id])
                insert_costs_route= [solution._delta_f[request_id][route_id] for 
                                   request_id in range(len(solution._delta_f))]
                insert_costs_route_wo_nan = [x for x,y in 
                                             enumerate(insert_costs_route) if 
                                             math.isnan(y) is False]
                value_list_insert_costs_route = [insert_costs_route[z] for z in
                                                 insert_costs_route_wo_nan]
                cheapest_insert_cost_route = min(value_list_insert_costs_route)
                request_id = insert_costs_route.index(cheapest_insert_cost_route)
                # Add request to route
                number_inserted +=1
                solution._assign_request(
                    request_id,
                    route_id,
                    solution._best_insert_position[request_id][route_id])
            i += 1
        # solution._delta_f = numpy.subtract(solution._delta_f, noisematrix)
        delta_f_without_noise = []
        delta_f_without_noise = [[solution._delta_f[x][y] + noisematrix[x][y] 
                                  for y in range(len(solution._delta_f[0]))] 
                                  for x in range(len(solution._delta_f))] 
        solution._delta_f = delta_f_without_noise
    return insert_greedy_sequential


def produce_insert_greedy_parallel():
    def insert_greedy_parallel(solution, noise, noise_parameter):
        """
        Insert unserved requests based on parallel greedy heuristic.
        (Basic greedy heuristic - Pisinger & Ropke (2007))
        """
        #Determine whether or not noise needs to be applied to the
        #repair heuristic
        delta_f_number_of_rows = sum(1 for x in solution._delta_f if 
                                     isinstance(x, list))
        delta_f_number_of_columns = sum(1 for x in solution._delta_f if 
                                        isinstance(x, list))
        noisematrix = [[0.0]*delta_f_number_of_columns]*delta_f_number_of_rows
        number_inserted = 0
        if noise is True:
            noisematrix = solution._calculate_noisematrix(noise_parameter)
            delta_f_with_noise= []
            delta_f_with_noise = [[solution._delta_f[x][y] + noisematrix[x][y] 
                                   for y in range(len(solution._delta_f[0]))] 
                                   for x in range(len(solution._delta_f))]
            solution._delta_f = delta_f_with_noise
        # Repeat as long as there are requests in the request bank and at least
        # one of these request can be added to a route
        while solution._is_next_insert_possible():
            # Determine the request_id and route_id which has the smallest
            # impact
            insert_costs= [solution._delta_f[request_id][route_id] for 
                           request_id in range(len(solution._delta_f))
            for route_id in range(len(solution._delta_f[0]))]
            insert_costs_wo_nan = [i for i,x in enumerate(insert_costs) if 
                                   math.isnan(x) is False]
            value_list_insert_costs = [insert_costs[x] for x in 
                                       insert_costs_wo_nan]
            minimum = min(value_list_insert_costs)
            requests_with_min_insert_costs = [i for i,x in enumerate(insert_costs)
                                        if x == minimum]

            min_arg = requests_with_min_insert_costs[0]
            n_routes = len(solution._delta_f[0])
            request_id = min_arg // n_routes
            route_id = min_arg % n_routes

            # assign request to route
            number_inserted += 1
            solution._assign_request(request_id,
                                     route_id,
                                     solution._best_insert_position[request_id]
                                                                    [route_id])

        # solution._delta_f = numpy.subtract(solution._delta_f, noisematrix)
        delta_f_without_noise = []
        delta_f_without_noise = [[solution._delta_f[x][y] - noisematrix[x][y] 
                                  for y in range(len(solution._delta_f[0]))] 
                                  for x in range(len(solution._delta_f))]
        solution._delta_f = delta_f_without_noise
    return (insert_greedy_parallel)


def produce_insert_regret_k(k=2, single_run=False):
    """
    Creates an insert_regret_k method to repair solutions.

    The method returned is the implementation of the regret repair function
    for a specific k-value, which either repairs the solution as complete as
    possible (until no requests can be added anymore), or only by a single step
    (adding one request from the request bank to its optimal position)

    Parameters
    ----------
    k : int
        The k value of the regret function. Note that this should cannot be
        greater than the number of routes. [Default value = 2]
    single_run : bool
        Whether or not the repair function should only add a single request
        from the request bank or should repair the solution as complete as
        possible. [Default value = False]

    Returns
    -------
    function
        A regret repair function for a specific k-value, which repairs one
        request at a time or until no longer possible (as complete as
        possible). This repair function requires a `Solution` object to operate
        on.
    """
    def insert_regret_k(solution, noise, noise_parameter):
        #Determine whether or not noise needs to be applied to the
        #repair heuristic
        noisematrix = [[0.0]*len(solution._delta_f[0])]*len(solution._delta_f)
        number_inserted = 0
        if noise is True:
            noisematrix = solution._calculate_noisematrix(noise_parameter)
            delta_f_with_noise= []
            delta_f_with_noise = [[solution._delta_f[x][y] + noisematrix[x][y] 
                                   for y in range(len(solution._delta_f[0]))] 
                                   for x in range(len(solution._delta_f))]
            solution._delta_f = delta_f_with_noise
        # Repeat as long as there are requests in the request bank and at least
        # one of these request can be added to a route
        while solution._is_next_insert_possible():
            #Test if regret_k is possible
            # if solution._delta_f.shape[1] < k
            if len(solution._delta_f[0]) < k:
                raise IndexError(
                    "Number of routes is less than k from regret_k")
            #Identify indices of empty routes
            empty_routes = []
            route_index = 0
            for route in solution.routes:
                if len(route) == 2:
                    empty_routes.append(route_index)
                    route_index += 1
                else: route_index += 1
            #Create a row sorted copy of delta_f
            # sorted_delta_f = solution._delta_f.copy()
            # sorted_delta_f.sort()
            sorted_delta_f = copy.deepcopy(solution._delta_f)
            #Give creation of new routes lowest priority by considering
            #them as infeasible options in regret calculation
            if len(empty_routes) > 1:
                for req_index in solution.request_bank:
                    for route_index in empty_routes[1:]:
                        sorted_delta_f[req_index][route_index] = 9999
            sorted_delta_f = [sorted(sublist) for sublist in sorted_delta_f]
            #Calculate the regret values for each row (= request)
            regret_array = [0.0]* len(solution._delta_f)
            f_1 = [request[0] for request in sorted_delta_f]
            testlist = [i for i, x in enumerate(f_1) if x == float('inf')]
            for i in range(0, len(testlist)):
                f_1[testlist[i]] = 9999
            for i in range(1, k):
                f_i = [request[i] for request in sorted_delta_f]
                testlist = [j for j, x in enumerate(f_i) if x == float('inf')]
                for j in range(0, len(testlist)):
                    f_i[testlist[j]] = 9999
                diff_f_i_and_f_1 = [a - b for a, b in zip(f_i, f_1)]
                zero_matrix = [0.0]*len(f_i)
                maximum_list = [max(a,b) for a,b in zip(diff_f_i_and_f_1, 
                                zero_matrix)]
                for index in range(len(regret_array)):
                    regret_array[index] += maximum_list[index]
            #Determine the request(s) with maximum regret value
            
            list_wo_nan = [i for i,x in enumerate(regret_array) 
                           if math.isnan(x) is False]
            value_list = [regret_array[x] for x in list_wo_nan]
            maximum = max(value_list)
            requests_with_max_regret = [i for i,x in enumerate(regret_array)
                                        if x == maximum]

            #In case of ties, select request with lowes insert cost
            if len(requests_with_max_regret) > 1:
                row_candidates = []
                for request in requests_with_max_regret:
                    row_candidates.append(solution._delta_f[request][:])

                minimum_list = [min(row) for row in row_candidates]
                request_id = requests_with_max_regret[minimum_list.index(min(minimum_list))]
                    
            else:
                request_id = requests_with_max_regret[0]
            if sorted_delta_f[request_id][1] == 9999:
                print empty_routes
                print request_id
                print maximum
                print sorted_delta_f[request_id]
            #Identify route with lowest insert cost for selected request
            all_routes_request_id = solution._delta_f[request_id]
            cheapest_insert_cost = min(all_routes_request_id)
            route_id = all_routes_request_id.index(cheapest_insert_cost)

            #Assign request to route
            number_inserted += 1
            solution._assign_request(request_id, route_id,
                                     solution._best_insert_position[request_id]
                                                                    [route_id])

            if single_run:
                delta_f_without_noise = []
                for request_delta_f, request_noisematrix in zip(solution._delta_f, noisematrix):
                    sublist = [request_delta_f[x] - request_noisematrix[x] for 
                               x in range(len(solution._delta_f[0]))]
                    delta_f_without_noise.append(sublist) 
                solution._delta_f = delta_f_without_noise
                break
        delta_f_without_noise = []
        delta_f_without_noise = [[solution._delta_f[x][y] - noisematrix[x][y] 
                                  for y in range(len(solution._delta_f[0]))] 
                                  for x in range(len(solution._delta_f))]
        solution._delta_f = delta_f_without_noise
    return (insert_regret_k)


def produce_insert_regret_2():
    """
    Creates a regret-2 repair function

    The returned regret-2 repair function will repairs the solution until no
    requests can be added anymore. This function exists for backward
    compatibility.
    """
    return produce_insert_regret_k()

def produce_insert_regret_3():
    """
    Creates a regret-3 repair function

    The returned regret-3 repair function will repairs the solution until no
    requests can be added anymore. This function exists for backward
    compatibility.
    """
    return produce_insert_regret_k(3)

def produce_insert_regret_4():
    """
    Creates a regret-4 repair function

    The returned regret-4 repair function will repairs the solution until no
    requests can be added anymore. This function exists for backward
    compatibility.
    """
    return produce_insert_regret_k(4)
