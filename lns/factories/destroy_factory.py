# -*- coding: utf-8 -*-

import random
import math


def produce_remove_random(number_of_requests, factor_lower_bound, 
                          factor_upper_bound):
    def remove_random(solution):
        """
        Remove n random requests from the solution.
        Update insert matrices only when all requests have been removed.

        Parameters
        ----------
        number_of_req_to_remove = int
            Number of requests that will be removed from the solution
        """
        #Listing the request id's of served requests
        request_ids = [id for id in solution.problem.P.keys() if id not in 
        solution.request_bank]
        #Randomly select the number of requets to remove. You cannot remove
        #more requests than the total number of served requests
        number_of_req_to_remove = min(len(request_ids), random.randint(int(number_of_requests*factor_lower_bound), 
                                                 int(number_of_requests*factor_upper_bound)))

        #Take random selection of requests (that belong to the solution)
        selected_request_ids = random.sample(request_ids, number_of_req_to_remove)
        selected_route_ids = []
        for request_id in selected_request_ids:
            selected_route_ids.append(
                solution._find_route_containing_request(request_id))
        #Remove all requests
        for request_id in selected_request_ids:
            solution._remove_request_no_update(request_id)
        #Update insert matrices
        for request_id in selected_request_ids:
            solution._update_best_insert_request(request_id)
        for route_id in selected_route_ids:
            solution._update_best_insert_route(route_id)
        #Move removed requests to request bank
        for request_id in selected_request_ids:
            solution.request_bank.append(request_id)
    return (remove_random)


def produce_remove_worst(number_of_requests, factor_lower_bound, 
                          factor_upper_bound, random_parameter):
    def remove_worst(solution):
        """
        Remove requests with high costs.

        Parameters
        ----------
        number_of_req_to_remove = int
            Number of requests that will be removed from the solution
        random_parameter = int
            Parameter (>= 1) introduces randomness in the selection of
            requests (a low value corresponds to much randomness)
        """
        #Listing the request id's of served requests
        request_ids = [id for id in solution.problem.P.keys() if id not in 
        solution.request_bank]
        #Randomly select the number of requets to remove. You cannot remove
        #more requests than the total number of served requests
        number_of_req_to_remove =  min(len(request_ids), random.randint(int(number_of_requests*factor_lower_bound), 
                                                 int(number_of_requests*factor_upper_bound)))
        p = random_parameter
        #create two-dimensional array listing the cost of all requests in the
        #solution as cost(i,s) = f(s) - f(s\i)
        cost_i_s = []
        for k in range(len(request_ids)):
            cost_i_s.append([])
        #eerste kolom van de array bevat de requests van de solution
        #tweede kolom van de array bevat de cost(i,s) van elke request
        i = 0
        for s in request_ids:
            route_id = solution._find_route_containing_request(s)
            position = (solution.routes[route_id].
                        route.index(s+solution.problem.n))
            f_s = solution.calculate_solution_cost()
            solution._remove_request(s)
            f_s_no_i = (solution.calculate_solution_cost()
                        - solution.cost_unserved_requests)
            c_i_s = f_s - f_s_no_i
            cost_i_s[i].append(s)
            cost_i_s[i].append(c_i_s)
            solution._assign_request(s, route_id, position)
            i += 1
        #sort array by descending cost c_i_s of the requests
        sorted_cost = sorted(cost_i_s,
                             key=lambda cost_i_s: cost_i_s[1], reverse=True)

        #Remove as many requests as defined by number_of_req_to_remove
        selected_route_ids = []
        selected_request_ids = []
        removed_requests = []
        while number_of_req_to_remove > 0:
            r = int((math.pow(random.random(), p))*len(request_ids))
            while r in removed_requests:
                r = int((math.pow(random.random(), p))*len(request_ids))
            request_id = sorted_cost[r][0]
            route_id = solution._find_route_containing_request(request_id)
            selected_request_ids.append(request_id)
            selected_route_ids.append(route_id)
            # remove request from solution (no update)
            solution._remove_request_no_update(request_id)
            number_of_req_to_remove = number_of_req_to_remove - 1
            removed_requests.append(r)
        #Update insert matrices
        for request_id in selected_request_ids:
            solution._update_best_insert_request(request_id)
        for route_id in selected_route_ids:
            solution._update_best_insert_route(route_id)
        #Move removed requests to request bank
        for request_id in selected_request_ids:
            solution.request_bank.append(request_id)
    return (remove_worst)
    
def produce_remove_related(number_of_requests, factor_lower_bound, 
                          factor_upper_bound, random_parameter):
    def remove_related(solution): 
        #Listing the request id's of served requests
        request_ids = [id for id in solution.problem.P.keys() if id not in 
        solution.request_bank]
        #Randomly select the number of requets to remove. You cannot remove
        #more requests than the total number of served requests
        number_of_req_to_remove =  min(len(request_ids), random.randint(int(number_of_requests*factor_lower_bound), 
                                                 int(number_of_requests*factor_upper_bound)))
        p = random_parameter
 
        #Listing the pickups/deliveries located at a terminal 
        terminal_pickup, terminal_delivery = (
        verify_pickup_delivery_on_terminal(solution, request_ids))
        #Randomly select a request from the solution to remove
        random_index = random.randint(0, len(request_ids)-1)
        id_request_to_remove = request_ids[random_index]
        solution._remove_request(id_request_to_remove)
        removed_requests = [id_request_to_remove]
        number_of_req_to_remove -= 1
        list_no_relatedness = []

        #Remove as many requests as defined by number_of_req_to_remove
        while number_of_req_to_remove > 0:
            random_request_index = random.randrange(0, len(removed_requests))
            random_request_id = removed_requests[random_request_index]
            list_of_requests_not_removed = [id for id in request_ids if id not 
            in removed_requests]
            
            #Calculate relatedness of random_request to each of the requests 
            #that hasn't been removed yet, and put it in a list: 
            #(id unremoved request, relatedness)
            relate_list = [[id, calculate_relatedness(solution, random_request_id,
                            id, terminal_pickup, terminal_delivery)] for id in 
                            list_of_requests_not_removed]
            #Remove None-values from list. For a VRP problem with 1 depot, 
            #relate_list = relatedness_list
            relatedness_list = [sublist for sublist in relate_list if None not
            in sublist]
            #Sort by descending relatedness
            sorted_relatedness_list = sorted(relatedness_list, 
                            key=lambda relatedness_list: relatedness_list[1])
            #Choose random related request: as p increases, the more related 
            #request is chosen
            random_related = int((math.pow(random.random(), p))*
                                len(sorted_relatedness_list))

            
            #If no related values, put id in list
            #When id's in list = removed id's, break
            if len(sorted_relatedness_list) > 0:
                related_id = sorted_relatedness_list[random_related][0]
                solution._remove_request(related_id)
                removed_requests.append(related_id)
                number_of_req_to_remove -= 1
            elif len(removed_requests) == len(list_no_relatedness):
                print "There are no related requests to remove"
                break
            else:
                list_no_relatedness.append(random_request_id)
                remove_duplicates = set(list_no_relatedness)
                list_no_relatedness = list(remove_duplicates)
                continue
    return (remove_related)

def produce_remove_timeoriented(number_of_requests, factor_lower_bound, 
                          factor_upper_bound, random_parameter, 
                           B_parameter):
    def remove_timeoriented(solution):
        #Listing the request id's of served requests
        request_ids = [id for id in solution.problem.P.keys() if id not in 
        solution.request_bank]
        #Randomly select the number of requets to remove. You cannot remove
        #more requests than the total number of served requests
        B = B_parameter
        number_of_req_to_remove =  min(B_parameter+1, random.randint(int(number_of_requests*factor_lower_bound), 
                                   int(number_of_requests*factor_upper_bound)))

        p = random_parameter
        #Listing the pickups/deliveries located at a terminal
        terminal_pickup, terminal_delivery = (
            verify_pickup_delivery_on_terminal(solution, request_ids))
        #Randomly select a request from the solution to remove
        #Before removing calculate its current start service time
        random_index = random.randint(0, len(request_ids)-1)
        id_request_to_remove = request_ids[random_index]
        delivery_to_remove = id_request_to_remove + solution.problem.n
        
        route_id_request_to_remove = solution._find_route_containing_request(id_request_to_remove)
        start_service = solution.routes[route_id_request_to_remove].calculate_earliest(0)
        for position in range(0, len(solution.routes[route_id_request_to_remove])):
            if id_request_to_remove == solution.routes[route_id_request_to_remove].route[position]:
                id_request_to_remove_position = position
                break        
        for position in range(0, len(solution.routes[route_id_request_to_remove])):
            if delivery_to_remove == solution.routes[route_id_request_to_remove].route[position]:
                delivery_to_remove_position = position
                break
        time_request_to_remove = start_service[id_request_to_remove_position]
        time_delivery_to_remove = start_service[delivery_to_remove_position]
        
        solution._remove_request(id_request_to_remove)
        removed_requests = [id_request_to_remove]
        number_of_req_to_remove -= 1
        

        #Make a list of B requests closest (with regards to distance)
        #to this removed random request (id unremoved request, relatedness)
        list_of_requests_not_removed = [id for id in request_ids if id not
                                        in removed_requests]
        relate_list = [[id, calculate_relatedness(solution, id_request_to_remove,
                        id, terminal_pickup, terminal_delivery)] for id in
                       list_of_requests_not_removed]
        relatedness_list = [sublist for sublist in relate_list if None not
                            in sublist]
        sorted_relatedness_list = sorted(relatedness_list,
                                         key=lambda relatedness_list:
                                         relatedness_list[1])
        b_requests = sorted_relatedness_list[0:B]
        b_request_ids = [column[0] for column in b_requests]
    

        list_no_relatedness_time = []

        #Remove as many requests as defined by n_requests
        while number_of_req_to_remove > 0:
            b_list_of_requests_not_removed = [id for id in b_request_ids if id
                                            not in removed_requests]
            #Calculate time relatedness of random_request to each of the
            #requests that hasn't been removed yet, and put it in a list:
            #(id unremoved request, relatedness)
            relate_list_time = [[id, calculate_time_relatedness(solution, time_request_to_remove, time_delivery_to_remove,
                                id)] for id in b_list_of_requests_not_removed]
            #Remove None-values from list
            relatedness_list_time = [sublist for sublist in relate_list_time if None not
                                     in sublist]
            #Sort by descending time relatedness
            sorted_relatedness_list_time = sorted(relatedness_list_time,
                            key=lambda relatedness_list_time: relatedness_list_time[1])
            #Choose random related request: as p increases, the more related
            #request is chosen
            random_related = int((math.pow(random.random(), p)) *
                                len(sorted_relatedness_list_time))
            #If no related values, put id in list
            #When id's in list = removed id's, break
            if len(sorted_relatedness_list_time) > 0:
                related_id = sorted_relatedness_list_time[random_related][0]
                solution._remove_request(related_id)
                removed_requests.append(related_id)
                number_of_req_to_remove -= 1
            elif len(removed_requests) == len(list_no_relatedness_time):
                print "There are no related requests to remove"
                break
            else:
                list_no_relatedness_time.append(id_request_to_remove)
                remove_duplicates = set(list_no_relatedness_time)
                list_no_relatedness_time = list(remove_duplicates)
                continue
    return remove_timeoriented

def produce_remove_neighbor_graph(number_of_requests, factor_lower_bound, 
                          factor_upper_bound, random_parameter):

    def remove_neighbor_graph(solution):
        #Listing the request id's of served requests
        request_ids = [id for id in solution.problem.P.keys() if id not in
                       solution.request_bank]
        #Randomly select the number of requests to remove. You cannot remove
        #more requests than the total number of served requests
        number_of_req_to_remove =  min(len(request_ids), 
                                       random.randint(int(number_of_requests*factor_lower_bound), 
                                        int(number_of_requests*factor_upper_bound)))
        p = random_parameter
        removed_requests = []
        
        #Remove as many requests as defined by number_of_req_to_remove
        while number_of_req_to_remove > 0:
            list_of_requests_not_removed = [id for id in request_ids if id not 
            in removed_requests]
            #For all requests in solution calculate their cost by summing the 
            #weights of edges incident to i and i+n using the neighbor graph
            #(id unremoved request, cost)
            cost_list = [[id, calculate_neighbor_costs(solution, id)] for id in
                         list_of_requests_not_removed]
            #Sort by ascending cost
            sorted_cost_list = sorted(cost_list, 
                            key=lambda cost_list: cost_list[1])
            #Choose random request: as p increases, requests with the lowest
            #cost are chosen
            random_cost = int((math.pow(random.random(), p))*
                                len(sorted_cost_list))
            cost_id = sorted_cost_list[random_cost][0]
            solution._remove_request(cost_id)
            removed_requests.append(cost_id)
            number_of_req_to_remove -= 1
    
    return remove_neighbor_graph


def verify_pickup_delivery_on_terminal(solution, request_ids):
    #Verifying whether a pickup/delivery is located at a terminal
    #Assumption start and end terminal at same location 
    start_terminals = [vehicle_id for vehicle_id in solution.problem.tau_k]
    pickups_on_terminal = []
    deliveries_on_terminal = []

    for pickup_id, vehicle_id in [(pickup_id, vehicle_id) for pickup_id in
    request_ids for vehicle_id in start_terminals]:
        delivery_id = pickup_id + solution.problem.n
        #If pickup has same coordinates as a terminal, add to list
        if (solution.problem.P[pickup_id].x_coord == 
            solution.problem.tau_k[vehicle_id].x_coord and
            solution.problem.P[pickup_id].y_coord ==
            solution.problem.tau_k[vehicle_id].y_coord and
            pickup_id not in pickups_on_terminal):
                pickups_on_terminal.append(pickup_id)
        #If delivery has same coordinates as a terminal, add to list
        if (solution.problem.D[delivery_id].x_coord ==
            solution.problem.tau_k[vehicle_id].x_coord and
            solution.problem.D[delivery_id].y_coord ==
            solution.problem.tau_k[vehicle_id].y_coord and
            delivery_id not in deliveries_on_terminal):
                deliveries_on_terminal.append(delivery_id)
        else:
            continue
    return pickups_on_terminal, deliveries_on_terminal


def calculate_relatedness(solution, pickup1, pickup2, terminal_pickup,
                          terminal_delivery):
    #Calculate how related 2 requests are
    #The lower the measure, the more related are the 2 requests
    #This function has been written in such a way that it can be used for
    #various types of routing problems
    delivery1 = pickup1 + solution.problem.n
    delivery2 = pickup2 + solution.problem.n

    distance_p1p2 = solution.problem.distancematrix[pickup1, pickup2]
    distance_p1d2 = solution.problem.distancematrix[pickup1, delivery2]
    distance_d1p2 = solution.problem.distancematrix[delivery1, pickup2]
    distance_d1d2 = solution.problem.distancematrix[delivery1, delivery2]

    count_conditions_true = 0

    if pickup1 in terminal_pickup:
        distance_p1p2 = 0
        distance_p1d2 = 0
        count_conditions_true += 1       
    if pickup2 in terminal_pickup:
        distance_p1p2 = 0
        distance_d1p2 = 0
        count_conditions_true += 1
    if delivery1 in terminal_delivery:
        distance_d1p2 = 0
        distance_d1d2 = 0
        count_conditions_true += 1
    if delivery2 in terminal_delivery:
        distance_p1d2 = 0
        distance_d1d2 = 0
        count_conditions_true += 1
   
   #For a VRP problem with 1 depot, this function will never return 'None',
   #because it doesn't make sense that pickup and delivery of a request would
   #both be located at the depot
    if (count_conditions_true >= 3) or (count_conditions_true == 2 and (
        (pickup1 in terminal_pickup and delivery1 in terminal_delivery) or
        (pickup2 in terminal_pickup and delivery2 in terminal_delivery))):
        relatedness = None
    else: 
        nonzero_terms = 4.0/(2**count_conditions_true)
        relatedness = (1/nonzero_terms)*(distance_p1p2 + distance_p1d2 + 
        distance_d1p2 + distance_d1d2)
    return relatedness


def calculate_time_relatedness(solution, pickup1_time, delivery1_time, pickup2):
    #Calculate how related 2 requests are in terms of start service time
    #The lower the measure, the more related are the 2 requests
    delivery2 = pickup2 + solution.problem.n

    route_id2 = solution._find_route_containing_request(pickup2)

    for position in range(0, len(solution.routes[route_id2])):
            if pickup2 == solution.routes[route_id2].route[position]:
                pickup2_position = position
                break
    for position in range(0, len(solution.routes[route_id2])):
            if delivery2 == solution.routes[route_id2].route[position]:
                delivery2_position = position
                break

    start_service2 = solution.routes[route_id2].calculate_earliest(0)

    time_pickup1 = pickup1_time
    time_delivery1 = delivery1_time
    time_pickup2 = start_service2[pickup2_position]
    time_delivery2 = start_service2[delivery2_position]

    time_relatedness = (abs(time_pickup1 - time_pickup2)
                        + abs(time_delivery1 - time_delivery2))
    return time_relatedness


def calculate_neighbor_costs(solution, request_id):
    #Calculate the cost of a request by summing the
    #weights of edges incident to i and i+n using the neighbor graph
    cost = 0
    pickup_id = request_id
    delivery_id = request_id + solution.problem.n
    route_id = solution._find_route_containing_request(pickup_id)
    for position in range(0, len(solution.routes[route_id])):
            if pickup_id == solution.routes[route_id].route[position]:
                pickup_position = position
                break
    for position in range(0, len(solution.routes[route_id])):
            if delivery_id == solution.routes[route_id].route[position]:
                delivery_position = position
                break
    cost += solution.neighbor_graph[(solution.routes[route_id].route[pickup_position-1],
                                     solution.routes[route_id].route[pickup_position])]
    cost += solution.neighbor_graph[(solution.routes[route_id].route[pickup_position],
                                     solution.routes[route_id].route[pickup_position+1])]
    cost += solution.neighbor_graph[(solution.routes[route_id].route[delivery_position-1],
                                     solution.routes[route_id].route[delivery_position])]
    cost += solution.neighbor_graph[(solution.routes[route_id].route[delivery_position],
                                     solution.routes[route_id].route[delivery_position+1])]
    return cost
