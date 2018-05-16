# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 15:54:42 2014

@author: lucp2487
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 04 10:02:32 2014

@author: lucp2487
"""

from lns.utils import input_output_helper
import time
from lns.algorithm import Alns_PR_Minimization, AlnsBase, Alns_Solver
import sys
from lns.solution import Solution



# Read problem
problem = input_output_helper.input_data(sys.argv[2])

# Parameters
id = int(sys.argv[1])
max_runtime = float(problem.runtime)
number_of_requests = len(problem.P)
seed = int(sys.argv[4])
random_parameter = int(sys.argv[5])

cooling_rate = float(sys.argv[6])
start_temp_control_param = float(sys.argv[7])

noise_parameter = float(sys.argv[8])
factor_lower_bound = 0.1
factor_upper_bound = 0.5
remove_random = (sys.argv[9])
remove_worst = (sys.argv[10])
remove_related = (sys.argv[11])
remove_time_oriented = False
remove_neighbor_graph = False
insert_greedy_parallel = (sys.argv[12])
insert_regret_sequential = False
insert_regret_2 = (sys.argv[13])
insert_regret_3 = False
insert_regret_4 = False
removal_condition = [remove_random, remove_worst, remove_related, remove_time_oriented, remove_neighbor_graph]
insertion_condition = [insert_greedy_parallel, insert_regret_sequential, insert_regret_2, insert_regret_3, insert_regret_4]


#Initialize alns_vm
alns_vm = Alns_PR_Minimization(seed, number_of_requests, factor_lower_bound, factor_upper_bound, random_parameter)
alns_vm.max_runtime = max_runtime
#alns_vm.random_parameter = random_parameter
alns_vm.destroy_heuristics = [heuristic for heuristic in 
                              alns_vm.list_of_all_destroy_heuristics if 
removal_condition[alns_vm.list_of_all_destroy_heuristics.index(heuristic)] == 'True']
alns_vm.repair_heuristics = [heuristic for heuristic in 
                             alns_vm.list_of_all_repair_heuristics if
insertion_condition[alns_vm.list_of_all_repair_heuristics.index(heuristic)] == 'True']

alns_vm.cooling_rate = cooling_rate
alns_vm.start_temp_control_param = start_temp_control_param
alns_vm.noise_parameter = noise_parameter


#Initialize alns_main
alns_main = AlnsBase(seed, number_of_requests, factor_lower_bound, factor_upper_bound, random_parameter)
alns_main.max_runtime = max_runtime
#alns_main.random_parameter = random_parameter
alns_main.destroy_heuristics = [heuristic for heuristic in 
                                alns_main.list_of_all_destroy_heuristics if 
removal_condition[alns_main.list_of_all_destroy_heuristics.index(heuristic)] == 'True']
alns_main.repair_heuristics = [heuristic for heuristic in 
                               alns_main.list_of_all_repair_heuristics if
insertion_condition[alns_main.list_of_all_repair_heuristics.index(heuristic)] == 'True']

alns_main.cooling_rate = cooling_rate
alns_main.start_temp_control_param = start_temp_control_param
alns_main.noise_parameter = noise_parameter

alg = Alns_Solver(alns_vm, alns_main, id)
start_time = time.clock()
solution = alg.solve(problem, start_time)
end_time = time.clock()
runtime = end_time - start_time

#Write output to txt-file (save path to be added as command line argument: ./output_%d.txt)
filename_output = sys.argv[14]%(id)
f = open(filename_output,'w')
f.write("id,total_cost, n_vehicles, runtime\n")
f.write("%d, %.2f, %d, %.2f\n"%(id, solution.calculate_solution_cost(), 
                            solution._number_of_used_vehicles(), runtime))
f.close()

#Write routes final solution tp txt-file (save path to be added as command line argument: ./solution_%d.txt)
filename_solution = sys.argv[15]%(id)
s = open(filename_solution, 'w')
s.write("experiment, problem_instance, algorithm_instance, run, vehicle, solution\n")
algorithm_instance = int(sys.argv[3])
problem_name = str(sys.argv[2])
split_problem_name = problem_name.split('_')
problem_instance = split_problem_name[0][18:]
for i in range(len(solution.routes)):
    vehicle_id = i
    string_route = " ".join(str(x - number_of_requests + 1) for x in solution.routes[i].route[1:-1] if x >= number_of_requests)
    s.write("%s, %d, %d, %.2f, [%s]\n" % (problem_instance, algorithm_instance, vehicle_id, solution.routes[i].cost, string_route))
s.close()

#print solution
print solution.calculate_solution_cost()
print "Runtime : ", end_time - start_time, "seconds"

