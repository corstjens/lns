# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:00:12 2015

"""

import random
import math
import os

#Specifcy folder where txt-files are to be saved
save_path = './Instances_Experiment_1'

#How many problem instances to generate
number_of_instances = 200

for instance_id in range(1,number_of_instances + 1):
    #Sample the number of customers to serve
    customers = random.randint(25,400)
    vehicles= customers
    #The capacity of each vehicle is fixed at 150 units
    vehicle_capacity = 150
    
    depot_id = 0
    depot_x_coord = random.randint(0,500)
    depot_y_coord = random.randint(0,500)
    depot_demand = 0
    #The depot is opened a fixed time window of 15 hours
    depot_start_tw = 0
    depot_end_tw = 900
    depot_service_time = 0
    
    #Sample the maximum CPU time the algorithm gets to solve the problem instance
    runtime = random.triangular(60,1800)
    
    #For these 3 characteristics we keep track of totals in order to
    #calculate an average measure across all customers
    total_service_time = 0
    total_time_window_width = 0
    total_demand = 0
    
    #create txt-file
    #filename = Instance + id + geographical distribution of 
    #customers (Random/Clustered/Semi-clustered)
    filename = 'Instance%d_' % (instance_id) + 'LNS_Random.txt' 
    complete_name = os.path.join(save_path, filename)
    f = open(complete_name,'w')
    f.write("%s\nruntime: %f" % (filename, runtime))
    f.write("\n\tVEHICLE\n\tNUMBER\t\tCAPACITY\n")
    f.write("\t%d\t\t\t%d\n\n" % (vehicle_number, vehicle_capacity))
    f.write("CUSTOMERS\nCUST ID\t XCOORD\t YCOORD\t DEMAND\t 
             START TIME WINDOW\t END TIME WINDOW\t SERVICE TIME\n\n")
    f.write("%d\t%d\t%d\t%d\t\t%d\t\t\t%d\t\t\t%d\n" % (depot_id, depot_x_coord, depot_y_coord, depot_demand, 
            depot_start_tw, depot_end_tw, depot_service_time))
    
    #For these characteristics we sample minimum and maximum values
    #from a uniform distribution
    min_service_time = random.uniform(10,30)
    max_service_time = random.uniform(30,50)
    min_width = random.randint(20,50)
    max_width = random.randint(50,80)
    
    #Create uniformly distributed customers
    for id in range(1, customer_number+1):
        feasible = False
        #As long as the generated problem instance is 'infeasible',
        #generate a new one
        while feasible == False:
            customer_id = id
            #Sample x and y coordinates for customer
            x_coord = random.randint(0,500)
            y_coord = random.randint(0,500)
            #Sample customer demand
            demand = random.randint(10, 50)
            
            #Sample service time at customer from triangular distribution
            service_time = int(random.triangular(min_service_time, max_service_time))
            distance = int(math.sqrt((depot_x_coord - x_coord) ** 2
                                     + (depot_y_coord - y_coord) ** 2))
            
            if (depot_start_tw + distance < depot_end_tw - distance -
                service_time):
                tw_centre = random.randint(depot_start_tw + distance, 
                                           depot_end_tw - distance - 
                                           service_time)
                                                    
                tw_width = random.randint(min_width, max_width)
                tw_start = time_window_centre - 0.5*tw_width
                tw_end = time_window_centre + 0.5*tw_width                      
                
                if ((tw_end + service_time + distance <= depot_end_tw)):
                    f.write("%d\t%d\t%d\t%d\t\t%d\t\t\t%d\t\t\t%d\n" % 
                           (customer_id, x_coord, y_coord, demand, 
                            tw_start, tw_end, service_time))
                    total_service_time += service_time
                    total_demand += demand
                    total_tw_width += tw_width
                    feasible = True
            else: feasible = False
    average_service_time = total_service_time/float(customers)
    average_tw_width = total_tw_width/float(customers)
    average_demand = total_demand/float(customers)
    f.write("average service time: %f - average time window width: %f 
            - average demand: %f\n" % (average_service_time, average_tw_width, average_demand))
    f.close()
