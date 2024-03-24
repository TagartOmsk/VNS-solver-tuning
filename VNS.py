import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import itertools
import os
import sys
import datetime as dt
import random
import time

min_solved_for_capping = 3
seed = 1000
random.seed(seed)
max_time = dt.timedelta(hours=1)
# instances = ['drilling_30_1', 'drilling_30_2', 'drilling_30_3', 'drilling_30_4', 'drilling_30_5', 'drilling_30_6', 'drilling_30_7', 'drilling_30_8', 'drilling_30_9', 'drilling_30_10']
instances = [] 
inst_dir_path = './instances'
discovered_points_file_path = 'discovered_points.csv'

for file in os.listdir(inst_dir_path):
    instances.append(os.path.join(inst_dir_path, file))

sets = './4_jobs_sets.txt'
params_options = {
    "threads": [1, 4, 8, 12], 
    "presolve": [-1, 0, 1, 2],
    "gomorypasses": [0, 1, 2, 5],
    "method": [-1, 0, 1, 2, 3, 4, 5],
    "minrelnodes": [0],  # relaxation nodes, discarded them for some reason
    "mipfocus": [0, 1, 2, 3]
}

def prepare_gurobi(**kwargs):
    params = [f"{k} {kwargs[k]}\n" for k in kwargs]
    with open('./gurobi.op2', 'w') as f:
        f.writelines(params)

def generate_random_point(params_options):
    res = {}
    for param in params_options:
        res[param] = np.random.choice(params_options[param])

    return res

def generate_neighborhood(point, params_options):
    neighborhood = [point]

    for param in params_options:
        for val in params_options[param]:
            if val == point[param]:
                continue
            
            new = point.copy()
            new[param] = val
            neighborhood.append(new)
    
    # convert list to dict
    res = {i:k for i,k in zip(range(len(neighborhood)), neighborhood)}
    
    return res

def generate_neighborhood_two(point, params_options):
    neighborhood = [point]

    for p1, p2 in itertools.product(params_options, params_options):
        if p1 == p2:
            continue
        
        for v1, v2 in itertools.product(params_options[p1], params_options[p2]):
            if v1 == point[p1] or v2 == point[p2]:
                continue
            
            new = point.copy()
            new[p1] = v1
            new[p2] = v2
            neighborhood.append(new)
    
    res = {i:k for i,k in zip(range(len(neighborhood)), neighborhood)}
    
    return res

def calculate_f(current_point, discovered_points):
    current_point_mask = np.ones(discovered_points.shape[0], dtype=bool)
    for param in current_point:
        current_point_mask = np.logical_and(current_point_mask, discovered_points[param] == current_point[param])

    if discovered_points[current_point_mask].empty or discovered_points[current_point_mask]['time_spent'].isna().sum() > 0:
        return sys.float_info.max
    
    return discovered_points[current_point_mask]['run_time'].sum()

def compute_point(current_point, discovered_points, instances, best_f):
    # check if neighbourhood already computed
    current_point_mask = np.ones(discovered_points.shape[0], dtype=bool)
    for param in current_point:
        current_point_mask = np.logical_and(current_point_mask, discovered_points[param] == current_point[param])

    res_dict = {}

    # если у нас нет записей об этой точке, то вычисляем 
    if discovered_points[current_point_mask].empty:
        # run
        prepare_gurobi(**current_point)
        time_history = []
        for instance in instances:
            cmd = f"faketime -f '@2021-01-01 01:01:00' gams sched.gms --instance {instance} --sets {sets} > gurobi_res.txt"

            print(f'Running {cmd}')

            run_start_time = time.time_ns()
            os.system(cmd)
            run_time = time.time_ns() - run_start_time

            # parse shit
            with open('gurobi_res.txt', 'r') as res_f:
                res_lines = res_f.readlines()

            res_dict = current_point.copy()
            res_dict['instance'] = instance
            res_dict['final_solve'] = sys.float_info.max
            res_dict['best_possible'] = sys.float_info.max
            res_dict['run_time'] = run_time

            for line in res_lines:

                if "MIP   Solution" in line:
                    res_dict["final_solve"] = float(line.split(': ')[1].split('(')[0].replace(' ', '').replace('\n', '').replace('\t', ''))
                if "Best possible" in line:
                    res_dict["best_possible"] = float(line.split(': ')[1].replace(' ', '').replace('\n', '').replace('\t', ''))
                if "Executing after solve: elapsed" in line:
                    res_dict["time_spent"] = line.split('elapsed')[1].replace(' ', '').replace('\n', '').replace('\t', '')

            res_dict["abs_gap"] = abs(res_dict['final_solve'] - res_dict['best_possible'])
            res_dict["rel_gap"] = res_dict['abs_gap'] / res_dict['final_solve']

            print(f"finished calculation with result {res_dict}")

            discovered_points = discovered_points.append(res_dict, ignore_index=True)
            del res_lines

            time_history.append(res_dict['run_time'])
            # accum_gap = (np.array(time_history) / len(instances)).sum()

            # capping
            if len(time_history) > min_solved_for_capping and np.array(time_history).sum() > best_f:
                break
    
        discovered_points.to_csv(discovered_points_file_path)

    return res_dict, discovered_points

print('init start')

nb_structure_functions = [generate_neighborhood, generate_neighborhood_two]
local_search_nb_function = nb_structure_functions[0]

discovered_points = pd.DataFrame(columns=['instance', 'time_spent', 'final_solve', 'best_possible', 'abs_gap', 'rel_gap', 'run_time'] + list(params_options.keys()))

best_point = {param:None for param in params_options}
best_f = sys.float_info.max

stop = False
start_time = dt.datetime.now()
print(f'start at {start_time}') 

current_point = {}
for param in params_options:
        current_point[param] = params_options[param][0]        

_, discovered_points = compute_point(current_point, discovered_points, instances, best_f)
best_point = current_point
best_f = calculate_f(current_point, discovered_points)

while(not stop):
    current_structure_id = 0

    while(current_structure_id < len(nb_structure_functions)):
        current_nb = nb_structure_functions[current_structure_id](current_point, params_options)

        # shake
        search_center_point_idx = np.random.choice(np.array(list(current_nb)))
        search_center_point = current_nb[search_center_point_idx]
        current_nb.pop(search_center_point_idx, None)
        
        local_nb = local_search_nb_function(search_center_point, params_options)
        local_cur_point_idx = 0  # all the functions write center first
        
        local_cur_f = sys.float_info.max
        local_cur_point = search_center_point

        _, discovered_points = compute_point(local_cur_point, discovered_points, instances, best_f)
        local_best_f = calculate_f(local_cur_point, discovered_points)
        local_best_point = local_cur_point


        # local search
        while(local_nb):
            
            local_cur_point_idx = np.random.choice(np.array(list(local_nb)))
            local_cur_point = local_nb[local_cur_point_idx]
            _, discovered_points = compute_point(local_cur_point, discovered_points, instances, best_f)
            local_cur_f = calculate_f(local_cur_point, discovered_points)

            if local_best_f > local_cur_f:
                local_best_f = local_cur_f
                local_best_point = local_cur_point

            local_nb.pop(local_cur_point_idx, None)

            

        # neighbourhood change
        if best_f > local_best_f:  # use best_f for each iteration of local search, keep global best_f as well
            best_f = local_best_f
            best_point = local_best_point
            current_structure_id = 0
            print(f'move to better point {best_point} with f = {best_f}')
        else:
            current_structure_id += 1

    if dt.datetime.now() - start_time >= max_time:
        stop = True




discovered_points.to_csv(discovered_points_file_path)
print(f"finished in point {best_point} with f = {best_f}")
with open('result_point.txt', 'w') as f:
    f.write(f"finished in point {best_point} with f = {best_f}")
