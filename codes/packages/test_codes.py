import os
import pickle
import time
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing
from packages import objects, actv_analysis

def prepare_anova(net=3, relu=2, epoch=30, num_test_units=6000):
    print(f'Network{net} Relu{relu} Epoch{epoch}')
    pickle_filename = f'network{net}_Relu{relu}_epoch{epoch}.pkl'

    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            units = pickle.load(f)
    else:
        print(f"Pickle file {pickle_filename} not found. Creating a new pkl..")
        actv_net = actv_analysis.get_actv_net(net=net, relu=relu, epoch=epoch)
        units = [objects.Unit(i) for i in range(actv_net.shape[0])]
        _ = [unit.add_network_info(epoch,net,relu) for unit in units]
        with open(pickle_filename, 'wb') as f:
            pickle.dump(units, f)

    min_sz_idx=3; max_sz_idx=9; numbers=range(2,21,2); inst=500
    sizes = np.arange(4,14)[min_sz_idx: max_sz_idx+1]

    actv_net = actv_analysis.get_actv_net(net=net, relu=relu, epoch=epoch)
    take = np.arange(0,100).reshape(10,10)[:,min_sz_idx:max_sz_idx+1].reshape(len(numbers)*(max_sz_idx-min_sz_idx+1))
    actv_szAtoB = actv_net[:,take,:]
    actv_2D = actv_szAtoB.reshape(actv_szAtoB.shape[0], actv_szAtoB.shape[1]*actv_szAtoB.shape[2])

    return units, actv_2D, numbers, sizes, inst

def print_and_add_anova2(u, units, actv_2D, numbers, sizes, inst, parallel=1):
    try:
        print(f'Processing unit: {u}, actv_2D shape: {actv_2D.shape}')
        return units[u].add_anova2(actv_2D, numbers, sizes, inst, parallel)
    except IndexError:
        print(f'IndexError when processing unit {u}')
        raise





def run_test_anova(num_cores, num_test_units, num_units_per_block, units, actv_2D, numbers, sizes, inst):
    start_time = time.time()
    total_completed_units = 0
    while total_completed_units < num_test_units:
        start_idx = total_completed_units
        end_idx = min(start_idx + num_units_per_block, num_test_units)
        active_bk = np.arange(start_idx, end_idx)
        print(f'Start idx: {start_idx}, End idx: {end_idx}, active_bk: {active_bk}')

        anova2_results = Parallel(n_jobs=num_cores)(
            delayed(print_and_add_anova2)(u, units, actv_2D, numbers, sizes, inst, parallel=1) for u in active_bk)

        total_completed_units += len(active_bk)
        print(f'{total_completed_units} units have been completed!')

    elapsed_time = time.time() - start_time
    print("--- %s seconds ---" % elapsed_time)
    return elapsed_time




def test_anova_speed():
    min_cores = 5
    max_cores = multiprocessing.cpu_count()
    num_test_units = 800
    block_ranges = np.array([100,200,400])

    units, actv_2D, numbers, sizes, inst = prepare_anova()

    actv_sums = actv_2D.sum(axis=1)
    active_units = np.nonzero(actv_sums)[0]

    if len(active_units) < num_test_units:
        print("Not enough units with non-zero responses")
        return None

    units = [units[i] for i in active_units[:num_test_units]]
    print(f'there are {len(units)} units')
    actv_2D = actv_2D[active_units[:num_test_units],:]

    df = pd.DataFrame(columns=block_ranges, index=range(min_cores, max_cores + 1))

    for num_cores in range(min_cores, max_cores + 1):
        print(f"testing {num_cores} cores..")
        for idx, num_units_per_block in enumerate(block_ranges):
            print(f'Processing block{idx}')
            start_time = time.time()
            time_taken = run_test_anova(num_cores, num_test_units, num_units_per_block, units, actv_2D, numbers, sizes, inst)
            df.loc[num_cores, num_units_per_block] = time_taken

    print(df)
    df.to_csv('anova2_execution_times.csv')
    return df



