import numpy as np
import os
import h5py
import pickle
import time
import math

def get_actv_net(net,relu,epoch):
    """
    INPUT:
      1. net: network ID
      2. epoch: training epoch
      3. relu: relu layer
    OUTPUT: Activity matrix corresponding to the input parameters
    """
    start_time = time.time()
    dir_path = os.path.dirname(os.path.realpath('../'))
    #print(dir_path)
    Unt_f500 = h5py.File(dir_path+'/data/raw_response/actv_f500_network'+str(net)+'_relu'+str(relu)+'_epoch'+str(epoch)+'.mat', 'r')
    actv_ = Unt_f500['actv'][:]
    actv = np.transpose(actv_, (2,1,0))

    print("--- %s seconds ---" % (time.time() - start_time))
    return actv


def get_PNs(actv, numbers, sizes, min_sz_idx, max_sz_idx):
    ###
    ## INPUT:
    #   1. actv: activity matrix
    #   2. numbers
    #   3. sizes
    #   4. min_sz_idx = index corresponding to the minimum size
    #   5. max_sz_idx = index corresponding to the maximum size
    ## OUTPUT:
    #   preferred numerosity by size, overall preferred numerosity
    ###
    avg_actv = np.nanmean(actv, axis=2)
    avg_actv_nxs_ = avg_actv.reshape(actv.shape[0], len(numbers), len(sizes))
    avg_actv_nxs = avg_actv_nxs_[:,:,min_sz_idx:max_sz_idx+1]
    PN_by_size = numbers[np.argmax(avg_actv_nxs, axis=1)]
    oPN = numbers[np.argmax(np.mean(avg_actv_nxs,axis=2),axis=1)]
    return PN_by_size, oPN


def get_PSs(actv, numbers, sizes, min_sz_idx, max_sz_idx):
    ###
    ## INPUT:
    #   1. actv: activity matrix
    #   2. numbers
    #   3. sizes
    #   4. min_sz_idx = index corresponding to the minimum size
    #   5. max_sz_idx = index corresponding to the maximum size
    ## OUTPUT:
    #   preferred numerosity by number, overall preferred size
    ###
    avg_actv = np.nanmean(actv, axis=2)
    avg_actv_nxs_ = avg_actv.reshape(actv.shape[0], len(numbers), len(sizes))
    avg_actv_nxs = avg_actv_nxs_[:,:,min_sz_idx:max_sz_idx+1]
    PS_by_size = sizes[min_sz_idx:max_sz_idx+1][np.argmax(avg_actv_nxs, axis=2)]
    oPS = sizes[min_sz_idx:max_sz_idx+1][np.argmax(np.mean(avg_actv_nxs,axis=1),axis=1)]
    return PS_by_size, oPS



def get_blocks(indices, block_size=3000):
    """
    Divide the given indices into blocks of a certain size.

    Parameters:
    - indices: The indices to be divided into blocks.
    - block_size: The number of indices in each block.

    Returns:
    - blocks: A list of lists, each sub-list contains the indices for a block.
    """
    if len(indices) == 0:
        return []

    blocks = [indices[i * block_size : (i + 1) * block_size] for i in range((len(indices) + block_size - 1) // block_size)]

    return blocks




def get_selective_units(net, relu, epoch, actv_thres=0.05, p_val=0.01, min_sz_idx=3, max_sz_idx=9, numbers=range(2,21,2), inst=500, PN=[], PS=[]):
    """
    The function loads actv_net and unit objects from the corresponding pickle files, 
    and identifies units that have at least actv_thres response rate within the specified 
    size range, and show selectivity to both number and size. 

    Parameters:
    - net, relu, epoch: identifiers for the network, layer (relu) and training epoch.
    - actv_thres: Minimum response rate to be considered as an active unit.
    - p_val: P-value threshold for selectivity. Default is 0.1
    - min_sz_idx, max_sz_idx: lower and upper bound of index of sizes that will be included in the analysis.
    - numbers: numbers used for the analysis. Default is range(2,21,2)
    - inst: number of image example for each combination of a number and a size. Default is 500.

    Returns: active selective units and their response vectors that meet the specified criteria.
    """

    # Load actv_net
    actv_net = get_actv_net(net=net, relu=relu, epoch=epoch)

    # Subset actv_net based on size range
    if min_sz_idx != max_sz_idx:
        take = np.arange(0,100).reshape(10,10)[:,min_sz_idx:max_sz_idx+1].reshape(len(numbers)*(max_sz_idx-min_sz_idx+1))
    else:
        take = np.arange(0,100).reshape(10,10)[:,min_sz_idx:max_sz_idx+1].flatten()

    actv_szAtoB = actv_net[:,take,:]

    # Calculate response rates
    response_rate = np.mean(actv_szAtoB > 0, axis=(1, 2))

    # Load units
    with open(f'network{net}_Relu{relu}_epoch{epoch}.pkl', 'rb') as f:
        units = pickle.load(f)

    # Check the selectivity to both number and size by checking the ANOVA2 results
    selective_units = [unit for unit in units if (unit.anova2_numbers is not None and unit.anova2_numbers < p_val) and 
                                           (unit.anova2_sizes is not None and unit.anova2_sizes < p_val)]

    # Identify units that meet both criteria: response rate and selectivity
    active_selective_units = [unit for unit in selective_units if response_rate[unit.id] >= actv_thres]

    # If PN or/and PS are not empty, narrow down units based on the inputs
    if len(PN)!=0:
        active_selective_units = [unit for unit in active_selective_units if unit.PN.value_counts().idxmax() in PN]

    if len(PS)!=0:
        active_selective_units = [unit for unit in active_selective_units if unit.PS.value_counts().idxmax() in PS]

    # Extract response vectors for active, selective units
    unit_ids = [unit.id for unit in active_selective_units]
    response_vectors = actv_szAtoB[unit_ids, :, :]
    avg_response_vectors = np.mean(response_vectors, axis=2)

    return active_selective_units, avg_response_vectors



def find_consistent_PN(net, relu, epochs=range(0, 91, 10), consistency='overall'):
    """
    This function finds units in an AlexNet whose Preferred Numbers (PNs) are consistent across different sizes
    and across multiple training epochs.

    Parameters:
    - net: Integer specifying the AlexNet to consider.
    - relu: Integer specifying the layer of the AlexNet to consider.
    - epochs: A range object specifying the training epochs to consider.
    - consistency: String indicating the type of consistency to consider. Can be 'absolute' or 'overall'.

    Returns:
    - consistent_across_epochs: A set of unit ids that have consistent PNs across all specified training epochs.
    - consistent_PNs_across_epochs: A dictionary where keys are the ids of units with consistent PNs across all 
                                    specified training epochs, and the values are the corresponding PNs.
    """
    # Create a list to store the unit ids with consistent PNs for each epoch
    consistent_across_epochs = []

    # Create a dictionary to store the PNs for each unit with consistent PNs across epochs
    consistent_PNs_across_epochs = {}

    for epoch in epochs:
        print(f'Loading network{net}_Relu{relu}_epoch{epoch}.pkl')
        with open(f'network{net}_Relu{relu}_epoch{epoch}.pkl', 'rb') as f:
            units = pickle.load(f)
            
            if consistency == 'absolute':
                consistent_units = {unit.id: unit.PN[0] for unit in units if unit.PN.nunique().sum() == 1}
            elif consistency == 'overall':
                consistent_units = {}
                for unit in units:
                    vc = unit.PN.value_counts()
                    if vc.iloc[0] > 0.5 * unit.PN.size:
                        consistent_units[unit.id] = vc.index[0]

        # Collect units with consistent PNs and their PNs
        if epoch == epochs[0]:
            consistent_across_epochs = set(consistent_units.keys())
            consistent_PNs_across_epochs = consistent_units
        else:
            consistent_across_epochs &= set(consistent_units.keys())
            for unit_id in consistent_across_epochs:
                consistent_PNs_across_epochs[unit_id] = consistent_units[unit_id]

    return consistent_across_epochs, consistent_PNs_across_epochs



def update_PNS(units, actv, numbers=np.arange(2,21,2), sizes = np.arange(4,14), min_sz_idx=3, max_sz_idx=9):
    PNs=get_PNs(actv, numbers, sizes, min_sz_idx, max_sz_idx)[1]
    PSs=get_PSs(actv, numbers, sizes, min_sz_idx, max_sz_idx)[1]
    for i in range(len(units)):
        units[i].PN = PNs[i]
        units[i].PS = PSs[i]
    return units


def update_response_rate(units, actv, subset=False):
    non_zero_counts = np.count_nonzero(actv, axis=(1, 2))

    # Get the total number of elements in the second and third dimensions
    total_elements = actv.shape[1] * actv.shape[2]

    # Compute the response rate for each sample
    response_rates = (non_zero_counts / total_elements) * 100
    
    for i in range(len(units)):
        if subset:
            units[i].response_rate_subset = response_rates[i]
        else:
            units[i].response_rate = response_rates[i]
    
    return units


def update_monotonicity(units, actv, numbers, min_sz_idx=3, max_sz_idx=9, subset=False):
    avg_actv = np.mean(actv, axis=2)
    avg_actv_nxs = avg_actv.reshape(actv.shape[0], 10,10)

    if subset:
        avg_actv_nxs_subset = avg_actv_nxs[:,:,min_sz_idx:max_sz_idx+1]
    else:
        avg_actv_nxs_subset = avg_actv_nxs

    # Average across sizes
    avg_across_sizes = np.mean(avg_actv_nxs_subset, axis=2)

    # Average across numbers
    avg_across_numbers = np.mean(avg_actv_nxs_subset, axis=1)

    for i, unit in enumerate(units):
        # Check for monotonicity across numbers
        if np.all(np.diff(avg_across_sizes[i]) >= 0):  
            unit.num_monotonicity = 1
        elif np.all(np.diff(avg_across_sizes[i]) <= 0): 
            unit.num_monotonicity = -1
        else:
            unit.num_monotonicity = 0

        # Check for monotonicity across sizes
        if np.all(np.diff(avg_across_numbers[i]) >= 0):  
            unit.size_monotonicity = 1
        elif np.all(np.diff(avg_across_numbers[i]) <= 0): 
            unit.size_monotonicity = -1
        else:
            unit.size_monotonicity = 0

        

        

