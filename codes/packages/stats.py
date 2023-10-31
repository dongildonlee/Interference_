import glob
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.formula.api import ols
import sys
import time
sys.path.append('../')
from packages import objects, actv_analysis, load_csv
from sklearn.manifold import TSNE


def anova2_unit(actv_2D, unit_obj, numbers, sizes, instances, parallel=0):
    """
    Input:
      1. act_2D: 2D activity matrix from DNN
      2. unit_obj: unit object (see objects folder for details)
      3. numbers
      4. sizes
      5. instances: number of instances per each (number,size) combnination
    Output: None. The function directly add 2-way ANOVA results to the object as its feature
    """
    unit = unit_obj.id
    #print(f'Current unit: {unit}, Shape of actv_2D: {actv_2D.shape}')

    # df = pd.DataFrame({'number': np.repeat(np.repeat(numbers,len(sizes)),instances),'size': np.tile(np.repeat(sizes,instances),len(numbers)),'activity':actv_2D[unit,:]})
    # model = ols('activity ~ C(number) + C(size) + C(number):C(size)', data=df).fit()
    # stat = sm.stats.anova_lm(model, typ=2)
    # #print(f"Stat for unit {unit_obj.id}: {stat}")
    # if parallel:
    #     return stat.iloc[:, 3].to_frame()
    # else:
    #     unit_obj.anova2 = stat.iloc[:, 3].to_frame()
    df = pd.DataFrame({'number': np.repeat(np.repeat(numbers,len(sizes)),instances),'size': np.tile(np.repeat(sizes,instances),len(numbers)),'activity':actv_2D[unit,:]})
    model = ols('activity ~ C(number) + C(size) + C(number):C(size)', data=df).fit()
    stat = sm.stats.anova_lm(model, typ=2)
    return stat.iloc[:, 3].to_frame()



def anova2_single(actv_2D, unit, numbers, sizes, instances):
    """
    Input:
      1. act_2D: 2D activity matrix from DNN
      2. unit: unit whose activities will be analyzied
      3. numbers
      4. sizes
      5. instances: number of instances per each (number,size) combnination
    Output: anova2 result for the unit
    """
    df = pd.DataFrame({'number': np.repeat(np.repeat(numbers,len(sizes)),instances),'size': np.tile(np.repeat(sizes,instances),len(numbers)),'activity':actv_2D[unit,:]})
    model = ols('activity ~ C(number) + C(size) + C(number):C(size)', data=df).fit()
    stat = sm.stats.anova_lm(model, typ=2)
    return stat


def get_selectivity(df_anova2):
    """
    Input:
      1. df_anova2: dataframe with anova2 (2-way ANOVA) results
    Output: a dataframe with selectivity for number, size and both
    """
    df_selectivity = pd.DataFrame(index=df_anova2.index.to_numpy(), columns=['selectivity'])
    unit_noresponse = df_anova2.index[np.sum(df_anova2.isnull().iloc[:,0:3],axis=1)==3].to_numpy()
    df_anova2 = df_anova2.drop(labels=unit_noresponse)
    number_selective = df_anova2.index[df_anova2.loc[:, 'number'] < 0.01].to_numpy()
    size_selective = df_anova2.index[df_anova2.loc[:, 'size'] < 0.01].to_numpy()
    interaction = df_anova2.index[df_anova2.loc[:, 'inter'] < 0.01].to_numpy()
    # Populate the dataframe with selectivity:
    df_selectivity.loc[np.setdiff1d(np.setdiff1d(number_selective, size_selective), interaction), 'selectivity'] = 'number'
    df_selectivity.loc[np.setdiff1d(np.setdiff1d(size_selective, number_selective), interaction), 'selectivity'] = 'size'
    NS_units = np.intersect1d(number_selective, size_selective)
    df_selectivity.loc[np.setdiff1d(NS_units, interaction), 'selectivity'] = 'NS NI'
    df_selectivity.loc[np.intersect1d(NS_units, interaction), 'selectivity'] = 'NS I'
    return df_selectivity



# def gaussian_curve_fit(numbers, sizes, uoi, actv_net):
#     """
#     INPUT:
#       1. numbers: an array of numbers
#       2. sizes: np array of sizes
#       3. uoi: np array of neuron indices that are of our interest
#       4. actv_net: raw activity data (e.g. 43264 x 100 x 100 shape)
#     OUTPUT: Fitted parameters
#     """

#     # average activity data across instances (e.g.n=100) and reshape the structure to (# of neurons)x(# of numbers)x(# of sizes)
#     x = np.log2(numbers)
#     avg_actv_10x10 = np.mean(actv_net,axis=2).reshape(actv_net.shape[0],len(numbers),len(sizes))
#     PNidx4each_size = np.argmax(avg_actv_10x10, axis=1)
#     PN4each_size = numbers[PNidx4each_size]
#     df_pn = pd.DataFrame(index=np.arange(43264), columns = sizes, data = PN4each_size)

#     popts_sz = []

#     for s in np.arange(len(sizes)):
#         print(sizes[s])

#         popts2 = pd.DataFrame(index = np.arange(43264), columns = ['a','x0','sigma','pcov','r2'])
#         avg_actv = avg_actv_10x10[:,:,s]
#         avg_actv_norm = normed_data(avg_actv)

#         for i in uoi:
#             if np.mod(i,1000)==0:
#                 print("size:", s, " unit:",i)
#             try:
#                 y = avg_actv_norm[i,:]
#                 # weighted arithmetic mean (corrected - check the section below),
#                 #mean = sum(x*y) / sum(y)\n",
#                 mean = np.log2(df_pn.loc[i,sizes[s]])
#                 #sigma = np.sqrt(sum(y*(x-mean)**2) / sum(y))\n",
#                 sigma=1
#                 popt2,pcov2 = sp.optimize.curve_fit(gaus, x, y, p0=[1,mean,sigma])
#                 y_pred = gaus(x,*popt2)
#                 r2 = r2_score(y,y_pred)
#                 #index.append(i)
#                 #ls_popt.append(popt)\n",
#                 popts2.iloc[i,0:3] = popt2
#                 popts2.loc[i,'pcov'] = pcov2
#                 popts2.loc[i,'r2'] = r2
#             except:
#                 continue
#         popts_sz.append(popts2)
#     return popts_sz




def pkl_anova2(net, relu, epoch, min_sz_idx=3, max_sz_idx=9, numbers=range(4,21,2), inst=500):
    """
    Function: This function performs an ANOVA2 analysis on units in a neural network that have nonzero activations.
    
    Parameters:
    - net: Network ID or information.
    - relu: Relu layer ID or information.
    - epoch: Epoch ID or information.
    - min_sz_idx (default = 3): The minimum size index for the analysis.
    - max_sz_idx (default = 9): The maximum size index for the analysis.
    - numbers (default = range(2,21,2)): The numbers to use for the analysis.
    - inst (default = 500): The number of instances for the analysis.

    Returns:
    The function does not return anything. It saves the updated units into a pickle file.
    """
    print(f'Network{net} Relu{relu} Epoch{epoch}')
    pickle_filename = f'network{net}_Relu{relu}_epoch{epoch}_4to20.pkl'

    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            units = pickle.load(f)

        incomplete_units = np.array([i for i in range(len(units)) if not units[i].no_response_subset and (units[i].anova2 is None or units[i].anova2.isna().all().all())])
        print(f'{len(incomplete_units)} units for a task completion.')

        if len(incomplete_units) / len(units) < 0.9:
            print(f'Less than 90% of units are incomplete. {len(incomplete_units) / len(units)}% units incomplete!')
            print('Skipping ANOVA2...')
            return

        print("More than 90% of units are incomplete. Proceeding with ANOVA2...")
        actv_net = actv_analysis.get_actv_net(net=net, relu=relu, epoch=epoch)

    else:
        print(f"Pickle file {pickle_filename} not found. Creating a new pkl..")
        actv_net = actv_analysis.get_actv_net(net=net, relu=relu, epoch=epoch)
        units = [objects.Unit(i) for i in range(actv_net.shape[0])]
        _ = [unit.add_network_info(epoch,net,relu) for unit in units]

    sizes = np.arange(4,14)[min_sz_idx: max_sz_idx+1]
    num_units = actv_net.shape[0]

    take = np.arange(0,100).reshape(10,10)[:,min_sz_idx:max_sz_idx+1].reshape(len(numbers)*(max_sz_idx-min_sz_idx+1))
    actv_szAtoB = actv_net[:,take,:]
    actv_2D = actv_szAtoB.reshape(actv_szAtoB.shape[0], actv_szAtoB.shape[1]*actv_szAtoB.shape[2])

    # Before starting the loop, calculate sums of activations for each unit
    actv_sums = actv_2D.sum(axis=1)

    # Get the indices of units with non-zero responses
    active_units = np.nonzero(actv_sums)[0]

    [setattr(units[i], 'no_response_subset', True) for i in np.setdiff1d(np.arange(num_units), active_units)]
    print("unit objects' no_reponse_subset has been updated.")

    # Calculate and print the percentage of all-zero units
    zero_units_percentage = (actv_sums == 0).mean() * 100
    print(f'{zero_units_percentage}% of units have zero responses.')

    # Divide active units into blocks for parallel processing
    blocks = actv_analysis.get_blocks(active_units)

    for bk_idx, block in enumerate(blocks):
        print(f"Processing net {net}, relu {relu}, epoch {epoch}, block {bk_idx + 1}/{len(blocks)}")
        try:
            start_time = time.time()
            # Only active units from the current block will be processed
            anova2_results = Parallel(n_jobs=-1)(delayed(units[u].add_anova2)(actv_2D, numbers, sizes, inst, parallel=1) for u in block)

            for i, u in enumerate(block):
                units[u].anova2 = anova2_results[i]
            print("--- %s seconds ---" % (time.time() - start_time))

            # Save to pickle file after each block is processed
            with open(pickle_filename, 'wb') as f:
                pickle.dump(units, f)

            backup_filename = f'backup_network{net}_Relu{relu}_epoch{epoch}_{bk_idx}.pkl'
            with open(backup_filename, 'wb') as f:
                pickle.dump(units, f)
            
            # Get list of all backup files
            backups = sorted(glob.glob(f'backup_network{net}_Relu{relu}_epoch{epoch}_*.pkl'))
            
            # Delete all but the last two
            for backup in backups[:-2]:
                os.remove(backup)

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

        # Save to pickle file after each block is processed, but outside of the try block
        with open(pickle_filename, 'wb') as f:
            pickle.dump(units, f)


def pkl_anova2_find_backup(net, relu, epoch):
    import re

    pattern = f"pkl_backup/backup_network{net}_Relu{relu}_epoch{epoch}_"
    # Get a list of all files in the current directory that match the pattern
    matching_files = [(os.path.getmtime(f), f) for f in os.listdir('.') if f.startswith(pattern) and f.endswith('.pkl')]
    
    # If no matching files, return None
    if not matching_files:
        print("No matching backup files found.")
        return None, 0

    # Sort the matching files based on the modification time in descending order
    matching_files.sort(reverse=True)

    # Try to load the most recent file first, if that fails move on to the next one
    for _, f in matching_files:
        try:
            with open(f, 'rb') as file:
                data = pickle.load(file)
            print(f"Successfully loaded file {f}")  # Add this line to print the file name when it is successfully loaded
            
            # Use regex to find the number in the filename
            number = int(re.findall(r'\d+', f)[0])

            # Add debugging print statements here
            print(f"The type of data loaded from {f} is: {type(data)}")
            print(f"The content of data loaded from {f} is: {data}")
            
            return data, number
        except Exception as e:
            print(f"Failed to load file {f}. Error: {e}")

    print("None of the backup files could be loaded.")
    return None, 0





def get_completed_units(filename):
    """
    Function: This function loads units from a given pickle file and returns those that have 
    completed the ANOVA2 analysis.
    
    Parameters:
    - filename: The name of the pickle file to load the units from.
    
    Returns:
    A list of units that have completed the ANOVA2 analysis.
    """
    with open(filename, 'rb') as f:
        units = pickle.load(f)
    completed_units = [unit for unit in units if unit.anova2 is not None and not unit.anova2.isna().all().all()]
    return completed_units



def pkl_no_resp(net, relu, epoch):
    """
    Check each unit object and see if its response for all images were 0.
    Must run pkl_anova2 before this since objects are created in that function.

    Parameters:
    - net, relu, epoch: explained above.

    Returns: pickled objects
    """
    print(f'network{net}_Relu{relu}_epoch{epoch}')
    actv = actv_analysis.get_actv_net(net=net,relu=relu,epoch=epoch)
    # Load pkl data:
    with open(f'network{net}_Relu{relu}_epoch{epoch}.pkl', 'rb') as f:
        data = pickle.load(f)
    # Get unit ids from the unit objects
    ids = [data[i].id for i in range(len(data))]
    idxs_no_resp = np.where(np.all(actv == 0, axis=(1, 2)))[0]
    # Check if ids are ascending order:    
    if ids == sorted(ids):
        for i, obj in enumerate(data):
        # If the id is within idxs, set 'no_response' to True, otherwise set it to False
            obj.no_response = i in idxs_no_resp
        with open(f'network{net}_Relu{relu}_epoch{epoch}.pkl', 'wb') as f:
            pickle.dump(data, f)
    else:
        print("Unit ids are not in ascending order!")



def generate_labels(min_sz_idx=7, max_sz_idx=13, min_num_idx=0, max_num_idx=9,
                    numbers=range(2,21,2), sizes=range(7,14)):
    labels = []
    num_sizes = max_sz_idx - min_sz_idx + 1
    for num in numbers:
        for sz in sizes:
            label = f'{num}-[{sz}]'
            labels.append(label)
    return labels



def cos_similarity(relu, epoch, nets=range(1,2), 
                   min_sz_idx=3, max_sz_idx=9, 
                   min_num_idx=0, max_num_idx=9,
                   numbers=range(2,21,2), sizes=range(4,14),
                   PN=[], PS=[],
                   main='number'):
    cos_sim_matrices = []
    numb = numbers[min_num_idx:max_num_idx+1]

    for net in nets:
        # Get selective units for current network and epoch
        # units, avg_response_vectors = actv_analysis.get_selective_units(net, relu, epoch, min_sz_idx=min_sz_idx, max_sz_idx=max_sz_idx, PN=PN,PS=PS)
        # units, avg_response_vectors = actv_analysis.get_random_active_units(net, relu, epoch, min_sz_idx=min_sz_idx, max_sz_idx=max_sz_idx, PN=PN,PS=PS)

        # Normalize the vectors
        normalized_vectors = normalize(avg_response_vectors)

        # Compute cosine similarity matrix
        cos_sim_matrix = cosine_similarity(normalized_vectors.T)

        # Add matrix to the list of matrices
        cos_sim_matrices.append(cos_sim_matrix)

    # Compute average cosine similarity matrix
    avg_cos_sim_matrix = np.mean(cos_sim_matrices, axis=0)

    # Create labels for the heatmap ticks
    x_labels = ['' for _ in range(avg_cos_sim_matrix.shape[0])]
    y_labels = ['' for _ in range(avg_cos_sim_matrix.shape[1])]

    if main == 'number':
        num_sizes = max_sz_idx - min_sz_idx + 1
        for num in numb:
            for sz in range(min_sz_idx, max_sz_idx + 1):
                idx = ((num - min(numb)) // 2) * num_sizes + (sz - min_sz_idx) - min(sz for sz in sizes)
                label = f'{num}-[{sz}]'
                x_labels[idx] = label
                y_labels[idx] = label


    elif main == 'size':
        num_numbers = max_num_idx - min_num_idx + 1
        for sz in sizes:
            for nb in range(min_num_idx, max_num_idx + 1):
                idx = ((sz - min(sizes)) - min_sz_idx) * num_numbers + (nb - min_num_idx)
                label = f'{sz}-[{nb}]'
                x_labels[idx] = label
                y_labels[idx] = label


    # Plot heatmap for each network
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(avg_cos_sim_matrix)  # Using default cmap
    ax.set_xticks(np.arange(avg_cos_sim_matrix.shape[0])) 
    ax.set_yticks(np.arange(avg_cos_sim_matrix.shape[1]))  
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticklabels(y_labels, rotation=0)
    plt.title(f'Average Cosine Similarity for ReLU {relu}, Epoch {epoch}, Sizes {min_sz_idx}-{max_sz_idx}')
    plt.tight_layout()
    plt.savefig(f'avg_cos_similarity_relu{relu}_epoch{epoch}_sizes{min_sz_idx}-{max_sz_idx}.pdf')
    plt.close()

    return avg_cos_sim_matrix


def plot_mds(cos_sim_matrix, relu, epoch, dim=2,x_lim=None, y_lim=None, numbers=range(2,21,2), sizes=range(7,14), annot=True):
    """
    Perform MDS on a cosine similarity matrix and plot the results.

    Parameters:
    - cos_sim_matrix: Cosine similarity matrix.
    - dim: Number of dimensions for MDS.
    - numbers: Range of numbers represented in the data.
    - sizes: Range of sizes represented in the data.
    """

    # Perform MDS
    mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=42)
    mds_results = mds.fit_transform(1 - cos_sim_matrix)  # We subtract from 1 to convert similarity to dissimilarity

    # Perform PCA on the results of MDS
    pca = PCA(n_components=dim)
    mds_results = pca.fit_transform(mds_results)

    # Create a colormap
    cmap = plt.cm.get_cmap('viridis')

    # Map numbers to colors and sizes to point sizes
    sizes_map = {sz: (i+1)*10 for i, sz in enumerate(sizes)}  # Map sizes to point sizes

    # Create figure and plot MDS results
    fig, ax = plt.subplots(figsize=(10, 10))

    if annot:
        # Create labels based on structure of cos_sim_matrix
        for i in range(cos_sim_matrix.shape[0]):
            num = numbers[i // len(sizes)]  # Calculate number based on row index
            sz = sizes[i % len(sizes)]  # Calculate size based on row index

            color = cmap((num - min(numbers)) / (max(numbers) - min(numbers)))  # Map number to color
            size = sizes_map[sz]  # Map size to point size

            label = f"{num}-{sz}"
            ax.scatter(mds_results[i, 0], mds_results[i, 1], color=color, s=size)
            ax.annotate(label, (mds_results[i, 0], mds_results[i, 1]))

    ax.set_title(f'MDS of RSA: number-size representation in relu{relu} epoch{epoch}', fontsize=20)
    ax.set_xlabel('Dimension 1', fontsize=14)
    ax.set_ylabel('Dimension 2', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    plt.tight_layout()
    plt.savefig(f'MDS_relu{relu}_epoch{epoch}.pdf')
    plt.show()




def process_block(block, units, actv_2D, numbers, sizes, inst, parallel=0):
    failed_units = []
    for i, u in enumerate(block):
        try:
            anova2_results = units[u].add_anova2(actv_2D, numbers, sizes, inst, parallel)
            units[u].anova2 = anova2_results
        except Exception as e:
            print(f"An error occurred in unit {u}: {e}")
            failed_units.append(u)
    return units, failed_units



def handle_failed_blocks(failed_sub_blocks, units, actv_2D, numbers, sizes, inst):
    success_blocks = []
    while failed_sub_blocks:
        new_failed_sub_blocks = []
        for sub_block in failed_sub_blocks:
            current_block_size = max(1, len(sub_block) // 2)
            print(f'current block size:{current_block_size}')
            sub_blocks = [sub_block[i:i + current_block_size] for i in range(0, len(sub_block), current_block_size)]
            for sb_idx, sub_block in enumerate(sub_blocks):
                if len(sub_block) <= 100:
                    # Process the units individually
                    print("processing units individually..")
                    units, failed_units = process_block(sub_block, units, actv_2D, numbers, sizes, inst)
                    sub_block = [u for u in sub_block if u not in failed_units]  # Remove failed units from the block
                    if sub_block:  # If there are still units left in the sub_block
                        success_blocks.append(sub_block)
                else:
                    # If block size is still greater than 100, we attempt to process it as a whole
                    try:
                        anova2_results = Parallel(n_jobs=-1)(delayed(units[u].add_anova2)(actv_2D, numbers, sizes, inst, parallel=1) for u in sub_block)
                        for i, u in enumerate(sub_block):
                            units[u].anova2_numbers = anova2_results[i].loc['C(number)', 'PR(>F)']
                            units[u].anova2_sizes = anova2_results[i].loc['C(size)', 'PR(>F)']
                            units[u].anova2_ns_int = anova2_results[i].loc['C(number):C(size)', 'PR(>F)']
                        success_blocks.append(sub_block)
                    except Exception as e:
                        print(f"An error occurred in sub block {sb_idx}: {e}")
                        new_failed_sub_blocks.append(sub_block)
        failed_sub_blocks = new_failed_sub_blocks
    return units, failed_sub_blocks, success_blocks



def save_updated_units(net, relu, epoch, units, last_backup_idx, pkl_filename, csv_filename):
    # Saving the updated units at the end of processing
    #pickle_filename = f'pkl/network{net}_Relu{relu}_epoch{epoch}.pkl'
    with open(pkl_filename, 'wb') as f:
        pickle.dump(units, f)
    print(f'Successfully saved the updated units to {pkl_filename}')

    # # Saving a backup copy of the updated units
    # #backup_filename = f'pkl_backup/backup_network{net}_Relu{relu}_epoch{epoch}_{last_backup_idx+1}.pkl'
    # backup_filename = pkl_filename + str(last_backup_idx+1) +
    # with open(backup_filename, 'wb') as f:
    #     pickle.dump(units, f)
    # print(f'Successfully saved a backup of the updated units to {backup_filename}')

    # Saving the updated units at the end of processing
    #csv_filename = f'csv/network{net}_Relu{relu}_epoch{epoch}.csv'
    load_csv.obj_to_csv(units, csv_filename)
    print(f'Successfully saved the updated units to {csv_filename}')

    # # Saving a backup copy of the updated units
    # backup_filename = f'csv_backup/backup_network{net}_Relu{relu}_epoch{epoch}_{last_backup_idx+1}.csv'
    # load_csv.obj_to_csv(units, backup_filename)
    # print(f'Successfully saved a backup of the updated units to {backup_filename}')



def pkl_anova2_V2(net, relu, epoch, min_sz_idx=3, max_sz_idx=9, min_nidx=0, max_nidx=9, numbers=np.arange(2,21,2), inst=500):
    """
    Function: This function performs an ANOVA2 analysis on units in a neural network that have nonzero activations.
    
    Parameters:
    - net: Network ID or information.
    - relu: Relu layer ID or information.
    - epoch: Epoch ID or information.
    - min_sz_idx (default = 3): The minimum size index for the analysis.
    - max_sz_idx (default = 9): The maximum size index for the analysis.
    - numbers (default = range(2,21,2)): The numbers to use for the analysis.
    - inst (default = 500): The number of instances for the analysis.

    Returns:
    The function does not return anything. It saves the updated units into a pickle file.
    """
    print(f'Network{net} Relu{relu} Epoch{epoch}')
    pkl_filename = f'pkl/4to20/network{net}_Relu{relu}_epoch{epoch}_4to20.pkl'
    csv_filename = f'csv/network{net}_Relu{relu}_epoch{epoch}_obj_4to20.csv'

    layer_numunits = {'relu1':290400, 'relu2':186624, 'relu3':64896, 'relu4':64896, 'relu5':43264}

    units = None
    actv_net = None
    last_backup_idx = 0

    # First, try to load the units from the pickle file
    if os.path.exists(pkl_filename):
        try:
            with open(pkl_filename, 'rb') as f:
                units = pickle.load(f)
            print("Pickle file found.")
        except Exception as e:
            print(f"Pickle file {pkl_filename} failed to load due to error: {e}. Looking for backup...")

    # If the pickle file does not exist or failed to load, try the CSV file
    if units is None and os.path.exists(csv_filename):
        print("Looking for csv file..")
        try:
            units = load_csv.csv_to_obj(csv_filename)
            print('csv file found. Converted to objects')
        except Exception as e:
            print(f"CSV file {csv_filename} failed to load due to error: {e}. Looking for backup...")

    # If neither the pickle file nor the CSV file exist or loaded successfully, load from backup
    if units is None:
        print("Looking for backup pkl file..")
        units, last_backup_idx = pkl_anova2_find_backup(net, relu, epoch)

    if units is None:
        print(f"Neither CSV file {csv_filename} nor backup file found. Creating a new list of unit objects..")
        units = [objects.Unit(i) for i in range(list(layer_numunits.values())[relu-1])]
        for unit in units:
            unit.add_network_info(epoch, net, relu)

    incomplete_units = np.array([i for i in range(len(units)) 
                             if not units[i].no_response_subset and units[i].anova2_failed is not True and
                             all(attr is None for attr in [units[i].anova2_numbers, units[i].anova2_sizes, units[i].selectivity_number])])

    if len(incomplete_units) ==0: ## if there is nothing to do, simply terminate the function
        return
    else:
        print(f'Loading activity data to see if anova2 is complete.')
        
        numbers = numbers[min_nidx:max_nidx+1] # take the subset of default numbers
        sizes = np.arange(4,14)[min_sz_idx: max_sz_idx+1]
        actv_net = actv_analysis.get_actv_net(net=net, relu=relu, epoch=epoch)

        inactive_units = np.arange(actv_net.shape[0])[np.all(actv_net == 0, axis=(1, 2))]
        for idx, unit in enumerate(units):
            unit.no_response = idx in inactive_units
        print("Updated no_response attributes of the units.")

        take = np.arange(0,100).reshape(10,10)[min_nidx:max_nidx+1,min_sz_idx:max_sz_idx+1].reshape(len(numbers)*(max_sz_idx-min_sz_idx+1))
        actv_szAtoB = actv_net[:,take,:]
        inactive_subset_units = np.arange(actv_net.shape[0])[np.all(actv_szAtoB == 0, axis=(1, 2))]
        for idx, unit in enumerate(units):
            unit.no_response_subset = idx in inactive_subset_units
        print("Updated no_response_subset attributes of the units.")

        units = actv_analysis.update_PNS(units, actv_net, numbers=np.arange(2,21,2), sizes = np.arange(4,14), min_nidx=min_nidx, max_nidx=max_nidx, min_sz_idx=min_sz_idx, max_sz_idx=max_sz_idx)
        print("Updated PN and PS attributes of the units.")
        
        units = actv_analysis.update_response_rate(units, actv_net, subset=False)
        units = actv_analysis.update_response_rate(units, actv_szAtoB, subset=True)
        print("Updated response rates of the units.")
        
        # Get the indices of units with non-zero responses
        active_subset_units = np.setdiff1d(np.arange(actv_net.shape[0]), inactive_subset_units)
        incomplete_units = np.array([i for i in range(len(units)) 
                             if not units[i].no_response_subset and 
                             all(attr is None for attr in [units[i].anova2_numbers, units[i].anova2_sizes, units[i].selectivity_number])])
        if len(incomplete_units)==0:
            print("2-way ANOVA is complete!")
            return
        print(f'{len(incomplete_units) / len(active_subset_units)*100}% units are incomplete!')

        actv_2D = actv_szAtoB.reshape(actv_szAtoB.shape[0], actv_szAtoB.shape[1]*actv_szAtoB.shape[2])

        block_size = 3000
        print(f'Starting with {block_size} number of units at a time.')

        blocks = actv_analysis.get_blocks(indices=incomplete_units, block_size=block_size)

        for bk_idx, block in enumerate(blocks):
            print(f"Processing net {net}, relu {relu}, epoch {epoch}, block {bk_idx + 1}/{len(blocks)}")

            current_block_size = min(len(block), 3000)  

            failed_sub_blocks = []

            while len(block) > 1:
                try:
                    start_time = time.time()
                    anova2_results = Parallel(n_jobs=-1)(delayed(units[u].add_anova2)(actv_2D, numbers, sizes, inst, parallel=1) for u in block)
                    for i, u in enumerate(block):
                        units[u].anova2_numbers = anova2_results[i].loc['C(number)', 'PR(>F)']
                        units[u].anova2_sizes = anova2_results[i].loc['C(size)', 'PR(>F)']
                        units[u].anova2_ns_int = anova2_results[i].loc['C(number):C(size)', 'PR(>F)']
                    print("--- %s seconds ---" % (time.time() - start_time))
                    block=[]

                except Exception as e:
                    print(f"An error occurred: {e}")
                    failed_sub_blocks.append(block)

                if failed_sub_blocks:
                    print(f'block size is {current_block_size}. Partitioning it into two sub-blocks')
                    units, failed_sub_blocks, success_blocks = handle_failed_blocks(failed_sub_blocks, units, actv_2D, numbers, sizes, inst)
                    block = [u for u in block if u not in [item for sublist in success_blocks for item in sublist]] # Update 'block' by removing successfully processed blocks
            
            # Saving the updated units at the end of each iteration
            save_updated_units(net, relu, epoch, units, last_backup_idx, pkl_filename, csv_filename)
            last_backup_idx += 1

            # # Saving the updated units at the end of processing
            # pickle_filename = f'pkl/network{net}_Relu{relu}_epoch{epoch}.pkl'
            # with open(pickle_filename, 'wb') as f:
            #     pickle.dump(units, f)
            # print(f'Successfully saved the updated units to {pickle_filename}')

            # # Saving a backup copy of the updated units
            # backup_filename = f'pkl_backup/backup_network{net}_Relu{relu}_epoch{epoch}_{last_backup_idx+1}.pkl'
            # with open(backup_filename, 'wb') as f:
            #     pickle.dump(units, f)
            # print(f'Successfully saved a backup of the updated units to {backup_filename}')

            # # Saving the updated units at the end of processing
            # csv_filename = f'csv/network{net}_Relu{relu}_epoch{epoch}.csv'
            # load_csv.obj_to_csv(units, csv_filename)
            # print(f'Successfully saved the updated units to {csv_filename}')

            # # Saving a backup copy of the updated units
            # backup_filename = f'csv/backup_network{net}_Relu{relu}_epoch{epoch}_{last_backup_idx+1}.csv'
            # load_csv.obj_to_csv(units, backup_filename)
            # print(f'Successfully saved a backup of the updated units to {backup_filename}')

            # last_backup_idx += 1



def tSNE(net, relu, epoch, units, units_sample_size, n_comp=2, min_sz_idx=3, max_sz_idx=9, numbers=range(2,21,2), random_state=42):
    """
    Plots the t-SNE of the given activations.

    Parameters:
    - actv_net: 4D array containing activations.
    - units_sample_size: The number of units to sample for the plot.
    - n_comp: The number of components to use in the technique.
    - random_state: Random seed for reproducibility.
    """
    sizes = np.arange(4,14)[min_sz_idx: max_sz_idx+1]
    actv_net = actv_analysis.get_actv_net(net=net,relu=relu,epoch=epoch)
    take = np.arange(0,100).reshape(10,10)[:,min_sz_idx:max_sz_idx+1].reshape(len(numbers)*(max_sz_idx-min_sz_idx+1))
    avg_actv_szAtoB = np.mean(actv_net[:,take,:],axis=2)

    # Sample a subset of your data. 
    np.random.seed(random_state)  # for reproducibility
    idx = np.random.choice(avg_actv_szAtoB.shape[0], units_sample_size, replace=False)
    data_sample = avg_actv_szAtoB[idx, :]
    pn_sample = [units[i].PN for i in idx]  # get corresponding pns
    ps_sample = [units[i].PS for i in idx]  # get corresponding pss

    # normalize PS values to match the desired range of point sizes
    ps_sample = np.array(ps_sample)
    ps_sample = (ps_sample - np.min(ps_sample)) / (np.max(ps_sample) - np.min(ps_sample)) * 50 + 50

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    tsne = TSNE(n_components=n_comp, random_state=random_state)
    tsne_results = tsne.fit_transform(data_sample)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=pn_sample, s=ps_sample)
    plt.title('t-SNE plot of unit activations', fontsize=20)
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    fig.colorbar(scatter, label='PN')

    plt.show()


from sklearn.decomposition import PCA

def apply_pca(net, relu, epoch, units, units_sample_size, min_sz_idx=3, max_sz_idx=9, numbers=range(2,21,2), n_comp=2, random_state=42):
    """
    Apply PCA on the given activations.

    Parameters:
    - net, relu, epoch: parameters to fetch activations using `get_actv_net`.
    - units_sample_size: The number of units to sample for the plot.
    - min_sz_idx, max_sz_idx, numbers: parameters to generate the `take` array.
    - n_comp: The number of components to use in PCA.
    - random_state: Random seed for reproducibility.
    """
    
    sizes = np.arange(4,14)[min_sz_idx: max_sz_idx+1]
    actv_net = actv_analysis.get_actv_net(net=net,relu=relu,epoch=epoch)
    take = np.arange(0,100).reshape(10,10)[:,min_sz_idx:max_sz_idx+1].reshape(len(numbers)*(max_sz_idx-min_sz_idx+1))
    avg_actv_szAtoB = np.mean(actv_net[:,take,:],axis=2)

    # Sample a subset of your data. 
    np.random.seed(random_state)  # for reproducibility
    idx = np.random.choice(avg_actv_szAtoB.shape[0], units_sample_size, replace=False)
    data_sample = avg_actv_szAtoB[idx, :]
    pn_sample = [units[i].PN for i in idx]  # get corresponding pns
    ps_sample = [units[i].PS for i in idx]  # get corresponding pss

    pca = PCA(n_components=n_comp, random_state=random_state)
    pca_results = pca.fit_transform(data_sample)
    
    # Create a new figure and axes for the PCA plot
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=pn_sample, s=np.array(ps_sample)*2)  # Adjust sizes using PS values
    ax.set_xscale("log")  # Set x-axis to log scale
    ax.set_yscale("log")  # Set y-axis to log scale
    plt.title('PCA plot of unit activations', fontsize=20)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    fig.colorbar(scatter, label='PN')
    plt.show()


def number_ratio(numbers, idx1, idx2):
    num1, num2 = numbers[idx1], numbers[idx2]
    ratio = np.minimum(num1, num2) / np.maximum(num1, num2)
    return ratio













