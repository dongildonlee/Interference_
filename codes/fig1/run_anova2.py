import numpy as np
import pandas as pd
import h5py
import sys
import os
import time
from joblib import Parallel, delayed
sys.path.append('../')
from packages import stats

############# Parameters ########################
relu=5
nets = np.arange(1,3)
epochs=np.arange(0,91,10)
units = np.arange(43264) # unique to Relu5 layer
num_blocks = 8
num_units_per_block = int(43264/num_blocks)

numbers = np.arange(2,21,2)
min_sz_idx=3; max_sz_idx=9
sizes = np.arange(4,14)[min_sz_idx: max_sz_idx+1]
inst = 500
#################################################

dir_path = os.path.dirname(os.path.realpath('../'))
save_to_folder = dir_path+'/dataframes/ANOVA2'

for net in nets:
    print("net:", net)
    for epoch in epochs:
        print("epoch:", epoch)

        # Load raw response of AlexNet and perform a transpose for a better readability:
        f500 = h5py.File(dir_path+'/data/raw_response/actv_f500_network'+str(net)+'_relu'+str(relu)+'_epoch'+str(epoch)+'.mat', 'r')
        actv_=f500['actv'][:]
        actv = np.transpose(actv_, (2,1,0))

        # Take activity corresponding to size 7 to 13:
        take = np.arange(0,100).reshape(10,10)[:,min_sz_idx:max_sz_idx+1].reshape(len(numbers)*(max_sz_idx-min_sz_idx+1))
        actv_szAtoB = actv[:,take,:]
        actv_2D = actv_szAtoB.reshape(actv_szAtoB.shape[0], actv_szAtoB.shape[1]*actv_szAtoB.shape[2])

        # Make a dataframe to store ANOVA2 results:
        df_anova2 = pd.DataFrame(index=units, columns = ['number', 'size', 'inter', 'residual'])

        ## Perform 2-way ANOVA with parallel computing:
        for bk in np.arange(num_blocks): # to minimize the impact of occassional occurrence of 'SVD did not converge' error
            print("block:", bk)
            try:
                unit_idx = np.arange(bk*num_units_per_block, (bk+1)*num_units_per_block)
                start_time = time.time()
                anova_results = Parallel(n_jobs=-1)(delayed(stats.anova2_single)(actv_2D, u, numbers, sizes, inst) for u in unit_idx)
                print("--- %s seconds ---" % (time.time() - start_time))

                ## Save the data (save every time a block is complete):
                anova_pval = pd.concat(anova_results).iloc[:,3].to_numpy().reshape(num_units_per_block,4)
                df_anova2.iloc[unit_idx,:]=anova_pval
                df_anova2.to_csv(save_to_folder+'/df_anova2 for He initialized net'+str(net)+'_relu'+str(relu)+'_epoch'+str(epoch)+' size 7to13 500inst.csv')
            except:
                continue
