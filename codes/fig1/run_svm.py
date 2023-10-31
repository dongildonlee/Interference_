# import numpy as np
# import pandas as pd
# import sys
# import os
# from joblib import Parallel, delayed
# sys.path.append('../')
# from packages import actv_analysis, svm, load_csv

# ############# Parameters ################
# relu=5
# epochs=np.arange(0,91,10)
# exps = np.arange(1,11)
# min_sz_idx=0; max_sz_idx=9
# selectivity =['number','size','both'][2]
# num_units = 100
# rate_threshold = 0.05
# #########################################

# dir_path = os.path.dirname(os.path.realpath('../'))
# path_for_units = dir_path+'/dataframes/SVM/units/'+str(num_units)+' units sampled from distribution higher than '+str(rate_threshold)+' response rate including PN2 and PN20'
# set_folder = dir_path+'/dataframes/SVM' # retrieve training/test sets
# save_to_folder = dir_path+'/dataframes/SVM_predictions'

# for epoch in epochs:
#     print("epoch:",epoch)
#     for net in np.arange(1,3):
#         print("net:",net)
#         actv_net = actv_analysis.get_actv_net(net=net,relu=relu,epoch=epoch)
#         actv = actv_net.reshape(43264,10,10,500)
#         units = load_csv.units_for_svm(path=path_for_units,num_units=num_units,net=net,epoch=epoch,relu=relu)

#         #start_time = time.time()
#         y_preds = Parallel(n_jobs=-1)(delayed(svm.SVM_fit)(dir_path=dir_path, units=units, actv=actv, exp=exp) for exp in exps)

#         #print("--- %s seconds ---" % (time.time() - start_time))
#         for exp in exps:
#             pd.Series(y_preds[exp-1]).to_csv(save_to_folder+'/SVM prediction of He untrained net'+str(net)+' relu'+str(relu)+' epoch'+str(epoch)+' '+str(num_units)+' '+str(selectivity)+' units that are randomly drawn from distribution exp' + str(exp)+ ' Jan2023.csv', index=True)

import os
import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed
from packages.actv_analysis import get_actv_net
from packages.svm import SVM_fit
from packages.load_csv import units_for_svm

# Parameters
relus = range(2,6)
epochs = np.arange(0, 91, 10)
exps = np.arange(1, 11)
num_units = 100
rate_threshold = 0.05

dir_path = os.path.dirname(os.path.realpath('../'))
#path_for_units = f"{dir_path}/dataframes/SVM/units/{num_units} units sampled from distribution higher than {rate_threshold} response rate including PN2 and PN20"
set_folder = f"{dir_path}/dataframes/SVM"  # retrieve training/test sets
save_to_folder = f"{dir_path}/dataframes/SVM_predictions"

for relu in relus:
    for epoch in epochs:
        for net in range(1, 3):
            pkl_filename = f'network{net}_Relu{relu}_epoch{epoch}.pkl'
            print(f'Loading {pkl_filename}..')
            with open(pkl_filename, 'rb') as f:
                units = pickle.load(f)
            actv_net = get_actv_net(net=net, relu=5, epoch=epoch)
            actv = actv_net.reshape(43264, 10, 10, 500)
            #units = units_for_svm(path=path_for_units, num_units=num_units, net=net, epoch=epoch, relu=5)

            y_preds = Parallel(n_jobs=-1)(delayed(SVM_fit)(dir_path=dir_path, units=units, actv=actv, exp=exp) for exp in exps)

            [pd.Series(y_preds[exp-1]).to_csv(f"{save_to_folder}/SVM prediction of He untrained net{net} relu5 epoch{epoch} {num_units} units that are randomly drawn from distribution exp{exp} Jan2023.csv", index=True) for exp in exps]

