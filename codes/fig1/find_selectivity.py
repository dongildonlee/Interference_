import numpy as np
import pandas as pd
import sys
import os
from joblib import Parallel, delayed
sys.path.append('../')
from packages import stats

############# Parameters ################
relu=5
nets = np.arange(1,3)
epochs=np.arange(0,90,30)
#########################################

dir_path = os.path.dirname(os.path.realpath('../'))
save_to_folder = dir_path+'/dataframes/selectivity'

for net in nets:
    print("net:",net)
    for epoch in epochs:
        print("epoch:",epoch)
        anova2 = pd.read_csv(dir_path+'/dataframes/ANOVA2/df_anova2 for He initialized net'+str(net)+'_relu'+str(relu)+'_epoch'+str(epoch)+' size 7to13 500inst.csv')
        df_selectivity = stats.get_selectivity(anova2)
        df_selectivity.to_csv(save_to_folder+'/selectivity of He untrained net'+str(net)+' relu'+str(relu)+' epoch'+str(epoch)+'.csv', index=True)
