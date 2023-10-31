import numpy as np
import pandas as pd
import sys
import os
sys.path.append('../')
from packages import svm

############# Parameters ################
relu=5
nets = np.arange(1,3)
epochs=np.arange(0,31,10)
exps = np.arange(1,11)

numbers = np.arange(2,21,2)
congruency=['congruent','incongruent']
exps = np.arange(1,11)
selectivity = 'both'
num_units=100
#########################################

dir_path = os.path.dirname(os.path.realpath('../'))
save_to_folder = dir_path+'/dataframes/SVM_analysis'

# Gather all SVM test trials:
df_all_exps = pd.concat([pd.read_csv(dir_path+'/dataframes/SVM/test_sets/test set idx for exp'+str(exp) + '.csv').drop(columns=['Unnamed: 0']) for exp in exps])
df_all_exps.index = np.arange(df_all_exps.shape[0])

# Make a dataframe to store all the info related to SVM:
df_all_info = pd.DataFrame(index=np.arange(df_all_exps.shape[0]*len(nets)*len(epochs)), columns=['num1', 'num2','sz1','sz2','img1','img2','net','epoch','congruency','number distance','number ratio','size distance','actv diff','correctly predicted'])
df_all_info.iloc[:,0:6] = pd.concat([df_all_exps]*len(nets)*len(epochs))

## Fill in information:

# 1. Fill in net:
df_all_info.loc[:,'net'] = np.tile(np.repeat(np.arange(1,len(nets)+1), df_all_exps.shape[0]), len(epochs))
# Fill in epoch:
df_all_info.loc[:,'epoch'] = np.repeat(epochs, df_all_exps.shape[0]*len(nets))
# 2. Fill in congruency:
exp_idx = np.arange(df_all_info.shape[0])
c1 = exp_idx[((df_all_info.loc[:,'num1'] < df_all_info.loc[:,'num2'])&(df_all_info.loc[:,'sz1'] < df_all_info.loc[:,'sz2']))]
c2 = exp_idx[((df_all_info.loc[:,'num1'] > df_all_info.loc[:,'num2'])&(df_all_info.loc[:,'sz1'] > df_all_info.loc[:,'sz2']))]
c = np.union1d(c1,c2)
ic = np.setdiff1d(exp_idx, c)
df_all_info.loc[c,'congruency'] = 'C'
df_all_info.loc[ic,'congruency'] = 'IC'
# 3. Fill in number and size distance:
df_all_info.loc[c,'number distance'] = -np.abs(np.arange(2,20,2)[df_all_info.loc[c,'num1'].to_numpy().astype(int)] - np.arange(2,20,2)[df_all_info.loc[c,'num2'].to_numpy().astype(int)])
df_all_info.loc[ic,'number distance'] = np.abs(np.arange(2,20,2)[df_all_info.loc[ic,'num1'].to_numpy().astype(int)] - np.arange(2,20,2)[df_all_info.loc[ic,'num2'].to_numpy().astype(int)])
df_all_info.loc[c,'size distance'] = -np.abs(df_all_info.loc[c,'sz1']-df_all_info.loc[c,'sz2'])
df_all_info.loc[ic,'size distance'] = np.abs(df_all_info.loc[ic,'sz1']-df_all_info.loc[ic,'sz2'])
# 4. Fill in number ratio:
df_all_info.loc[:,'number ratio'] = numbers[np.amin(df_all_info.iloc[:,0:2].to_numpy(), axis=1).astype(int)]/numbers[np.amax(df_all_info.iloc[:,0:2].to_numpy(), axis=1).astype(int)]
# 5. Fill in prediction:
all_y = svm.get_y(df_all_exps).to_numpy()

# 6. Fill in prediction accuracy:
for epoch in epochs:
    for net in nets:
        file = 'SVM prediction of He untrained net'+str(net)+' relu'+str(relu)+' epoch'+str(epoch)+' '+str(num_units)+ ' '+selectivity+' units that are randomly drawn from distribution '
        net_idx = df_all_info[(df_all_info['epoch']==epoch)&(df_all_info['net']==net)].index.to_numpy()
        df_all_preds = pd.concat([pd.read_csv(dir_path+'/dataframes/SVM_predictions/'+file+'exp'+str(exp)+ ' Jan2023.csv').drop(columns=['Unnamed: 0']) for exp in exps])
        correct_or_not = all_y == df_all_preds.iloc[:,0].to_numpy()
        df_all_info.loc[net_idx,'correctly predicted'] = correct_or_not
df_all_info.to_csv(save_to_folder+'/SVM all info.csv', index=True)
