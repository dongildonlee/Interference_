import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
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
save_to_folder = dir_path+'/figures'

# Load data:

data = pd.read_csv(dir_path+'/dataframes/SVM_analysis/SVM all info.csv').drop(columns=['Unnamed: 0'])
num_ratios = np.unique(data['number ratio'].to_numpy())

# Make a dataframe for the plot:
df_numratio_vs_acc = pd.DataFrame(index=np.arange(len(nets)*len(epochs)*len(num_ratios)*len(congruency)), columns= ['net', 'network', 'num ratios', 'congruency','inaccuracy'])
df_numratio_vs_acc.loc[:,'net'] = np.tile(np.repeat(nets, len(congruency)*len(num_ratios)), len(epochs))
df_numratio_vs_acc.loc[:,'epoch'] = np.repeat(epochs, len(nets)*len(congruency)*len(num_ratios))
df_numratio_vs_acc.loc[:,'num ratios'] = np.tile(num_ratios, len(nets)*len(epochs)*len(congruency))
df_numratio_vs_acc.loc[:,'congruency'] = np.tile(np.repeat(np.array(['C','IC']),len(num_ratios)),len(nets)*len(epochs))

# Simple sorting and add relevant results for plotting to the dataframe:
for item in np.arange(df_numratio_vs_acc.shape[0]):
    _data = data[(data['net']==df_numratio_vs_acc.loc[item, 'net'])*(data['epoch']==df_numratio_vs_acc.loc[item, 'epoch'])*(data['number ratio']==df_numratio_vs_acc.loc[item, 'num ratios'])*(data['congruency']==df_numratio_vs_acc.loc[item, 'congruency'])]
    df_numratio_vs_acc.loc[item, 'inaccuracy'] = (_data.shape[0] - np.sum(_data['correctly predicted']))/_data.shape[0]

# Plot a lineplot:
# x-axis = number ratio (smaller/larger number)
# y-axis = SVM performance (in terms of inaccuracy)
sns.set_style('white')
sns.lineplot(data = df_numratio_vs_acc, x='num ratios', y='inaccuracy', hue='congruency')
plt.xlim(0,0.9); plt.ylim(0,0.35)
plt.savefig(save_to_folder+"/lineplot for number ratios vs number comparison inaccuracy.pdf", transparent=True)
