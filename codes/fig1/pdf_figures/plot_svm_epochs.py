import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../')
from packages import svm

############# Parameters ################
relu=5;
nets=np.array([1,2]);
epochs=np.arange(0,31,10)
exps = np.arange(1,11) # SVM test trials

selectivity = 'both'
num_units = 100
congruency=np.array(['C','IC'])
#########################################

dir_path = os.path.dirname(os.path.realpath('../'))
save_to_folder1 = dir_path+'/dataframes/SVM_analysis'
save_to_folder2 = dir_path+'/figures'

# Make a dataframe to store SVM performance:
num_rows = len(nets)*len(epochs)*len(congruency)
df_SVM_accuracy = pd.DataFrame(index=np.arange(num_rows), columns=['epoch','net','congruency','accuracy'])
df_SVM_accuracy.loc[:,'epoch'] = np.repeat(epochs, len(nets)*len(congruency))
df_SVM_accuracy.loc[:,'net'] = np.tile(np.repeat(np.arange(1,1+len(nets)),2), len(epochs))
df_SVM_accuracy.loc[:,'congruency'] = np.tile(congruency, len(nets)*len(epochs))


# Load all test sets for SVM and get congruent and incongruent trial indices:

all_test_X = svm.get_all_text_X(exps)
c, ic = svm.get_congruency(all_test_X, 'dot size')


i=0
for epoch in epochs:
    print("epoch:", epoch)
    for net in nets:
        print("net:", net)
        all_preds = svm.get_all_preds(net=net, relu=relu, epoch=epoch, selectivity=selectivity, num_units=num_units, exps=exps)
        acc_C = svm.get_SVM_accuracy(c, all_test_X, all_preds)
        acc_IC = svm.get_SVM_accuracy(ic, all_test_X, all_preds)
        df_SVM_accuracy.iloc[i:i+2,3] = [acc_C, acc_IC]
        i+=2
df_SVM_accuracy.to_csv(save_to_folder+'/SVM_accuracy.csv', index=True)


# Plot training epoch vs SVM performance

sns.set_style("white")
ax=sns.lineplot(data=df_SVM_accuracy, x="epoch", y="accuracy", hue="congruency", palette=['g', 'r'], style='congruency', markers=False)
plt.xlabel('Training epoch',fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.7,1)
plt.tight_layout()
plt.savefig(save_to_folder+'/training epoch vs SVM accuracy.pdf', transparent=True)
