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
import statsmodels.api as sm
from statsmodels.formula.api import ols
import sys
import time
sys.path.append('../')
from packages import objects, actv_analysis
#from umap.umap_ import UMAP
#from sklearn.manifold import TSNE


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

    df = pd.DataFrame({'number': np.repeat(np.repeat(numbers,len(sizes)),instances),'size': np.tile(np.repeat(sizes,instances),len(numbers)),'activity':actv_2D[unit,:]})
    model = ols('activity ~ C(number) + C(size) + C(number):C(size)', data=df).fit()
    stat = sm.stats.anova_lm(model, typ=2)
    #print(f"Stat for unit {unit_obj.id}: {stat}")
    if parallel:
        return stat.iloc[:, 3].to_frame().T
    else:
        unit_obj.anova2 = stat.iloc[:, 3].to_frame().T
    #return unit_obj.anova2
