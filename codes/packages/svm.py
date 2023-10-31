import numpy as np
import pandas as pd
import random
import os
from itertools import combinations, product
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

def gen_SVM_input(nidx, sidx, rep_per_ns_combo, img_inst):
    """
    Generates a DataFrame with examples of number, size, and image indices 
    for a pair of images being compared.

    Parameters:
    nidx: Index of numbers used for SVM training and test.
    sidx: Index of sizes used for SVM training and test.
    rep_per_ns_combo: Number of instances for each unique (number,size) combination.
    img_inst: Image instances available (e.g. 500 images for each combination).

    Returns:
    DataFrame with number, size and image indices for a pair of images being compared.
    """
    # ncombs = np.array([[a, b] for idx, a in enumerate(nidx) for b in nidx[idx + 1:]]) # num1 cannot be equal to num2
    # ncombs = np.concatenate([ncombs,ncombs[:,[1,0]]],axis=0)
    # scombs = np.array([[a, b] for idx, a in enumerate(sidx) for b in sidx[:]]) # sz1 CAN be equal to sz2
    # num_rows = len(ncombs)*len(scombs)*rep_per_ns_combo
    # # Make a dataframe:
    # pd_ns_idx = pd.DataFrame(index = np.arange(num_rows), columns = ['num1','num2','sz1','sz2','img1','img2'])
    # pd_ns_idx.iloc[:,0:2] = np.repeat(ncombs, len(scombs)*rep_per_ns_combo,axis=0)
    # pd_ns_idx.iloc[:,2:4] = np.tile(np.repeat(scombs, rep_per_ns_combo, axis=0), (len(ncombs),1))
    # pd_ns_idx.loc[:,'img1'] = random.choices(np.arange(img_inst), k = num_rows)
    # pd_ns_idx.loc[:,'img2'] = random.choices(np.arange(img_inst), k = num_rows)
    # # Shuffle rows:
    # pd_ns_idx = pd_ns_idx.sample(frac=1).reset_index(drop=True)
    # return pd_ns_idx
    ncombs = list(combinations(nidx, 2))
    ncombs += [(b, a) for a, b in ncombs]
    scombs = list(product(sidx, repeat=2))
    
    num_rows = len(ncombs) * len(scombs) * rep_per_ns_combo
    
    pd_ns_idx = pd.DataFrame(index=np.arange(num_rows), columns=['num1','num2','sz1','sz2','img1','img2'])
    
    pd_ns_idx['num1'], pd_ns_idx['num2'] = np.repeat(ncombs, len(scombs)*rep_per_ns_combo,axis=0).T
    pd_ns_idx['sz1'], pd_ns_idx['sz2'] = np.tile(np.repeat(scombs, rep_per_ns_combo, axis=0), (len(ncombs),1)).T
    pd_ns_idx['img1'] = random.choices(range(img_inst), k=num_rows)
    pd_ns_idx['img2'] = random.choices(range(img_inst), k=num_rows)
    
    pd_ns_idx = pd_ns_idx.sample(frac=1, replace=False)
    
    return pd_ns_idx.reset_index(drop=True)


# def get_SVM_actv(df_idx, units, actv):
#     ###########################################################
#     ## INPUT:
#     #   1. df_idx: Dataframe with examples of number, size and image indices for a pair of iamges being compared
#     #   2. units: unit IDs used in SVM
#     #   3. actv: activity matrix
#     ## OUTPUT:
#     #   1. X: activity corresponding to the input dataframe
#     ###########################################################
#     d = df_idx
#     actv_list=[]
#     for im in d.index:
#         #print(im)
#         actv_left = actv[units, d.loc[im,'num1'], d.loc[im,'sz1'], d.loc[im,'img1']]
#         actv_right = actv[units, d.loc[im,'num2'], d.loc[im,'sz2'], d.loc[im,'img2']]
#         actv_ = np.concatenate((actv_left, actv_right))
#         actv_list.append(actv_)
#     X=np.vstack(actv_list)
#     return X
# 

def get_SVM_actv(df_idx, units, actv):
    """
    Extracts activity values from actv array using indices from df_idx.

    Parameters:
    df_idx: DataFrame with number, size and image indices for a pair of images being compared.
    units: Unit IDs used for SVM training.
    actv: Activity matrix.

    Returns:
    An array with activity values.
    """
    actv_list = []
    
    for idx_row in df_idx.itertuples():
        num1, sz1, img1, num2, sz2, img2 = idx_row.num1, idx_row.sz1, idx_row.img1, idx_row.num2, idx_row.sz2, idx_row.img2
        actv_left = actv[units, num1, sz1, img1]
        actv_right = actv[units, num2, sz2, img2]
        actv_ = np.concatenate((actv_left, actv_right))
        actv_list.append(actv_)
        
    X = np.vstack(actv_list)
    
    return X



# def get_y(df_idx):
#     ###########################################################
#     ## INPUT:
#     #   1. df_idx: Dataframe with examples of number, size and image indices for a pair of iamges being compared
#     ## OUTPUT:
#     #   1. y: An array of correct responses
#     ###########################################################
#     is_left_larger = df_idx.loc[:,'num1'] > df_idx.loc[:,'num2']
#     y = is_left_larger*2-1
#     return y
def get_y(df_idx):
    """
    Generates the response array.
    
    Parameters:
    df_idx: DataFrame with indices for a pair of images.
    
    Returns:
    y: An array of correct responses.
    """
    y = (df_idx['num1'] > df_idx['num2']).astype(int) * 2 - 1
    return y


# def SVM_fit(set_path, units, actv, exp):
#     ###########################################################
#     ## INPUT:
#     #   1. set_path: directory where data is read from
#     #   2. units: unit IDs used for SVM training
#     #   3. actv: activity matrix
#     #   4. exp: experiment #
#     ## OUTPUT:
#     #   An array of prediction from the classifier
#     ###########################################################
#     # Get classifier:
#     clf = make_pipeline(LinearSVC(random_state=1234, tol=1e-5, max_iter=1000000))
#     scaler = StandardScaler()
#     # Get training and test sets for experiment (exp):
#     df_train = pd.read_csv(set_path+'/training set idx for exp'+str(exp)+'.csv',index_col=0)
#     df_test = pd.read_csv(set_path+'/test set idx for exp'+str(exp) + '.csv', index_col=0)

#     # Process training data and fit classifier:
#     X_tr = get_SVM_actv(df_train, units, actv)
#     X_tr_scld = scaler.fit_transform(X_tr)
#     y_tr = get_y(df_train)
#     clf.fit(X_tr_scld, y_tr)

#     # Process test data and get prediction:
#     X_tst = get_SVM_actv(df_test, units, actv)
#     X_tst_scld = scaler.transform(X_tst)
#     y_test = get_y(df_test)
#     y_pred = np.array([clf.predict([X_tst_scld[i,:]])[0] for i in np.arange(X_tst_scld.shape[0])])

#     return y_pred
def SVM_fit(units, actv, exp):
    """
    Trains and predicts with SVM.
    
    Parameters:
    set_path: Directory where data is read from.
    units: Unit IDs used for SVM training.
    actv: Activity matrix.
    exp: Experiment number.
    
    Returns:
    y_pred: An array of predictions from the classifier.
    """
    # Get classifier:
    clf = make_pipeline(MinMaxScaler(), LinearSVC(random_state=1234, tol=1e-5, max_iter=1000000, dual=False))
    
    # # Load training and test sets:
    # df_train = pd.read_csv(f'csv/svm_training_set{exp}_4to20.csv', index_col=0)
    # df_test = pd.read_csv(f'csv/svm_test_set{exp}_4to20.csv', index_col=0)
    dir_path = os.path.join('..', '..', 'csv')
    train_file_path = os.path.join(dir_path, f'svm_training_set{exp}_4to20.csv')
    test_file_path = os.path.join(dir_path, f'svm_test_set{exp}_4to20.csv')

    # Load the datasets
    df_train = pd.read_csv(train_file_path, index_col=0)
    df_test = pd.read_csv(test_file_path, index_col=0)
    
    # Process training data and fit classifier:
    X_tr = get_SVM_actv(df_train, units, actv)
    y_tr = get_y(df_train)
    clf.fit(X_tr, y_tr)

    # Process test data and get prediction:
    X_tst = get_SVM_actv(df_test, units, actv)
    y_pred = clf.predict(X_tst)
    
    return y_pred


def SVM_fit_shuffled(units, actv, exp, mode="rows"):
    """
    Trains and predicts with SVM after shuffling data in different modes.
    
    Parameters:
    units: Unit IDs used for SVM training.
    actv: Activity matrix.
    exp: Experiment number.
    mode: If set to 'cols', half of the rows of X_tr are shuffled and the first half columns are swapped with the second half columns.
          If set to 'rows', only the rows of X_tr are shuffled.
    
    Returns:
    y_pred: An array of predictions from the classifier.
    """
    # Get classifier:
    clf = make_pipeline(MinMaxScaler(), LinearSVC(random_state=1234, tol=1e-5, max_iter=1000000))
    
    # Load training and test sets:
    df_train = pd.read_csv(f'csv/svm_training_set{exp}_4to20.csv', index_col=0)
    df_test = pd.read_csv(f'csv/svm_test_set{exp}_4to20.csv', index_col=0)
    
    # Process training data and fit classifier:
    X_tr = get_SVM_actv(df_train, units, actv)
    y_tr = get_y(df_train)

    # Shuffle if mode is "rows" or "cols"
    num_rows = X_tr.shape[0] // 2  # half of the rows
    num_cols = X_tr.shape[1] // 2  # half of the columns

    if mode == "rows":
        # Shuffle rows
        np.random.shuffle(X_tr)
    elif mode == "cols":
        # Choose random rows
        random_rows = np.random.choice(X_tr.shape[0], size=num_rows, replace=False)

        # Swap first half of columns with second half
        X_tr[random_rows, :num_cols], X_tr[random_rows, num_cols:] = X_tr[random_rows, num_cols:].copy(), X_tr[random_rows, :num_cols].copy()
    
    # Fit the model
    clf.fit(X_tr, y_tr)

    # Process test data and get prediction:
    X_tst = get_SVM_actv(df_test, units, actv)
    y_pred = clf.predict(X_tst)
    
    return y_pred



def SVM_fit_with_seed(exp, units, actv):
    random.seed(exp)
    y_pred = SVM_fit(units=units, actv=actv, exp=exp)
    return y_pred


def SVM_fit_shuffled_with_seed(exp, units, actv, mode="rows"):
    random.seed(exp)
    y_pred = SVM_fit_shuffled(units=units, actv=actv, exp=exp, mode="rows")
    return y_pred



def SVM_fit_shuff(set_path, units, actv, exp, axis):
    ###########################################################
    ## INPUT:
    #   1. set_path: directory where data is read from
    #   2. units: unit IDs used for SVM training
    #   3. actv: activity matrix
    #   4. exp: experiment #
    ## OUTPUT:
    #   An array of prediction from the classifier
    ###########################################################
    # Get classifier:
    clf = make_pipeline(LinearSVC(random_state=1234, tol=1e-5, max_iter=1000000))
    scaler = StandardScaler()
    # Get training and test sets for experiment (exp):
    df_train = pd.read_csv(set_path+'/training set idx for exp'+str(exp)+'.csv',index_col=0)
    df_test = pd.read_csv(set_path+'/test set idx for exp'+str(exp) + '.csv', index_col=0)

    # Process training data and fit classifier:
    X_tr = get_SVM_actv(df_train, units, actv)
    X_tr_scld = scaler.fit_transform(X_tr)
    y_tr = get_y(df_train)
    clf.fit(X_tr_scld, y_tr)

    # Process test data and get prediction:
    X_tst = get_SVM_actv(df_test, units, actv)
    if axis == 'row':
        X_tst_scld = np.random.shuffle(scaler.transform(X_tst))
    if axis == 'column':
        X_tst_scld = np.random.shuffle((scaler.transform(X_tst)).T)
    y_test = get_y(df_test)
    y_pred = np.array([clf.predict([X_tst_scld[i,:]])[0] for i in np.arange(X_tst_scld.shape[0])])

    return y_pred


def get_all_test_X(set_path, exps):
    ###########################################################
    ## INPUT:
    #   1. directory where data is read from
    #   2. exps: SVM test trial #
    ## OUTPUT:
    #   SVM test trials with number size index in a dataframe
    ###########################################################

    dir_path = os.path.dirname(os.path.realpath('../'))
    all_test_X = pd.concat([pd.read_csv(set_path+'/test set idx for exp'+str(exp) + '.csv').drop(columns=['Unnamed: 0']) for exp in exps])
    all_test_X.index = np.arange(all_test_X.shape[0])
    return all_test_X


def get_all_preds(set_path, net, relu, epoch, num_units, selectivity, exps):
    ###########################################################
    ## INPUT:
    #   1. set_path: directory where data is read from
    #   2. net: network ID #
    #   3. relu: relu layer #
    #   4. epoch: training epoch #
    #   5. num_units: number of units used in SVM
    #   6. selevtivity: number, size or NS (number and size)
    #   7. exps: SVM test trials
    ## OUTPUT:
    #   predictions on the SVM test trials
    ###########################################################
    dir_path = os.path.dirname(os.path.realpath('../'))
    all_preds=pd.concat([pd.read_csv(set_path+'/SVM prediction of He untrained net'+str(net)+' relu'+str(relu)+' epoch'+str(epoch)+' '+str(num_units)+' '+str(selectivity)+' units that are randomly drawn from distribution exp' + str(exp)+ '.csv').drop(columns=['Unnamed: 0']) for exp in exps])
    all_preds.index = np.arange(all_preds.shape[0])
    return all_preds



def get_SVM_accuracy(pair_idx, all_test_X, all_preds):
    ###########################################################
    ## INPUT:
    #   1. pair_idx: SVM test trial #
    #   2. indices corresponding to congruent/incongruent trials
    #   3. all_test_X: test SVM trials
    #   4. all_preds: predictions on the SVM trials
    ## OUTPUT:
    #   SVM accuracy
    ###########################################################
    test_arr = ((all_test_X['num1'] > all_test_X['num2'])*2-1).to_numpy()[pair_idx]
    pred_arr = all_preds['0'].to_numpy()[pair_idx]
    accuracy = sum(test_arr == pred_arr)/len(test_arr)
    return accuracy


# def get_all_preds_temp(file):
#     all_preds=pd.concat([pd.read_csv(file+' exp' + str(exp)+ ' Dec12.csv').drop(columns=['Unnamed: 0']) for exp in np.arange(1,11)])
#     all_preds.index = np.arange(all_preds.shape[0])
#     return all_preds


def get_congruency(test_X, variable):
    ###########################################################
    ## INPUT:
    #   1. test_X: SVM test set in dataframe
    #   2. variable: currently only 'dot size' available
    ## OUTPUT:
    #   congruent and incongruent trial indices
    ###########################################################
    if variable == 'dot size':
        c1 = test_X.index.to_numpy()[(test_X['num1']<test_X['num2'])*(test_X['sz1']<test_X['sz2'])]
        c2 = test_X.index.to_numpy()[(test_X['num1']>test_X['num2'])*(test_X['sz1']>test_X['sz2'])]
        c = np.union1d(c1,c2)
        ic = np.setdiff1d(test_X.index.to_numpy(), c)

    return c, ic



def get_svm_matrix(test_csv, pred_csv):
    """
    Generates a matrix that represents the average congruency and incongruency score for each pair of numbers (num1, num2) in SVM test and prediction data.

    Congruency and incongruency is defined based on the pair of numbers (num1, num2) and size pairs (sz1, sz2). Congruency is defined when:
    1. num1 < num2 and sz1 < sz2, or
    2. num1 > num2 and sz1 > sz2.

    Incongruency is defined when the above conditions are not met. 

    The function generates a matrix where each element at (i, j) for i < j contains the average congruency score for the pair (i, j), and each element at (i, j) for i > j contains the average incongruency score for the pair (j, i). 

    The score is computed as the mean of equal elements between prediction and test results, considering only the congruent or incongruent pairs for the upper and lower triangle, respectively.

    Parameters
    ----------
    test_csv : str
        Path to the CSV file containing the test data. It must contain columns 'num1', 'num2', 'sz1', 'sz2'.
    
    pred_csv : str
        Path to the CSV file containing the prediction data. It should contain a single column '0' with predicted values.

    Returns
    -------
    score_matrix : pandas.DataFrame
        A DataFrame where each cell at (i, j) contains the average congruency/incongruency score for the pair (i, j).
        The score is computed as the mean of equal elements between prediction and test results, 
        considering only the congruent pairs for the upper triangle and the incongruent pairs for the lower triangle.

    Notes
    -----
    If num1 == num2 or sz1 == sz2, those cases are not included in the computation and will appear as NaN values in the returned DataFrame.
    """

    # Load the data
    test = pd.read_csv(test_csv).drop('Unnamed: 0', axis=1)
    pred = pd.read_csv(pred_csv)['0'].to_numpy()
    ans = get_y(pd.read_csv(test_csv).drop('Unnamed: 0', axis=1))  # Assumes that get_y is defined elsewhere
    # Check for equality element-wise:
    equal_elements = np.equal(pred, ans)
    test['equal_elements'] = equal_elements
    # Prepare an empty matrix to hold the result
    score_matrix = np.zeros((10, 10)) * np.nan
    
    # Fill in congruency
    for i in np.arange(10):
        avg_scores=[]
        for j in np.arange(i+1,10):
            cong1 = test[((test['num1']==i)&(test['num2']==j))&(test['sz1']<test['sz2'])].index
            cong2 = test[((test['num1']==j)&(test['num2']==i))&(test['sz1']>test['sz2'])].index
            cong = np.union1d(cong1,cong2)
            equal_elements = np.equal(pred[cong], ans[cong])
            avg_scores.append(np.mean(equal_elements))
        score_matrix[i,i+1:]=avg_scores
            
    # Fill in incongruency
    for i in np.arange(1,10):
        avg_scores=[]
        for j in np.arange(0,i):
            incong1 = test[((test['num1']==j)&(test['num2']==i))&(test['sz1']>test['sz2'])].index
            incong2 = test[((test['num1']==i)&(test['num2']==j))&(test['sz1']<test['sz2'])].index
            incong = np.union1d(incong1,incong2)
            equal_elements = np.equal(pred[incong], ans[incong])
            avg_scores.append(np.mean(equal_elements))
        score_matrix[i,0:i]=avg_scores

    return pd.DataFrame(score_matrix)


def get_random_units(units, num_units, sort_by, percentage_true=1, upper_bound=None, **quadrants):
    if sort_by == 'fit':
        fit = pd.DataFrame(np.array([[unit.coeff1, unit.coeff2, unit.r_sqrd] for unit in units]), 
                           columns=['coeff1', 'coeff2', 'rsqrd'])

        # Define upper bound conditions
        conditions = {}
        for quad, cond in quadrants.items():
            if upper_bound and quad in upper_bound:
                upper_bound_1, upper_bound_2 = upper_bound[quad]
                
                if quad in ['Q1', 'Q3']:
                    bound_conditions = ((fit['coeff1'].abs() <= upper_bound_1), (fit['coeff2'].abs() <= upper_bound_2))
                elif quad == 'Q2':
                    bound_conditions = ((fit['coeff1'] >= upper_bound_1) & (fit['coeff1'] <= 0), (fit['coeff2'].abs() <= upper_bound_2))
                elif quad == 'Q4':
                    bound_conditions = ((fit['coeff1'] <= upper_bound_1) & (fit['coeff1'] >= 0), (fit['coeff2'].abs() <= upper_bound_2))
            else:
                bound_conditions = (True, True)
                
            conditions[quad] = (fit['rsqrd'] > 0.1) & bound_conditions[0] & bound_conditions[1]

    elif sort_by == 'ktau':
        fit = pd.DataFrame(np.array([[unit.kendall_stats['averaged_across_sizes']['tau'],
                                      unit.kendall_stats['averaged_across_numbers']['tau']] for unit in units]),
                           columns=['ktau_num', 'ktau_size'])

        # Define upper bound conditions
        conditions = {}
        for quad, cond in quadrants.items():
            if upper_bound and quad in upper_bound:
                upper_bound_1, upper_bound_2 = upper_bound[quad]
                
                if quad in ['Q1', 'Q3']:
                    bound_conditions = ((fit['ktau_num'].abs() <= upper_bound_1), (fit['ktau_size'].abs() <= upper_bound_2))
                elif quad == 'Q2':
                    bound_conditions = ((fit['ktau_num'] >= upper_bound_1) & (fit['ktau_num'] <= 0), (fit['ktau_size'].abs() <= upper_bound_2))
                elif quad == 'Q4':
                    bound_conditions = ((fit['ktau_num'] <= upper_bound_1) & (fit['ktau_num'] >= 0), (fit['ktau_size'].abs() <= upper_bound_2))
            else:
                bound_conditions = (True, True)
                
            conditions[quad] = bound_conditions[0] & bound_conditions[1]

    else:
        print("Invalid sort_by value. Please use 'fit' or 'ktau'.")
        return None

    # Filter indices based on specified quadrants for true units
    selected_indices_true = []
    for quad, cond in conditions.items():
        if quadrants.get(quad, False):
            selected_indices_true.extend(fit[cond].index.tolist())
    
    # Define conditions for false units
    conditions_false = {quad: ~cond for quad, cond in conditions.items()}
    
    # Filter indices based on specified quadrants for false units
    selected_indices_false = []
    for quad, cond in conditions_false.items():
        if quadrants.get(quad, False):
            selected_indices_false.extend(fit[cond].index.tolist())
    
    # Ensure unique indices
    selected_indices_true = list(set(selected_indices_true))
    selected_indices_false = list(set(selected_indices_false))

    # Determine number of units to sample from each group
    num_true = int(num_units * percentage_true)
    num_false = num_units - num_true
    
    # Check if there are enough units to sample from
    if num_true > len(selected_indices_true):
        print(f"Not enough true units to sample from. Required: {num_true}, Available: {len(selected_indices_true)}")
        return None
    if num_false > len(selected_indices_false):
        print(f"Not enough false units to sample from. Required: {num_false}, Available: {len(selected_indices_false)}")
        return None

    # Randomly sample indices
    random_indices_true = np.random.choice(selected_indices_true, num_true, replace=False) if num_true > 0 else []
    random_indices_false = np.random.choice(selected_indices_false, num_false, replace=False) if num_false > 0 else []

    # Combine and return indices as a NumPy array
    random_indices = np.concatenate([random_indices_true, random_indices_false]).astype(int)
    return random_indices
