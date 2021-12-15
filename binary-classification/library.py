import numpy as np
import scipy.io as sio
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from math import floor
import statistics as s

def count_NaN(data):
    no_NaN_data = data.notna()
    countNaN = np.zeros((np.shape(data)[1]))
    for k in range(np.shape(data)[1]):
        for i in range (np.shape(data)[0]):
            if no_NaN_data.iloc[i,k] == False:
                countNaN[k] +=1
    return countNaN

def replace_NaN(data):
    no_NaN_data = data
    mean_data = np.mean(data)
    count_replaced_values = np.zeros((np.shape(data)[1]))
    for k in data:
        for i in range (len(data[k])):
            va_k = no_NaN_data[k]
            if va_k[i] == np.NaN:
                count_replaced_values[k] += 1
                va_k[i] = mean_data[k] #replace with mean
    return no_NaN_data, count_replaced_values

#In practice we often ignore the shape of the distribution and just transform the data to center it by removing 
# the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.

def center_and_normalize(data):
    center_and_normalize = data
    mean_data, std_data = np.mean(data), np.std(data)
    types_data = data.dtypes
    for k in center_and_normalize:
        if types_data[k] == float: #on se souhaite pas centrée réduire les labels
            center_and_normalize[k] = (center_and_normalize[k] - mean_data[k])*(1/std_data[k])
    return center_and_normalize

def clean_data_f(data):
    data_na=data.notna() #renvoie une dataframe booléen
    data_types=data.dtypes #datatype of each column
    for k in data:
        for i in range(len(data_na[k])):
            #Il est possible que certains string aient des \t ou des " ", il faut les enlever
            if type(data[k][i])==str:
                data.at[i,k]=data[k][i].replace(" ","")
                data.at[i,k]=data[k][i].replace("\t","")
                #Si un des NaN avait ce genre de caractères alors ils n'étaient pas repérés et comptaient
                #Pour une valeur: On modifie donc la table data_na 
                if data[k][i]=="?":
                    data_na.at[i,k]=False
    for index in data:
        if data_types[index]==object:
            clear_data_String(data,index,data_na)
        else:
            if data_types[index]==int:
                clear_data_Float_Int(data,index,int)
            else:
                clear_data_Float_Int(data,index,float)
    data = center_and_normalize(data)
    replace_by_Int(data)
    return data

def clear_data_String(data,k,data_na):
    list_value={}
    data_na=data_na[k]
    for value in range(len(data_na)):
        if data_na[value]:
            if data[k][value] not in list_value:
                list_value[data[k][value]]=0
            else:
                list_value[data[k][value]]+=1
    moy,Max=data[k][0],0
    for value in list_value:
        if list_value[value]>Max:
            Max,moy=list_value[value],value
    for value in range(len(data)):
        if not data_na[value]:
            data.at[value,k]=moy

def clear_data_Float_Int(data,k,int_or_float):
    moy=data[k].mean()
    data_na=data[k].isna()
    if int_or_float==int:
        if moy-floor(moy)<0.5:
                moy=int(moy)
        else:
                moy=int(moy + 1)
    for value in range(len(data_na)):
        if data_na[value]:
            data.at[value,k]=moy

def replace_by_Int(data):
    data_types=data.dtypes #datatype of each column
    for k in data:
        if data_types[k]==object:
            list_value={}
            data_na=data[k].isna()
            number=0
            for value in range(len(data_na)):
                if not data_na[value]:
                    if data[k][value] not in list_value:
                        list_value[data[k][value]]=number
                        number+=1
                    data.at[value,k]=list_value[data[k][value]]
            data[k] = data[k].astype(int)


def split_data_df(data, test_size):
    features = list(data.columns[:-1])
    labels   = data.iloc[:,-1]
    X        = data[features]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size)
    return X_train, X_test, y_train, y_test


def split_data_and_pca(data, test_size):

    features = list(data.columns[:-1])
    labels   = data.iloc[:,-1]
    X        = data[features]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size)

    X_tot = np.concatenate((X_train, X_test))
    pca = PCA(n_components = 0.99)
    pca.fit(X_tot)
    X_tot = pca.transform(X_tot)

    X_train_PCA = X[:len(X_train)]
    X_test_PCA = X[len(X_train):]
    return X_train_PCA, X_test_PCA, y_train, y_test, pca.n_components_


def shapiro_test(data):
    n = np.shape(data)[1]
    p_values = np.zeros((n))
    for k in range(n):
        data_k = data.iloc[:,k]
        shapiro_test = shapiro(data_k)
        p_values[k] = shapiro_test.pvalue
    return p_values



def precision_recall_multilabels(y_true, y_pred, labels):
    recalls = []
    precisions = []
    for label in labels:

        pos_true = y_true == label
        pos_pred = y_pred == label

        # By hand
        # true_pos = pos_pred & pos_true
        # recalls.append(np.sum(true_pos) / np.sum(pos_true))
        # precisions.append(np.sum(true_pos) / np.sum(pos_pred))

        # With sklearn
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))

    return precisions, recalls


def kfold_precisions_recalls(X, y, labels, clf, kf: KFold):
    """Returns the history of precisions and recalls through K-fold training

    Parameters
    ----------
    X, y : data
    labels : list[int]
    clf : classifier
    kf : KFold instance

    Returns
    -------
    precisions : list[list], shape (num_folds, len(labels))
    recalls : list[list], shape (num_folds, len(labels))
    """
    precisions, recalls = [], []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        precisions_, recalls_ = precision_recall_multilabels(y_test, y_pred, labels)

        precisions.append(precisions_)
        recalls.append(recalls_)

    return precisions, recalls

# Once you have filled the `clfs_results` dictionnary below
# call `kfold_multimodels_report(clfs_results)`


def kfold_multimodels_report(clfs_results):
    """
    Prints a report for the results of experiments on multiple models,
    each one evaluated using k-fold cross-validation.

    The results of the experiments should be given as the 'clfs_stats'
    argument, with the following structure:

    {
        "clf_name1": {"metric1": list[list], "metric2": list[list], ...},
        "clf_name2": {"metric1": list[list], "metric2": list[list], ...},
        ...
    }

    with each list[list] being of shape (num_folds, num_classes).
    """
    clfs_stats = kfold_summarize_results(clfs_results)
    with np.printoptions(precision=2, floatmode="fixed"):
        for clf_name, clf_stats in clfs_stats.items():
            print(f"{clf_name:<15}")
            for metric_name, stats in clf_stats.items():
                print(f"{metric_name:>15}")
                for stat_name, data in stats.items():
                    print(f"{stat_name:>20}: {data}")


def kfold_summarize_results(clfs_results):
    """Computes stats on results of multi-models k-folds experiments.

    Takes:

    {
        "clf_name1": {"metric1": list[list], "metric2": list[list], ...},
        "clf_name2": {"metric1": list[list], "metric2": list[list], ...},
        ...
    }

    Returns:

    {
        "clf_name1": {"metric1": {"mean": value, "std": value ...}, ...},
        ...
    }
    """
    clfs_stats = {clf_name: {} for clf_name in clfs_results}
    for clf_name, clf_results in clfs_results.items():
        for metric, data in clf_results.items():
            clfs_stats[clf_name][metric] = {
                "mean": np.mean(data, axis=0),
                "std": np.std(data, axis=0),
            }
    return clfs_stats

def dataset_to_numpy(data, first_col=0, last_col=-1):
    set_of_data=[]
    for col in data.columns:
        tmp_data = data[col]
        set_of_data.append(tmp_data.to_numpy().T)
    return np.vstack(tuple(set_of_data[first_col:last_col])).T


def pca(data):
    pca = PCA(n_components = 0.99)
    X=np.array(data)
    pca.fit(X)
    return pca.transform(X)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score

def Logistic_regression(data,test_size = 0.3):
    X_train, X_test, y_train, y_test = split_data_df(data, test_size)
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
    y_pred = LR.predict(X_test)
    jaccard_scor = jaccard_score(y_test, y_pred, average='weighted')
    f1_scor = f1_score(y_test, y_pred, average='weighted')
    return jaccard_scor, f1_scor

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def KNN(data, k=4, test_size = 0.3):
    #X_train, X_test, y_train, y_test = split_data_pca(data, test_size)
    X_train, X_test, y_train, y_test = split_data_df(data, test_size)
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    Train_set_Accuracy = metrics.accuracy_score(y_train, neigh.predict(X_train))
    Test_set_Accuracy = metrics.accuracy_score(y_test, y_pred)
    return Train_set_Accuracy, Test_set_Accuracy



""" DECISION FOREST - AUBIN JULIEN """

# Import scikit-learn functions
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
# from sklearn.tree import export_graphviz
# from graphviz import Source
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score

# Decision forest
def trainDecisionForest(X_train,y_train,n_trees) :

    # Cross-validation procedure
    cvp = ShuffleSplit(n_splits=100, test_size=1/3, train_size=2/3)

    # Define the max depths between 1 and 20
    n_depths = 20
    depths = np.linspace(1,n_depths,n_depths//2)

    # Loop on the max_depth parameter and compute median Neg_log_loss
    tab_score_tree = np.zeros(n_depths)
    for i in range(len(depths)):
        class_tree = DecisionTreeClassifier(max_depth=depths[i])
        # Accuracy score
        tab_score_tree[i] = s. median(cross_val_score(class_tree, X_train, y_train, scoring='accuracy', cv=cvp)) 
        
        # Neg_log_loss : tab_score_tree[i] = s.median(-cross_val_score(class_tree, X, y, scoring='neg_log_loss', cv=cvp))
        # Need to minimize log_loss - but we have error on dimension
    
    opt = (np.argmax(tab_score_tree) + 1) * 2 # depth for which we get the minimum score -> need to max Accuracy if used

    # Train Decision forest :
    class_forest = RandomForestClassifier(n_estimators=n_trees, max_depth=opt)
    class_forest.fit(X_train, y_train)
    return class_forest


# Ada Boost 
def trainAdaBoost(X,y,n_trees):
    class_adaB = AdaBoostClassifier(n_estimators=n_trees)
    class_adaB.fit(X,y)
    return class_adaB

#Test Decision Forest
def testDecisionForest(DFclf, X_test) :
    class_forest_predict = DFclf.predict(X_test)
    return class_forest_predict


# Test Ada Boost
def testAdaBoost(ABclf, X_test) :
    class_adaB_predict = ABclf.predict(X_test)
    return class_adaB_predict


# Export the tree to "plot_tree.pdf"
def plotTree(class_tree, data):
    plot_tree = export_graphviz(class_tree, out_file=None, feature_names=list(data.columns)[1:], filled=True) #data
    graph = Source(plot_tree) 
    graph.render("class_tree")

    # Plot the tree
    return graph




""" MODEL VALIDATION - AUBIN JULIEN"""

import seaborn as sns

def confusionMatrix(y_test, y_predicted):
    conf_mat = confusion_matrix(y_test, y_predicted)
    conf_map = sns.heatmap(conf_mat, annot=True)
    return conf_map

def validateModel(y_test, y_pred):
    lloss = log_loss(y_test, y_pred, normalize=True)
    acc = accuracy_score(y_test, y_pred, normalize=True)*100
    rec = recall_score(y_test, y_pred, average = 'binary') * 100
    f1 = f1_score(y_test, y_pred)
    print(f'Your model has a log_loss of : {lloss}%\nYour model has an accuracy of : {acc}%\nYour model has a recall of : {rec}%\nYour model has a F1 score = {f1} ')
