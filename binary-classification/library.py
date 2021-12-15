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
    # for k in data:
    #     for i in range(len(data_na[k])):
    #         #Il est possible que certains string aient des \t ou des " ", il faut les enlever
    #         if type(data[k][i])==str:
    #             data.at[i,k]=data[k][i].replace(" ","")
    #             data.at[i,k]=data[k][i].replace("\t","")
    #             #Si un des NaN avait ce genre de caractères alors ils n'étaient pas repérés et comptaient
    #             #Pour une valeur: On modifie donc la table data_na 
    #             if data[k][i]=="?":
    #                 data_na.at[i,k]=False
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



def pca(data):
    pca = PCA(n_components = 0.99)
    X=np.array(data)
    pca.fit(X)
    return pca.transform(X)


#Faire aussi cross validation
def split_data(data, test_size):
    labels = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)
    return X_train, X_test, y_train, y_test

def split_data_df(data, test_size):
    features = list(data.columns[:-1])
    labels   = data.iloc[:,-1]
    X        = data[features]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size)
    return X_train, X_test, y_train, y_test

#En statistique, le test de Shapiro–Wilk teste l'hypothèse nulle selon laquelle un échantillon x 1 , … , x n 
#est issu d'une population normalement distribuée. 
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

def dataset_to_numpy(data):
    set_of_data=[]
    for col in data.columns:
        tmp_data = data[col]
        set_of_data.append(tmp_data.to_numpy().T)
    return np.vstack(tuple(set_of_data[1:-1])).T

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
    X_train, X_test, y_train, y_test = split_data_pca(data, test_size)
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    Train_set_Accuracy = metrics.accuracy_score(y_train, neigh.predict(X_train))
    Test_set_Accuracy = metrics.accuracy_score(y_test, y_pred)
    return Train_set_Accuracy, Test_set_Accuracy