" Overall import "
import numpy as np
import scipy.io as sio
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from math import floor
import statistics as s

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro
from sklearn.model_selection import KFold

" Logistic Regression"
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, svm

" Decision Forest "
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

" Model Validation "
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, log_loss, precision_score


""" PREPROCESSING DATASET - RIVIERE CLEMENT """

def count_NaN(data):
    """Returns the count of NaNs in the Dataframe
    Parameters
    ----------
    data : pandas Dataframe

    Returns
    -------
    countNaN : int
    """
    no_NaN_data = data.notna()
    countNaN = np.zeros((np.shape(data)[1]))
    for k in range(np.shape(data)[1]):
        for i in range (np.shape(data)[0]):
            if no_NaN_data.iloc[i,k] == False:
                countNaN[k] +=1
    return countNaN


def replace_NaN(data):
    """Replace the NaNs of the Dataframe by the mean of the data
    Parameters
    ----------
    data : pandas Dataframe

    Returns
    -------
    no_NaN_data           : pandas Dataframe
    count_replaced_values : int
    """
    no_NaN_data = data
    mean_data = np.mean(data)
    count_replaced_values = np.zeros((np.shape(data)[1]))
    for k in data:
        for i in range (len(data[k])):
            va_k = no_NaN_data[k]
            if va_k[i] == np.NaN:
                count_replaced_values[k] += 1
                va_k[i] = mean_data[k] # replace with mean
    return no_NaN_data, count_replaced_values

#In practice we often ignore the shape of the distribution and just transform the data to center it by removing 
# the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.

def center_and_normalize(data):
    """Center and Normalize a pandas Dataframe
    Parameters
    ----------
    data : pandas Dataframe

    Returns
    -------
    center_and_normalize : pandas Dataframe
    """
    center_and_normalize = data
    mean_data, std_data = np.mean(data), np.std(data)
    types_data = data.dtypes
    for k in center_and_normalize:
        if types_data[k] == float: #on se souhaite pas centrée réduire les labels
            center_and_normalize[k] = (center_and_normalize[k] - mean_data[k])*(1/std_data[k])
    return center_and_normalize


def clean_data_f(data):
    """Cleaning function for the datasets, that center and normalize the data and 
       replace strings by integers
    Parameters
    ----------
    data : pandas Dataframe

    Returns
    -------
    data : pandas Dataframe
    """
    data_na=data.notna() # renvoie une dataframe booléen
    data_types=data.dtypes # datatype of each column
    for k in data:
        for i in range(len(data_na[k])):
            # Il est possible que certains string aient des \t ou des " ", il faut les enlever
            if type(data[k][i])==str:
                data.at[i,k]=data[k][i].replace(" ","")
                data.at[i,k]=data[k][i].replace("\t","")
                # Si un des NaN avait ce genre de caractères alors ils n'étaient pas repérés et comptaient
                # Pour une valeur: On modifie donc la table data_na 
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
    """
    Parameters
    ----------
    data : pandas Dataframe

    Returns
    -------
    data : pandas Dataframe
    """
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
    """
    Parameters
    ----------
    data : pandas Dataframe

    Returns
    -------
    data : pandas Dataframe
    """
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
    """
    Parameters
    ----------
    data : pandas Dataframe

    Returns
    -------
    data : pandas Dataframe
    """
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


""" DATASET MANAGEMENT - SELLAMI AZIZ """

def dataset_to_numpy(data, first_col=0, last_col=-1):
    """Convert dataframe to a numpy array and select the columns to keep
    Parameters
    ----------
    data      : pandas Dataframe
    first_col : (int) first column to select
    last_col  : (int) last column to select

    Returns
    -------
    np.array
    """
    set_of_data=[]
    for col in data.columns:
        tmp_data = data[col]
        set_of_data.append(tmp_data.to_numpy().T)
    return np.vstack(tuple(set_of_data[first_col:last_col])).T


""" SPLIT DATASET & PCA - RIVIERE CLEMENT """

def split_data_df(data, test_size):
    """Split training and testing data from a Dataframe
    Parameters
    ----------
    data      : pandas Dataframe
    test_size : int

    Returns
    -------
    X_train : np.array
    X_test  : np.array
    y_train : np.array
    y_test  : np.array
    """
    features = list(data.columns[:-1])
    labels   = data.iloc[:,-1]
    X        = data[features]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size)
    return X_train, X_test, y_train, y_test


def split_data_and_pca(data, test_size):
    """Split training and testing data from a Dataframe and apply PCA
    Parameters
    ----------
    data      : pandas Dataframe
    test_size : int

    Returns
    -------
    X_train : np.array
    X_test  : np.array
    y_train : np.array
    y_test  : np.array
    """
    features = list(data.columns[:-1])
    labels   = data.iloc[:,-1]
    X        = data[features]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size)

    X_tot = np.concatenate((X_train, X_test))
    pca = PCA(n_components = 0.99)
    pca.fit(X_tot)
    X_tot = pca.transform(X_tot)

    X_train_PCA = X_tot[:len(X_train)]
    X_test_PCA = X_tot[len(X_train):]
    return X_train_PCA, X_test_PCA, y_train, y_test, pca.n_components_


def pca(data):
    """Apply PCA
    Parameters
    ----------
    data      : pandas Dataframe

    Returns
    -------
    np.array
    """
    pca = PCA(n_components = 0.99)
    X=np.array(data)
    pca.fit(X)
    return pca.transform(X)


""" SHAPIRO TEST - RIVIERE CLEMENT """

def shapiro_test(data):
    """Apply Shapiro Test, to test gaussianity of the data
    Parameters
    ----------
    data      : pandas Dataframe

    Returns
    -------
    np.array
    """
    n = np.shape(data)[1]
    p_values = np.zeros((n))
    for k in range(n):
        data_k = data.iloc[:,k]
        shapiro_test = shapiro(data_k)
        p_values[k] = shapiro_test.pvalue
    return p_values

""" MODEL VALIDATION : PRECISION & RECALL - SELLAMI AZIZ """

def precision_recall(y_true, y_pred, labels=[0,1]):
    """Compute precision and recall for each classes
    Parameters
    ----------
    y_true : (np.array) True labels
    y_pred : (np.array) predicted labels

    Returns
    -------
    precisions : list
    recalls : list
    """
    recalls = []
    precisions = []
    for label in labels:
        pos_true = y_true == label
        pos_pred = y_pred == label

        true_pos = pos_pred & pos_true
        recalls.append(np.sum(true_pos) / np.sum(pos_true))
        precisions.append(np.sum(true_pos) / np.sum(pos_pred))
    return precisions, recalls

""" CROSS-VALIDATION - SELLAMI AZIZ """

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

        precisions_, recalls_ = precision_recall(y_test, y_pred, labels)

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
    print("classes:             [   0    1]")
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

def crossValidation(X,y,clfs,classes_labels=[0,1], n_spl=10):
    clfs_results = {clf_name: {"precisions": None, "recalls": None} for clf_name in clfs}
    kf = KFold(n_splits=n_spl, shuffle=True, random_state=34)
    for clf_name, clf in clfs.items():
        precisions, recalls = kfold_precisions_recalls(X, y, classes_labels, clf, kf)
        clfs_results[clf_name]["precisions"] = precisions
        clfs_results[clf_name]["recalls"] = recalls
    kfold_multimodels_report(clfs_results)


""" SVM - SELLAMI AZIZ """
def testSVM(X,y):
    """Test SVM classification with simple classifier
    Parameters
    ----------
    X : (np.array) Features
    y : (np.array) labels

    Returns
    -------
    precisions : float
    recalls : float
    """
    clf = svm.SVC()
    train_size = 100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, shuffle=True
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precisions, recalls = precision_recall(y_test, y_pred)
    print(f"precision for class 0,1 : {[ round(elem, 2) for elem in precisions ] }")
    print(f"recalls   for class 0,1 : {[ round(elem, 2) for elem in recalls ] }")
    return precisions, recalls


""" LOGISTIC REGRESSION - BEN AYED MARWEN """

def Logistic_regression(X_train,X_test,y_train,y_test):
    """Compute Logistic Regression
    Parameters
    ----------
    X_train : np.array
    X_test  : np.array
    y_train : np.array
    y_test  : np.array

    Returns
    -------
    y_pred  : np.array
    """
    LR = LogisticRegression().fit(X_train,y_train)
    y_pred = LR.predict(X_test)
    return y_pred


""" K-NEAREST NEIGHBOUR - BEN AYED MARWEN """

def KNN(data, k=4, test_size = 0.3):
    """Compute KNN
    Parameters
    ----------
    X_train   : data
    k         : int
    test_size : float (<1)

    Returns
    -------
    Train_set_Accuracy  : float
    Test_set_Accuracy   : float
    """
    #X_train, X_test, y_train, y_test = split_data_pca(data, test_size)
    X_train, X_test, y_train, y_test = split_data_df(data, test_size)
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    y_pred = neigh.predict(X_test)
    Train_set_Accuracy = metrics.accuracy_score(y_train, neigh.predict(X_train))
    Test_set_Accuracy = metrics.accuracy_score(y_test, y_pred)
    return Train_set_Accuracy, Test_set_Accuracy


""" DECISION FOREST & ADA BOOST - AUBIN JULIEN """

# Decision forest
# We define first the optimal depth using Cross-validation with accuracy score and the apply a random forest with optimal depth.
def trainDecisionForest(X_train,y_train,n_trees) :
    """Train and return a Decision Forest
    Parameters
    ----------
    X_train   : (np.array) training data
    y_train   : (np.array) training labels
    n_trees   : int

    Returns
    -------
    class_forest  : decision forest classifier
    """
    # Cross-validation procedure
    cvp = ShuffleSplit(n_splits=100, test_size=1/3, train_size=2/3)

    # Define the max depths between 1 and 20
    n_depths = 20
    depths = np.linspace(1,n_depths,n_depths//2)    # Depths are define with a stepsize of 2, to lower time calculation

    # Loop on the max_depth parameter and compute median Neg_log_loss
    tab_score_tree = np.zeros(n_depths)
    for i in range(len(depths)):
        class_tree = DecisionTreeClassifier(max_depth=depths[i])
        # Accuracy score
        tab_score_tree[i] = s. median(cross_val_score(class_tree, X_train, y_train, scoring='accuracy', cv=cvp)) 
        
        # Neg_log_loss : tab_score_tree[i] = s.median(-cross_val_score(class_tree, X, y, scoring='neg_log_loss', cv=cvp))
        # If we use it : need to minimize log_loss - but we have error on dimension
    
    opt = (np.argmax(tab_score_tree) + 1) * 2 # depth for which we get the minimum score -> need to max Accuracy if used
    print("Optimal Depth =", opt)

    # Train Decision forest :
    class_forest = RandomForestClassifier(n_estimators=n_trees, max_depth=opt)
    class_forest.fit(X_train, y_train)
    return class_forest


# Ada Boost 
# Using scikit function, we simply define the Ada Boost classifier.
def trainAdaBoost(X,y,n_trees):
    """Train and return a AdaBoost Classifier
    Parameters
    ----------
    X   : (np.array) training data
    y   : (np.array) training labels
    n_trees   : int

    Returns
    -------
    class_adaB  : AdaBoost classifier
    """
    class_adaB = AdaBoostClassifier(n_estimators=n_trees)
    class_adaB.fit(X,y)
    return class_adaB

#Test Decision Forest & Ada Boost
# With the Random Forest Classifier produced by trainDecisionForest(X_train,y_train,n_trees) or trainAdaBoost(X,y,n_trees),
# we predict the labels of the data.
def testClassifier(classifier, X_test) :
    """Test Decision Forest & Ada Boost
    Parameters
    ----------
    classifier : classifier to test
    X_test     : (np.array) test data

    Returns
    -------
    classifier_predict  : (np.array) prediction
    """
    classifier_predict = classifier.predict(X_test)
    return classifier_predict


""" MODEL VALIDATION - AUBIN JULIEN"""

# Confusion Matrix
# We create the confusion matrix and then return the plot using heatmap.
def confusionMatrix(y_test, y_predicted):
    """Plot a Confusion Matrix
    Parameters
    ----------
    y_test      : (np.array)
    y_predicted : (np.array)

    Returns
    -------
    conf_map  : plot
    """
    conf_mat = confusion_matrix(y_test, y_predicted)
    conf_map = sns.heatmap(conf_mat, annot=True)
    return conf_map

# Model's Metrics
# Using scikit metrics, we calculate various metrics to estimate our model's performance.
def validateModel(y_test, y_pred):
    """Using scikit metrics, we calculate various metrics to estimate our model's performance.
    Parameters
    ----------
    y_test      : (np.array)
    y_predicted : (np.array)
    """
    lloss = log_loss(y_test, y_pred, normalize=True)
    acc = accuracy_score(y_test, y_pred, normalize=True)*100        #Percentage
    rec = recall_score(y_test, y_pred, average = 'binary')*100      #Percentage
    f1 = f1_score(y_test, y_pred)
    print(f'Your model has a log_loss of : {lloss:.2f}%\nYour model has an accuracy of : {acc:.2f}%\nYour model has a recall of : {rec:.2f}%\nYour model has a F1 score = {f1:.2f} ')
