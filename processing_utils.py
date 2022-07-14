from tkinter import N
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix,  roc_auc_score
from sklearn import metrics
import math
from xgboost import XGBClassifier
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import jieba
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from optbinning import OptimalBinning
import seaborn as sns

def get_fig(df, label, start):
    ax = sns.barplot(x="model", y=label, hue="text", data=df, palette="Set2", ci=None)
    ax.set_ybound(start)

def binning(df, column, num, method, target, dtype):
    x = df[column[num]]
    y = target
    optb = OptimalBinning(name=column[num], dtype = dtype, solver="cp", prebinning_method = method)
    df[column[num]] = optb.fit_transform(x, y, metric = "indices")


def remove_outliers(df, cont):
    df_before = df.copy().astype(float)
    for col in cont:
        df_col = df_before[col]
        s = df_col.describe()
        q1 = s['25%']
        q3 = s['75%']
        iqr = q3 - q1
        mi = q1 - 1.5 * iqr
        ma = q3 + 1.5 * iqr
        df_before.loc[df_before.index.isin(df_col[(df_col < mi)].index), col] = mi
        df_before.loc[df_before.index.isin(df_col[(df_col > ma)].index), col] = ma
    return df_before

def cat_cont_split(df): 
    feature_cols = df.columns.values.tolist()
    cat = []
    cont = []
    for col in feature_cols:
        n = df[col].nunique()
        if n > 15:
            cont.append(col)
        else:
            cat.append(col)
    return cat, cont

def one_hot_encoder(text):
    word_list = []
    for line in text:
        seg = jieba.lcut(line)
        word_list.append(seg)
    multi_one_hot = MultiLabelBinarizer()
    one_hot_vec = multi_one_hot.fit_transform(word_list)
    return one_hot_vec

def fill_value(df, strategy):
    cols = df.columns
    if strategy == 'mean':
        df_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        df = df_mean.fit_transform(df)
    if strategy == 'median':
        df_median = SimpleImputer(missing_values=np.nan, strategy='median')
        df = df_median.fit_transform(df)
    if strategy == 'most_frequent':
        df_0 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        df = df_0.fit_transform(df)
    df = pd.DataFrame(df, dtype='float')
    df.columns = cols
    return df


def get_metrics(y_test, y_predict_rate):

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_predict_rate)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    precision = precisions[best_f1_score_index]
    recall = recalls[best_f1_score_index]
    threshold = thresholds[best_f1_score_index]
    y_predict = np.where(y_predict_rate > threshold, 1, 0)

    aucprc = metrics.auc(recalls, precisions)
    
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_rate)

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()

    NPV = tn / (fn + tn)            #NPV
    Specificity = tn / (fp + tn)    #Specificity
 
    aucroc = roc_auc_score(y_test, y_predict_rate)

    accuracy = accuracy_score(y_test, y_predict)

    print('precision: %10.5f \t recall: %10.5f \t f1: %10.5f \t accuracy: %10.5f \t aucprc: %10.5f \t aucroc: %10.5f \t NPV: %10.5f \t Specificity: %10.5f \t ' 
    % (precision, recall, best_f1_score, accuracy, aucprc, aucroc, NPV, Specificity))
    return [precision, recall, best_f1_score, accuracy, aucprc, aucroc, NPV, Specificity]

def get_result(results):
    """_summary_

    Args:
        results (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_result = pd.DataFrame(results, columns=['precision', 'recall', 'f1', 'accuracy', 'aucprc', 'aucroc', 'NPV', 'Specificity'])
    df_result_ci = pd.DataFrame([df_result.mean(),df_result.mean() - df_result.std()/math.sqrt(len(df_result))*1.96, df_result.mean() + df_result.std()/math.sqrt(len(df_result))*1.96], index=['均值', '置信区间-左', '置信区间-右'])
    return df_result_ci

def get_model(name):
    if name == 'lasso':
        return Lasso(alpha=0.01, max_iter=100000)
    elif name == 'xgb':
        return XGBClassifier(eval_metric=['logloss','auc','error'])
    elif name == 'RF':
        return RandomForestRegressor(max_depth=1, random_state=625)
    elif name == 'LR':
        return LogisticRegression(random_state=625)

def train(model_name, data, y, X_test, y_test, text_use, text, test):

    if text_use == "text":
        one_hot = one_hot_encoder(text)
        data = pd.concat([pd.DataFrame(data.values) ,pd.DataFrame(one_hot[:13904])], axis=1).values
        X_test = pd.concat([pd.DataFrame(X_test.values) ,pd.DataFrame(one_hot[13904:])], axis=1).values
    else:
        data = data.values
        X_test = X_test.values

    kf = KFold(n_splits=5, shuffle=True, random_state=625)
    results = []
    print('\n' + model_name + '\n')
    for train_index, valid_index in kf.split(data):
        X_train, X_valid = data[train_index], data[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        if test == None:
            X_test = X_valid
            y_test = y_valid
        cls = get_model(model_name)
        cls.fit(X_train, y_train)
        if model_name == 'LR':
            y_predict_rate = cls.predict_proba(X_test)[:,1]
        else:
            y_predict_rate = cls.predict(X_test)
        result = get_metrics(y_test, y_predict_rate)
        results.append(result)
    
    df_result = get_result(results)
    return df_result

def StandardScaler(data):
    data=(data-data.mean())/data.std()
    return data

def time_split(df, text, words_ids, y, time_point):

    df_train = df[:time_point]
    df_test = df[time_point:]
    y_train = y[:time_point]
    y_test = y[time_point:]
    words_ids_train = words_ids[:time_point]
    words_ids_test = words_ids[time_point:]
    text_train = text[:time_point]
    text_test = text[time_point:]

    return df_train, df_test, y_train, y_test, text_train, text_test, words_ids_train, words_ids_test