import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger=setup_logging('model_training')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
import pickle

def knn(X_train,y_train,X_test,y_test):
    try:
        global knn_fpr,knn_tpr
        knn_reg = KNeighborsClassifier(n_neighbors=5)
        knn_reg.fit(X_train,y_train)
        logger.info(f'Test accuracy of KNN :{accuracy_score(y_test,knn_reg.predict(X_test))}')
        logger.info(f'confusion matrix of KNN:\n{confusion_matrix(y_test,knn_reg.predict(X_test))}')
        logger.info(f'classification report:\n{classification_report(y_test,knn_reg.predict(X_test))}')
        predictions = knn_reg.predict_proba(X_test)[:,1]
        knn_fpr, knn_tpr, knn_thre = roc_curve(y_test,predictions)
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
def lg(X_train,y_train,X_test,y_test):
    try:
        global lg_fpr, lg_tpr
        lg_reg = LogisticRegression()
        lg_reg.fit(X_train,y_train)
        logger.info(f'Test accuracy of lg :{accuracy_score(y_test,lg_reg.predict(X_test))}')
        logger.info(f'confusion matrix of lg:\n{confusion_matrix(y_test,lg_reg.predict(X_test))}')
        logger.info(f'classification report:\n{classification_report(y_test,lg_reg.predict(X_test))}')
        predictions = lg_reg.predict_proba(X_test)[:,1]
        lg_fpr,lg_tpr, lg_thre= roc_curve(y_test,predictions)
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
def nb(X_train,y_train,X_test,y_test):
    try:
        global nb_fpr, nb_tpr
        nb_reg = GaussianNB()
        nb_reg.fit(X_train,y_train)
        logger.info(f'Test accuracy of nb :{accuracy_score(y_test,nb_reg.predict(X_test))}')
        logger.info(f'confusion matrix of nb:\n{confusion_matrix(y_test,nb_reg.predict(X_test))}')
        logger.info(f'classification report:\n{classification_report(y_test,nb_reg.predict(X_test))}')
        predictions = nb_reg.predict_proba(X_test)[:,1]
        nb_fpr, nb_tpr,nb_thre = roc_curve(y_test,predictions)
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
def dt(X_train,y_train,X_test,y_test):
    try:
        global dt_fpr, dt_tpr
        dt_reg = DecisionTreeClassifier(criterion='entropy')
        dt_reg.fit(X_train,y_train)
        logger.info(f'Test accuracy of dt :{accuracy_score(y_test, dt_reg.predict(X_test))}')
        logger.info(f'confusion matrix of dt:\n{confusion_matrix(y_test, dt_reg.predict(X_test))}')
        logger.info(f'classification report:\n{classification_report(y_test,dt_reg.predict(X_test))}')
        predictions = dt_reg.predict_proba(X_test)[:,1]
        dt_fpr, dt_tpr,dt_thre = roc_curve(y_test,predictions)
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
def rf(X_train,y_train,X_test,y_test):
    try:
        global rf_fpr, rf_tpr
        rf_reg = RandomForestClassifier(n_estimators=5, criterion='entropy')
        rf_reg.fit(X_train, y_train)
        logger.info(f'Test accuracy of rf :{accuracy_score(y_test,rf_reg.predict(X_test))}')
        logger.info(f'confusion matrix of rf:\n{confusion_matrix(y_test,rf_reg.predict(X_test))}')
        logger.info(f'classification report:\n{classification_report(y_test,rf_reg.predict(X_test))}')
        predictions = rf_reg.predict_proba(X_test)[:,1]
        rf_fpr, rf_tpr,rf_ther = roc_curve(y_test,predictions)
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
def ada(X_train,y_train,X_test,y_test):
    try:
        global ada_fpr, ada_tpr
        t = LogisticRegression()
        ada_reg = AdaBoostClassifier(estimator=t, n_estimators=5)
        ada_reg.fit(X_train, y_train)
        logger.info(f'Test accuracy of ada :{accuracy_score(y_test, ada_reg.predict(X_test))}')
        logger.info(f'confusion matrix of ada:\n{confusion_matrix(y_test, ada_reg.predict(X_test))}')
        logger.info(f'classification report:\n{classification_report(y_test, ada_reg.predict(X_test))}')
        predictions = ada_reg.predict_proba(X_test)[:,1]
        ada_fpr, ada_tpr , ada_ther= roc_curve(y_test,predictions)
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
def gb(X_train,y_train,X_test,y_test):
    try:
        global gb_fpr, gb_tpr,gb_reg
        gb_reg = GradientBoostingClassifier(n_estimators=5)
        gb_reg.fit(X_train, y_train)
        logger.info(f'Test accuracy of gb :{accuracy_score(y_test, gb_reg.predict(X_test))}')
        logger.info(f'confusion matrix of gb:\n{confusion_matrix(y_test, gb_reg.predict(X_test))}')
        logger.info(f'classification report:\n{classification_report(y_test, gb_reg.predict(X_test))}')
        predictions = gb_reg.predict_proba(X_test)[:,1]
        gb_fpr, gb_tpr,gb_ther = roc_curve(y_test,predictions)
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
def xgb(X_train,y_train,X_test,y_test):
    try:
        global xgb_fpr, xgb_tpr
        xgb_reg = XGBClassifier()
        xgb_reg.fit(X_train, y_train)
        logger.info(f'Test accuracy of xgb :{accuracy_score(y_test,xgb_reg.predict(X_test))}')
        logger.info(f'confusion matrix of xgb:\n{confusion_matrix(y_test, xgb_reg.predict(X_test))}')
        logger.info(f'classification report:\n{classification_report(y_test, xgb_reg.predict(X_test))}')
        predictions = xgb_reg.predict_proba(X_test)[:,1]
        xgb_fpr, xgb_tpr,xgb_ther = roc_curve(y_test,predictions)
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
def svm(X_train,y_train,X_test,y_test):
    try:
        global svm_fpr, svm_tpr
        svm_reg = SVC(kernel='rbf', probability=True)
        svm_reg.fit(X_train, y_train)
        logger.info(f'Test accuracy of svm :{accuracy_score(y_test,svm_reg.predict(X_test))}')
        logger.info(f'confusion matrix of svm:\n{confusion_matrix(y_test,svm_reg.predict(X_test))}')
        logger.info(f'classification report:\n{classification_report(y_test,svm_reg.predict(X_test))}')
        predictions = svm_reg.predict_proba(X_test)[:,1]
        svm_fpr, svm_tpr,svm_ther = roc_curve(y_test,predictions)
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')


def plot():
    try:
        plt.figure(figsize=(5, 3))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(knn_fpr, knn_tpr, label='knn')
        plt.plot(lg_fpr, lg_tpr, label='lg')
        plt.plot(nb_fpr, nb_tpr, label='nb')
        plt.plot(dt_fpr, dt_tpr, label='dt')
        plt.plot(rf_fpr, rf_tpr, label='rf')
        plt.plot(ada_fpr, ada_tpr, label='ada')
        plt.plot(gb_fpr, gb_tpr, label='gb')
        plt.plot(xgb_fpr, xgb_tpr, label='xgb')
        #plt.plot(svm_fpr, svm_tpr, label='svm')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title("ROC Curve - ALL Models")
        plt.legend()
        plt.show()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

def all_models(X_train,y_train,X_test,y_test):
    try:
        logger.info(f'{X_train.shape}')
        logger.info(f'{X_test.shape}')
        logger.info(f'{y_train.shape}')
        logger.info(f'{y_test.shape}')
        logger.info('==============KNN=================')
        knn(X_train,y_train,X_test,y_test)
        logger.info('==============LG=================')
        lg(X_train,y_train,X_test,y_test)
        logger.info('==============NB=================')
        nb(X_train,y_train,X_test,y_test)
        logger.info('==============DT=================')
        dt(X_train,y_train,X_test,y_test)
        logger.info('==============RF=================')
        rf(X_train,y_train,X_test,y_test)
        logger.info('==============ADA=================')
        ada(X_train,y_train,X_test,y_test)
        logger.info('==============GB=================')
        gb(X_train,y_train,X_test,y_test)
        logger.info('==============XGB=================')
        xgb(X_train,y_train,X_test,y_test)
        #logger.info('==============SVM=================')
        # svm(X_train,y_train,X_test,y_test)
        plot()
        with open('credit_card.pkl', 'wb') as f:
            pickle.dump(gb_reg, f)

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
