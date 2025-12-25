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
logger=setup_logging('data_balance')
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle
from model_training import all_models

def data_balancing(X_train,y_train,X_test,y_test):
    try:
        logger.info(f"Before Number of Rows for Good class : {sum(y_train == 1)}")
        logger.info(f"Before Number of Rows for Bad class : {sum(y_train == 0)}")
        sm_res = SMOTE(random_state=42)
        X_train_bal,y_train_bal = sm_res.fit_resample(X_train,y_train)
        logger.info(f"After Number of Rows for Good class : {sum(y_train_bal == 1)}")
        logger.info(f"After Number of Rows for Bad class : {sum(y_train_bal == 0)}")
        logger.info(f"{X_train_bal.shape}")
        logger.info(f"{y_train_bal.shape}")
        logger.info(f"{X_train_bal.sample(10)}")
        sc = StandardScaler()
        sc.fit(X_train_bal)
        X_train_bal_scaled = sc.transform(X_train_bal)
        X_test_scaled = sc.transform(X_test)
        logger.info(f"{X_train_bal_scaled}")
        with open('scalar.pkl', 'wb') as f:
            pickle.dump(sc, f)
        all_models(X_train_bal_scaled,y_train_bal,X_test_scaled,y_test)
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
