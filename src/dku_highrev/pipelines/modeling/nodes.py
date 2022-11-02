"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.18.3
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

import joblib
import logging

def prepare_data_for_modeling(df):
    '''Prepare data for modeling
    
    Inputs:
    df: a pandas dataframe with customers data
        
    Outputs:
    df: a pandas dataframe with customers data ready for modeling
        X: a pandas dataframe with customer features
        y: a pandas series with customer labels, encoded as integers (0 = low revenue, 1 = high revenue)

    '''
    # Suppress "a copy of slice from a DataFrame is being made" warning
    pd.options.mode.chained_assignment = None
    
    # prepare dataset for classification
    features = df.columns[1:-1]
    X = df[features]
    y = df['high_revenue']

    # identify numeric features in X
    numeric_features = X.select_dtypes(include=[np.number]).columns
    # identify categorical features in X
    categorical_features = X.select_dtypes(exclude=[np.number]).columns

    # Normalize numeric features with MinMaxScaler
    scaler = MinMaxScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=categorical_features)

    # transform bool in y to 0/1
    y = y.astype(int)
    
    #create a dataframe with the features and the labels
    data_prepared = pd.concat([X, y], axis=1)
    return data_prepared

def split_data(df):
    """Splits data into features and targets training and test sets.

    Inputs:
        df: Data containing features and target.

    Output:
        Split data.
    """
    # split dataset into train and test

    features = df.columns[:-1]
    X = df[features]
    y = df['high_revenue']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):    
    '''Train a model predicting high-revenue customers from features and labels
    
    Inputs:
    X_train: a pandas dataframe with customers features
    y_train: a pandas series with customers labels
        
    Outputs:
    model: a trained model
    
    '''
    
    # Suppress "a copy of slice from a DataFrame is being made" warning
    pd.options.mode.chained_assignment = None

    #model = DecisionTreeClassifier()
    model = LogisticRegression(solver='lbfgs', max_iter=200)
    #model = RandomForestClassifier()
    '''
    model = LogisticRegression(C=0.056, class_weight={}, dual=False, fit_intercept=True,
                intercept_scaling=1, l1_ratio=None, max_iter=1000,
                multi_class='auto', n_jobs=None, penalty='l2',
                random_state=123, solver='lbfgs', tol=0.0001, verbose=0,
                warm_start=False)
    '''
    
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    '''Evaluate a model predicting high-revenue customers from features and labels
    
    Inputs:
    model: a trained model
    X_test: a pandas dataframe with customers features

        
    Outputs:
    None
    
    '''
    
    labels = y_test.unique()
    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)[:,1]
    
    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probas)
    print('ROC AUC: %.3f' % roc_auc)
    print('Accuracy: %.3f' % accuracy)

    #printout the results
    logger = logging.getLogger(__name__)
    logger.info("Model has an accuracy of %.3f on test data.", accuracy)
    logger.info("Model has an ROC AUC of %.3f on test data.", roc_auc)
