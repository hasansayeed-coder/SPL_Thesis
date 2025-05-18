import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.decomposition import PCA
import xgboost as xg
import random
import os
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

DATASET_PATH = "/content/drive/My Drive/Software_Cost_Data/cocomo81.csv"

cols = ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 'acap', 'aexp',
        'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced', 'loc', 'actual']

def read_csv(file_path):
    return pd.read_csv(file_path, names=cols)

def check_null(df):
    print('\nAny null entry: ', df.isnull().values.any())

def describe_dataset(df):
    print('\nDataset info:\n')
    print(df.info())
    print('\nDataset describe:\n', df.describe())

def visualize_boxplot(df):
    for col in df.columns:
        if col != 'actual':
            plt.boxplot(df[col], vert=False)
            plt.title(col)
            label = 'Min: {}, Mean: {:.2f}, Max: {}'.format(
                    df[col].min(), df[col].mean(), df[col].max())
            plt.xlabel(label)
            plt.show()


def visualize_histogram(df):
    for col in df.columns:
        if col != 'actual':
            plt.hist(df[col], bins=10, edgecolor='black')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()


def visualize_scatter(df):
    print('Choose 2 column number to view scatter:')
    for i in range(df.shape[1]):
        print(i, ':', df.columns[i], end='  \t')
    print()
    ch = 'y'
    while ch == 'y' or ch == 'Y':
        print('\nEnter column1 number:')
        col1 = int(input())
        print('\nEnter column2 number:')
        col2 = int(input())
        plt.scatter(df.iloc[:, col1], df.iloc[:, col2])
        plt.xlabel(df.columns[col1])
        plt.ylabel(df.columns[col2])
        plt.show()
        print('\nWant to see another scatter plot? Enter y for Yes or any other key for no:')
        ch = input()

def normalize(df):
    for col in df.columns:
        if col != 'actual':
            max_val = df[col].max()
            min_val = df[col].min()
            df[col] = df[col].apply(lambda x: ((x - min_val)/(max_val - min_val)))
    return df


def pca_transform_data(df, components=9):
    pca = PCA(n_components=components)
    pca.fit(df.drop('actual', axis=1))
    pca_data = pd.DataFrame(pca.transform(df.drop('actual', axis=1))[:, :components])
    return pca_data

def split_data_into_folds(df, pca_data):
    folds = {}
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for counter, (train_index, test_index) in enumerate(kf.split(pca_data)):
        folds[counter] = {'train': train_index, 'test': test_index}
    return folds

def train_test_split_fold(folds, counter, df, pca_data):
    trainX = pca_data.iloc[folds[counter]['train'], :]
    trainY = df['actual'].iloc[folds[counter]['train']]
    testX = pca_data.iloc[folds[counter]['test'], :]
    testY = df['actual'].iloc[folds[counter]['test']]
    return trainX, trainY, testX, testY

def print_stats(predictions, actual, model):
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    mmre = np.mean(np.abs(np.array(actual) - np.array(predictions)) / np.array(actual))
    r2 = sm.r2_score(actual, predictions)
    print(f'{model} - RMSE: {rmse}, MAE: {mae}, MMRE: {mmre}, R2: {r2}')
    return {'rmse': rmse, 'mae': mae, 'mmre': mmre, 'r2': r2}

def run_model(model, folds, df, pca_data, model_name):
    predictions = []
    actual = []
    testCounter = 0
    while testCounter < 10:
        trainX, trainY, testX, testY = train_test_split_fold(folds, testCounter, df, pca_data)
        model.fit(trainX, trainY)
        predictions.extend(model.predict(testX))
        actual.extend(testY)
        testCounter += 1
    return print_stats(predictions, actual, model_name)

def run_all_models(df, pca_data, folds):
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': RidgeCV(),
        'Lasso': LassoCV(),
        'KNN': KNeighborsRegressor(),
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(),
        'XgBoost': xg.XGBRegressor(),
        'DecisionTree': tree.DecisionTreeRegressor(),
    }
    stats = {}
    for name, model in models.items():
        stats[name] = run_model(model, folds, df, pca_data, name)
    return stats

def cocomo_pipeline(file_path):
    df = read_csv(file_path)
    print('\nDataset head:\n', df.head(3))

    check_null(df)
    describe_dataset(df)
    visualize_boxplot(df)
    visualize_histogram(df)

    df_normalized = normalize(df)

    pca_data = pca_transform_data(df_normalized)
    folds = split_data_into_folds(df, pca_data)

    model_stats = run_all_models(df, pca_data, folds)

    print('\nModel stats:', model_stats)

cocomo_pipeline(DATASET_PATH)