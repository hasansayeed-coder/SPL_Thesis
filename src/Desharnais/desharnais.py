import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import numpy as np
from scipy.stats import norm
import math
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge , Lasso , LassoCV
import numpy as np
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold , cross_val_score
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error
import sklearn.metrics as sm
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
import warnings
from matplotlib.pyplot import figure
warnings.filterwarnings('ignore')
from google.colab import drive


drive.mount('/content/drive')

DATASET_PATH = "/content/drive/My Drive/Software_Cost_Data/desharnais.csv"


def readCsv(path) :
  columns = ['Proj', 'TeamExp', 'ManagerExp', 'YearEnd', 'Len', 'Effort', 'Transac', 'Entities', 'PtsNonAdjust',
               'Adjust', 'PtsAjust', 'Lang']
  df = pd.read_csv(path , names=columns)

  columns[columns.index('Effort')] = columns[columns.index('Lang')]
  columns[len(columns) - 1] = 'Effort'
  df = df[columns]
  print(df.head(1))
  return df

def read_and_preprocess(path):
    df = pd.read_csv(path)
    print("Initial data preview:")
    print(df.head(3))

    print("\nColumn names and types:")
    print(df.dtypes)

    if 'Proj' in df.columns:
        df = df.drop('Proj', axis=1)
    if 'Lang' in df.columns:
        df = df.drop('Lang', axis=1)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if df.isnull().values.any():
        print("\nNull values found, filling with 0")
        df.fillna(0, inplace=True)

    print("\nData after preprocessing:")
    print(df.head(3))
    print("\nData types after preprocessing:")
    print(df.dtypes)

    df.columns = df.columns.str.strip().str.lower()

    return df

def checkNullFill(df) :
  print("\nDoes the data contain null values:" , df.isnull().values.any() , end="\n\n")


  df['teamexp'] = pd.to_numeric(df['teamexp'] , errors='coerce')
  df['managerexp'] = pd.to_numeric(df['managerexp'] , errors='coerce')
  df.fillna(0 , inplace = True)

def describeData(df):
    desc = df.describe()  
    numeric_cols = desc.columns

    for col in numeric_cols:
        eachColStats = pd.DataFrame(desc[col]).transpose()
        print(f"Statistics for attribute '{col}':\n{eachColStats}\n")

    print("\nDatatype of each column:\n")
    print(df.info())


def boxPlot(df , cols) :
  for i in range(0 , len(cols) , 3) :
    df[cols[i:i + 3]].plot.box(subplots = True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def barPlot(df, cols):
    for i in range(0, len(cols), 4):
        current_cols = cols[i:i + 4]
        n = len(current_cols)

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))  
        
        if n == 1:
            axes = [axes]

        for ax, col in zip(axes, current_cols):
            ax.hist(df[col].dropna(), bins=10, edgecolor="black")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

        fig.tight_layout(pad=2.0)
        plt.show()


def scatterPlotVsEffort(df):
    effort_col = 'effort'
    num_cols = df.shape[1]
    for j in range(0, num_cols, 4):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
        axes[0].scatter(df[effort_col], df.iloc[:, j], color='red')
        axes[0].set_xlabel(effort_col)
        axes[0].set_ylabel(df.columns[j])

        if j + 1 < num_cols:
            axes[1].scatter(df[effort_col], df.iloc[:, j + 1], color='red')
            axes[1].set_xlabel(effort_col)
            axes[1].set_ylabel(df.columns[j + 1])
        else:
            axes[1].set_visible(False)

        if j + 2 < num_cols:
            axes[2].scatter(df[effort_col], df.iloc[:, j + 2], color='red')
            axes[2].set_xlabel(effort_col)
            axes[2].set_ylabel(df.columns[j + 2])
        else:
            axes[2].set_visible(False)

        if j + 3 < num_cols:
            axes[3].scatter(df[effort_col], df.iloc[:, j + 3], color='red')
            axes[3].set_xlabel(effort_col)
            axes[3].set_ylabel(df.columns[j + 3])
        else:
            axes[3].set_visible(False)

        fig.tight_layout()
        plt.show()

def scatterAmongAll(df):
  print("Want to see all scatter plots between every attribute ?")
  ans = input()

  if ans.lower() == "yes" or ans.lower() == "y" :
    print("\nOptions : \n1. View each pair plot in detail \n2. View zoomed out pairwise plot")
    ans2 = int(input())
    if ans2 == 1 :
      for i in range(0 , df.shape[1]) :
        for j in range(0 , df.shape[1] , 3) :
          fig , axes = plt.subplots(nrows=1 , ncols=3, figsize=(12,3))
          axes[0].scatter(df.iloc[: , i] , df.iloc[:,j])
          axes[0].set_xlabel(df.columns[i])
          axes[0].set_ylabel(df.columns[j])

          # Check if j+1 is within bounds before accessing
          if j + 1 < df.shape[1]:
            axes[1].scatter(df.iloc[:,i] , df.iloc[:,j+1])
            axes[1].set_xlabel(df.columns[i])
            axes[1].set_ylabel(df.columns[j+1])
          else:
            axes[1].set_visible(False)  # Hide if out of bounds

          # Check if j+2 is within bounds before accessing
          if j + 2 < df.shape[1]:
            axes[2].scatter(df.iloc[:, i] , df.iloc[: , j+2])
            axes[2].set_xlabel(df.columns[i])
            axes[2].set_ylabel(df.columns[j+2])
          else:
            axes[2].set_visible(False)  # Hide if out of bounds

          fig.tight_layout()
          plt.show()

    elif ans2 == 2 :
      sns.pairplot(df)


def plotMetrics(model, metric, res, color):
    typeModel = list(res.keys())
    metricValues = list(res.values())

    # Filter out None values from metricValues
    metricValues = [val for val in metricValues if val is not None]
    # Filter out corresponding typeModel values
    typeModel = [typeModel[i] for i, val in enumerate(metricValues) if val is not None]

    # Check if any metric values are left after filtering
    if not metricValues:
        print(f"No values for {model} - {metric}")
        return  # Exit early if no values

    plt.figure(figsize=(9, 5))
    low = min(metricValues)
    high = max(metricValues)
    title = model + " " + metric.upper()

    if metric not in ["r2", "mmre"]:
        plt.ylim([math.ceil(low - 0.5 * (high - low)), math.ceil(high + 0.1 * (high - low))])

    # Ensure typeModel and metricValues have the same length
    plt.bar(range(len(metricValues)), metricValues, tick_label=typeModel, color=color, edgecolor="black", width=0.7)
    plt.title(title)
    plt.show()


def plotMetricsDriver(forPlotMetrics):
    modelNames = [
        "LinearRegression",
        "LassoRegression",
        "RidgeRegression",
        "KNNRegression",
        "SVRRegression",
        "RandomForestRegression",
        "GradientBoostingRegression",
        "BayesianRidgeRegression"
    ]
    metrics = ["r2", "rmse", "mae", "mmre"]
    colors = ['orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta', 'brown']  # Extended colors for all models

    for i in range(len(modelNames)):
        for eachMetric in metrics:
            res = {}
            for eachType, eachTypeVal in forPlotMetrics.items():
                # Safely get metrics for new models, skip if missing
                if modelNames[i] in eachTypeVal:
                    res[eachType] = eachTypeVal[modelNames[i]][eachMetric]
                else:
                    res[eachType] = None  # or np.nan if you prefer
            plotMetrics(modelNames[i], eachMetric, res, colors[i])


def barPlotResults(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_bars = len(data)
    bar_width = total_width / n_bars
    bars = []
    for i, (name, values) in enumerate(data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        for x, y in enumerate(values):
            # Check if y is None and replace with 0 if it is
            if y is None:
                y = 0
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
        bars.append(bar[0])
    if legend:
        ax.legend(bars, data.keys(),loc=(1.04, 0.72))


def barPlotsResultsDriver(forPlotMetrics):
    arr = {}
    metrics = ["r2", "rmse", "mae", "mmre"]
    modelNames = [
        "LinearRegression",
        "LassoRegression",
        "RidgeRegression",
        "KNNRegression",
        "SVRRegression",
        "RandomForestRegression",
        "GradientBoostingRegression",
        "BayesianRidgeRegression"
    ]

    for i in range(len(metrics)):
        buf = {model: [] for model in modelNames}

        for eachKey, valDict in forPlotMetrics.items():
            for model in modelNames:
                # Safely append metric if model exists for this result type
                if model in valDict:
                    buf[model].append(valDict[model][metrics[i]])
                else:
                    buf[model].append(None)  # or np.nan if you want

        arr[metrics[i]] = buf

    for eachKey, eachVal in arr.items():
        fig, ax = plt.subplots()
        barPlotResults(ax, eachVal, total_width=.8, single_width=.9)
        plt.title(eachKey.upper())
        plt.tick_params(axis='x', bottom=False, labelbottom=False)
        x_label = f"{' '*4}all_features{' '*7}normalized{' '*8}select_features{' '*9}PCA"
        plt.xlabel(x_label, horizontalalignment='left', x=0.01)
        plt.show()


def correlationData(df) :
  pd.set_option('display.expand_frame_repr' , False)
  print(f"Correlation matrix:\n{df.corr().round(decimals=1)}")

  f,ax = plt.subplots(figsize=(15,15))
  sns.heatmap(df.corr(), annot=True , linewidths=.5 , fmt='.2f')
  plt.show()


def effortDistribution(df) :
  plt.figure(figsize=(10 , 6))
  plt.subplot(1,2,1)
  # Changed 'Effort' to 'effort' to match the lowercase column name
  sns.distplot(df['effort'] , fit=norm)

  plt.subplot(1 , 2, 2)
  # Changed 'Effort' to 'effort' to match the lowercase column name
  res = stats.probplot(df['effort'] , plot=plt)
  plt.subplots_adjust(wspace=0.4)
  plt.show()

def normalize(df):
    x = df.values
    y = df['effort'] # Changed from 'Effort' to 'effort'
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_norm = pd.DataFrame(x_scaled)
    df_norm.columns = df.columns
    if 'Proj' in df_norm.columns: df_norm = df_norm.drop('Proj', axis=1)
    df_norm = df_norm.drop('effort', axis=1) # Changed from 'Effort' to 'effort'
    return df_norm, y

def LinReg(x_train , y_train , x_test) :
  model_LR = LinearRegression()
  model_LR.fit(x_train , y_train)
  y_test_pred = model_LR.predict(x_test)
  return y_test_pred

def LassoReg(x_train , y_train , x_test) :
  alphas = [0.01 , 0.1 , 1 , 10 , 100]
  cv_mse = []

  for each in alphas :
    model_Lasso = Lasso(alpha=each)
    ten_fold_mse = TenFold(model_Lasso , x_train , y_train)
    cv_mse.append(ten_fold_mse)

  ind = cv_mse.index(min(cv_mse))
  model_Lasso = Lasso(alpha=alphas[ind])
  model_Lasso.fit(x_train , y_train)
  y_test_pred = model_Lasso.predict(x_test)
  return y_test_pred

def LassoRegCV(x_train, y_train, x_test):
    model_lassocv = LassoCV(alphas=None, cv=10, max_iter=100000)
    model_lassocv.fit(x_train, y_train)
    y_test_pred = model_lassocv.predict(x_test)
    return y_test_pred

def RidgeReg(x_train , y_train , x_test) :
  alphas = [0.01 , 0.1 , 1 , 10 , 100]
  cv_mse = []

  for each in alphas :
    model_ridge = Ridge(alpha=each)
    ten_fold_mse = TenFold(model_ridge , x_train , y_train)
    cv_mse.append(ten_fold_mse)

  ind = cv_mse.index(min(cv_mse))
  model_ridge = Ridge(alpha=alphas[ind])
  model_ridge.fit(x_train , y_train)
  y_test_pred = model_ridge.predict(x_test)
  return y_test_pred


def SVRReg(x_train, y_train, x_test):
    model_svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    model_svr.fit(x_train, y_train)
    y_test_pred = model_svr.predict(x_test)
    return y_test_pred

def RandomForestReg(x_train, y_train, x_test):
    model_rf = RandomForestRegressor(n_estimators=100, random_state=30)
    model_rf.fit(x_train, y_train)
    y_test_pred = model_rf.predict(x_test)
    return y_test_pred

def GradientBoostingReg(x_train, y_train, x_test):
    model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=30)
    model_gb.fit(x_train, y_train)
    y_test_pred = model_gb.predict(x_test)
    return y_test_pred

def BayesianRidgeReg(x_train, y_train, x_test):
    model_br = BayesianRidge()
    model_br.fit(x_train, y_train)
    y_test_pred = model_br.predict(x_test)
    return y_test_pred

def knnReg(x_train, y_train, x_test, k):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    ten_fold_mse = TenFold(model, x_train, y_train)
    return y_test_pred

def mmre(y_actual , y_pred):
  if isinstance(y_actual , np.ndarray) :
    y_actual = np.array(y_actual.tolist())
  else :
    y_actual = np.array(y_actual)

  y_subtracted = abs(np.subtract(y_actual , y_pred)) / y_actual

  mmreSum = 0
  for i in range(len(y_actual)) :
    if(y_actual[i] != 0) :
      mmreSum += abs(y_actual[i] - y_pred[i]) / y_actual[i]
    else :
      mmreSum += abs(y_actual[i] - y_pred[i])

  mmre = mmreSum / len(y_actual)
  return mmre

def KNNBest(x_train , y_train , x_test , k=12):
  knn_k = []
  cv_mse = []

  for k in range(2 , k+1 , 2) :
    model = KNeighborsRegressor(n_neighbors=k)
    ten_fold_mse = TenFold(model , x_train , y_train)
    cv_mse.append(ten_fold_mse)
    knn_k.append(k)

  minCvRmse = min(cv_mse)
  minK = knn_k[cv_mse.index(minCvRmse)]
  model = KNeighborsRegressor(n_neighbors=minK)
  model.fit(x_train , y_train)
  y_test_pred = model.predict(x_test)

  return y_test_pred

def pca_calc(comp, x, y):
    res = {}
    pca = PCA(n_components=comp)
    pca.fit(x)
    num_Of_PCA_Comps = ['Comp_1', 'Comp_2', 'Comp_3', 'Comp_4', 'Comp_5', 'Comp_6', 'Comp_7', 'Comp_8', 'Comp_9',
                        'Comp_10']
    df_pca = pd.DataFrame(pca.transform(x), columns=num_Of_PCA_Comps[:comp])

    y_pca = y
    x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(df_pca, y_pca, test_size=0.25, random_state=30)

    res['LinearRegression'] = LinReg(x_train_pca, y_train_pca, x_test_pca)
    res['LassoRegression'] = LassoReg(x_train_pca, y_train_pca, x_test_pca)
    res['RidgeRegression'] = RidgeReg(x_train_pca, y_train_pca, x_test_pca)
    res['LassoCVRegression'] = LassoRegCV(x_train_pca, y_train_pca, x_test_pca)
    res['KNNRegression'] = KNNBest(x_train_pca, y_train_pca, x_test_pca)

    return res, y_test_pca

def TenFold(model , X , y):
  cv = KFold(n_splits=10 , random_state=30, shuffle=True)
  score = cross_val_score(model , X , y , scoring='neg_mean_squared_error',cv=cv)
  return math.sqrt(sum(-1 * score)/len(X))

def calcMetrics(ytest , ypred) :
  res = {}
  res['r2'] = round(r2_score(ytest , ypred) , 4)
  res['rmse'] = round(math.sqrt(mean_squared_error(ytest , ypred)) , 4)
  res['mae'] = round(mean_absolute_error(ytest , ypred) , 4)
  res['mmre'] = round(mmre(ytest , ypred) , 5)

  return res


def printMetrics(modelDetails , res , misc=""):
  print(f"\nFor model {modelDetails} {misc}:")

  for key , value in res.items():
    print(f"{key} : {value}")


def pcaPeakRecalc(resultMetrics, newResults):
    for model, vals in resultMetrics.items():
        for eachMetric, val in resultMetrics[model].items():
            if eachMetric == "r2":
                if resultMetrics[model]["r2"] < newResults[model]["r2"]:
                    resultMetrics[model]["r2"] = newResults[model]["r2"]
            else:
                if resultMetrics[model][eachMetric] > newResults[model][eachMetric]:
                    resultMetrics[model][eachMetric] = newResults[model][eachMetric]

def allDataModels(df):
    X = df.loc[:, df.columns != 'effort'] # Change 'Effort' to 'effort'
    Y = df['effort'] # Change 'Effort' to 'effort'
    resultMetrics = {}
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=30)

    models = {
        "LinearRegression": LinReg,
        "LassoRegression": LassoReg,
        "RidgeRegression": RidgeReg,
        "KNNRegression": KNNBest ,
        "SVRRegression": SVRReg,
        "RandomForestRegression": RandomForestReg,
        "GradientBoostingRegression": GradientBoostingReg,
        "BayesianRidgeRegression": BayesianRidgeReg
    }

    for modelName, model in models.items():
        y_test_pred = models[modelName](x_train, y_train, x_test)
        resMetrics = calcMetrics(y_test, y_test_pred)
        modelStr, modelStrDetails = modelName, "with all data and no normalization"
        printMetrics(modelStr, resMetrics, modelStrDetails)
        resultMetrics[modelName] = resMetrics
    return resultMetrics

def allFeatNormalized(df):
    x, y = normalize(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=30)
    resultMetrics = {}

    models = {
        "LinearRegression": LinReg,
        "LassoRegression": LassoReg,
        "RidgeRegression": RidgeReg,
        "KNNRegression": KNNBest ,
        "SVRRegression": SVRReg,
        "RandomForestRegression": RandomForestReg,
        "GradientBoostingRegression": GradientBoostingReg,
        "BayesianRidgeRegression": BayesianRidgeReg
    }
    
    for modelName, model in models.items():
        y_test_pred = models[modelName](x_train, y_train, x_test)
        resMetrics = calcMetrics(y_test, y_test_pred)
        modelStr, modelStrDetails = modelName, "with all features and normalization"
        printMetrics(modelStr, resMetrics, modelStrDetails)
        resultMetrics[modelName] = resMetrics
    return resultMetrics


def selectFeats(df):
    # Changed column names to lowercase to match DataFrame
    # Added .str.strip() to remove any leading/trailing spaces
    max_corr_features = [col.strip() for col in ['len', 'transac', 'entities', 'ptsnonadjust', 'ptsajust', 'effort']]
    xfeats, y = normalize(df[[col for col in max_corr_features if col in df.columns]])  # Select only if they exist
    x_train, x_test, y_train, y_test = train_test_split(xfeats, y, test_size=0.25, random_state=30)
    resultMetrics = {}

    models = {
        "LinearRegression": LinReg,
        "LassoRegression": LassoReg,
        "RidgeRegression": RidgeReg,
        "KNNRegression": KNNBest ,
        "SVRRegression": SVRReg,
        "RandomForestRegression": RandomForestReg,
        "GradientBoostingRegression": GradientBoostingReg,
        "BayesianRidgeRegression": BayesianRidgeReg
    }

    for modelName, model in models.items():
        y_test_pred = models[modelName](x_train, y_train, x_test)
        resMetrics = calcMetrics(y_test, y_test_pred)
        modelStr, modelStrDetails = modelName, "with select features and normalization"
        printMetrics(modelStr, resMetrics, modelStrDetails)
        resultMetrics[modelName] = resMetrics
    return resultMetrics

def pcaPlot(df, components):
    pca = PCA(n_components=10)
    df = df.loc[:, df.columns != 'Effort']
    pca_fit = pca.fit(df)

    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Variance')
    plt.show()

def pcaTrain(comps, df):
    x, y = normalize(df)
    resultMetrics = {}
    result, y_test = pca_calc(comps, x, y)
    for model, modelPred in result.items():
        resMetrics = calcMetrics(y_test, modelPred)
        modelStrDetails = "with normalization and PCA components for training"
        printMetrics(model, resMetrics, modelStrDetails)
        resultMetrics[model] = resMetrics
    return resultMetrics

def effortDistribution(df) :
  plt.figure(figsize=(10 , 6))
  plt.subplot(1,2,1)
  # Changed 'Effort' to 'effort' to match the lowercase column name
  sns.distplot(df['effort'] , fit=norm)

  plt.subplot(1 , 2, 2)
  # Changed 'Effort' to 'effort' to match the lowercase column name
  res = stats.probplot(df['effort'] , plot=plt)
  plt.subplots_adjust(wspace=0.4)
  plt.show()

def visualization(df):
  cols = df.select_dtypes(include='number').columns
  if len(cols) == 0:
    print("No numeric columns to visualize.")
    return
  boxPlot(df, cols)
  barPlot(df, cols)
  scatterPlotVsEffort(df)
  scatterAmongAll(df)
  correlationData(df)
  effortDistribution(df)


def models(df):
    forPlotMetrics = {}
    forPlotMetrics['all_features'] = allDataModels(df)
    forPlotMetrics['all_feats_normalized'] = allFeatNormalized(df)
    forPlotMetrics['select_features'] = selectFeats(df)
    pcaPlot(df, df.shape[1])

    pcaPeakResults = {}
    pcaPeakResults = pcaTrain(3, df)

    for i in range(3, 10):
        print(f"\n*******Using {i} principal components:*******")
        newRes = pcaTrain(i, df)
        pcaPeakRecalc(pcaPeakResults, newRes)

    forPlotMetrics['PCA'] = pcaPeakResults
    plotMetricsDriver(forPlotMetrics)
    barPlotsResultsDriver(forPlotMetrics)

df = read_and_preprocess(DATASET_PATH)
checkNullFill(df)
describeData(df)
visualization(df)
models(df)

###Deep Learning Model for the Desharnais Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings
from google.colab import drive

# Suppress warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
drive.mount('/content/drive')

# Data Preprocessing
def read_and_preprocess(path):
    df = pd.read_csv(path)

    # Clean column names BEFORE dropping
    df.columns = df.columns.str.strip().str.lower()  # Clean column names first

    # Now drop non-numeric columns using the cleaned names
    # Check if 'proj' and 'lang' exist before dropping to avoid KeyError
    cols_to_drop = []
    if 'proj' in df.columns:
        cols_to_drop.append('proj')
    if 'lang' in df.columns:
        cols_to_drop.append('lang')

    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)
    else:
        print("Warning: Neither 'proj' nor 'lang' (case-insensitive, stripped) found in columns.")
        print("Current columns:", df.columns.tolist())


    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric
    df.fillna(0, inplace=True)  # Handle missing values

    return df

# Normalizing data
def normalize(df):
    # Check if 'effort' column exists before dropping
    if 'effort' not in df.columns:
        raise KeyError("'effort' column not found in the DataFrame.")

    x = df.drop(columns=['effort'])
    y = df['effort']
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_norm = pd.DataFrame(x_scaled, columns=x.columns)
    return df_norm, y

# Splitting data for training and testing
def train_test_split_fold(folds, counter, df, df_norm):
    # Ensure 'effort' column exists in the original df before accessing
    if 'effort' not in df.columns:
         raise KeyError("'effort' column not found in the DataFrame.")
    trainX = df_norm.iloc[folds[counter]['train'], :]
    trainY = df['effort'].iloc[folds[counter]['train']]
    testX = df_norm.iloc[folds[counter]['test'], :]
    testY = df['effort'].iloc[folds[counter]['test']]
    return trainX, trainY, testX, testY

# Define the DNN model with regularization
def create_dnn_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))  # Dropout for regularization
    model.add(BatchNormalization())  # Batch Normalization
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))  # Dropout for regularization
    model.add(BatchNormalization())  # Batch Normalization
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# Model Training and Evaluation
def run_dnn_model(df, df_norm):
    folds = {}
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    counter = 0
    for train_index, test_index in kf.split(df_norm):
        folds[counter] = {'train': train_index, 'test': test_index}
        counter += 1

    model = create_dnn_model(df_norm.shape[1])  # Create DNN model
    predictions = []
    actual = []

    for fold in range(10):
        trainX, trainY, testX, testY = train_test_split_fold(folds, fold, df, df_norm)
        model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=0)  # Train the model
        y_pred = model.predict(testX)

        predictions.extend(y_pred.flatten())  # Flatten for consistency
        actual.extend(testY)

    # Calculate Metrics
    # Ensure actual is not empty to avoid division by zero or other issues in metric calculations
    if not actual:
        print("Warning: No actual values collected during training/prediction. Skipping metric calculation.")
        return None, None, None, None # Return None for metrics if no data

    actual_np = np.array(actual)
    predictions_np = np.array(predictions)

    # Handle potential zeros in 'actual' for MMRE calculation
    # Replace zeros or near-zeros with a small value or filter them out
    non_zero_actual_indices = actual_np != 0
    if np.any(non_zero_actual_indices):
        mmre = np.mean(np.abs(actual_np[non_zero_actual_indices] - predictions_np[non_zero_actual_indices]) / actual_np[non_zero_actual_indices])
    else:
        mmre = np.nan # Or 0, depending on desired behavior if all actual are zero


    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)


    print(f"DNN RMSE: {rmse}")
    print(f"DNN MAE: {mae}")
    print(f"DNN MMRE: {mmre}")
    print(f"DNN R²: {r2}")

    return rmse, mae, mmre, r2

# Visualizing Results
def plot_results(model_name, metrics):
    # metrics contains [RMSE, MAE, MMRE, R²]
    metrics_names = ['RMSE', 'MAE', 'MMRE', 'R²']

    # Handle None or NaN values in metrics
    metrics = [metric if pd.notna(metric) and metric is not None else 0 for metric in metrics]  # Replace None/NaN with 0


    # Make sure there's valid data in metrics before plotting
    if len(metrics) == 4:
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_names, metrics, color=['orange', 'blue', 'green', 'red'])
        plt.title(f"{model_name} Performance Metrics")
        plt.xlabel("Metrics")
        plt.ylabel("Values")
        plt.show()
    else:
        print(f"Invalid metrics values for {model_name}, skipping plot.")


# Main Execution Function
def main():
    # Adjust path according to where your Desharnais dataset is in Google Drive
    DATASET_PATH = "/content/drive/My Drive/Software_Cost_Data/desharnais.csv"  # Adjust path as needed
    df = read_and_preprocess(DATASET_PATH)

    # Only proceed if the dataframe is not empty after preprocessing and has the 'effort' column
    if df is not None and not df.empty and 'effort' in df.columns:
        df_norm, y = normalize(df)

        # Run DNN Model
        rmse, mae, mmre, r2 = run_dnn_model(df, df_norm)

        # Plot Results - only plot if metrics were calculated
        if all(m is not None for m in [rmse, mae, mmre, r2]):
             plot_results("DNN", [rmse, mae, mmre, r2])
    else:
        print("DataFrame is empty after preprocessing or 'effort' column is missing. Cannot proceed with model training.")


if __name__ == '__main__':
    main()