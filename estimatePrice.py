import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from sklearn.linear_model import Ridge

import analysisPlotter

def outputResult(outfile):

    with open(outfile , 'w') as f:
        f.write('Id,preco')

def rmspe(correct, prediction):
    length = np.min([len(correct),len(prediction)])
    totalError = 0
    errors = []

    for c, p in zip(correct, prediction):
        error = ((p - c)/p)**2
        totalError += error
        errors.append(error)
    totalError = np.sqrt(totalError/length)
    return totalError, errors
    
def printResults(results, print_coefs=False):
    for key in results.keys():
        print()
        print(key)
        if print_coefs:
            print(results[key]['coef'])
        print('\nDesempenho no conjunto de treinamento:')
        print('RMSPE = %.3f' % results[key]['rmspe_train'])

        print('\nDesempenho no conjunto de teste:')
        print('RMSPE = %.3f' % results[key]['rmspe_test'])

def kFoldCrossValidation(x, y, predictor, splits = 10):
    kf = KFold(n_splits=splits)
    kf.get_n_splits(x)
    i = 0
    results = {}

    train_error = 0
    test_error = 0
    for train_index, test_index in kf.split(x):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        predictor.fit(x_train, y_train)
        y_pred_train = predictor.predict(x_train)
        y_pred_test = predictor.predict(x_test)
        rmspe_train, errors_train = rmspe(y_train, y_pred_train)
        rmspe_test, errors_test = rmspe(y_test , y_pred_test)

        train_error += rmspe_train
        test_error += rmspe_test
        
        i = i + 1
        results['Fold {}'.format(i)] = {'rmspe_train': rmspe_train, 'rmspe_test': rmspe_test, 'coef': predictor.coef_}

    return results, train_error/splits, test_error/splits

if __name__ == '__main__':

    dataset = pd.read_csv('data/train.csv')
    
    # input data
    dataset = dataset.loc[:, dataset.columns != 'diferenciais']

    # Remove entries with outlier values on the preco column
    dataset = dataset[np.abs(dataset.preco - dataset.preco.mean()) <= (4*dataset.preco.std())]

    x = dataset.iloc[:,1:-1].values
    # output data
    y = dataset['preco'].values

    labelEncoder = LabelEncoder()
    oneHotEncoder = OneHotEncoder(categorical_features=[0,1,2])

    x[:,0] = labelEncoder.fit_transform(x[:,0])
    x[:,1] = labelEncoder.fit_transform(x[:,1])
    x[:,2] = labelEncoder.fit_transform(x[:,2])

    x = oneHotEncoder.fit_transform(x).toarray()


    x_train, x_test, y_train, y_test = train_test_split(
            x, 
            y, 
            test_size = 0.9 #,
            #random_state = 2018
    )

    # polyFeat = PolynomialFeatures(degree=3)
    # xPoly = polyFeat.fit_transform(x.toarray())

    linearRegressor = LinearRegression()
    linearRegressor.fit(x_train, y_train)
    y_pred_train = linearRegressor.predict(x_train)
    y_pred_test = linearRegressor.predict(x_test)

    lr_ridge = Ridge(alpha=10000)
    lr_ridge.fit(x_train, y_train)
    y_pred_train_ridge = lr_ridge.predict(x_train)
    y_pred_test_ridge = lr_ridge.predict(x_test)


    results = {}

    rmspe_train, errors_train = rmspe(y_train, y_pred_train)
    rmspe_test, errors_test = rmspe(y_test , y_pred_test)

    rmspe_train_ridge, errors_train_ridge = rmspe(y_train, y_pred_train_ridge)
    rmspe_test_ridge, errors_test_ridge = rmspe(y_test , y_pred_test_ridge)

    results['LinearRegression'] = {'rmspe_train': rmspe_train, 'rmspe_test': rmspe_test, 'coef': linearRegressor.coef_}
    results['LinearRegression with Rigde'] = {'rmspe_train': rmspe_train_ridge, 'rmspe_test': rmspe_test_ridge, 'coef': lr_ridge.coef_}

    printResults(results)


    import heapq

    largest = heapq.nlargest(6, errors_train_ridge)
    for large in largest:
        idx = errors_train_ridge.index(large)
        print(idx, y_train[idx])
    # analysisPlotter.plotHistogram(errors_train_ridge)

    
    result, train_error, test_error = kFoldCrossValidation(x,y,lr_ridge)
    # printResults(result)
    print(train_error, test_error)

    result, train_error, test_error = kFoldCrossValidation(x,y,linearRegressor)
    # printResults(result)
    print(train_error, test_error)

    analysisPlotter.plotAscending(x_train[:,75], y_pred_train)
    analysisPlotter.plotAscending(x_train[:,75], y_pred_train_ridge)
    analysisPlotter.plotAscending(y_pred_train_ridge, np.array(errors_train_ridge))
    plt.show()
    