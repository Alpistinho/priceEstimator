import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

import analysisPlotter

def rmspe(correct, prediction):
    length = np.min([len(correct),len(prediction)])
    totalError = 0
    errors = []

    for c, p in zip(correct, prediction):
        error = ((p - c)/p)**2
        totalError += error
        errors.append(error)
    totalError = np.sqrt(totalError/length)
    return totalError, np.array(errors)

if __name__ == '__main__':

    dataset = pd.read_csv('data/train.csv')
    
    # input data
    dataset = dataset.loc[:, dataset.columns != 'diferenciais']
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
            test_size = 0.4 #,
            #random_state = 2018
    )

    # polyFeat = PolynomialFeatures(degree=3)
    # xPoly = polyFeat.fit_transform(x.toarray())

    linearRegressor = LinearRegression()
    linearRegressor.fit(x_train, y_train)
    y_pred_train = linearRegressor.predict(x_train)
    y_pred_test = linearRegressor.predict(x_test)

    lr_ridge = Ridge(alpha=1000)
    lr_ridge.fit(x_train, y_train)
    y_pred_train_ridge = lr_ridge.predict(x_train)
    y_pred_test_ridge = lr_ridge.predict(x_test)

    rmspe_train, errors_train = rmspe(y_train, y_pred_train)
    rmspe_test, errors_test = rmspe(y_test , y_pred_test)

    rmspe_train_ridge, errors_train_ridge = rmspe(y_train, y_pred_train_ridge)
    rmspe_test_ridge, errors_test_ridge = rmspe(y_test , y_pred_test_ridge)

    print('LinearRegression')
    print(linearRegressor.coef_)
    print('\nDesempenho no conjunto de treinamento:')
    print('RMSPE = %.3f' % rmspe_train)

    print('\nDesempenho no conjunto de teste:')
    print('RMSPE = %.3f' % rmspe_test)

    print('LinearRegression with Rigde')
    print(lr_ridge.coef_)
    print('\nDesempenho no conjunto de treinamento:')
    print('RMSPE = %.3f' % rmspe_train_ridge)

    print('\nDesempenho no conjunto de teste:')
    print('RMSPE = %.3f' % rmspe_test_ridge)

    
    
    analysisPlotter.plotAscending(x_train[:,75], y_pred_train)
    analysisPlotter.plotAscending(x_train[:,75], y_pred_train_ridge)
    plt.show()