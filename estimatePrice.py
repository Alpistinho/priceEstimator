import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

    x = oneHotEncoder.fit_transform(x)

    polyFeat = PolynomialFeatures(degree=3)
    # xPoly = polyFeat.fit_transform(x.toarray())

    linearRegressor = LinearRegression()
    linearRegressor.fit(x, y)

    y_pred = linearRegressor.predict(x)


    # bairros = dataset['bairro'].values
    # tipos = dataset['tipo'].values
    # tiposVendedor = dataset['tipo_vendedor'].values

    # bairrosLabels = labelEncoder.fit_transform(bairros)
    # tiposLabels = labelEncoder.fit_transform(tipos)
    # tiposVendedorLabels = labelEncoder.fit_transform(tiposVendedor)
    
    # oheBairros = oneHotEncoder.fit_transform(bairrosLabels[:,np.newaxis])
    # oheTipos = oneHotEncoder.fit_transform(tiposLabels[:,np.newaxis])
    #oheTipos.toarray() to get the columns encoded


    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, 
    #     y, 
    #     test_size = 0.4 #,
    #     #random_state = 2018
    # )