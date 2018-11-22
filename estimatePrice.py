import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy import stats
from sklearn import svm, tree
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, BayesianRidge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor

import analysisPlotter
from utils import *


k_features = 20

if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) == 1: 
		print('Using default dataset')
		dataset = pd.read_csv('data/train.csv')
		x, y, x_train, x_test, y_train, y_test, filtered_dataset = setDatasets(dataset, k_features=20, minimum_variance=None)
	elif len(sys.argv) == 2: 
		print('Using dataset {}'.format(sys.argv[1]))
		dataset = pd.read_csv(sys.argv[1])
		x, y, x_train, x_test, y_train, y_test, filtered_dataset = setDatasets(dataset, k_features=k_features)
	elif len(sys.argv) == 3: 
		print('Using dataset {} for training'.format(sys.argv[1]))
		print('Using dataset {} for testing'.format(sys.argv[2]))
		dataset = pd.read_csv(sys.argv[1])
		test_dataset = pd.read_csv(sys.argv[2])
		x_train, x_test, y_train, filtered_dataset = setDatasets(dataset, test_dataset, k_features=k_features)
	else:
		print('Wrong number of arguments')
		exit()

	print('Using {} features from dataset'.format(filtered_dataset.columns))
	# polyFeat = PolynomialFeatures(degree=3)
	# xPoly = polyFeat.fit_transform(x.toarray())

	# scaler = StandardScaler()
	# print(x_train[:10,3:5])
	# # analysisPlotter.plotHistogram(x_train[:,3:5])
	# scaler.fit(x_train[:,3:5])
	# print(scaler.transform(x_train[:,3:5]))

	# analysisPlotter.plotHistogram(scaler.transform(x_train[:,3:5])[:,0])
	# plt.show()

	linearRegressor = LinearRegression()
	
	

	linearRegressor.fit(x_train, y_train)
	y_pred_train = linearRegressor.predict(x_train)
	y_pred_test = linearRegressor.predict(x_test)

	# lr_ridge = Ridge(alpha=10000)
	# lr_ridge = svm.SVR(gamma='scale')
	lr_ridge = RandomForestRegressor()
	# lr_ridge = BayesianRidge(n_iter=300, lambda_1=1, lambda_2=1)
	lr_ridge.fit(x_train, y_train)
	y_pred_train_ridge = lr_ridge.predict(x_train)
	y_pred_test_ridge = lr_ridge.predict(x_test)

	# analysisPlotter.plotAscending(np.arange(y_train, y_train)
	# order = y_train.argsort()
	# plt.plot(np.arange(y_train.shape[0]), y_train[order], np.arange(y_train.shape[0]), y_pred_train_ridge[order])

	# percents = np.abs(y_pred_train_ridge[order] - y_train[order])/y_train[order]

	# fig1, ax1 = plt.subplots()
	# ax1.semilogx(y_train[order],percents[order])


	rmspe_train, errors_train = rmspe(y_train, y_pred_train)
	rmspe_train_ridge, errors_train_ridge = rmspe(y_train, y_pred_train_ridge)
	
	results = {}
	results['LinearRegression'] = {'rmspe_train': rmspe_train, 'coef': linearRegressor.coef_}
	try:
		results['LinearRegression with Rigde'] = {'rmspe_train': rmspe_train_ridge, 'coef': lr_ridge.coef_}
	except AttributeError:
		results['LinearRegression with Rigde'] = {'rmspe_train': rmspe_train_ridge}

	if len(sys.argv) == 3:
		from time import gmtime, strftime
		now = strftime("%Y%m%d%H%M%S", gmtime())
		print('Outputing submission data')
		outputResult('submissions/submission{}.csv'.format(now), y_pred_test_ridge)
	else: 

		rmspe_test, errors_test = rmspe(y_test , y_pred_test)
		rmspe_test_ridge, errors_test_ridge = rmspe(y_test , y_pred_test_ridge)

		results['LinearRegression'] = {'rmspe_train': rmspe_train, 'rmspe_test': rmspe_test, 'coef': linearRegressor.coef_}
		try:
			results['LinearRegression with Rigde'] = {'rmspe_train': rmspe_train_ridge, 'rmspe_test': rmspe_test_ridge, 'coef': lr_ridge.coef_}
		except AttributeError:
			results['LinearRegression with Rigde'] = {'rmspe_train': rmspe_train_ridge, 'rmspe_test': rmspe_test_ridge}

		result, train_error, test_error = kFoldCrossValidation(x,y,lr_ridge,splits=10)
		print('kFold cross validation with {} folds (Ridge)'.format(10))
		print(train_error, test_error)

		result, train_error, test_error = kFoldCrossValidation(x,y,linearRegressor,splits=10)
		print('kFold cross validation with {} folds (Linear)'.format(10))
		print(train_error, test_error)

	print('Cross validation using train/test separation from same dataset:')
	printResults(results)
	# analysisPlotter.plotAscending(x_train[:,75], y_pred_train)
	# analysisPlotter.plotAscending(x_train[:,75], y_pred_train_ridge)
	# analysisPlotter.plotAscending(y_pred_train_ridge, np.array(errors_train_ridge))
	plt.show()

	# values = []
	# for alpha in np.logspace(0, 4, num = 25):
	# 	lr_ridge = Ridge(alpha=alpha)
	# 	lr_ridge.fit(x_train, y_train)
	# 	y_pred_train_ridge = lr_ridge.predict(x_train)
	# 	y_pred_test_ridge = lr_ridge.predict(x_test)
	# 	_, train_error, test_error = kFoldCrossValidation(x,y,linearRegressor,splits=10)
	# 	values.append([alpha, train_error, test_error])

	# values = np.array(values)
	# plt.plot(values[:,0], values[:,1],values[:,0], values[:,2])
	# plt.show()