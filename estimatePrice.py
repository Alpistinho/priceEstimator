import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy import stats
from sklearn import svm, tree
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, BayesianRidge, RANSACRegressor
from sklearn.model_selection import KFold, KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFECV


import analysisPlotter
from utils import *


k_features = None
correct_preco = True
if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) == 1: 
		print('Using default dataset (data/train.csv) for training and cross validation')
		dataset = pd.read_csv('data/train.csv')
		x, y, x_train, x_test, y_train, y_test, filtered_dataset = setDatasets(dataset, k_features=k_features, minimum_variance=None, log_correct_preco=correct_preco)
	elif len(sys.argv) == 2: 
		print('Using dataset {} for training and cross validation'.format(sys.argv[1]))
		dataset = pd.read_csv(sys.argv[1])
		x, y, x_train, x_test, y_train, y_test, filtered_dataset = setDatasets(dataset, k_features=k_features, log_correct_preco=correct_preco)
	elif len(sys.argv) == 3: 
		print('Using dataset {} for training'.format(sys.argv[1]))
		print('Using dataset {} for testing'.format(sys.argv[2]))
		dataset = pd.read_csv(sys.argv[1])
		test_dataset = pd.read_csv(sys.argv[2])
		x_train, x_test, y_train, filtered_dataset = setDatasets(dataset, test_dataset, k_features=k_features, log_correct_preco=correct_preco)
	else:
		print('Wrong number of arguments')
		print('Correct usage:')
		print('No arguments: Use data/train.csv for training and crossvalidation')
		print('One argument: Use dataset specified for training and crossvalidation')
		print('Two arguments: First dataset is used for training while the second is used for producing the submission. Those are saved to the submissions directory')
		exit()

	print('Using {} features from dataset'.format(','.join([column for column in filtered_dataset.columns])))

	# print(x_train.shape)
	# polyFeat = PolynomialFeatures(degree=2)
	# x_train = polyFeat.fit_transform(x_train)
	# x_test = polyFeat.transform(x_test)
	# print(x_train.shape)
	
	# regressor = Ridge(alpha=10000)
	# regressor = svm.SVR(gamma='scale')
	regressor = RandomForestRegressor(n_estimators=10)
	# regressor = BayesianRidge(n_iter=300, lambda_1=1, lambda_2=1)
	rfecv = RFECV(estimator=regressor, step=1, cv=KFold())
	x_train = rfecv.fit_transform(x_train, y_train)
	x_test = rfecv.transform(x_test)
	print("Optimal number of features : %d" % rfecv.n_features_)

	# Plot number of features VS. cross-validation scores
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.show()

	# scaler = StandardScaler()
	# print(x_train[:10,3:5])
	# # analysisPlotter.plotHistogram(x_train[:,3:5])
	# scaler.fit(x_train[:,3:5])
	# print(scaler.transform(x_train[:,3:5]))

	# analysisPlotter.plotHistogram(scaler.transform(x_train[:,3:5])[:,0])
	# plt.show()

	regressor.fit(x_train, y_train)
	y_pred_train = regressor.predict(x_train)
	y_pred_test_ridge = regressor.predict(x_test)


	rmspe_train, errors_train = rmspe(y_train, y_pred_train)
	
	results = {}
	try:
		results['Regression'] = {'rmspe_train': rmspe_train, 'coef': regressor.coef_}
	except AttributeError:
		results['Regression'] = {'rmspe_train': rmspe_train}

	print('Cross validation using train/test separation from same dataset:')
	printResults(results)

	if len(sys.argv) == 3:
		from time import gmtime, strftime
		now = strftime("%Y%m%d%H%M%S", gmtime())
		print('Outputing submission data')
		outputResult('submissions/submission{}.csv'.format(now), y_pred_test_ridge)
	else: 
		rmspe_test, errors_test_ridge = rmspe(y_test , y_pred_test_ridge)

		try:
			results['Regression'] = {'rmspe_train': rmspe_train, 'rmspe_test': rmspe_test, 'coef': regressor.coef_}
		except AttributeError:
			results['Regression'] = {'rmspe_train': rmspe_train, 'rmspe_test': rmspe_test}

		result, train_error, test_error = kFoldCrossValidation(x,y,regressor,splits=10)
		print('kFold cross validation with {} folds'.format(10))
		print(train_error, test_error)

	analysisPlotter.plotAscending(y_train, np.array(errors_train))
	plt.show()