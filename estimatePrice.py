import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import analysisPlotter

def outputResult(outfile, results):

	with open(outfile , 'w') as f:
		f.write('Id,preco\n')
		lines = []
		Id = 0
		for result in results:
			lines.append(','.join([str(Id), str(result)]) + '\n')
			Id += 1
		f.writelines(lines)

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
		try:
			print('\nDesempenho no conjunto de teste:')
			print('RMSPE = %.3f' % results[key]['rmspe_test'])
		except KeyError:
			pass

def kFoldCrossValidation(x, y, predictor, splits = 10):
	kf = KFold(n_splits=splits, shuffle=True)
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

def setDatasets(train_dataset, test_dataset=None, remove_outliers=True):
	pd.options.mode.chained_assignment = None

	train_dataset = train_dataset.loc[:, train_dataset.columns != 'diferenciais']

	# Remove entries with outlier values on the preco column
	if remove_outliers:
		train_dataset = train_dataset[np.abs(train_dataset.preco - train_dataset.preco.mean()) <= (2*train_dataset.preco.std())]

	# output data
	y = train_dataset['preco'].values
	# Remove columns not used in training
	x_dataset = train_dataset.drop(['preco'], axis=1, inplace=False)
	
	if test_dataset is not None:
		test_dataset = test_dataset.loc[:, test_dataset.columns != 'diferenciais']
		complete_dataset = x_dataset.append(test_dataset)
	else:
		complete_dataset = x_dataset

	tipoCategories = complete_dataset.tipo.unique()
	bairroCategories = complete_dataset.bairro.unique()
	tipoVendedorCategories = complete_dataset.tipo_vendedor.unique()

	
	# Do one hot encoding on the categorical columns
	x_dataset['tipo'] = x_dataset.tipo.astype(pd.api.types.CategoricalDtype(categories=tipoCategories))
	x_dataset['bairro'] = x_dataset.bairro.astype(pd.api.types.CategoricalDtype(categories=bairroCategories))
	x_dataset['tipo_vendedor'] = x_dataset.tipo_vendedor.astype(pd.api.types.CategoricalDtype(categories=tipoVendedorCategories))

	x_dataset = pd.get_dummies(x_dataset)
	# input data
	x = x_dataset.iloc[:, 1:].values


	if test_dataset is not None:
		
		test_dataset['tipo'] = test_dataset.tipo.astype(pd.api.types.CategoricalDtype(categories=tipoCategories))
		test_dataset['bairro'] = test_dataset.bairro.astype(pd.api.types.CategoricalDtype(categories=bairroCategories))
		test_dataset['tipo_vendedor'] = test_dataset.tipo_vendedor.astype(pd.api.types.CategoricalDtype(categories=tipoVendedorCategories))
		
		test_dataset = pd.get_dummies(test_dataset)

		x_test = test_dataset.iloc[:,1:].values

		return x, x_test, y

	else:
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.9)
		return x, y, x_train, x_test, y_train, y_test

if __name__ == '__main__':
	print(sys.argv)
	if len(sys.argv) == 1: 
		print('Using default dataset')
		dataset = pd.read_csv('data/train.csv')
		x, y, x_train, x_test, y_train, y_test = setDatasets(dataset)
	elif len(sys.argv) == 2: 
		print('Using dataset {}'.format(sys.argv[1]))
		dataset = pd.read_csv(sys.argv[1])
		x, y, x_train, x_test, y_train, y_test = setDatasets(dataset)
	elif len(sys.argv) == 3: 
		print('Using dataset {} for training'.format(sys.argv[1]))
		print('Using dataset {} for testing'.format(sys.argv[2]))
		dataset = pd.read_csv(sys.argv[1])
		test_dataset = pd.read_csv(sys.argv[2])
		x_train, x_test, y_train = setDatasets(dataset, test_dataset)
	else:
		print('Wrong number of arguments')
		exit()

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

	lr_ridge = Ridge(alpha=200)
	lr_ridge.fit(x_train, y_train)
	y_pred_train_ridge = lr_ridge.predict(x_train)
	y_pred_test_ridge = lr_ridge.predict(x_test)

	# analysisPlotter.plotAscending(np.arange(y_train, y_train)
	order = y_train.argsort()
	plt.plot(np.arange(y_train.shape[0]), y_train[order], np.arange(y_train.shape[0]), y_pred_train_ridge[order])

	percents = np.abs(y_pred_train_ridge[order] - y_train[order])/y_train[order]

	fig1, ax1 = plt.subplots()
	ax1.semilogx(y_train[order],percents[order])


	rmspe_train, errors_train = rmspe(y_train, y_pred_train)
	rmspe_train_ridge, errors_train_ridge = rmspe(y_train, y_pred_train_ridge)
	
	results = {}
	results['LinearRegression'] = {'rmspe_train': rmspe_train, 'coef': linearRegressor.coef_}
	results['LinearRegression with Rigde'] = {'rmspe_train': rmspe_train_ridge, 'coef': lr_ridge.coef_}

	if len(sys.argv) == 3:
		from time import gmtime, strftime
		now = strftime("%Y%m%d%H%M%S", gmtime())
		print('Outputing submission data')
		outputResult('submissions/submission{}.csv'.format(now), y_pred_test_ridge)
	else: 

		rmspe_test, errors_test = rmspe(y_test , y_pred_test)
		rmspe_test_ridge, errors_test_ridge = rmspe(y_test , y_pred_test_ridge)

		results['LinearRegression'] = {'rmspe_train': rmspe_train, 'rmspe_test': rmspe_test, 'coef': linearRegressor.coef_}
		results['LinearRegression with Rigde'] = {'rmspe_train': rmspe_train_ridge, 'rmspe_test': rmspe_test_ridge, 'coef': lr_ridge.coef_}

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