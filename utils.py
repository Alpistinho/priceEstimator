import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold, RFE

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
	kf = KFold(n_splits=splits, shuffle=False)
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
		try:
			results['Fold {}'.format(i)] = {'rmspe_train': rmspe_train, 'rmspe_test': rmspe_test, 'coef': predictor.coef_}
		except AttributeError:
			results['Fold {}'.format(i)] = {'rmspe_train': rmspe_train, 'rmspe_test': rmspe_test}
	return results, train_error/i, test_error/i

def selectFeatures(feature_selector, x_train, y_train, x_test, complete_dataset):

	x_train = feature_selector.fit_transform(x_train, y_train)
	x_test = feature_selector.transform(x_test)

	x = complete_dataset.loc[:, complete_dataset.columns != 'Id']
	x = feature_selector.transform(x)

	mask = feature_selector.get_support()
	columns = complete_dataset.loc[:, complete_dataset.columns != 'Id'].columns
	complete_dataset = pd.DataFrame(x, columns=columns[mask])

	return x_train, x_test, x, complete_dataset

def setDatasets(train_dataset, test_dataset=None, remove_outliers=True, k_features=None, minimum_variance=None):
	pd.options.mode.chained_assignment = None

	train_dataset = train_dataset.loc[:, train_dataset.columns != 'diferenciais']

	# Remove entries with outlier values on the preco column
	if remove_outliers:
		train_dataset = train_dataset[np.abs(train_dataset.preco - train_dataset.preco.mean()) <= (2*train_dataset.preco.std())]

	# output data
	y = train_dataset['preco'].values

	# Remove columns not used in training
	train_dataset.drop(['preco'], axis=1, inplace=True)
	
	if test_dataset is None:
		complete_dataset = train_dataset
	else:
		test_dataset = test_dataset.loc[:, test_dataset.columns != 'diferenciais']
		complete_dataset = train_dataset.append(test_dataset)
		

	tipoCategories = complete_dataset.tipo.unique()
	bairroCategories = complete_dataset.bairro.unique()
	tipoVendedorCategories = complete_dataset.tipo_vendedor.unique()

	complete_dataset['tipo'] = complete_dataset.tipo.astype(pd.api.types.CategoricalDtype(categories=tipoCategories))
	complete_dataset['bairro'] = complete_dataset.bairro.astype(pd.api.types.CategoricalDtype(categories=bairroCategories))
	complete_dataset['tipo_vendedor'] = complete_dataset.tipo_vendedor.astype(pd.api.types.CategoricalDtype(categories=tipoVendedorCategories))

	# Do one hot encoding on the categorical columns
	complete_dataset = pd.get_dummies(complete_dataset)

	if test_dataset is None:
		train_dataset['tipo'] = train_dataset.tipo.astype(pd.api.types.CategoricalDtype(categories=tipoCategories))
		train_dataset['bairro'] = train_dataset.bairro.astype(pd.api.types.CategoricalDtype(categories=bairroCategories))
		train_dataset['tipo_vendedor'] = train_dataset.tipo_vendedor.astype(pd.api.types.CategoricalDtype(categories=tipoVendedorCategories))
		
		# Do one hot encoding on the categorical columns
		train_dataset = pd.get_dummies(train_dataset)
		
		# input data
		x = train_dataset.loc[:, complete_dataset.columns != 'Id'].values

		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.9)
		
	else:
		train_dataset['tipo'] = train_dataset.tipo.astype(pd.api.types.CategoricalDtype(categories=tipoCategories))
		train_dataset['bairro'] = train_dataset.bairro.astype(pd.api.types.CategoricalDtype(categories=bairroCategories))
		train_dataset['tipo_vendedor'] = train_dataset.tipo_vendedor.astype(pd.api.types.CategoricalDtype(categories=tipoVendedorCategories))

		test_dataset['tipo'] = test_dataset.tipo.astype(pd.api.types.CategoricalDtype(categories=tipoCategories))
		test_dataset['bairro'] = test_dataset.bairro.astype(pd.api.types.CategoricalDtype(categories=bairroCategories))
		test_dataset['tipo_vendedor'] = test_dataset.tipo_vendedor.astype(pd.api.types.CategoricalDtype(categories=tipoVendedorCategories))

		# Do one hot encoding on the categorical columns
		train_dataset = pd.get_dummies(train_dataset)
		test_dataset = pd.get_dummies(test_dataset)

		# input data
		x_train = train_dataset.iloc[:,1:]
		x_test = test_dataset.iloc[:,1:]
		x = x_train

		y_train = y

	if minimum_variance is not None:
		varianceThereshold = VarianceThreshold(threshold=minimum_variance)
		x_train, x_test, x, complete_dataset = selectFeatures(varianceThereshold, x_train, y_train, x_test, complete_dataset)
	if k_features is not None:
		selectKBest = SelectKBest(score_func=f_regression, k=k_features)
		x_train, x_test, x, complete_dataset = selectFeatures(selectKBest, x_train, y_train, x_test, complete_dataset)

	try:
		return x, y, x_train, x_test, y_train, y_test, complete_dataset
	except UnboundLocalError:
		return x_train, x_test, y, complete_dataset