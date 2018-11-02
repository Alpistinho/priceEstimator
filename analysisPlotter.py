import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from scipy import stats

# Remove outliers using z-score
# Receives list of all data which shall be filtered
# Returns an index list
def removeOutliers(datum):

    intersection = range(len(datum[0]))
    for data in datum:
        z = np.abs(stats.zscore(data))
        idx = np.where(z < 3)
        intersection = np.intersect1d(intersection,idx)

    return intersection

def plotHistogram(data):
    fig1, ax1 = plt.subplots()
    idx = removeOutliers([data])
    data = data[idx]
    ax1.hist(data, bins='auto')

def plotAscending(x,y):
    fig1, ax1 = plt.subplots()
    idx = removeOutliers([x, y])
    x = x[idx]
    y = y[idx]
    order = x.argsort()
    ax1.plot(x[order],y[order])

if __name__ == '__main__':

    dataset = pd.read_csv('data/train.csv')
    prices = dataset['preco'].values
    areas = dataset['area_util'].values
    plotHistogram(prices)
    plotAscending(areas, prices)
    plt.show()