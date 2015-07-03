__author__ = 'jennyyuejin'

import sys

import recsys.algorithm
recsys.algorithm.VERBOSE = True

from recsys.algorithm.factorize import SVD, SVDNeighbourhood
from recsys.datamodel.data import Data
from recsys.evaluation.prediction import RMSE, MAE

from main import evaluate

if __name__ == '__main__':

    svd = SVDNeighbourhood()
    # svd = SVD()
    svd.load_model('./model/svd.obj.zip')

    #Dataset
    PERCENT_TRAIN = 100
    data = Data()
    data.load('/Users/jennyyuejin/recommender/Data/movieData/u.data',
              sep='\t',
              format={'col':0, 'row':1, 'value':2, 'ids':int})
    # About format parameter:
    #   'row': 1 -> Rows in matrix come from column 1 in ratings.dat file
    #   'col': 0 -> Cols in matrix come from column 0 in ratings.dat file
    #   'value': 2 -> Values (Mij) in matrix come from column 2 in ratings.dat file
    #   'ids': int -> Ids (row and col ids) are integers (not strings)

    #Train & Test data
    _, test = data.split_train_test(percent=PERCENT_TRAIN, shuffle_data=True)

    evaluate(svd, test)