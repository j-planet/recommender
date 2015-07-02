__author__ = 'jennyyuejin'

import sys

import recsys.algorithm
recsys.algorithm.VERBOSE = True

from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data
from recsys.evaluation.prediction import RMSE, MAE

# svd = pickle.load(open('model/svd.obj', 'r'))
svd = SVD()
svd.load_model('./model/svd.obj.zip')

#Dataset
PERCENT_TRAIN = 0
data = Data()
data.load('/Users/jennyyuejin/recommender/Data/u.data',
          sep='\t',
          format={'col':0, 'row':1, 'value':2, 'ids':int})
# About format parameter:
#   'row': 1 -> Rows in matrix come from column 1 in ratings.dat file
#   'col': 0 -> Cols in matrix come from column 0 in ratings.dat file
#   'value': 2 -> Values (Mij) in matrix come from column 2 in ratings.dat file
#   'ids': int -> Ids (row and col ids) are integers (not strings)

#Train & Test data
_, test = data.split_train_test(percent=PERCENT_TRAIN, shuffle_data=False)

#Evaluation using prediction-based metrics
rmse = RMSE()
mae = MAE()
for rating, item_id, user_id in test.get():
    try:
        pred_rating = svd.predict(item_id, user_id)
        rmse.add(rating, pred_rating)
        mae.add(rating, pred_rating)
    except KeyError:
        continue

print 'RMSE=%s' % rmse.compute()
print 'MAE=%s' % mae.compute()