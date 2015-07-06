import sys

import recsys.algorithm
recsys.algorithm.VERBOSE = True

from recsys.algorithm.factorize import SVD, SVDNeighbourhood
from recsys.datamodel.data import Data
from recsys.evaluation.prediction import RMSE, MAE


#Evaluation using prediction-based metrics
def evaluate(clf, _testData, verbose = False):

    rmse = RMSE()
    mae = MAE()
    numErrors = 0

    for rating, item_id, user_id in _testData.get():
        try:
            pred_rating = clf.predict(item_id, user_id)
            rmse.add(rating, pred_rating)
            mae.add(rating, pred_rating)

            if verbose:
                print item_id, user_id, rating, pred_rating
        except KeyError as e:
            if verbose:
                print 'ERROR occurred:', e.message
            numErrors += 1

    print '\n%i/%i data points raised errors.' % (numErrors, len(_testData))
    print 'RMSE=%s' % rmse.compute()
    print 'MAE=%s' % mae.compute()



def test_save_n_load(percent_train,
         modelKlass = SVD,
         dataFname ='/Users/jennyyuejin/recommender/Data/movieData/u.data',
         dataFormat = {'col':0, 'row':1, 'value':2, 'ids':int}):

    data = Data()
    data.load(dataFname, sep='\t', format=dataFormat)

    print '------ evaluating original'
    train, test = data.split_train_test(percent=percent_train, shuffle_data=False)
    print len(train), 'training data points;', len(test), 'testing data points'

    #Create SVD
    K=100
    svd = modelKlass()
    svd.set_data(train)
    svd.compute(k=K, min_values=5, pre_normalize=None, mean_center=True, post_normalize=True)
    evaluate(svd, test)

    svd.save_model('./model/svd.obj.zip',
                   {'k': K, 'min_values': 5,
                    'pre_normalize': None, 'mean_center': True, 'post_normalize': True})


    print '------ evaluating copy'
    data2 = Data()
    data2.load(dataFname, sep='\t', format=dataFormat)
    _, test2 = data2.split_train_test(percent=percent_train, shuffle_data=False)   # reload data
    print len(test2), 'testing data points'

    svd_pred = modelKlass()
    svd_pred.load_model('./model/svd.obj.zip')

    evaluate(svd_pred, test2)



if __name__ == '__main__':

    test_save_n_load(80)

