__author__ = 'jennyyuejin'

from recsys.datamodel.item import Item
from recsys.datamodel.user import User
from recsys.datamodel.data import Data

from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

from recsys.algorithm.factorize import SVD, SVDNeighbourhood



# create an item
itemId = 0
item = Item(itemId)
item.add_data({'name': 'project0',
               'popularity': 0.5,
               'tags': [0, 0, 1]
               })

item2 = Item(1)
item2.add_data({'name': 'project1',
               'popularity': 0.5,
               'tags': [0, 0, 1]
               })

# create a user
userId = 0
user = User(userId)

# link an item with a user
rating = 1
user.add_item(itemId, rating)

data = Data()
data.add_tuple((rating, itemId, userId))
data.add_tuple((10, 1, 2))


svd = SVD()
svd.set_data(data)
svd.compute(k=100, min_values=0, pre_normalize=None, mean_center=True, post_normalize=True)

svd.similarity(0, 0)

l1 = ['a', 0, 1, 1]
l2 = ['b', 0, 1, 1]
print 1- spatial.distance.cosine(l1, l2)
cosine_similarity(l1, l2)