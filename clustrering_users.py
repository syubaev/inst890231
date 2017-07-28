import numpy as np
import pandas as pd
from numba import jit, guvectorize, int64
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import Pipeline

from my_classes import DataSet
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from nltk.cluster.gaac import GAAClusterer

N_COMPONENTS = 200
N_TAKE_TOP = 15
N_CLUSTERS = 12
LAST_N_PRIORS = 5

dt = DataSet(20000)

class PipClassSVDTakeTop:
    def __init__(self, n_take_top):
        self.n_take_top = n_take_top

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return X[:, :self.n_take_top]

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('svd', TruncatedSVD(n_components=N_COMPONENTS, n_iter=7, random_state=42)),
    ('svd_top', PipClassSVDTakeTop(N_TAKE_TOP)),
    ('cluster', AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='cosine', linkage='average'))
])

# USER PRIOR N LAST ORDERS


print('Calc products merged str. LAST_N_PRIORS=', LAST_N_PRIORS)
users_prior = pd.DataFrame()
if LAST_N_PRIORS is not None:

    orders_numb_top = dt.orders[['user_id', 'order_number', 'order_id']].\
        sort_values(['user_id', 'order_number'], ascending=[1, 0]).\
        groupby('user_id').head(LAST_N_PRIORS)['order_id'].\
        values

    priors_filtered = dt.priors[dt.priors.order_id.isin(orders_numb_top)]
    users_prior['all_products'] = priors_filtered.groupby('user_id')['product_id'].apply(set)
else:
    users_prior['all_products'] = dt.priors.groupby('user_id')['product_id'].apply(set)

user_products = users_prior.all_products.apply(
    lambda x: " ".join([str(prod_id) for prod_id in x]))

clusters = pipeline.fit_predict(user_products)


ar_clust, ar_cnt = np.unique(clusters, return_counts=True)
max_clust = np.argmax(ar_cnt)
for cl, cnt in zip(ar_clust, ar_cnt):
    if cnt < 500:
        clusters[clusters == cl] = max_clust

# test the GAAC clusterer with 4 clusters
#clusterer = GAAClusterer(N_CLUSTERS, normalise=False)
#clusters = clusterer.cluster(X_svd, True)
pd.DataFrame({'user_id': user_products.index, 'cluster':clusters}).to_csv('../tmp/user_by_cluster.csv', index=False)
print('Done clustering', np.unique(clusters, return_counts=True))

#
#(1 / (|A||B|)) * SUMxA SUMyB d(x,y)
