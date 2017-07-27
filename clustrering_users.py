import numpy as np
import pandas as pd
from my_classes import DataSet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.cluster.gaac import GAAClusterer

N_COMPONENTS = 200
N_TAKE_TOP = 15
N_CLUSTERS = 10
LAST_N_PRIORS = 4

dt = DataSet(2000)

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

print('Calc CountVectorizer')
user_products_cnt_vector = CountVectorizer().fit_transform(user_products)
user_products_cnt_vector = (user_products_cnt_vector > 0).astype(np.int8)

# TRY TruncatedSVD

svd = TruncatedSVD(n_components=N_COMPONENTS, n_iter=7, random_state=42)

X_svd = svd.fit_transform(user_products_cnt_vector)
X_svd = X_svd[:, :N_TAKE_TOP]

#clst = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='cosine', linkage='average')
#clusters = clst.fit_predict(X_C_svd[:40000, :])


# test the GAAC clusterer with 4 clusters
clusterer = GAAClusterer(N_CLUSTERS, normalise=False)
clusters = clusterer.cluster(X_svd, True)
pd.DataFrame({'user_id': user_products.index, 'cluster':clusters}).to_csv('../tmp/user_by_cluster.csv', index=False)